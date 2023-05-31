from joblib import Memory
import backoff

from pathlib import Path
import openai
import openai.error
import os
import argparse
import json
import tiktoken
from copy import deepcopy
import datasets
import glob
from tqdm import tqdm
from sklearn.metrics import classification_report

from src.evaluate_icsr_extraction import evaluate_icsr
from src import Icsr


# Set up your OpenAI API credentials
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set up caching
cachedir = os.environ.get("DSP_CACHEDIR") or os.path.join(
    Path.home(), "cachedir_joblib"
)
CacheMemory = Memory(location=cachedir, verbose=0)

# Set model length
model_to_length = {"gpt-4": 8192, "gpt-3.5-turbo": 4096}

# Backoff
def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


# Generate a response with caching and backoff
@CacheMemory.cache
def _generate_response(messages, **kwargs):
    if messages[-1]["role"] != "user":
        raise ValueError("Last message should always be a user message.")

    response = openai.ChatCompletion.create(messages=messages, **kwargs)

    return response


@backoff.on_exception(
    backoff.expo,
    (openai.error.RateLimitError, openai.error.ServiceUnavailableError),
    max_time=1000,
    on_backoff=backoff_hdlr,
)
def generate_response(messages, **kwargs):
    return _generate_response(messages, **kwargs)


# Truncate a conversation
def _truncate_conversation(system_message, user_messages, enc, **kwargs):
    max_prompt_length = model_to_length[kwargs["model"]]
    max_tokens = kwargs["max_tokens"]

    # get the total max length of our inputs
    buffer = (max_tokens * len(user_messages)) + (2 * len(user_messages) + 1)
    max_input_length = max_prompt_length - buffer

    # get the actual length of our inputs
    inputs = [system_message] + user_messages
    inputs_enc = [enc.encode(m["content"]) for m in inputs]
    inputs_len = [len(e) for e in inputs_enc]
    total_inputs_len = sum(inputs_len)

    extra_tokens = total_inputs_len - max_input_length

    # print('total_inputs_len: ', total_inputs_len)
    # print('max_input_length: ', max_input_length)
    # print('extra_tokens: ', extra_tokens)

    if extra_tokens > 0:
        # always truncate the first message, it contains the paper
        new_first_message = {
            "role": "user",
            "content": enc.decode(inputs_enc[1][:-extra_tokens]),
        }
        return system_message, [new_first_message, *user_messages[1:]]
    else:
        return system_message, user_messages


# Generate a conversation
def generate_conversation(system_message, user_messages, enc, **kwargs):
    # truncate the conversation
    system_message, user_messages = _truncate_conversation(
        system_message, user_messages, enc, **kwargs
    )

    # run the conversation
    conversation_messages = [system_message]
    responses = []
    for user_message in user_messages:
        conversation_messages.append(user_message)

        response = generate_response(conversation_messages, **kwargs)
        responses.append(response)

        conversation_messages.append(dict(response["choices"][0]["message"]))

    return conversation_messages, responses


# Parse a conversation into a canonical ICSR string
def conversation_to_icsr(conversation):
    assistant_messages = [m for m in conversation if m["role"] == "assistant"]

    serious = "2" if "not serious" in assistant_messages[0]["content"] else "1"
    patientsex = "2" if "female" in assistant_messages[0]["content"] else "1"
    drugs = assistant_messages[1]["content"]
    reactions = assistant_messages[2]["content"]

    # format strings
    drugs = drugs.upper()
    reactions = reactions.split(", ")
    reactions = (", ").join([r.capitalize() for r in reactions])

    return f"""serious: {serious}\npatientsex: {patientsex}\ndrugs: {drugs}\nreactions: {reactions}"""


def get_messages(text_input):
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. You read biomedical texts and concisely answer user questions about adverse drug events. You give the most specific answer supported by the text.",
    }

    user_messages = [
        {
            "role": "user",
            "content": f"Answer the question given the text. \n\nQuestion: This text describes an adverse drug event with regard to a patient or cohort. What is the sex of the patient? Was the adverse event serious (the adverse event resulted in death, a life threatening condition, hospitalization, disability, congenital anomaly, or other serious condition).Produce an answer in the following format: 'The patient is a {{male|female}} and the adverse event is {{serious|not serious}}'. If no weight or sex values can be identified, fill in 'N/A'.\n\n{text_input}",
        },
        {
            "role": "user",
            "content": "Give an alphabetized list of all active substances of drugs taken by the patient who experienced an adverse drug reaction, that could have caused this reaction. For every drug, give the most specific active substance that is supported by the text. Answer only with a comma-separated list. Do not include generic drug classes.",
        },
        {
            "role": "user",
            "content": "Give an alphabetized list of all adverse reactions the patient experienced. Answer with the MedDRA preferred term. For every reaction, give the most specific reaction that is supported by the text. Answer only with a comma-separated list.",
        },
    ]
    return system_message, user_messages


def get_run_number(output_dir):
    run_dirs = glob.glob(os.path.join(output_dir, "run*"))
    runs = [os.path.split(run)[-1].strip("run") for run in run_dirs]
    if runs:
        max_run = max(runs)
    else:
        max_run = "0"
    next_run = int(max_run) + 1
    return next_run


def run(
    max_dev_samples, output_dir, model_name, temperature, top_p, n, stop, max_tokens
):
    enc = tiktoken.encoding_for_model(model_name)

    # load data
    split = "validation"
    icsr_dataset = datasets.load_dataset("BioDEX/BioDEX-ICSR")[split]
    icsr_dataset = icsr_dataset[:max_dev_samples]

    inputs = icsr_dataset["fulltext_processed"]
    targets = icsr_dataset["target"]

    # run the model
    model_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "max_tokens": max_tokens,
    }
    conversations = []
    responses = []
    for text_input, _ in tqdm(zip(inputs, targets), total=len(inputs)):
        system_message, user_messages = get_messages(text_input)
        conversation, respons = generate_conversation(
            system_message, user_messages, enc, **model_kwargs
        )
        conversations.append(conversation)
        responses.append(respons)

    conversations_icsr = [conversation_to_icsr(c) for c in conversations]

    # metrics
    (precision, recall, f1), fails = evaluate_icsr(conversations_icsr, targets)
    parse_percent = 100 * (1 - (fails / len(conversations_icsr)))

    metrics = {
        "predict_icsr_precision": precision,
        "predict_icsr_recall": recall,
        "predict_icsr_f1": f1,
        "predict_icsr_parse_percent": parse_percent,
    }

    # experiment with per-attribute metrics
    # TODO: work this into the ICSR code later
    def get_set_precision_and_recalls(l1, l2):
        s1 = set(l1)
        s2 = set(l2)

        intersect = s1.intersection(s2)

        p = len(intersect) / len(s1)
        r = len(intersect) / len(s2)

        return p, r

    pred_icsrs = [Icsr.from_string(pred) for pred in conversations_icsr]
    target_icsrs = [Icsr.from_string(target) for target in targets]

    drug_metrics = []
    reaction_metrics = []
    for pred_icsr, target_icsr in zip(pred_icsrs, target_icsrs):

        drug_metrics.append(
            get_set_precision_and_recalls(pred_icsr.drugs, target_icsr.drugs)
        )
        reaction_metrics.append(
            get_set_precision_and_recalls(pred_icsr.reactions, target_icsr.reactions)
        )

    pred_patientsex = [i.patientsex for i in pred_icsrs]
    target_patientsex = [i.patientsex for i in target_icsrs]
    pred_serious = [i.serious for i in pred_icsrs]
    target_serious = [i.serious for i in target_icsrs]

    patientsex_report = classification_report(
        target_patientsex, pred_patientsex, output_dict=True
    )
    patientsex_precision = patientsex_report["weighted avg"]["precision"]
    patientsex_recall = patientsex_report["weighted avg"]["recall"]
    serious_report = classification_report(
        target_serious, pred_serious, output_dict=True
    )
    serious_precision = serious_report["weighted avg"]["precision"]
    serious_recall = serious_report["weighted avg"]["recall"]

    drug_precision = sum([m[0] for m in drug_metrics]) / len(targets)
    drug_recall = sum([m[1] for m in drug_metrics]) / len(targets)
    reaction_precision = sum([m[0] for m in reaction_metrics]) / len(targets)
    reaction_recall = sum([m[1] for m in reaction_metrics]) / len(targets)

    metrics.update(
        {
            "predict_drug_precision": drug_precision,
            "predict_drug_recall": drug_recall,
            "predict_reaction_precision": reaction_precision,
            "predict_reaction_recall": reaction_recall,
            "predict_patientsex_precision": patientsex_precision,
            "predict_patientsex_recall": patientsex_recall,
            "predict_serious_precision": serious_precision,
            "predict_serious_recall": serious_recall,
        }
    )

    print("Results:")
    for k, v in metrics.items():
        print("{:<30} {:<15}".format(k, v))

    # Get ouput dir
    output_dir = os.path.join(output_dir, model_name)
    next_run_number = get_run_number(output_dir)
    output_dir = os.path.join(output_dir, f"run{next_run_number:02d}")

    # Save predictions
    print("Logging to directory:")
    print(output_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "generated_eval_predictions.txt"), "w") as fp:
        for pred in conversations_icsr:
            pred = pred.replace("\n", " ").strip()
            fp.write(pred + "\n")

    with open(os.path.join(output_dir, "eval_labels.txt"), "w") as fp:
        for target in targets:
            target = target.replace("\n", " ").strip()
            fp.write(target + "\n")

    # Save metrics
    with open(os.path.join(output_dir, "eval_results.json"), "w") as fp:
        json.dump(metrics, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GPT3.5/GPT4 on BioDEX in a conversational manner."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="The maximum number of tokens to generate with the LM.",
    )
    parser.add_argument(
        "--max_dev_samples",
        type=int,
        default=None,
        help="The maximum number of samples to use from the development set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the output should be stored.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the LM to use.",
    )

    args = parser.parse_args()
    print("Arguments:")
    print(json.dumps(vars(args), indent=4))

    # set the parameters
    kwargs = {
        "max_dev_samples": args.max_dev_samples,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "temperature": 0.7,
        "top_p": 1.0,
        "n": 1,
        "stop": None,
        "max_tokens": args.max_tokens,
    }

    run(**kwargs)
