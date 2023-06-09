import os
import argparse
import datasets
from tqdm import tqdm
from typing import List
from src import Icsr
import dsp
import glob
import json
import tiktoken

from src.evaluate_icsr_extraction import evaluate_icsr


def load_data(validation_split: str, max_dev_samples: int, fulltext: bool) -> tuple:
    dataset = datasets.load_dataset("BioDEX/BioDEX-ICSR")

    question = "What adverse drug event was described in the following context?"
    train = [
        dsp.Example(
            question=question,
            context=process_text(example["abstract"]),
            answer=process_text(example["target"]),
        )
        for example in dataset["train"]
    ]
    dev = [
        dsp.Example(
            question=question,
            context=process_text(
                example["abstract"] if not fulltext else example["fulltext_processed"]
            ),
            answer=process_text(example["target"]),
        )
        for example in dataset[validation_split]
    ]
    if max_dev_samples:
        dev = dev[:max_dev_samples]
    return train, dev


def process_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("<strong>", "")
    text = text.replace("</strong>", "")
    text = text.lower()
    text = text.strip()
    return text


def vanilla_LM_QA(
    question: str,
    context: str,
    train: List[dsp.Example],
    n_demos: int,
    model_name: str,
    max_prompt_length: int,
    max_gen_length: int,
) -> str:
    demos = dsp.sample(train, k=n_demos)
    example = dsp.Example(question=question, context=context, demos=demos)
    qa_template = dsp.Template(
        instructions="Read a biomedical paper and extract information about the adverse drug event mentioned by the authors. Return a serious value ('1' for serious, '2' for not serious). Return a patientsex value ('1' for male, '2' for female). Return a list of drugs taken and reactions experienced.",
        question=dsp.Type(prefix="Question:", desc=f"${{{question}}}"),
        context=dsp.Type(
            prefix="Context:",
            desc="${biomedical paper that describes adverse drug events}",
            format=dsp.passages2text,
        ),
        answer=dsp.Type(
            prefix="Answer:",
            desc="${the adverse drug event described in the context}",
            format=dsp.format_answers,
        ),
    )
    example, completions = dsp.generate(
        qa_template, model_name, max_prompt_length, max_gen_length
    )(example, stage="qa")
    return process_text(completions.answer)


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
    max_tokens: int,
    max_prompt_length: int,
    n_demos: int,
    validation_split: str,
    max_dev_samples: int,
    output_dir: str,
    model_name: str,
    chat_model: bool,
    fulltext: bool,
    dontsave: bool,
):
    # Load data
    train, dev = load_data(validation_split, max_dev_samples, fulltext)

    # Configure language model
    os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(output_dir, "cache")
    model_type = "chat" if chat_model else "text"
    lm = dsp.GPT3(model=model_name, model_type=model_type, **{"max_tokens": max_tokens})
    dsp.settings.configure(lm=lm)

    # Load tokenizer
    enc = tiktoken.encoding_for_model(model_name)

    # Logging first three examples
    for example in dev[:3]:
        print("----------")
        prediction = vanilla_LM_QA(
            example.question,
            example.context,
            train,
            n_demos,
            model_name,
            max_prompt_length,
            max_tokens,
        )
        answer = example.answer

        print(example.question)
        print(example.context)
        print("Prediction: \t", prediction)
        print("Label: \t\t", answer)

        prediction_icsr = Icsr.from_string(prediction)
        answer_icsr = Icsr.from_string(answer)

        if prediction_icsr and answer_icsr:
            print("Similarity: \t", prediction_icsr.score(answer_icsr))
        else:
            print("Similarity: \t", "Failed to parse ICSR")

        print(
            "Amount of prompt tokens: ",
            lm.history[-1]["response"]["usage"]["prompt_tokens"],
        )

    # Log a full history
    print("Logging a full prompt:")
    lm.inspect_history(n=1)

    # Evaluate
    prompts, predictions, labels = [], [], []
    prompt_tokens, completion_tokens, total_tokens = [], [], []
    demo_context_tokens, inference_context_tokens = [], []

    for example in tqdm(dev):
        prediction = vanilla_LM_QA(
            example.question,
            example.context,
            train,
            n_demos,
            model_name,
            max_prompt_length,
            max_tokens,
        )
        prompt = lm.history[-1]["prompt"]

        # save predictions
        predictions.append(prediction)
        prompts.append(prompt)
        labels.append(example.answer)

        # save token lengths
        completion_tokens.append(
            lm.history[-1]["response"]["usage"]["completion_tokens"]
        )
        prompt_tokens.append(lm.history[-1]["response"]["usage"]["prompt_tokens"])
        total_tokens.append(lm.history[-1]["response"]["usage"]["total_tokens"])

        # get avg context length for demo and and inference
        inference_context = prompt.strip("\nAnswer:").split("Context:")[-1]
        demo_contexts = prompt.strip("\nAnswer:").split("Context:")[-1 - n_demos : -1]
        demo_contexts = [c.split("\nAnswer")[0] for c in demo_contexts]

        inference_context = len(enc.encode(inference_context))
        demo_contexts = [len(enc.encode(c)) for c in demo_contexts]

        inference_context_tokens.append(inference_context)
        demo_context_tokens.extend(demo_contexts)

    (precision, recall, f1), fails = evaluate_icsr(predictions, labels)
    parse_percent = 100 * (1 - (fails / len(predictions)))
    metrics = {
        "number_demo_samples": n_demos,
        "number_predict_samples": len(predictions),
        "predict_icsr_precision": precision,
        "predict_icsr_recall": recall,
        "predict_icsr_f1": f1,
        "predict_icsr_parse_percent": parse_percent,
        "predict_completion_token_avg": sum(completion_tokens) / len(completion_tokens),
        "predict_prompt_tokens_avg": sum(prompt_tokens) / len(prompt_tokens),
        "predict_total_tokens_avg": sum(total_tokens) / len(total_tokens),
        "predict_completion_token_max": max(completion_tokens),
        "predict_prompt_tokens_max": max(prompt_tokens),
        "predict_total_tokens_max": max(total_tokens),
        "predict_inference_context_tokens_avg": sum(inference_context_tokens)
        / len(inference_context_tokens),
        "predict_demo_context_tokens_avg": sum(demo_context_tokens)
        / len(demo_context_tokens),
        "predict_inference_context_tokens_max": max(inference_context_tokens),
        "predict_demo_context_tokens_max": max(demo_context_tokens),
    }

    print("Results:")
    for k, v in metrics.items():
        print("{:<30} {:<15}".format(k, v))

    # Get ouput dir
    output_dir = os.path.join(output_dir, model_name)
    next_run_number = get_run_number(output_dir)
    output_dir = os.path.join(output_dir, f"run{next_run_number:02d}")

    if dontsave == True:
        return

    # Save predictions
    print("Logging to directory:")
    print(output_dir)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(
        os.path.join(output_dir, f"generated_{validation_split}_predictions.txt"), "w"
    ) as f:
        for pred in predictions:
            f.write(pred + "\n")

    with open(os.path.join(output_dir, f"{validation_split}_labels.txt"), "w") as fp:
        fp.writelines("\n".join(labels))

    prompt_dict = {i: p for i, p in enumerate(prompts)}
    with open(os.path.join(output_dir, f"{validation_split}_prompts.json"), "w") as fp:
        json.dump(prompt_dict, fp, indent=4)

    # Save config
    with open(
        os.path.join(output_dir, f"lm_{validation_split}_config.json"), "w"
    ) as fp:
        json.dump(lm.kwargs, fp, indent=4)

    # Save metrics
    with open(os.path.join(output_dir, f"{validation_split}_results.json"), "w") as fp:
        json.dump(metrics, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT3 on BioDEX.")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="The maximum number of tokens to generate with GPT3.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=4096,
        help="The maximum number of tokens that fit in the models context window.",
    )
    parser.add_argument(
        "--n_demos",
        type=int,
        default=7,
        help="The number of demonstrative examples to use in question answering.",
    )
    parser.add_argument(
        "--validation_split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="The split to use for validation.",
    )
    parser.add_argument(
        "--max_dev_samples",
        type=int,
        default=None,
        help="The maximum number of samples to use from the set used for validation.",
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
        help="The name of the GPT3 model to use.",
    )
    parser.add_argument(
        "--chat_model",
        type=bool,
        default=False,
        help="If true, use the OpanAI chat API instead of the text api.",
    )
    parser.add_argument(
        "--fulltext",
        type=bool,
        default=False,
        help="Whether to use (truncated) fulltext input for inference.",
    )
    parser.add_argument(
        "--dontsave",
        type=bool,
        default=False,
        help="If true, dont save any of the results.",
    )

    args = parser.parse_args()
    print("Arguments:")
    print(json.dumps(vars(args), indent=4))

    run(
        max_tokens=args.max_tokens,
        max_prompt_length=args.max_prompt_length,
        n_demos=args.n_demos,
        validation_split=args.validation_split,
        max_dev_samples=args.max_dev_samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
        chat_model=args.chat_model,
        fulltext=args.fulltext,
        dontsave=args.dontsave,
    )
