import os
import argparse
import datasets
from tqdm import tqdm
from typing import List
from src import Icsr
import dsp
import glob
import json

from evaluate_icsr_extraction import evaluate_icsr


def load_data(max_dev_samples: int, fulltext: bool) -> tuple:
    dataset = datasets.load_dataset("FAERS-PubMed/BioDEX-ICSR")

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
        for example in dataset["validation"]
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
    question: str, context: str, train: List[dsp.Example], n_demos: int, model_name: str
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
    example, completions = dsp.generate(qa_template, model_name=model_name)(
        example, stage="qa"
    )
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
    n_demos: int,
    max_dev_samples: int,
    output_dir: str,
    model_name: str,
    fulltext: bool,
    dontsave: bool,
):
    # Load data
    train, dev = load_data(max_dev_samples, fulltext)

    # Configure language model
    os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(output_dir, "cache")
    lm = dsp.GPT3(model=model_name, **{"max_tokens": max_tokens})
    dsp.settings.configure(lm=lm)

    # Logging first three examples
    for example in dev[:3]:
        print("----------")
        prediction = vanilla_LM_QA(
            example.question, example.context, train, n_demos, model_name
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

    for example in tqdm(dev):
        try:
            prediction = vanilla_LM_QA(
                example.question, example.context, train, n_demos, model_name
            )
        except:
            print("Warning: trying with fewer shots for context length")
            prediction = vanilla_LM_QA(
                example.question, example.context, train, n_demos - 1, model_name
            )
        prompt = lm.history[-1]["prompt"]

        predictions.append(prediction)
        prompts.append(prompt)
        labels.append(example.answer)

        completion_tokens.append(
            lm.history[-1]["response"]["usage"]["completion_tokens"]
        )
        prompt_tokens.append(lm.history[-1]["response"]["usage"]["prompt_tokens"])
        total_tokens.append(lm.history[-1]["response"]["usage"]["total_tokens"])

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
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, "generated_eval_predictions.txt"), "w") as f:
        for pred in predictions:
            f.write(pred + "\n")

    with open(os.path.join(output_dir, "eval_labels.txt"), "w") as fp:
        fp.writelines("\n".join(labels))

    prompt_dict = {i: p for i, p in enumerate(prompts)}
    with open(os.path.join(output_dir, "eval_prompts.json"), "w") as fp:
        json.dump(prompt_dict, fp, indent=4)

    # Save config
    with open(os.path.join(output_dir, "lm_config.json"), "w") as fp:
        json.dump(lm.kwargs, fp, indent=4)

    # Save metrics
    with open(os.path.join(output_dir, "eval_results.json"), "w") as fp:
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
        "--n_demos",
        type=int,
        default=7,
        help="The number of demonstrative examples to use in question answering.",
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
        help="The name of the GPT3 model to use.",
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
        n_demos=args.n_demos,
        max_dev_samples=args.max_dev_samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
        fulltext=args.fulltext,
        dontsave=args.dontsave,
    )
