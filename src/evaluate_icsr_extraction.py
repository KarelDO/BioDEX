import argparse
from datasets import load_dataset

from src import Icsr
import logging

logger = logging.getLogger(__name__)


def evaluate_icsr_from_dataset(pred_strings, dataset_name, dataset_split):
    dataset = load_dataset(dataset_name)
    gold_strings = dataset[dataset_split]["target"]

    return evaluate_icsr(pred_strings, gold_strings)


def evaluate_icsr(pred_strings, gold_strings):
    # Remove newlines
    pred_strings = [line.strip() for line in pred_strings]
    gold_strings = [line.strip() for line in gold_strings]

    # Parse in ICSR format
    pred_icsrs = [Icsr.from_string(x) for x in pred_strings]
    gold_icsrs = [Icsr.from_string(x) for x in gold_strings]

    # report if we failed to parse predictions
    failed = len([p for p in pred_icsrs if p == None])
    logger.info(
        f"Evaluate: Failed to parse {failed:,}/{len(pred_icsrs):,} predictions."
    )

    # Evaluate
    scores = [
        p.score(g) if p else (0.0, 0.0, 0.0) for p, g in zip(pred_icsrs, gold_icsrs)
    ]

    precision = 100 * sum([s[0] for s in scores]) / len(scores)
    recall = 100 * sum([s[1] for s in scores]) / len(scores)
    f1 = 100 * sum([s[2] for s in scores]) / len(scores)

    logger.info(f"Evaluate: precision: {precision}")
    logger.info(f"Evaluate: recall: {recall}")
    logger.info(f"Evaluate: f1: {f1}")

    return (precision, recall, f1), failed


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("file_path", type=str, help="Path to the text file")
    parser.add_argument(
        "dataset_name", type=str, help="Name of the Hugging Face dataset"
    )
    parser.add_argument(
        "dataset_split", type=str, help="Split to evaluate on, either 'eval' or 'test'."
    )
    args = parser.parse_args()

    # Read text file
    with open(args.file_path, "r") as file:
        pred_strings = file.readlines()

    # Download Hugging Face dataset and get the gold targets
    (precision, recall, f1), failed = evaluate_icsr_from_dataset(
        pred_strings, args.dataset_name, args.dataset_split
    )

    print(f"Evaluate: precision: {precision}")
    print(f"Evaluate: recall: {recall}")
    print(f"Evaluate: f1: {f1}")
    print(f"Evaluate: failed: {failed}")
