import argparse
from datasets import load_dataset

from src import Icsr
import logging

logger = logging.getLogger(__name__)


def evaluate_icsr_from_dataset(pred_strings, dataset_name, dataset_split, detangled=False):
    dataset = load_dataset(dataset_name)
    gold_strings = dataset[dataset_split]["target"]

    if not detangled:
        return evaluate_icsr(pred_strings, gold_strings)
    if detangled:
        return evaluate_icsr_detangled(pred_strings, gold_strings)


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


def evaluate_icsr_detangled(pred_strings, gold_strings):
    # Remove newlines
    pred_strings = [line.strip() for line in pred_strings]
    gold_strings = [line.strip() for line in gold_strings]

    # Parse in ICSR format
    pred_icsrs = [Icsr.from_string(x) for x in pred_strings]
    gold_icsrs = [Icsr.from_string(x) for x in gold_strings]

    # Get detangled metrics
    patientsex_accuracy = []
    serious_accuracy = []
    drug_precision = []
    drug_recall = []
    reaction_precision = []
    reaction_recall = []

    for pred, gold in zip(pred_icsrs, gold_icsrs):
        if pred:
            patientsex, serious, drug, reaction = pred.score_detangled(gold)
        else:
            patientsex, serious, drug, reaction = (
                0,
                0,
                (0, 0),
                (
                    0,
                    0,
                ),
            )
        patientsex_accuracy.append(patientsex)
        serious_accuracy.append(serious)
        drug_precision.append(drug[0])
        drug_recall.append(drug[1])
        reaction_precision.append(reaction[0])
        reaction_recall.append(reaction[1])

    avg_patientsex_accuracy = sum(patientsex_accuracy) / len(patientsex_accuracy)
    avg_serious_accuracy = sum(serious_accuracy) / len(serious_accuracy)
    avg_drug_precision = sum(drug_precision) / len(drug_precision)
    avg_drug_recall = sum(drug_recall) / len(drug_recall)
    avg_reaction_precision = sum(reaction_precision) / len(reaction_precision)
    avg_reaction_recall = sum(reaction_recall) / len(reaction_recall)

    return (
        avg_patientsex_accuracy,
        avg_serious_accuracy,
        avg_drug_precision,
        avg_drug_recall,
        avg_reaction_precision,
        avg_reaction_recall,
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("file_path", type=str, help="Path to the text file")
    parser.add_argument(
        "dataset_name", type=str, help="Name of the Hugging Face dataset"
    )
    parser.add_argument(
        "dataset_split",
        type=str,
        help="Split to evaluate on, either 'validation' or 'test'.",
    )
    parser.add_argument(
        "--detangled",
        type=bool,
        default=False,
        help="If yes, return the attribute metrics separately.",
    )
    args = parser.parse_args()

    # Read text file
    with open(args.file_path, "r") as file:
        pred_strings = file.readlines()

    # Download Hugging Face dataset and get the gold targets
    (precision, recall, f1), failed = evaluate_icsr_from_dataset(
        pred_strings, args.dataset_name, args.dataset_split, detangled=False
    )

    print(f"Evaluate: precision: {precision}")
    print(f"Evaluate: recall: {recall}")
    print(f"Evaluate: f1: {f1}")
    print(f"Evaluate: failed: {failed}")

    if args.detangled:
        (
            avg_patientsex_accuracy,
            avg_serious_accuracy,
            avg_drug_precision,
            avg_drug_recall,
            avg_reaction_precision,
            avg_reaction_recall,
        ) = evaluate_icsr_from_dataset(
            pred_strings, args.dataset_name, args.dataset_split, detangled=True
        )

        print(f"Evaluate: avg_patientsex_accuracy: {avg_patientsex_accuracy}")
        print(f"Evaluate: avg_serious_accuracy: {avg_serious_accuracy}")
        print(f"Evaluate: avg_drug_precision: {avg_drug_precision}")
        print(f"Evaluate: avg_drug_recall: {avg_drug_recall}")
        print(f"Evaluate: avg_reaction_precision: {avg_reaction_precision}")
        print(f"Evaluate: avg_reaction_recall: {avg_reaction_recall}")
