import argparse
from datasets import load_dataset

from src import Icsr

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('dataset_name', type=str, help='Name of the Hugging Face dataset')
    args = parser.parse_args()

    # Read text file
    with open(args.file_path, 'r') as file:
        lines = file.readlines()

    # Get predicted targets, remove newlines
    pred_strings = [line.strip() for line in lines]

    # Download Hugging Face dataset and get the gold targets
    dataset = load_dataset(args.dataset_name)
    gold_strings = dataset['test']['target']

    # Parse in ICSR format
    pred_icsrs = [Icsr.from_string(x) for x in pred_strings]
    gold_icsrs = [Icsr.from_string(x) for x in gold_strings]

    # report if we failed to parse predictions
    failed = [p for p in pred_icsrs if p == None]
    print(f'Failed to parse {len(failed):,}/{len(pred_icsrs):,} predictions.')

    # Evaluate
    scores = [p.score(g) if p else (.0, .0, .0) for p,g in zip(pred_icsrs, gold_icsrs)]

    precision = sum([s[0] for s in scores]) / len(scores)
    recall = sum([s[1] for s in scores]) / len(scores)
    f1 = sum([s[2] for s in scores]) / len(scores)

    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')