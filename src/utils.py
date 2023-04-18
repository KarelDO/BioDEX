import numpy as np
from . import Match, Article, Report


def convert_arrays_to_lists(data):
    """
    Recursively converts all NumPy arrays in a dictionary to lists.
    """
    if isinstance(data, np.ndarray):
        # Convert NumPy array to list
        return [convert_arrays_to_lists(item) for item in data.tolist()]
    elif isinstance(data, dict):
        # Recursively iterate through dictionary entries
        return {key: convert_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively iterate through list elements
        return [convert_arrays_to_lists(item) for item in data]
    else:
        # Return data as is
        return data


def get_matches(dataset):
    """
    Casts a HuggingFace dataset to an array of Match objects
    """
    dataset = dataset.to_pandas()
    dataset["reports"] = dataset["reports"].apply(
        lambda reports: [convert_arrays_to_lists(r) for r in reports]
    )

    matches = []
    for _, row in dataset.iterrows():
        article = Article.parse_obj(row["article"])
        reports = [Report.parse_obj(r) for r in row["reports"]]
        match = Match(article=article, reports=reports)
        matches.append(match)

    return matches
