import argparse
import os
import numpy as np
import tiktoken

from datasets import load_dataset, DatasetDict

tokenizer = tiktoken.get_encoding("cl100k_base")

def map_generations(row):
    """
    Keeps only the shortest and longest correct generations.
    Adds 'generations_length' column.
    """
    correct_generations = [
        i for i, correct in enumerate(row['correctness_math_verify']) if correct
    ]
    
    if len(correct_generations) < 2:
        return None
    
    correct_lengths = [len(tokenizer.encode(row['generations'][i])) for i in correct_generations]
    min_id = correct_lengths.index(min(correct_lengths))
    max_id = correct_lengths.index(max(correct_lengths))
    
    row['generations'] = [row['generations'][min_id], row['generations'][max_id]]
    row['min_length'] = correct_lengths[min_id]
    row['max_length'] = correct_lengths[max_id]
    row['length_difference'] = correct_lengths[max_id] - correct_lengths[min_id]
    
    return row

def filter_pairs(
    ds: DatasetDict,
    max_length: int
) -> DatasetDict:
    """
    Filters dataset to contain only rows where there are 2 correct generations
    """
    train_dataset = ds['train']
    filtered_train_dataset = train_dataset.filter(lambda x: sum(x["correctness_math_verify"]) >= 2)
    filtered_train_dataset = filtered_train_dataset.map(map_generations)
    filtered_train_dataset = filtered_train_dataset.filter(
        lambda x: x["min_length"] <= max_length and x["max_length"] <= max_length and x["length_difference"] > 0
    )

    filtered_dataset = DatasetDict({
        'train': filtered_train_dataset
    })
    return filtered_dataset


def filter_shortest_k_generations(
    ds: DatasetDict,
    topK: int,
    method: str = 'shortest'
) -> DatasetDict:
    """
    Returns a filtered dataset containing the top k shortest generation samples
    based on the lengths in 'generations_length'.
    """
    train_ds = ds['train']
    
    # Sort the dataset based on the lengths in 'generations_length'
    if method == 'shortest':
        # Sort based on the shortest length in 'generations_length' (index 0)
        sorted_dataset = train_ds.sort('min_length')
    elif method == 'longest':
        # Sort based on the longest length in 'generations_length' (index 1)
        sorted_dataset = train_ds.sort('max_length')
    elif method == 'diff':
        # Sort based on the difference in length between the generations
        sorted_dataset = train_ds.sort('length_difference')
    else:
        raise ValueError(f"Invalid method: {method}")
    
    if topK != -1:
        sorted_dataset = sorted_dataset.select(range(topK))
    
    filtered_dataset = DatasetDict({
        'train': sorted_dataset
    })
    return filtered_dataset


def filter_ds(
    ds: DatasetDict, 
    topK: int, 
    method: str,
    max_length: int,
    min_length_difference: int
) -> DatasetDict:
    """
    Filters the dataset to:
    1. Contain exactly two correct generations.
    2. Contain the top k shortest generations.
    """
    pair_ds = filter_pairs(ds, max_length)
    pair_ds = pair_ds.filter(lambda x: x["length_difference"] >= min_length_difference)
    filtered_ds = filter_shortest_k_generations(pair_ds, topK, method)
    return filtered_ds


def add_comparison_columns(example):
    """
    Adds chosen, rejected, chosen_score, rejected_score columns.
    """
    assert example["min_length"] <= example["max_length"]

    # Construct 'chosen' and 'rejected'
    example['chosen'] = [
        {"content": example['problem'], "role": "user"},
        {"content": example["generations"][0], "role": "assistant"}
    ]
    example['rejected'] = [
        {"content": example['problem'], "role": "user"},
        {"content": example["generations"][1], "role": "assistant"}
    ]

    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter OpenR1-Math-220k dataset.")
    parser.add_argument('--dataset_name', type=str, default='open-r1/OpenR1-Math-220k', help='Name of the dataset to load')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save filtered dataset')
    parser.add_argument('--topK', type=int, default=-1, help='Number of shortest samples to keep')
    parser.add_argument('--method', type=str, default='shortest', help='Method to filter samples (shortest or diff)')
    parser.add_argument('--max_length', type=int, default=8192, help='Max length')
    parser.add_argument('--min_length_difference', type=int, default=300, help='Max length')

    args = parser.parse_args()

    # Load and process dataset
    ds = load_dataset(args.dataset_name, 'default')
    filtered_ds = filter_ds(ds, args.topK, args.method, args.max_length, args.min_length_difference)

    # Add comparison columns
    filtered_ds = filtered_ds.map(add_comparison_columns)

    # Save to disk
    os.makedirs(args.data_dir, exist_ok=True)
    save_path = os.path.join(args.data_dir, f'{args.dataset_name}_filtered')
    filtered_ds.save_to_disk(save_path)

    print(f"Filtered dataset saved to '{save_path}'")

    # Extract max generation lengths into a list
    max_lengths = filtered_ds['train']['max_length']
    min_lengths = filtered_ds['train']['min_length']
    length_differences = filtered_ds['train']['length_difference']

    # Compute statistics
    print(f'\nStatistics for Max Generation Lengths:')
    print(f'Number of samples: {len(max_lengths)}')
    print(f'Min generation length: {np.min(max_lengths)}')
    print(f'Max generation length: {np.max(max_lengths)}')
    print(f'Mean generation length: {np.mean(max_lengths):.2f}')
    print(f'Median generation length: {np.median(max_lengths)}')
    print(f'Standard deviation: {np.std(max_lengths):.2f}')

    print(f'\nStatistics for Min Generation Lengths:')
    print(f'Number of samples: {len(min_lengths)}')
    print(f'Min generation length: {np.min(min_lengths)}')
    print(f'Max generation length: {np.max(min_lengths)}')
    print(f'Mean generation length: {np.mean(min_lengths):.2f}')
    print(f'Median generation length: {np.median(min_lengths)}')
    print(f'Standard deviation: {np.std(min_lengths):.2f}')

    # Calculate and print statistics for the length difference
    print(f'\nStatistics for Length Difference:')
    print(f'Min length difference: {np.min(length_differences)}')
    print(f'Max length difference: {np.max(length_differences)}')
    print(f'Mean length difference: {np.mean(length_differences):.2f}')
    print(f'Median length difference: {np.median(length_differences)}')
    print(f'Standard deviation (length differences): {np.std(length_differences):.2f}')
