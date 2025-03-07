import argparse
import os
import numpy as np

from datasets import load_dataset, DatasetDict


def filter_pairs(ds: DatasetDict) -> DatasetDict:
    """
    Filters dataset to contain only rows where there are 2 correct generations
    """
    train_dataset = ds['train']
    filtered_train_dataset = train_dataset.filter(lambda x: x['correctness_count'] == 2 and len(x['generations']) == 2)
    filtered_dataset = DatasetDict({
        'train': filtered_train_dataset
    })
    return filtered_dataset


def add_length_data(example):
    """
    Adds the column 'max_generation_length' and 'length_difference' based on the largest
    and shortest generation length (in characters).
    """
    gen_lengths = [len(g) for g in example['generations']]
    example['max_generation_length'] = max(gen_lengths)
    example['length_difference'] = abs(gen_lengths[0] - gen_lengths[1])
    return example


def filter_shortest_k_generations(
    ds: DatasetDict,
    topK: int,
    method: str = 'shortest'
) -> DatasetDict:
    """
    Returns a filtered dataset containing the top k shortest generation samples.
    """
    train_ds = ds['train']
    train_ds = train_ds.map(add_length_data)
    if method == 'shortest':
        sorted_dataset = train_ds.sort('max_generation_length')
    elif method == 'diff':
        sorted_dataset = train_ds.sort('length_difference', reverse=True)
    else:
        raise ValueError(f"Invalid method: {method}")
    if topK != -1:
        sorted_dataset = sorted_dataset.select(range(topK))
    filtered_dataset = DatasetDict({
        'train': sorted_dataset
    })
    return filtered_dataset


def filter_ds(ds: DatasetDict, topK: int, method: str) -> DatasetDict:
    """
    Filters the dataset to:
    1. Contain exactly two correct generations.
    2. Contain the top k shortest generations.
    """
    pair_ds = filter_pairs(ds)
    filtered_ds = filter_shortest_k_generations(pair_ds, topK, method)
    return filtered_ds


def add_comparison_columns(example):
    """
    Adds chosen, rejected, chosen_score, rejected_score columns.
    """
    try:
        gen1, gen2 = example['generations']
        len1, len2 = len(gen1), len(gen2)
    except:
        breakpoint()
    if len1 <= len2:
        generation_short, generation_large = gen1, gen2
    else:
        generation_short, generation_large = gen2, gen1

    # Construct 'chosen' and 'rejected'
    example['chosen'] = [
        {"content": example['problem'], "role": "user"},
        {"content": generation_short, "role": "assistant"}
    ]
    example['rejected'] = [
        {"content": example['problem'], "role": "user"},
        {"content": generation_large, "role": "assistant"}
    ]

    # Compute scores
    length_diff = (len(generation_large) - len(generation_short))/len(generation_large) * 100
    rejected_score = min(100, max(0, length_diff))  # cap between 0-100
    example['rejected_score'] = rejected_score
    example['chosen_score'] = rejected_score + 100

    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter OpenR1-Math-220k dataset.")
    parser.add_argument('--dataset_name', type=str, default='open-r1/OpenR1-Math-220k', help='Name of the dataset to load')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save filtered dataset')
    parser.add_argument('--topK', type=int, default=-1, help='Number of shortest samples to keep')
    parser.add_argument('--method', type=str, default='shortest', help='Method to filter samples (shortest or diff)')

    args = parser.parse_args()

    # Load and process dataset
    ds = load_dataset(args.dataset_name, 'default')
    filtered_ds = filter_ds(ds, args.topK, args.method)

    # Add comparison columns
    filtered_ds = filtered_ds.map(add_comparison_columns)

    # Extract max generation lengths into a list
    max_lengths = filtered_ds['train']['max_generation_length']

    # Compute statistics
    print(f'Number of samples: {len(max_lengths)}')
    print(f'Min generation length: {np.min(max_lengths)}')
    print(f'Max generation length: {np.max(max_lengths)}')
    print(f'Mean generation length: {np.mean(max_lengths):.2f}')
    print(f'Median generation length: {np.median(max_lengths)}')
    print(f'Standard deviation: {np.std(max_lengths):.2f}')

    # Save to disk
    os.makedirs(args.data_dir, exist_ok=True)
    save_path = os.path.join(args.data_dir, f'{args.dataset_name}_filtered')
    filtered_ds.save_to_disk(save_path)

    print(f"Filtered dataset saved to '{save_path}'")
