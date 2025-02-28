import numpy as np

from datasets import load_dataset, DatasetDict


def filter_pairs(ds: DatasetDict) -> DatasetDict:
    """
    Filters dataset to contain only rows where there are 2 correct
    generations
    """
    train_dataset = ds['train']
    filtered_train_dataset = train_dataset.filter(lambda x: x['correctness_count'] == 2)
    filtered_dataset = DatasetDict({
        'train': filtered_train_dataset
    })
    
    return filtered_dataset

def add_max_generation_length(example):
    """
    Adds the column 'max_generation_lenght' based on the largest
    generation length measured in number of characters.
    """
    gen_lengths = [len(g) for g in example['generations']]
    example['max_generation_length'] = max(gen_lengths)
    return example


def filter_shortest_k_generations(
    ds: DatasetDict,
    topK: int
) -> DatasetDict:
    """
    Returns a filtered dataset containing the top k
    shortest generation samples. The shortest generations are measured
    by the longest generation of the two correct generations in each row.
    """
    train_ds = ds['train']
    train_ds = train_ds.map(add_max_generation_length)
    sorted_dataset = train_ds.sort('max_generation_length')
    shortest_generations_dataset = sorted_dataset.select(range(topK))
    filtered_dataset = DatasetDict({
        'train': shortest_generations_dataset
    })
    
    return filtered_dataset

def filter_ds(ds: DatasetDict, topK: int) -> DatasetDict:
    """
    Filters the specified dataset to:
    1. Contain exactly two correct generation.
    2. Contain the top k shortest generations.
    """
    pair_ds = filter_pairs(ds)
    filtered_ds = filter_shortest_k_generations(pair_ds, topK)
    return filtered_ds

if __name__ == '__main__':
    topK = 200
    ds = load_dataset('open-r1/OpenR1-Math-220k', 'default')
    filtered_ds = filter_ds(ds, topK)
    # Extract all max generation lengths into a list
    max_lengths = filtered_ds['train']['max_generation_length']

    # Compute statistics
    print(f'Number of samples: {len(max_lengths)}')
    print(f'Min generation length: {np.min(max_lengths)}')
    print(f'Max generation length: {np.max(max_lengths)}')
    print(f'Mean generation length: {np.mean(max_lengths):.2f}')
    print(f'Median generation length: {np.median(max_lengths)}')
    print(f'Standard deviation: {np.std(max_lengths):.2f}')