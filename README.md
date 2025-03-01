# efficient-reasoning

**efficient-reasoning** is a project designed to train language models using **Direct Preference Optimization (DPO)** to generate **correct but shorter responses** rather than **correct but longer responses**. This approach is particularly useful for math and reasoning tasks where brevity and precision are desired.

The project works by processing the `OpenR1-Math-220k` dataset, filtering it to focus only on pairs of correct responses with varying lengths, and preferring the shorter response for training.

---

## Installation

To set up the environment, use Conda to create and activate a virtual environment using `env.yaml`.

```bash
conda env create -f env.yaml
conda activate efficient-reasoning
```

## Quick Start

### Step 1: Create the DPO dataset

```
bash scripts/dataset_creation.sh
```
This script will:

1. Load the OpenR1-Math-220k dataset from Hugging Face.
Filter the dataset to only keep pairs where both responses are correct.
2. Select the top-k pairs where the shorter generation is used as the "chosen" response, and the longer generation is used as the "rejected" response.
3. Add fields needed for DPO training (chosen, rejected, chosen_score, rejected_score).
4. Save the filtered dataset to data/open-r1/OpenR1-Math-220k_filtered/.

### Step 2: Train the model using DPO
Run the training script using:

```
bash scripts/train_dpo_deepseek.sh
```

This script will:

1. Load the processed dataset from Step 1.
2. Run DPO training using a target language model (by default, DeepSeek).
3. Save checkpoints and training logs for future analysis.
