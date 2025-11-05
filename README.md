# Bert_From_Scratch


# Simplified BERT Pre-training from Scratch

This project is a simplified, from-scratch implementation of the BERT model in PyTorch. It is designed to replicate the two main pre-training objectives from the original paper:

1.  **Masked Language Modeling (MLM)**
2.  **Next Sentence Prediction (NSP)**

The model is trained on the `wikitext-2-v1` dataset.

---

## Core Components

* **`model.py`**: Contains the `SimplifiedBERT` architecture built from scratch.
    * `BertEmbeddings`: Sums token, position, and segment embeddings.
    * `MultiHeadSelfAttention`: A custom implementation of the attention mechanism.
    * `BertEncoderLayer`: The standard Transformer encoder block.
    * `MLMHead` & `NSPHead`: Output layers for the two pre-training tasks.

* **`dataset.py`**: A custom `torch.utils.data.Dataset` for the `wikitext-2-v1` corpus.
    * Fetches and processes text from the Hugging Face `datasets` library.
    * Creates 50/50 positive and negative NSP pairs.
    * Applies the 15% MLM masking strategy (80% `[MASK]`, 10% random, 10% same).
    * Includes a `collate_fn` for dynamic batch padding.

* **`train.py`**: The main training script.
    * Loads the model and dataset.
    * Runs the training loop, calculating a combined `total_loss = mlm_loss + nsp_loss`.
    * Saves the trained model weights (`simplified_bert.pth`).

---

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install torch transformers datasets tqdm
    ```

2.  **Start the training:**
    ```bash
    python train.py
    ```

---

## Development Note

This project represents my genuine effort to understand and implement the core components of BERT from the ground up. While I utilized an LLM for assistance with code generation and boilerplate, the core logic, model architecture, and data-handling strategies are my own work.