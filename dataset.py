import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import random
from tqdm import tqdm

# --- Configuration ---
# Use the same config as our model
from model import config
MAX_LEN = config["max_len"]
VOCAB_SIZE = config["vocab_size"]

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MASK_TOKEN_ID = tokenizer.mask_token_id
CLS_TOKEN_ID = tokenizer.cls_token_id
SEP_TOKEN_ID = tokenizer.sep_token_id
PAD_TOKEN_ID = tokenizer.pad_token_id


class BertPretrainingDataset(Dataset):
    def __init__(self, split='train'):
        print("Loading and processing dataset...")
        #loading the dataset WikiText-2
        #Train and Validation splits
        dataset = load_dataset("wikitext", "wikitext-2-v1")
        train_data = dataset["train"]
        val_data = dataset["validation"]
        
        # 2. Tokenize and find all sentences
        # We'll treat a "document" as a single article (separated by \n \n = \n \n)
        self.sentences = []
        #creating sentences
        for doc in tqdm(train_data['text']):
            if len(doc.strip()) > 0: # Skip empty docs
                # A simple sentence split. This is not perfect but good enough.
                # We also add sentences from the same doc.
                doc_sents = []
                for sent in doc.split('. '): # Split by period
                    if len(sent.strip()) > 10: # Filter short/empty sentences
                        doc_sents.append(sent.strip())
                
                if len(doc_sents) > 1:
                    self.sentences.append(doc_sents)

        # 3. Create NSP pairs (as (doc_index, sent_index))
        # This is a lightweight way to store pairs without duplicating text
        self.nsp_pairs = []
        print("Creating NSP pairs...")
        for doc_idx, doc in enumerate(tqdm(self.sentences)):
            for sent_idx in range(len(doc) - 1):
                # 50% chance of a positive pair
                if random.random() < 0.5:
                    # Positive pair
                    self.nsp_pairs.append((doc_idx, sent_idx, sent_idx + 1, 1))
                else:
                    # Negative pair
                    # Find a random document and random sentence
                    random_doc_idx = doc_idx
                    while random_doc_idx == doc_idx:
                        random_doc_idx = random.randint(0, len(self.sentences) - 1)
                    
                    random_sent_idx = random.randint(0, len(self.sentences[random_doc_idx]) - 1)
                    self.nsp_pairs.append((doc_idx, sent_idx, (random_doc_idx, random_sent_idx), 0))

        print(f"Created {len(self.nsp_pairs)} training pairs.")

    def __len__(self):
        return len(self.nsp_pairs)

    def __getitem__(self, idx):
        # 4. Get the pair
        doc_idx_a, sent_idx_a, sent_idx_b, nsp_label = self.nsp_pairs[idx]
        
        sent_a = self.sentences[doc_idx_a][sent_idx_a]
        
        if nsp_label == 1:
            sent_b = self.sentences[doc_idx_a][sent_idx_b]
        else:
            # sent_idx_b is a tuple (random_doc_idx, random_sent_idx)
            random_doc_idx, random_sent_idx = sent_idx_b
            sent_b = self.sentences[random_doc_idx][random_sent_idx]
            
        # 5. Tokenize
        # [CLS] sent_a [SEP] sent_b [SEP]
        token_a = tokenizer.tokenize(sent_a)
        token_b = tokenizer.tokenize(sent_b)
        
        # Truncate to fit in MAX_LEN
        # -3 for [CLS], [SEP], [SEP]
        truncate_seq_pair(token_a, token_b, MAX_LEN - 3)

        tokens = [tokenizer.cls_token] + token_a + [tokenizer.sep_token] + token_b + [tokenizer.sep_token]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        segment_ids = [0] * (len(token_a) + 2) + [1] * (len(token_b) + 1)
        
        # 6. Apply MLM
        # , 10% random, 10% same]
        mlm_input_ids, mlm_labels = self.mask_tokens(token_ids)
        
        return {
            "input_ids": mlm_input_ids,
            "segment_ids": segment_ids,
            "mlm_labels": mlm_labels,
            "nsp_label": nsp_label
        }
    
    def mask_tokens(self, token_ids):
        """
        Applies the BERT masking strategy.
        15% of tokens are masked.
        Of those 15%:
          - 80% are replaced with [MASK]
          - 10% are replaced with a random token
          - 10% are left unchanged
        """
        mlm_input_ids = list(token_ids)
        # We use -100 as the "ignore" label for CrossEntropyLoss
        mlm_labels = [-100] * len(token_ids)
        
        # Find indices we can mask (don't mask special tokens)
        special_token_ids = {CLS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID}
        candidate_indices = [
            i for i, token_id in enumerate(token_ids)
            if token_id not in special_token_ids
        ]
        
        # Calculate number of tokens to mask (15% of candidates)
        num_to_mask = max(1, int(len(candidate_indices) * 0.15))
        random.shuffle(candidate_indices)
        masked_indices = candidate_indices[:num_to_mask]
        
        for i in masked_indices:
            # Store the true token ID as the label
            mlm_labels[i] = token_ids[i]
            
            rand = random.random()
            if rand < 0.8:
                # 80% replace with [MASK]
                mlm_input_ids[i] = MASK_TOKEN_ID
            elif rand < 0.9:
                # 10% replace with random token
                mlm_input_ids[i] = random.randint(0, VOCAB_SIZE - 1)
            else:
                # 10% keep original (do nothing)
                pass
                
        return mlm_input_ids, mlm_labels

# Helper to truncate token pairs
def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# --- Collate Function ---
# This function is passed to the DataLoader to dynamically pad batches.

def collate_fn(batch):
    # Find max length in this batch
    max_batch_len = max(len(item["input_ids"]) for item in batch)

    # Pad all sequences to max_batch_len
    batch_input_ids = []
    batch_segment_ids = []
    batch_attention_mask = []
    batch_mlm_labels = []
    batch_nsp_labels = []

    for item in batch:
        input_ids = item["input_ids"]
        segment_ids = item["segment_ids"]
        mlm_labels = item["mlm_labels"]
        
        padding_len = max_batch_len - len(input_ids)
        
        # Pad input_ids, segment_ids, and mlm_labels
        padded_input_ids = input_ids + [PAD_TOKEN_ID] * padding_len
        padded_segment_ids = segment_ids + [0] * padding_len # Pad with 0
        padded_mlm_labels = mlm_labels + [-100] * padding_len # Pad with -100
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        
        batch_input_ids.append(padded_input_ids)
        batch_segment_ids.append(padded_segment_ids)
        batch_attention_mask.append(attention_mask)
        batch_mlm_labels.append(padded_mlm_labels)
        batch_nsp_labels.append(item["nsp_label"])

    # Convert to tensors
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "segment_ids": torch.tensor(batch_segment_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "mlm_labels": torch.tensor(batch_mlm_labels, dtype=torch.long),
        "nsp_label": torch.tensor(batch_nsp_labels, dtype=torch.long)
    }

# --- Example usage (for testing) ---
if __name__ == "__main__":
    dataset = BertPretrainingDataset(split='train')
    
    # Test __getitem__
    print("\n--- Example Item ---")
    item = dataset[0]
    print(f"Input IDs (masked): {item['input_ids']}")
    print(f"Segment IDs: {item['segment_ids']}")
    print(f"MLM Labels: {item['mlm_labels']}")
    print(f"NSP Label: {item['nsp_label']}")
    
    # Test collate_fn and DataLoader
    print("\n--- Example Batch ---")
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_fn
    )
    
    batch = next(iter(data_loader))
    print(f"Batch Input IDs shape: {batch['input_ids'].shape}")
    print(f"Batch Segment IDs shape: {batch['segment_ids'].shape}")
    print(f"Batch Attention Mask shape: {batch['attention_mask'].shape}")
    print(f"Batch MLM Labels shape: {batch['mlm_labels'].shape}")
    print(f"Batch NSP Labels shape: {batch['nsp_label'].shape}")