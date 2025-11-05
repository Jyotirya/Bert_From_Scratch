import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Import our custom modules ---
from model import SimplifiedBERT, config
from dataset import BertPretrainingDataset, collate_fn, tokenizer

# --- Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 
NUM_EPOCHS = 3
SAVE_MODEL_PATH = "simplified_bert.pth"

# --- 1. Load Model ---
print(f"Loading model on {DEVICE}...")
model = SimplifiedBERT(config).to(DEVICE)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- 2. Load Dataset ---
# We use 'train' for both as wikitext-2 'validation' is very small
train_dataset = BertPretrainingDataset(split='train') 

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True
)

# --- 3. Setup Optimizer and Loss ---
# Adam optimizer (as used in the BERT paper)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss functions
# For MLM: CrossEntropyLoss. The -100 labels are automatically ignored.
mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100) 
# For NSP: Binary CrossEntropy (or just CrossEntropy with 2 classes)
nsp_loss_fn = nn.CrossEntropyLoss()

# --- 4. Training Loop ---

print("\n--- Starting Training ---")
model.train() # Set model to training mode

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    loop = tqdm(train_loader, leave=True)
    
    for batch in loop:
        # Zero gradients
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(DEVICE)
        segment_ids = batch["segment_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        mlm_labels = batch["mlm_labels"].to(DEVICE)
        nsp_label = batch["nsp_label"].to(DEVICE)
        
        # --- Forward Pass ---
        mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)
        
        # --- Calculate Loss ---
        # 1. MLM Loss
        # We need to flatten the batch and sequence dimensions
        # mlm_logits: (Batch, SeqLen, VocabSize) -> (Batch * SeqLen, VocabSize)
        # mlm_labels: (Batch, SeqLen) -> (Batch * SeqLen)
        mlm_loss = mlm_loss_fn(mlm_logits.view(-1, config["vocab_size"]), mlm_labels.view(-1))
        
        # 2. NSP Loss
        nsp_loss = nsp_loss_fn(nsp_logits, nsp_label)
        
        # 3. Total Loss
        total_loss = mlm_loss + nsp_loss
        
        # --- Backward Pass ---
        total_loss.backward()
        
        # --- Update Weights ---
        optimizer.step()
        
        # Update progress bar description
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(mlm_loss=mlm_loss.item(), nsp_loss=nsp_loss.item(), total_loss=total_loss.item())

# --- 5. Save the model ---
print(f"\nTraining complete. Saving model to {SAVE_MODEL_PATH}")
torch.save(model.state_dict(), SAVE_MODEL_PATH)


# --- 6. Demonstration (How to use the model) ---
print("\n--- Model Demonstration ---")
model.eval() # Set model to evaluation mode

def predict_masked_token(text):
    # Tokenize and find mask index
    tokenized_text = tokenizer.tokenize(text)
    try:
        mask_index = tokenized_text.index('[MASK]')
    except ValueError:
        print("Text must include '[MASK]' token.")
        return

    # Prepare inputs
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [0] * len(token_ids) # Only one sentence

    # Convert to tensors
    input_ids_tensor = torch.tensor([token_ids]).to(DEVICE)
    segment_ids_tensor = torch.tensor([segment_ids]).to(DEVICE)
    # Attention mask (all 1s, no padding)
    attention_mask_tensor = torch.ones_like(input_ids_tensor).to(DEVICE)

    with torch.no_grad():
        mlm_logits, _ = model(input_ids_tensor, segment_ids_tensor, attention_mask_tensor)
    
    # Get predictions for the [MASK] token
    mask_logits = mlm_logits[0, mask_index, :]
    top_5_tokens = torch.topk(mask_logits, 5, dim=0).indices.tolist()
    
    print(f"Input: {text}")
    for token_id in top_5_tokens:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        print(f"> {token}")

def check_nsp(sent_a, sent_b, actual_is_next):
    # Prepare inputs
    token_a = tokenizer.tokenize(sent_a)
    token_b = tokenizer.tokenize(sent_b)
    
    tokens = [tokenizer.cls_token] + token_a + [tokenizer.sep_token] + token_b + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * (len(token_a) + 2) + [1] * (len(token_b) + 1)
    
    # Convert to tensors
    input_ids_tensor = torch.tensor([token_ids]).to(DEVICE)
    segment_ids_tensor = torch.tensor([segment_ids]).to(DEVICE)
    attention_mask_tensor = torch.ones_like(input_ids_tensor).to(DEVICE)
    
    with torch.no_grad():
        _, nsp_logits = model(input_ids_tensor, segment_ids_tensor, attention_mask_tensor)
        
    # Get prediction (0 = IsNext, 1 = NotNext... wait, or is it 0=NotNext, 1=IsNext?)
    # Our dataset set 1 = IsNext, 0 = NotNext.
    # We should check the dataset.py... Yes, 1 = IsNext, 0 = NotNext.
    # The nsp_loss_fn (CrossEntropy) expects class indices.
    # torch.argmax will give us the predicted class (0 or 1)
    
    prediction_idx = torch.argmax(nsp_logits, dim=1).item()
    result = "IsNext" if prediction_idx == 1 else "NotNext"
    is_correct = (prediction_idx == 1) == actual_is_next
    
    print(f"\nSentence A: {sent_a}")
    print(f"Sentence B: {sent_b}")
    print(f"Model Prediction: {result}")
    print(f"Correct: {is_correct}")

# Run demonstrations
# Note: A model trained for 3 epochs on a small dataset will be very random.
# These predictions will likely be poor, but it proves the code runs.
predict_masked_token("The capital of France is [MASK] .")
predict_masked_token("Hello, I am a [MASK] model.")

# Positive NSP Example
check_nsp("The dog ran.", "It was fast.", actual_is_next=True)
# Negative NSP Example
check_nsp("The dog ran.", "Quantum physics is complex.", actual_is_next=False)