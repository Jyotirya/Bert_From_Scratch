import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Configuration ---
# We'll use a smaller, "simplified" config
# for this bootcamp project to train faster.
config = {
    "vocab_size": 30522,      # Standard BERT vocab size
    "hidden_size": 256,       # BERT-base is 768, we use 256
    "num_layers": 4,          # BERT-base is 12, we use 4
    "num_heads": 4,           # BERT-base is 12, we use 4
    "ff_size": 1024,          # Feed-forward size (usually 4*hidden_size)
    "max_len": 512,           # Max sequence length
    "dropout_prob": 0.1
}

# --- 1. Embedding Layer ---
# Combines Token, Position, and Segment embeddings

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, type_vocab_size=2):
        super().__init__()
        # Token Embeddings
        self.tok_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # Position Embeddings (learnable)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        # Segment Embeddings (A vs B sentence)
        self.seg_embed = nn.Embedding(type_vocab_size, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(config["dropout_prob"])

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        
        # Create position IDs (0, 1, 2, ..., seq_len-1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get the embeddings
        tok_embeddings = self.tok_embed(input_ids)
        pos_embeddings = self.pos_embed(pos_ids)
        seg_embeddings = self.seg_embed(segment_ids)
        
        # Add them all up
        embeddings = tok_embeddings + pos_embeddings + seg_embeddings
        
        # Apply LayerNorm and Dropout
        embeddings = self.norm(embeddings)
        embeddings = self.drop(embeddings)
        return embeddings

# --- 2. Multi-Head Self-Attention ---
# 
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size not divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # Output linear layer
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(config["dropout_prob"])

    def _split_heads(self, x, batch_size):
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        # Then transpose to (batch_size, num_heads, seq_len, head_dim)
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x, batch_size):
        # Transpose back to (batch_size, seq_len, num_heads, head_dim)
        # Then reshape to (batch_size, seq_len, hidden_size)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.num_heads * self.head_dim)

    def forward(self, x, attention_mask):
        batch_size = x.size(0)
        
        # 1. Project Q, K, V
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # 2. Split heads
        q = self._split_heads(q, batch_size)  # (B, n_heads, L, head_dim)
        k = self._split_heads(k, batch_size)  # (B, n_heads, L, head_dim)
        v = self._split_heads(v, batch_size)  # (B, n_heads, L, head_dim)
        
        # 3. Scaled Dot-Product Attention
        # (B, n_heads, L, head_dim) @ (B, n_heads, head_dim, L) -> (B, n_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        # Apply the attention mask
        # attention_mask is (B, 1, 1, L) or (B, 1, L, L)
        # It has -1e9 for masked positions, 0 for others
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # Softmax
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.drop(attn_probs)
        
        # (B, n_heads, L, L) @ (B, n_heads, L, head_dim) -> (B, n_heads, L, head_dim)
        context = torch.matmul(attn_probs, v)
        
        # 4. Combine heads
        context = self._combine_heads(context, batch_size)
        
        # 5. Output linear layer
        output = self.out_linear(context)
        return output

# --- 3. Feed-Forward Network ---
class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.activation = nn.GELU() # BERT uses GELU
        self.drop = nn.Dropout(config["dropout_prob"])

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x

# --- 4. Transformer Encoder Layer ---
# 
class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size, ff_size)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.drop1 = nn.Dropout(config["dropout_prob"])
        self.drop2 = nn.Dropout(config["dropout_prob"])

    def forward(self, x, attention_mask):
        # 1. Self-Attention + Residual + Norm
        attn_output = self.attention(x, attention_mask)
        x = self.norm1(x + self.drop1(attn_output)) # Add & Norm
        
        # 2. Feed-Forward + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_output)) # Add & Norm
        
        return x

# --- 5. Output Heads ---

# Head for Masked Language Modeling (MLM)
class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        # This final layer projects from hidden_size to vocab_size
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.decoder(x)
        return x

# Head for Next Sentence Prediction (NSP)
class NSPHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # This takes the [CLS] token's output
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, cls_output):
        return self.seq_relationship(cls_output)

# --- 6. The Full BERT Model ---
class SimplifiedBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = BertEmbeddings(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            max_len=config["max_len"]
        )
        
        self.encoder_layers = nn.ModuleList([
            BertEncoderLayer(
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                ff_size=config["ff_size"]
            ) for _ in range(config["num_layers"])
        ])
        
        self.mlm_head = MLMHead(config["hidden_size"], config["vocab_size"])
        self.nsp_head = NSPHead(config["hidden_size"])

    def forward(self, input_ids, segment_ids, attention_mask):
        # 1. Create the attention mask for self-attention
        # Input mask is (B, L) with 1s for tokens, 0s for padding
        # We need (B, 1, 1, L) with 0.0 for tokens, -1e9 for padding
        ext_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        ext_attention_mask = (1.0 - ext_attention_mask) * -1e9
        
        # 2. Get embeddings
        x = self.embeddings(input_ids, segment_ids)
        
        # 3. Pass through all encoder layers
        for layer in self.encoder_layers:
            x = layer(x, ext_attention_mask)
            
        # 4. Get outputs for the heads
        # The output for the [CLS] token (first token)
        cls_output = x[:, 0] 
        
        # The output for all tokens (for MLM)
        token_outputs = x
        
        # 5. Get logits
        mlm_logits = self.mlm_head(token_outputs)
        nsp_logits = self.nsp_head(cls_output)
        
        return mlm_logits, nsp_logits