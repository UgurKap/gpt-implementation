# From Chapter 4
import torch
from torch import nn
from mha import MultiHeadAttention


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Use the last `context_size` tokens
        with torch.no_grad():
            logits = model(idx_cond)  # GPT prediction
        logits = logits[:, -1, :]  # Get the last row of predictions
        probas = torch.softmax(logits, dim=-1)  # Token probabilities
        idx_next = torch.argmax(
            probas, dim=-1, keepdim=True
        )  # Index of the most likely token
        idx = torch.cat((idx, idx_next), dim=1)  # Add the token to the input sequence

    return idx


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=cfg["emb_dim"], out_features=4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(in_features=4 * cfg["emb_dim"], out_features=cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg.get("drop_rate_mha", cfg["drop_rate"]),
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg.get("drop_rate_ff", cfg["drop_rate"]))

    def forward(self, x):
        shortcut = x  # Remember the original input
        x = self.norm1(x)  # Apply the first LayerNorm
        x = self.att(x)  # Apply causal multi-head attention
        x = self.drop_shortcut(x)  # Apply dropout to the context vectors
        x = x + shortcut  # Skip connection by re-adding the original input

        shortcut = x  # Another skip connection anchor
        x = self.norm2(x)  # LayerNorm
        x = self.ff(x)  # Feedforward with GELU
        x = self.drop_shortcut(x)  # Apply dropout
        x = x + shortcut  # Skip connection
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super(GPTModel, self).__init__()
        self.tok_emb = nn.Embedding(
            num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=cfg["context_length"], embedding_dim=cfg["emb_dim"]
        )
        self.drop_emb = nn.Dropout(cfg.get("drop_rate_emb", cfg["drop_rate"]))

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            in_features=cfg["emb_dim"], out_features=cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
