from torch import nn
import torch

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)  # (batch_size, seq_length, hidden_dim)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)  # (batch_size, seq_length, hidden_dim)
        x = self.norm1(attn_output + x)
        x = self.ffn(x)
        x = self.norm2(x + attn_output)
        return x  # (batch_size, seq_length, hidden_dim)

class TransformerExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim: int = 32, num_heads: int = 4, n_layers: int = 4, min_tokens: int = 0, max_tokens: int = 10):
        super(TransformerExtractor, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn_block = nn.ModuleList([
            AttentionBlock(hidden_dim, hidden_dim, num_heads) for _ in range(n_layers)
        ])
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert input_ids.size(1) <= self.max_tokens, f"Input tokens with size {input_ids.size()} exceed max_tokens: {self.max_tokens}"
        assert input_ids.size(1) >= self.min_tokens, f"Input tokens with size {input_ids.size()} less than min_tokens: {self.min_tokens}"
        
        x = self.embed(input_ids)  # (batch_size, seq_length, hidden_dim)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_length + 1, hidden_dim)
        attention_mask = torch.cat((torch.zeros((B, 1), dtype=torch.bool, device=x.device), attention_mask), dim=1)  # (batch_size, seq_length + 1)
        for layer in self.attn_block:
            x = layer(x, attention_mask)
        pooled_output = x[:, 0, :]  # Take the [CLS] token representation
        return pooled_output  # (batch_size, hidden_size)
    
    def train_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        random_mask = (torch.rand(attention_mask.shape[:2]) < 0.5).to(attention_mask.device)
        random_mask = attention_mask & random_mask
        
        original_feat = self.forward(input_ids, attention_mask)
        random_feat = self.forward(input_ids, random_mask)
        drift_loss = nn.functional.mse_loss(original_feat, random_feat)
        mean = torch.mean(original_feat, dim=0)
        std = torch.std(original_feat, dim=0)
        
        dist_loss = torch.mean((mean - 0.0) ** 2) + torch.mean((std - 1.0) ** 2)
        
        loss = drift_loss + dist_loss 
        loss = torch.tensor(0.0, device=input_ids.device)
        return original_feat, loss