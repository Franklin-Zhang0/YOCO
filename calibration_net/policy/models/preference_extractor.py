"""Preference extractor for HyperLoRA conditioning.

This module encodes paired biased/true poses into a conditioning vector for HyperLoRA.
Uses Combined_CNN_MLP_MIXER to encode each pose, then aggregates pair embeddings via
a transformer with RoPE positional encoding.
"""

from torch import nn
import torch
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformer attention.
    
    Reference: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin for max sequence length
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (max_seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len, dim) input tensor
            seq_len: sequence length (if None, use x.shape[1])
        
        Returns:
            cos, sin: (1, seq_len, dim) cosine and sine embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        return (
            self.cos_cached[:seq_len].unsqueeze(0),  # (1, seq_len, dim)
            self.sin_cached[:seq_len].unsqueeze(0),  # (1, seq_len, dim)
        )


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor.
    
    Args:
        x: (B, seq_len, dim) input
        cos: (1, seq_len, dim) cosine embedding
        sin: (1, seq_len, dim) sine embedding
    
    Returns:
        x_rotated: (B, seq_len, dim) with RoPE applied
    """
    # Split x into two halves
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation
    # Interleave cos/sin
    cos_half = cos[..., ::2]
    sin_half = sin[..., ::2]
    
    # Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
    x_rotated = torch.stack([
        x1 * cos_half - x2 * sin_half,
        x1 * sin_half + x2 * cos_half
    ], dim=-1)
    
    # Flatten back
    x_rotated = x_rotated.flatten(-2)
    
    return x_rotated


class TransformerEncoderLayerWithRoPE(nn.Module):
    """TransformerEncoderLayer with RoPE positional encoding."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(d_model)
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (B, seq_len, d_model)
            src_key_padding_mask: (B, seq_len) True = ignore, False = attend
        
        Returns:
            output: (B, seq_len, d_model)
        """
        # Apply RoPE
        cos, sin = self.rope(src)
        src_rope = apply_rotary_pos_emb(src, cos, sin)
        
        # Self-attention with RoPE
        src2, _ = self.self_attn(
            src_rope, src_rope, src,  # Q, K with RoPE; V without
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class PairedPoseExtractor(nn.Module):
    """Paired-pose preference extractor with token-level processing, role embeddings, and RoPE.

    Architecture:
        1. Encode each pose (biased, true) using shared CNN-MLP-MIXER encoder -> token features (B, 6, D)
        2. Add role/type embeddings to distinguish biased vs true tokens
        3. Concatenate biased tokens + true tokens per pair -> (B, K, 12, D) sequence
        4. Add CLS token and flatten to (B, 1 + K*12, D)
        5. Apply N-layer TransformerEncoder with RoPE (shared across HyperLoRA modules)
        6. Pool into conditioning vector(s):
           - Default: use CLS token -> (B, output_dim)
           - Per-module pooling: learn M query vectors that attend over the shared transformer
             output -> (B, M, output_dim). Each HyperLoRA HyperBlock consumes its own slice.

    Args:
        pose_encoder: Shared encoder (e.g., Combined_CNN_MLP_MIXER) with extract_features() method
        pair_fusion_dim: Hidden dimension for transformer (default: 256)
        output_dim: Final conditioning vector dimension (default: 256)
        num_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        per_module_pooling: If True, produce per-HyperLoRA-module conditioning using attention pooling.
        num_conditioning_heads: Number of per-module conditioning vectors to produce (M).
    """

    def __init__(
        self,
        pose_encoder: nn.Module,
        pair_fusion_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        per_module_pooling: bool = False,
        num_conditioning_heads: int = 1,
    ):
        super().__init__()
        
        self.pose_encoder = pose_encoder
        self.n_layers = n_layers
        
        # Infer encoder output dimension (num_tokens, token_dim)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 21, 3)
            encoder_out = self.pose_encoder.extract_features(dummy_input)  # (1, num_tokens, token_dim)
            self.num_tokens_per_pose = encoder_out.shape[1]  # Should be 6
            encoder_token_dim = encoder_out.shape[2]  # Should be 320
        
        # Project encoder tokens to transformer hidden dim
        self.token_proj = nn.Linear(encoder_token_dim, pair_fusion_dim)
        
        # Role/type embeddings: 0 = biased, 1 = true
        self.role_embeddings = nn.Embedding(2, pair_fusion_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, pair_fusion_dim))
        
        # Transformer encoder with RoPE (N layers) - shared
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                d_model=pair_fusion_dim,
                nhead=num_heads,
                dim_feedforward=pair_fusion_dim * 4,
                dropout=dropout,
                activation='gelu',
            )
            for _ in range(n_layers)
        ])

        # Optional: per-module attention pooling (shared transformer, different pooling per module)
        self.per_module_pooling = bool(per_module_pooling)
        self.num_conditioning_heads = int(num_conditioning_heads) if num_conditioning_heads is not None else 1
        if self.num_conditioning_heads < 1:
            raise ValueError(f"num_conditioning_heads must be >= 1, got {self.num_conditioning_heads}")

        if self.per_module_pooling and self.num_conditioning_heads > 1:
            # Learnable query vectors (M, D) that attend over transformer outputs.
            self.pool_queries = nn.Embedding(self.num_conditioning_heads, pair_fusion_dim)
            self.pool_attn = nn.MultiheadAttention(
                pair_fusion_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.pool_queries = None
            self.pool_attn = None

        # Final projection (shared across heads; works for (B, D) and (B, M, D))
        self.output_proj = nn.Sequential(
            nn.Linear(pair_fusion_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.pair_fusion_dim = pair_fusion_dim
        self.output_dim = output_dim
    
    def forward(self, paired_poses: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            paired_poses: (B, K, 2, 21, 3) where K=num_pairs, dim=2 is [biased, true]
            pair_mask: (B, K) boolean mask, True for valid pairs, False for padding
            
        Returns:
            conditioning_vector: (B, output_dim)
        """
        B, K, _, _, _ = paired_poses.shape
        
        # Reshape to (B*K*2, 21, 3) for batch encoding
        poses_flat = paired_poses.reshape(B * K * 2, 21, 3)
        
        # Encode all poses to token features
        pose_token_feats = self.pose_encoder.extract_features(poses_flat)  # (B*K*2, num_tokens, token_dim)
        num_tokens = pose_token_feats.shape[1]
        
        # Reshape back to (B, K, 2, num_tokens, token_dim)
        pose_token_feats = pose_token_feats.reshape(B, K, 2, num_tokens, -1)
        
        # Project tokens to transformer dimension
        pose_token_feats = self.token_proj(pose_token_feats)  # (B, K, 2, num_tokens, pair_fusion_dim)
        
        # Add role embeddings: 0 for biased, 1 for true
        # role_emb_biased = self.role_embeddings(torch.tensor(0, device=pose_token_feats.device))  # (pair_fusion_dim,)
        # role_emb_true = self.role_embeddings(torch.tensor(1, device=pose_token_feats.device))  # (pair_fusion_dim,)
        
        # pose_token_feats[:, :, 0, :, :] = pose_token_feats[:, :, 0, :, :] + role_emb_biased
        # pose_token_feats[:, :, 1, :, :] = pose_token_feats[:, :, 1, :, :] + role_emb_true
        
        # Concatenate biased + true tokens per pair: (B, K, 2*num_tokens, pair_fusion_dim)
        pair_tokens = pose_token_feats.reshape(B, K, 2 * num_tokens, self.pair_fusion_dim)
        
        # Flatten pairs into sequence: (B, K * 2*num_tokens, pair_fusion_dim)
        token_sequence = pair_tokens.reshape(B, K * 2 * num_tokens, self.pair_fusion_dim)
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, pair_fusion_dim)
        token_sequence = torch.cat([cls_tokens, token_sequence], dim=1)  # (B, 1 + K*2*num_tokens, pair_fusion_dim)
        
        # Create attention mask for padding
        # pair_mask: (B, K) True = valid, False = padding
        # We need to mask out all tokens from padded pairs
        # token_mask: (B, 1 + K*2*num_tokens) False = attend, True = ignore
        token_mask = torch.zeros(B, 1 + K * 2 * num_tokens, dtype=torch.bool, device=pair_mask.device)
        
        # CLS token is never masked (index 0)
        # For each pair k, tokens are at indices [1 + k*2*num_tokens : 1 + (k+1)*2*num_tokens]
        for k in range(K):
            start_idx = 1 + k * 2 * num_tokens
            end_idx = 1 + (k + 1) * 2 * num_tokens
            # If pair k is padding (pair_mask[:, k] == False), mask its tokens
            token_mask[:, start_idx:end_idx] = ~pair_mask[:, k].unsqueeze(1)
        
        # Apply transformer layers with RoPE
        for layer in self.transformer_layers:
            token_sequence = layer(token_sequence, src_key_padding_mask=token_mask)

        # Pool into conditioning
        if self.per_module_pooling and self.num_conditioning_heads > 1:
            # Queries: (B, M, D)
            queries = self.pool_queries.weight.unsqueeze(0).expand(B, -1, -1)
            pooled, _ = self.pool_attn(
                queries,
                token_sequence,
                token_sequence,
                key_padding_mask=token_mask,
                need_weights=False,
            )  # (B, M, D)
            conditioning_vector = self.output_proj(pooled)  # (B, M, output_dim)
            return conditioning_vector

        # Default: use CLS token
        cls_output = token_sequence[:, 0, :]  # (B, pair_fusion_dim)
        conditioning_vector = self.output_proj(cls_output)  # (B, output_dim)
        return conditioning_vector
    
    def train_forward(self, paired_poses: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward with optional regularization loss.
        
        Returns:
            conditioning_vector: (B, output_dim)
            loss: scalar regularization loss (currently zero)
        """
        conditioning_vector = self.forward(paired_poses, pair_mask)
        
        # Optional: add regularization
        loss = torch.tensor(0.0, device=conditioning_vector.device, dtype=conditioning_vector.dtype)
        
        return conditioning_vector, loss
