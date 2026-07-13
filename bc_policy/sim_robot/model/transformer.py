from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=x.device) * -scale)
        emb = x[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        cond_dim: int,
        n_cond_tokens: int,
        n_layer: int = 16,
        n_head: int = 16,
        n_emb: int = 1024,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        n_cond_layers: int = 2,
        causal_attn: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_cond_tokens = n_cond_tokens
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_cond_tokens + 1, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        if causal_attn:
            mask = torch.triu(torch.ones(horizon, horizon), diagonal=1).bool()
            self.register_buffer("mask", mask.float().masked_fill(mask, float("-inf")))
        else:
            self.mask = None

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, sample: torch.Tensor, time: torch.Tensor | float, cond_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = sample.shape[0]
        if not torch.is_tensor(time):
            time = torch.full((batch_size,), float(time), device=sample.device, dtype=sample.dtype)
        elif time.ndim == 0:
            time = time[None].expand(batch_size).to(sample.device)
        else:
            time = time.to(sample.device)

        time_token = self.time_emb(time).unsqueeze(1)
        obs_tokens = self.cond_obs_emb(cond_tokens)
        cond = torch.cat([time_token, obs_tokens], dim=1)
        cond = self.drop(cond + self.cond_pos_emb[:, : cond.shape[1]])
        memory = self.encoder(cond)

        x = self.input_emb(sample)
        x = self.drop(x + self.pos_emb[:, : x.shape[1]])
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask)
        return self.head(self.ln_f(x))

