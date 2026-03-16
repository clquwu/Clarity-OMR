"""Stage-B DaViT encoder + RoPE decoder model with DoRA target wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "PyTorch is required for Stage-B model execution."
        ) from exc
    return torch, nn, F


def _require_timm():
    try:
        import timm
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "timm is required for Stage-B encoder. Install timm to proceed."
        ) from exc
    return timm


def _maybe_flash_attn():
    if str(os.environ.get("OMR_DISABLE_FLASH_ATTN", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return None
    try:
        from flash_attn import flash_attn_func
    except ModuleNotFoundError:
        return None
    return flash_attn_func


torch, nn, F = _require_torch()


@dataclass(frozen=True)
class StageBModelConfig:
    """Configuration values aligned with omr-final-plan Stage-B defaults."""

    vocab_size: int = 380
    encoder_dim: int = 768
    decoder_dim: int = 768
    decoder_layers: int = 8
    decoder_heads: int = 12
    max_decode_length: int = 512
    dora_rank: int = 32
    image_height: int = 192
    max_image_width: int = 2048
    pretrained_backbone: str = "davit_base.msft_in1k"
    contour_classes: int = 3

    @property
    def image_size(self) -> int:
        """Backward-compatible alias for legacy callers."""
        return self.image_height


def list_dora_target_modules() -> list[str]:
    """Return linear module names targeted for DoRA adaptation."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "cross_attn_q",
        "cross_attn_k",
        "cross_attn_v",
        "cross_attn_out",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def build_dora_config(rank: int = 32) -> dict[str, object]:
    """Return an adapter configuration dictionary consumed by training code."""
    return {
        "adapter_type": "dora",
        "rank": rank,
        "target_modules": list_dora_target_modules(),
        "alpha": rank,
        "dropout": 0.10,
    }


class _SpatialSineEmbedding(nn.Module):
    """Fixed 2D sine/cos positional encoding projected to encoder dim."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature_map.shape
        device = feature_map.device
        dtype = feature_map.dtype

        num_frequencies = max(1, (self.channels + 3) // 4)
        freq_positions = torch.arange(num_frequencies, device=device, dtype=dtype)
        if num_frequencies == 1:
            inv_freq = torch.ones_like(freq_positions)
        else:
            inv_freq = 1.0 / (10000 ** (freq_positions / float(num_frequencies - 1)))

        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        phase_y = (2.0 * torch.pi) * y.unsqueeze(-1) * inv_freq.unsqueeze(0)
        phase_x = (2.0 * torch.pi) * x.unsqueeze(-1) * inv_freq.unsqueeze(0)

        sin_y = phase_y.sin().unsqueeze(1).expand(-1, width, -1)
        cos_y = phase_y.cos().unsqueeze(1).expand(-1, width, -1)
        sin_x = phase_x.sin().unsqueeze(0).expand(height, -1, -1)
        cos_x = phase_x.cos().unsqueeze(0).expand(height, -1, -1)

        embedding = torch.cat([sin_y, cos_y, sin_x, cos_x], dim=-1).permute(2, 0, 1)
        if embedding.shape[0] < self.channels:
            repeats = (self.channels + embedding.shape[0] - 1) // embedding.shape[0]
            embedding = embedding.repeat(repeats, 1, 1)
        embedding = embedding[: self.channels, :, :]
        return embedding.unsqueeze(0).repeat(batch, 1, 1, 1)


class RMSNorm(nn.Module):
    """Root mean square normalization with learned scale only."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight


class RotaryEmbedding(nn.Module):
    """Applies rotary positional embedding on projected Q/K tensors."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, tensor: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        if tensor.dim() == 3:
            batch, length, dim = tensor.shape
            heads = None
        elif tensor.dim() == 4:
            batch, heads, length, dim = tensor.shape
        else:
            raise ValueError(f"RoPE expects 3D or 4D tensor, got rank {tensor.dim()}")
        if dim != self.dim:
            raise ValueError(f"RoPE input dim mismatch: expected {self.dim}, got {dim}")

        start = int(max(0, position_offset))
        pos = torch.arange(start, start + length, device=tensor.device, dtype=tensor.dtype)
        freqs = torch.einsum("n,d->nd", pos, self.inv_freq.to(dtype=tensor.dtype))
        emb = torch.cat((freqs, freqs), dim=-1)
        if heads is None:
            cos = emb.cos().unsqueeze(0).expand(batch, -1, -1)
            sin = emb.sin().unsqueeze(0).expand(batch, -1, -1)
        else:
            cos = emb.cos().unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)
            sin = emb.sin().unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)

        half = dim // 2
        first = tensor[..., :half]
        second = tensor[..., half:]
        rotated = torch.cat((-second, first), dim=-1)
        return (tensor * cos) + (rotated * sin)


def _reshape_to_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, dim = tensor.shape
    head_dim = dim // heads
    return tensor.view(batch, length, heads, head_dim).transpose(1, 2)


def _reshape_from_heads(tensor: torch.Tensor) -> torch.Tensor:
    batch, heads, length, head_dim = tensor.shape
    return tensor.transpose(1, 2).contiguous().view(batch, length, heads * head_dim)


def _run_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    flash_attn_func = _maybe_flash_attn()
    if flash_attn_func is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q_fa, k_fa, v_fa, causal=causal)
        return out.transpose(1, 2).contiguous()

    if q.is_cuda:
        disable_torch_flash = str(os.environ.get("OMR_DISABLE_TORCH_FLASH_SDP", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        disable_torch_mem_efficient = str(
            os.environ.get("OMR_DISABLE_TORCH_MEM_EFFICIENT_SDP", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        force_torch_math_only = str(os.environ.get("OMR_FORCE_TORCH_MATH_SDP", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        enable_flash = (not disable_torch_flash) and (not force_torch_math_only)
        enable_mem_efficient = (not disable_torch_mem_efficient) and (not force_torch_math_only)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=True,
            enable_mem_efficient=enable_mem_efficient,
        ):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class DavitEncoder(nn.Module):
    """Loads pretrained DaViT features and projects to plan encoder dim."""

    def __init__(
        self,
        encoder_dim: int,
        backbone_name: str,
        pretrained: bool,
    ):
        super().__init__()
        timm = _require_timm()
        disable_timm_fused = str(os.environ.get("OMR_DISABLE_TIMM_FUSED_ATTN", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        set_fused_attn = getattr(timm.layers, "set_fused_attn", None)
        if set_fused_attn is None and not disable_timm_fused:
            raise RuntimeError(
                "timm.layers.set_fused_attn is required to enforce encoder FlashAttention acceleration."
            )
        if set_fused_attn is not None:
            set_fused_attn(not disable_timm_fused)
        disable_backbone_pretrained = str(os.environ.get("OMR_DISABLE_BACKBONE_PRETRAINED", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        effective_pretrained = bool(pretrained and not disable_backbone_pretrained)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=effective_pretrained,
            features_only=True,
            out_indices=[-1],
            in_chans=3,
        )
        fused_attention_modules = 0
        for module in self.backbone.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = not disable_timm_fused
                fused_attention_modules += 1
        if fused_attention_modules == 0 and not disable_timm_fused:
            raise RuntimeError(
                f"Backbone '{backbone_name}' has no fused-attention modules; "
                "encoder FlashAttention path is unavailable."
            )
        self.fused_attention_modules = fused_attention_modules
        out_channels = int(self.backbone.feature_info.channels()[-1])
        self.proj = nn.Conv2d(out_channels, encoder_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        images = images.clamp(0.0, 1.0)
        features = self.backbone(images)
        feat = features[-1]
        return self.proj(feat)


class DeformableContextBlock(nn.Module):
    """2D learned-offset deformable sampler over encoder feature maps."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.offset_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, heads * 2),
        )
        self.value_proj = nn.Linear(dim, dim)
        self.mix = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, sequence: torch.Tensor, height: int, width: int) -> torch.Tensor:
        residual = sequence
        normed = self.norm(sequence)
        batch, token_count, channels = normed.shape
        if token_count != height * width:
            raise ValueError(
                f"DeformableContextBlock expected {height * width} tokens, got {token_count}."
            )

        offset_logits = self.offset_mlp(normed).view(batch, height, width, self.heads, 2)
        offsets = offset_logits.mean(dim=3)
        offset_x = offsets[..., 0].tanh() * (2.0 / max(1, width - 1))
        offset_y = offsets[..., 1].tanh() * (2.0 / max(1, height - 1))
        sampled_offsets = torch.stack([offset_x, offset_y], dim=-1)

        y_axis = torch.linspace(-1.0, 1.0, height, device=sequence.device, dtype=sequence.dtype)
        x_axis = torch.linspace(-1.0, 1.0, width, device=sequence.device, dtype=sequence.dtype)
        yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        sampling_grid = (base_grid + sampled_offsets).clamp(-1.0, 1.0)

        value_map = self.value_proj(normed).transpose(1, 2).reshape(batch, channels, height, width)
        sampled = F.grid_sample(
            value_map,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled_sequence = sampled.flatten(2).transpose(1, 2)
        mixed = self.mix(sampled_sequence)
        return residual + self.out(mixed)


class PositionalBridge(nn.Module):
    """Projects encoder feature map to decoder memory sequence."""

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.norm = nn.LayerNorm(decoder_dim)
        self.spatial_embedding = _SpatialSineEmbedding(encoder_dim)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, height, width = feature_map.shape
        enriched = feature_map + self.spatial_embedding(feature_map)
        flattened = enriched.flatten(2).transpose(1, 2)
        projected = self.proj(flattened)
        return self.norm(projected), (height, width)


class DecoderBlock(nn.Module):
    """Decoder block with RoPE self-attn and cross-attn to encoder memory."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("decoder dim must be divisible by number of heads")
        self.heads = heads
        self.norm1 = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = RMSNorm(dim)
        self.cross_attn_q = nn.Linear(dim, dim)
        self.cross_attn_k = nn.Linear(dim, dim)
        self.cross_attn_v = nn.Linear(dim, dim)
        self.cross_attn_out = nn.Linear(dim, dim)

        self.norm3 = RMSNorm(dim)
        ffn_hidden_dim = (dim * 8) // 3
        self.gate_proj = nn.Linear(dim, ffn_hidden_dim)
        self.up_proj = nn.Linear(dim, ffn_hidden_dim)
        self.down_proj = nn.Linear(ffn_hidden_dim, dim)
        # Backward-compatible aliases for older references.
        self.ffn_gate = self.gate_proj
        self.ffn_up = self.up_proj
        self.ffn_down = self.down_proj
        self.rope = RotaryEmbedding(dim // heads)
        self.dropout = nn.Dropout(0.1)

    def _self_attention(
        self,
        hidden: torch.Tensor,
        *,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed = self.norm1(hidden)
        q = _reshape_to_heads(self.q_proj(normed), self.heads)
        k_new = _reshape_to_heads(self.k_proj(normed), self.heads)
        v_new = _reshape_to_heads(self.v_proj(normed), self.heads)

        offset = 0
        if past_key_value is not None:
            offset = int(past_key_value[0].shape[2])

        q = self.rope(q, position_offset=offset)
        k_new = self.rope(k_new, position_offset=offset)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k_new), dim=2)
            v = torch.cat((past_v, v_new), dim=2)
            # Incremental decoding has no future positions in K/V.
            causal = False
        else:
            k = k_new
            v = v_new
            causal = True

        attn = _run_attention(q, k, v, causal=causal)
        attn = self.out_proj(_reshape_from_heads(attn))
        new_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            new_past = (k, v)
        return hidden + self.dropout(attn), new_past

    def prepare_cross_attention_memory(
        self,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = _reshape_to_heads(self.cross_attn_k(memory), self.heads)
        v = _reshape_to_heads(self.cross_attn_v(memory), self.heads)
        return k, v

    def _cross_attention(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        cross_attn_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        query = self.norm2(hidden)
        q = _reshape_to_heads(self.cross_attn_q(query), self.heads)
        if cross_attn_kv is None:
            k, v = self.prepare_cross_attention_memory(memory)
        else:
            k, v = cross_attn_kv
        attn = _run_attention(q, k, v, causal=False)
        attn = self.cross_attn_out(_reshape_from_heads(attn))
        return hidden + self.dropout(attn)

    def _ffn(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.norm3(hidden)
        gated = F.silu(self.gate_proj(normed)) * self.up_proj(normed)
        out = self.down_proj(gated)
        return hidden + self.dropout(out)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cross_attn_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden, new_past = self._self_attention(hidden, past_key_value=past_key_value, use_cache=use_cache)
        hidden = self._cross_attention(hidden, memory, cross_attn_kv=cross_attn_kv)
        return self._ffn(hidden), new_past


class StageBModel(nn.Module):
    """DaViT encoder + transformer decoder for staff token generation."""

    def __init__(self, config: StageBModelConfig):
        super().__init__()
        self.config = config
        self.encoder = DavitEncoder(
            encoder_dim=config.encoder_dim,
            backbone_name=config.pretrained_backbone,
            pretrained=True,
        )
        self.deformable_attention = DeformableContextBlock(
            dim=config.encoder_dim, heads=config.decoder_heads
        )
        self.positional_bridge = PositionalBridge(
            encoder_dim=config.encoder_dim, decoder_dim=config.decoder_dim
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.decoder_dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config.decoder_dim, config.decoder_heads) for _ in range(config.decoder_layers)]
        )
        self.decoder_norm = RMSNorm(config.decoder_dim)
        self.lm_head = nn.Linear(config.decoder_dim, config.vocab_size)
        self.contour_head = nn.Sequential(
            nn.Linear(config.decoder_dim, 128),
            nn.GELU(),
            nn.Linear(128, config.contour_classes),
        )
        self.max_decode_length = config.max_decode_length

    def encode_staff(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.encoder(images)
        batch, channels, height, width = feature_map.shape
        sequence = feature_map.flatten(2).transpose(1, 2)
        sequence = self.deformable_attention(sequence, height, width)
        sequence = sequence.transpose(1, 2).reshape(batch, channels, height, width)
        memory, (memory_height, memory_width) = self.positional_bridge(sequence)
        contour_logits = self.contour_head(memory.mean(dim=1))
        return memory, contour_logits

    def prepare_decoder_memory(
        self,
        memory: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(block.prepare_cross_attention_memory(memory) for block in self.decoder_blocks)

    def decode_tokens(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        encoder_kv_cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    ]:
        hidden = self.token_embedding(decoder_input_ids)
        if past_key_values is not None and len(past_key_values) != len(self.decoder_blocks):
            raise ValueError(
                f"past_key_values length mismatch: expected {len(self.decoder_blocks)}, got {len(past_key_values)}."
            )
        if encoder_kv_cache is not None and len(encoder_kv_cache) != len(self.decoder_blocks):
            raise ValueError(
                f"encoder_kv_cache length mismatch: expected {len(self.decoder_blocks)}, got {len(encoder_kv_cache)}."
            )
        next_past: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, block in enumerate(self.decoder_blocks):
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            layer_encoder_kv = encoder_kv_cache[layer_idx] if encoder_kv_cache is not None else None
            hidden, layer_next_past = block(
                hidden,
                memory,
                past_key_value=layer_past,
                use_cache=use_cache,
                cross_attn_kv=layer_encoder_kv,
            )
            if use_cache:
                if layer_next_past is None:
                    raise RuntimeError("Decoder block returned no cache while use_cache=True.")
                next_past.append(layer_next_past)
        hidden = self.decoder_norm(hidden)
        cache_tuple = tuple(next_past) if use_cache else None
        return self.lm_head(hidden), hidden, cache_tuple

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        *,
        return_aux: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **_: object,
    ):
        if images is None:
            images = pixel_values
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        if images is None or decoder_input_ids is None:
            raise ValueError("StageBModel.forward requires image tensor and decoder/input token tensor.")
        memory, contour_logits = self.encode_staff(images)
        logits, _, _ = self.decode_tokens(decoder_input_ids, memory)
        if return_aux:
            return {"logits": logits, "contour_logits": contour_logits}
        return logits


def build_stage_b_model(config: StageBModelConfig | None = None):
    """Construct the Stage-B model with plan-aligned default dimensions."""
    if config is None:
        config = StageBModelConfig()
    return StageBModel(config)


def run_stage_b_shape_smoke_test(
    *,
    batch_size: int = 2,
    image_height: int = 192,
    image_width: int = 1600,
    seq_len: int = 64,
) -> dict[str, Sequence[int] | int]:
    """Run a deterministic shape smoke test for Stage-B forward."""
    model = build_stage_b_model()
    image_width = max(1, min(image_width, model.config.max_image_width))
    images = torch.rand(batch_size, 1, image_height, image_width)
    tokens = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, seq_len))
    logits = model(images, tokens)
    return {
        "batch_size": batch_size,
        "image_height": image_height,
        "image_width": image_width,
        "seq_len": seq_len,
        "vocab_size": model.config.vocab_size,
        "logits_shape": list(logits.shape),
    }

