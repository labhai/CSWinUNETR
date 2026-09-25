"""Shifted cross-shaped stripe attention with detail-enhanced multi-scale K/V."""

from itertools import product
from math import prod

import torch
from monai.networks.layers import DropPath
from torch import nn
from torch.nn import functional as F


def partition_windows(x: torch.Tensor, window: tuple[int, ...]) -> torch.Tensor:
    """Channels-last feature map -> (batch * windows, tokens, channels)."""
    batch, *spatial, channels = x.shape
    shape = [batch]
    for size, width in zip(spatial, window):
        shape.extend((size // width, width))
    shape.append(channels)
    dims = len(window)
    order = [0] + list(range(1, 2 * dims, 2)) + list(range(2, 2 * dims + 1, 2)) + [2 * dims + 1]
    return x.reshape(shape).permute(order).reshape(-1, prod(window), channels)


def reverse_windows(
    x: torch.Tensor, spatial: tuple[int, ...], window: tuple[int, ...]
) -> torch.Tensor:
    groups = tuple(size // width for size, width in zip(spatial, window))
    batch = x.shape[0] // prod(groups)
    dims = len(window)
    x = x.reshape(batch, *groups, *window, x.shape[-1])
    order = [0]
    for i in range(dims):
        order.extend((1 + i, 1 + dims + i))
    return x.permute(*order, 2 * dims + 1).reshape(batch, *spatial, x.shape[-1])


class MultiScaleKV(nn.Module):
    """Fuse axial, local, pooled and high-pass branches with a spatial softmax."""

    def __init__(self, channels: int, spatial_dims: int, pool_ratio: int, strip_kernel: int):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.pool_ratio = pool_ratio
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.strips = nn.ModuleList()
        for axis in range(spatial_dims):
            kernel = tuple(strip_kernel if d == axis else 1 for d in range(spatial_dims))
            self.strips.append(
                conv(
                    channels,
                    channels,
                    kernel,
                    padding=tuple(k // 2 for k in kernel),
                    groups=channels,
                    bias=False,
                )
            )
        self.local = conv(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pooled = conv(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.detail = conv(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gate = conv(4 * channels, 4, 1)
        self.norm = nn.LayerNorm(channels)
        self.kv = nn.Linear(channels, 2 * channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gate._zero_init = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.movedim(-1, 1)
        pool = F.avg_pool2d if self.spatial_dims == 2 else F.avg_pool3d
        ratio = tuple(min(self.pool_ratio, size) for size in x.shape[2:])
        pooled = self.pooled(pool(x, ratio, stride=ratio, ceil_mode=True))
        if pooled.shape[2:] != x.shape[2:]:
            mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
            pooled = F.interpolate(pooled, size=x.shape[2:], mode=mode, align_corners=False)
        high_pass = x - pool(x, 3, stride=1, padding=1, count_include_pad=False)
        branches = [
            sum(layer(x) for layer in self.strips),
            self.local(x),
            pooled,
            self.detail(high_pass),
        ]
        weights = self.gate(torch.cat(branches, dim=1)).float().softmax(1).to(x.dtype)
        fused = sum(weights[:, i : i + 1] * branch for i, branch in enumerate(branches))
        return self.kv(self.norm(fused.movedim(1, -1))).chunk(2, dim=-1)


class StripeAttention(nn.Module):
    """Stripe attention with multi-scale K/V and depthwise local positional encoding."""

    def __init__(
        self, channels: int, heads: int, spatial_dims: int, pool_ratio: int, strip_kernel: int
    ):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(channels, channels)
        self.multiscale = MultiScaleKV(channels, spatial_dims, pool_ratio, strip_kernel)
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.lepe = conv(channels, channels, 3, padding=1, groups=channels)

    def forward(
        self, x: torch.Tensor, window: tuple[int, ...], mask: torch.Tensor | None
    ) -> torch.Tensor:
        spatial = x.shape[1:-1]
        keys, values = self.multiscale(x)
        q, k, v = [partition_windows(t, window) for t in (self.q(x), keys, values)]
        batch_windows, tokens, channels = v.shape
        v_image = v.reshape(batch_windows, *window, channels).movedim(-1, 1)
        lepe = self.lepe(v_image).movedim(1, -1).reshape_as(v)
        q, k, v = [
            t.reshape(batch_windows, tokens, self.heads, channels // self.heads).transpose(1, 2)
            for t in (q, k, v)
        ]
        if mask is not None:
            mask = mask.repeat(x.shape[0], 1, 1).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(batch_windows, tokens, channels) + lepe
        return reverse_windows(out, spatial, window)


def attention_mask(spatial, window, shift, valid):
    """Exclude wrapped stripe neighbors and padded keys; True means allowed."""
    labels = torch.zeros((1, *spatial, 1), device=valid.device, dtype=torch.int32)
    regions = []
    for size, width in zip(spatial, window):
        if shift == 0 or width == size:
            regions.append((slice(None),))
        else:
            regions.append((slice(0, -width), slice(-width, -shift), slice(-shift, None)))
    for label, region in enumerate(product(*regions)):
        labels[(slice(None), *region, slice(None))] = label
    labels = partition_windows(labels, window).squeeze(-1)
    allowed = labels.unsqueeze(1) == labels.unsqueeze(2)
    valid_keys = partition_windows(valid, window).squeeze(-1)
    return allowed & valid_keys.unsqueeze(1)


class CSWinBlock(nn.Module):
    """Pre-norm Transformer block with optional shifted stripe attention."""

    def __init__(
        self,
        channels,
        heads,
        spatial_dims,
        stripe_width,
        pool_ratio,
        strip_kernel,
        shifted,
        global_attention,
        drop_path,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.stripe_width = stripe_width
        self.global_attention = global_attention
        self.shift = stripe_width // 2 if shifted and not global_attention else 0
        branches = 1 if global_attention else spatial_dims
        self.norm1 = nn.LayerNorm(channels)
        self.attentions = nn.ModuleList(
            [
                StripeAttention(
                    channels // branches, heads // branches, spatial_dims, pool_ratio, strip_kernel
                )
                for _ in range(branches)
            ]
        )
        self.proj = nn.Linear(channels, channels)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(channels)
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[2:]
        shortcut = x.movedim(1, -1)
        normalized = self.norm1(shortcut).movedim(-1, 1)
        padding = tuple((-size) % self.stripe_width for size in spatial)
        pad = tuple(value for amount in padding[::-1] for value in (0, amount))
        padded = F.pad(normalized, pad).movedim(1, -1)
        valid = F.pad(torch.ones((1, 1, *spatial), device=x.device, dtype=torch.bool), pad).movedim(
            1, -1
        )
        axes = tuple(range(1, self.spatial_dims + 1))
        if self.shift:
            padded = torch.roll(padded, (-self.shift,) * self.spatial_dims, axes)
            valid = torch.roll(valid, (-self.shift,) * self.spatial_dims, axes)
        padded_shape = padded.shape[1:-1]
        channel_groups = padded.chunk(len(self.attentions), dim=-1)
        outputs = []
        for axis, (attention, branch) in enumerate(zip(self.attentions, channel_groups)):
            window = tuple(
                size if self.global_attention or d == axis else self.stripe_width
                for d, size in enumerate(padded_shape)
            )
            mask = (
                attention_mask(padded_shape, window, self.shift, valid)
                if self.shift or any(padding)
                else None
            )
            outputs.append(attention(branch, window, mask))
        out = self.proj(torch.cat(outputs, dim=-1))
        if self.shift:
            out = torch.roll(out, (self.shift,) * self.spatial_dims, axes)
        out = out[(slice(None), *(slice(0, size) for size in spatial), slice(None))]
        out = shortcut + self.drop_path(out)
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        return out.movedim(-1, 1).contiguous()
