"""CSWinUNETR encoder and decoder for thin-structure segmentation."""

import math
from itertools import pairwise

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention import CSWinBlock
from .sdsconv import SDSConv


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims, kernel):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv = conv(in_channels, out_channels, kernel, stride=2, padding=kernel // 2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x).movedim(1, -1)).movedim(-1, 1).contiguous()


def _residual_block(in_channels, out_channels, spatial_dims):
    return UnetrBasicBlock(
        spatial_dims, in_channels, out_channels, 3, 1, "instance", res_block=True
    )


class EncoderStage(nn.Module):
    """Residual convolution, SDSConv, CSWin blocks, and downsampling."""

    def __init__(
        self,
        channels,
        heads,
        spatial_dims,
        use_sdsconv,
        kernel_size,
        stripe_width,
        pool_ratio,
        strip_kernel,
        global_attention,
        drop_paths,
        mlp_ratio,
    ):
        super().__init__()
        self.refine = _residual_block(channels, channels, spatial_dims)
        self.sdsconv = (
            SDSConv(channels, spatial_dims, kernel_size=kernel_size) if use_sdsconv else None
        )
        self.blocks = nn.Sequential(
            *[
                CSWinBlock(
                    channels,
                    heads,
                    spatial_dims,
                    stripe_width=stripe_width,
                    pool_ratio=pool_ratio,
                    strip_kernel=strip_kernel,
                    shifted=bool(block % 2),
                    global_attention=global_attention,
                    drop_path=drop_path,
                    mlp_ratio=mlp_ratio,
                )
                for block, drop_path in enumerate(drop_paths)
            ]
        )
        self.merge = ConvEmbedding(channels, 2 * channels, spatial_dims, kernel=3)

    def forward(self, x):
        x = self.refine(x)
        if self.sdsconv is not None:
            x = x + self.sdsconv(x)
        return self.merge(self.blocks(x))


class CSWinUNETR(nn.Module):
    """Hierarchical CSWin encoder with a convolutional UNETR decoder.

    Args:
        in_channels: Input image channels.
        out_channels: Number of classes, including background.
        spatial_dims: 2 for images or 3 for volumes.
        use_checkpoint: Recompute encoder stages during backward to save memory.
        sds_stages: Stage-wise SDSConv flags, from shallow to deep.
        sds_kernel_sizes: SDSConv kernel lengths in the same stage order.
        feature_size: Base embedding channels, doubled at each merge.
        depths: Transformer block counts for the four encoder stages.
        num_heads: Attention heads per stage; None uses dimension-specific defaults.
        stripe_widths: Attention stripe widths per stage.
        pool_ratios: Multi-scale K/V pooling ratios per stage.
        strip_kernel_sizes: Axial depthwise kernel lengths per stage.
        mlp_ratio: Transformer MLP hidden width relative to stage channels.
        drop_path_rate: Maximum stochastic-depth probability.

    Inputs are NCHW or NCDHW; each spatial dimension must be a multiple of 32
    and at least 64. Outputs are unnormalized logits at the input resolution.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=2,
        spatial_dims=2,
        use_checkpoint=False,
        sds_stages=(True, False, False, False),
        sds_kernel_sizes=(11, 9, 7, 5),
        *,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=None,
        stripe_widths=(1, 2, 7, 7),
        pool_ratios=(8, 4, 2, 1),
        strip_kernel_sizes=(9, 7, 5, 3),
        mlp_ratio=4.0,
        drop_path_rate=0.2,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        if in_channels < 1 or out_channels < 2:
            raise ValueError(
                "in_channels must be positive; out_channels must include background and foreground"
            )
        if (
            not isinstance(sds_stages, (tuple, list))
            or len(sds_stages) != 4
            or any(not isinstance(flag, (bool, int)) or flag not in (0, 1) for flag in sds_stages)
        ):
            raise ValueError("sds_stages must contain four booleans or 0/1 flags")
        if not isinstance(feature_size, int) or feature_size < 1:
            raise ValueError("feature_size must be a positive integer")
        if num_heads is None:
            num_heads = tuple((2 if spatial_dims == 2 else 3) * 2**i for i in range(4))
        for name, values in (
            ("depths", depths),
            ("num_heads", num_heads),
            ("stripe_widths", stripe_widths),
            ("pool_ratios", pool_ratios),
            ("strip_kernel_sizes", strip_kernel_sizes),
            ("sds_kernel_sizes", sds_kernel_sizes),
        ):
            if (
                not isinstance(values, (tuple, list))
                or len(values) != 4
                or any(not isinstance(value, int) or value < 1 for value in values)
            ):
                raise ValueError(f"{name} must contain four positive integers")
        if any(k < 5 or k % 2 == 0 for k in sds_kernel_sizes):
            raise ValueError("sds_kernel_sizes must contain four odd integers of at least 5")
        if any(k % 2 == 0 for k in strip_kernel_sizes):
            raise ValueError("strip_kernel_sizes must be odd")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(feature_size * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden width")
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")
        channels = [feature_size * 2**i for i in range(5)]
        for i, heads in enumerate(num_heads):
            branches = 1 if i == 3 else spatial_dims
            if channels[i] % heads or heads % branches:
                raise ValueError(
                    f"Stage {i + 1}: channels ({channels[i]}) must be divisible by heads ({heads}), "
                    f"and heads must be divisible by attention branches ({branches})"
                )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.use_checkpoint = use_checkpoint
        self.feature_size = feature_size
        self.patch_embed = ConvEmbedding(in_channels, feature_size, spatial_dims, kernel=7)
        total_blocks = sum(depths)
        drop_paths = [drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]
        stages = []
        offset = 0
        for i, depth in enumerate(depths):
            stages.append(
                EncoderStage(
                    channels=channels[i],
                    heads=num_heads[i],
                    spatial_dims=spatial_dims,
                    use_sdsconv=sds_stages[i],
                    kernel_size=sds_kernel_sizes[i],
                    stripe_width=stripe_widths[i],
                    pool_ratio=pool_ratios[i],
                    strip_kernel=strip_kernel_sizes[i],
                    global_attention=i == 3,
                    drop_paths=drop_paths[offset : offset + depth],
                    mlp_ratio=mlp_ratio,
                )
            )
            offset += depth
        self.stages = nn.ModuleList(stages)
        self.patch_embed.apply(self._initialize_encoder)
        self.stages.apply(self._initialize_encoder)
        self.input_skip = _residual_block(in_channels, feature_size, spatial_dims)
        self.skip_blocks = nn.ModuleList(
            [_residual_block(c, c, spatial_dims) for c in channels[:3]]
        )
        self.bottleneck = _residual_block(channels[-1], channels[-1], spatial_dims)
        decoder_channels = channels[::-1] + [feature_size]
        self.decoder = nn.ModuleList(
            [
                UnetrUpBlock(spatial_dims, cin, cout, 3, 2, "instance", res_block=True)
                for cin, cout in pairwise(decoder_channels)
            ]
        )
        self.output = UnetOutBlock(spatial_dims, feature_size, out_channels)

    @staticmethod
    def _initialize_encoder(module):
        if getattr(module, "_zero_init", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
            fan_out = math.prod(module.kernel_size) * module.out_channels // module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != self.spatial_dims + 2 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected a {self.spatial_dims}D input with {self.in_channels} channels; got {tuple(x.shape)}"
            )
        if any(size < 64 or size % 32 for size in x.shape[2:]):
            raise ValueError("Spatial dimensions must be multiples of 32 and at least 64")
        features = [self.patch_embed(x)]
        for stage in self.stages:
            current = features[-1]
            current = (
                checkpoint(stage, current, use_reentrant=False)
                if self.use_checkpoint and self.training
                else stage(current)
            )
            features.append(current)

        # Normalize decoder features without changing the encoder's residual stream.
        features = [
            F.layer_norm(feature.movedim(1, -1), (feature.shape[1],)).movedim(-1, 1)
            for feature in features
        ]
        skips = (
            [self.input_skip(x)]
            + [block(feature) for block, feature in zip(self.skip_blocks, features[:3])]
            + [features[3]]
        )
        current = self.bottleneck(features[4])
        for upsample, skip in zip(self.decoder, reversed(skips)):
            current = upsample(current, skip)
        return self.output(current)
