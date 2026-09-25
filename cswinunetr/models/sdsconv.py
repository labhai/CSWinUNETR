"""Sparse-control dynamic snake convolution."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    groups = next(g for g in range(min(32, channels), 0, -1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class SDSConv(nn.Module):
    """Curvilinear sampling from sparse RBF-interpolated control points.

    Axial branches share control locations and coordinate-offset fields;
    each branch uses the offsets orthogonal to its traversal direction.
    """

    def __init__(self, channels: int, spatial_dims: int, kernel_size: int = 11):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        if kernel_size < 5 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 5")
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.control_points = kernel_size // 2
        self.sigma = 0.35
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        offset_channels = (spatial_dims + 1) * self.control_points
        self.offset_conv = nn.Sequential(
            conv(channels, channels, 3, padding=1, groups=channels),
            conv(channels, offset_channels, 1),
        )
        self.offset_norm = group_norm(offset_channels)
        self.register_buffer("axis", torch.arange(kernel_size).float() - kernel_size // 2)
        self.register_buffer("control_base", torch.linspace(-1, 1, self.control_points))
        self.log_scope_base = nn.Parameter(torch.zeros(spatial_dims))
        self.log_scope_amp = nn.Parameter(torch.full((spatial_dims,), -3.0))

        self.aggregators = nn.ModuleList()
        for axis in range(spatial_dims):
            kernel = tuple(kernel_size if d == axis else 1 for d in range(spatial_dims))
            self.aggregators.append(
                nn.Sequential(
                    conv(channels, channels, kernel, stride=kernel, groups=channels),
                    conv(channels, channels, 1),
                    group_norm(channels),
                    nn.ReLU(),
                )
            )
        hidden = min(channels, max(channels // 8, 32))
        self.fuse = nn.Sequential(
            conv(spatial_dims * channels, hidden, 1, bias=False),
            group_norm(hidden),
            nn.GELU(),
            conv(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            group_norm(hidden),
            nn.GELU(),
            conv(hidden, channels, 1),
        )
        self.output_norm = group_norm(channels)
        # Initialize straight trajectories and a zero residual.
        for layer in (self.offset_conv[-1], self.fuse[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer._zero_init = True

    def dense_offsets(self, controls: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """RBF interpolation and center anchoring: (B, axes, M, ...) -> (B, axes, P, ...)."""
        extra = (1,) * self.spatial_dims
        base = self.control_base.view(1, self.control_points, *extra)
        mu = (base + positions / (self.control_points - 1)).clamp(-1, 1)
        xi = (self.axis / (self.kernel_size // 2)).view(1, 1, self.kernel_size, *extra)
        weights = (-(xi - mu.unsqueeze(2)).square() / (2 * self.sigma**2)).softmax(1)
        dense = (controls.unsqueeze(3) * weights.unsqueeze(1)).sum(2)
        center = self.kernel_size // 2
        return dense - dense[:, :, center : center + 1]

    def scope(self) -> torch.Tensor:
        """Positive, bounded, monotonic distance ramp for each axial branch."""
        distance = self.axis.abs() / (self.kernel_size // 2)
        log_scale = (
            self.log_scope_base[:, None] + F.softplus(self.log_scope_amp)[:, None] * distance
        )
        return (log_scale.tanh() * math.log(4.0)).exp()

    def _sample(self, x: torch.Tensor, offsets: torch.Tensor, axis: int) -> torch.Tensor:
        batch, _, *spatial = x.shape
        points = self.kernel_size
        support = self.axis.view(1, points, *((1,) * self.spatial_dims))
        scale = self.scope()[axis].view(1, points, *((1,) * self.spatial_dims))
        # Interleave each center with its P samples along the traversal axis.
        order = [0] + list(range(2, self.spatial_dims + 2))
        order.insert(axis + 2, 1)
        output_shape = list(spatial)
        output_shape[axis] *= points
        coordinates = []
        for dimension, size in enumerate(spatial):
            shape = [1, 1] + [1] * self.spatial_dims
            shape[dimension + 2] = size
            base = torch.arange(size, device=x.device, dtype=x.dtype).view(shape)
            delta = support if dimension == axis else offsets[:, dimension] * scale
            coordinate = (base + delta).expand(batch, points, *spatial)
            coordinate = coordinate.permute(order).reshape(batch, *output_shape)
            coordinates.append(2 * coordinate / max(size - 1, 1) - 1)
        grid = torch.stack(coordinates[::-1], dim=-1)  # grid_sample expects x,y[,z].
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted = self.offset_norm(self.offset_conv(x)).tanh()
        # Keep subpixel coordinates in float32 under mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            *controls, positions = predicted.float().split(self.control_points, dim=1)
            offsets = self.dense_offsets(torch.stack(controls, dim=1), positions)
            sampling_input = x.float()
            samples = [
                self._sample(sampling_input, offsets, axis) for axis in range(self.spatial_dims)
            ]
        responses = [layer(sample.to(x.dtype)) for layer, sample in zip(self.aggregators, samples)]
        return self.output_norm(self.fuse(torch.cat(responses, dim=1)))
