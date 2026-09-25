"""Segmentation evaluation metrics."""

import torch


def foreground_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image binary Dice; a jointly empty mask receives a score of one."""
    prediction, target = prediction.bool().flatten(1), target.bool().flatten(1)
    intersection = (prediction & target).sum(1)
    total = prediction.sum(1) + target.sum(1)
    return (2 * intersection + 1e-6) / (total + 1e-6)


def thin_structure_metrics(prediction, target):
    """2D clDice, Betti errors (8-connected FG / 4-connected BG), and HD95.

    HD95 uses the 95th percentile of pooled bidirectional surface distances,
    in pixels. A single empty mask receives the image diagonal as its distance.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt, label
    from skimage.morphology import skeletonize

    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Expected aligned 2D binary masks")

    def betti(mask):
        components = label(mask, structure=np.ones((3, 3)))[1]
        background, count = label(~mask)
        border = np.unique(
            np.concatenate((background[0], background[-1], background[:, 0], background[:, -1]))
        )
        holes = count - np.count_nonzero(border)
        return components, holes

    prediction_betti, target_betti = betti(prediction), betti(target)
    errors = [abs(a - b) for a, b in zip(prediction_betti, target_betti)]
    if not prediction.any() and not target.any():
        cldice, hd95 = 1.0, 0.0
    elif not prediction.any() or not target.any():
        cldice, hd95 = 0.0, float(np.linalg.norm(np.array(target.shape) - 1))
    else:
        prediction_skeleton, target_skeleton = skeletonize(prediction), skeletonize(target)
        precision = (target & prediction_skeleton).sum() / prediction_skeleton.sum()
        sensitivity = (prediction & target_skeleton).sum() / target_skeleton.sum()
        cldice = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        )
        prediction_surface = prediction ^ binary_erosion(prediction)
        target_surface = target ^ binary_erosion(target)
        distances = np.concatenate(
            (
                distance_transform_edt(~target_surface)[prediction_surface],
                distance_transform_edt(~prediction_surface)[target_surface],
            )
        )
        hd95 = float(np.percentile(distances, 95))
    return {
        "cldice": float(cldice),
        "betti0_error": int(errors[0]),
        "betti1_error": int(errors[1]),
        "betti_error": int(sum(errors)),
        "hd95": hd95,
    }
