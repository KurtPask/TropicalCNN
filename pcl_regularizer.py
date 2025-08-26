#!/usr/bin/env python
import torch
import torch.nn as nn


class Proximity(nn.Module):
    def __init__(
        self,
        device,
        num_classes: int = 100,
        feat_dim: int = 1024,
        distance_metric: str = "l2",  # "l2" or "tropical"
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if distance_metric not in {"l2", "tropical"}:
            raise ValueError(
                f"Unsupported metric {distance_metric!r}. Choose 'l2' or 'tropical'."
            )
        self.distance_metric = distance_metric

        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim, device=device)
        )

    def forward(self, x: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """
        Compute mean distance between features and their class centers.
        Supports standard L2 or tropical symmetric distance.
        """
        B, C = x.size(0), self.num_classes

        if self.distance_metric == "l2":
            x_norm2 = x.pow(2).sum(dim=1, keepdim=True)
            c_norm2 = self.centers.pow(2).sum(dim=1, keepdim=True).T
            distmat = x_norm2 + c_norm2
            distmat = distmat.addmm(x, self.centers.t(), beta=1, alpha=-2)
            class_range = torch.arange(C, device=x.device)
            mask = labels.view(-1, 1) == class_range.view(1, -1)
            target_dists = distmat[mask]
            target_dists = target_dists.clamp(min=1e-12, max=1e12)
            return target_dists.mean()
        else:
            centers_labels = self.centers[labels]
            diff = x - centers_labels
            min_diff, _ = diff.min(dim=1)
            max_diff, _ = diff.max(dim=1)
            tropical_dists = max_diff - min_diff
            return tropical_dists.mean()


class ConProximity(nn.Module):
    def __init__(
        self,
        device,
        num_classes: int = 100,
        feat_dim: int = 1024,
        distance_metric: str = "l2",  # "l2" or "tropical"
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if distance_metric not in {"l2", "tropical"}:
            raise ValueError(
                f"Unsupported metric {distance_metric!r}. Choose 'l2' or 'tropical'."
            )
        self.distance_metric = distance_metric

        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim, device=device)
        )

    def forward(self, x: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """
        Compute mean distance to non-target class centers using
        either L2 or tropical symmetric distance.
        """
        B, C = x.size(0), self.num_classes
        class_range = torch.arange(C, device=x.device)
        mask_target = labels.view(-1, 1) == class_range.view(1, -1)

        if self.distance_metric == "l2":
            x_norm2 = x.pow(2).sum(dim=1, keepdim=True)
            c_norm2 = self.centers.pow(2).sum(dim=1, keepdim=True).T
            distmat = x_norm2 + c_norm2
            distmat = distmat.addmm(x, self.centers.t(), beta=1, alpha=-2)
            non_target_dists = distmat[~mask_target]
            non_target_dists = non_target_dists.clamp(min=1e-12, max=1e12)
            return non_target_dists.mean()
        else:
            diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
            diff_min = diff.min(dim=2).values
            diff_max = diff.max(dim=2).values
            trop_mat = diff_max - diff_min
            non_target_trop = trop_mat[~mask_target]
            return non_target_trop.mean()
