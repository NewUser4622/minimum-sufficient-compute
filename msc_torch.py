"""
msc_torch.py -- model-side components for MSC-KD.

Contains the four pieces that touch the network:

  1. ExitHead / MultiExitWrapper -- the depth budget axis
  2. OrdinalSufficiencyHead      -- monotone sufficiency curve, by construction
  3. MSCLoss                     -- the three-term objective
  4. learn_then_test_threshold   -- distribution-free routing calibration

Companion to msc_core.py, which handles the oracle and all analysis.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Depth budget axis
# ---------------------------------------------------------------------------

class ExitHead(nn.Module):
    """Lightweight classifier attached at an intermediate depth.

    Kept deliberately small -- pool, normalise, project. A heavier head would
    do its own representation learning, which would confound the measurement:
    we want to read off what the backbone has computed by this depth, not
    what a capable head can recover from it.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.fc(self.norm(x))


class MultiExitWrapper(nn.Module):
    """Attach K exit heads to a frozen backbone.

    The backbone MUST be frozen while heads are trained. If it adapts, each
    exit is reading a different network and the "same model under reduced
    compute" interpretation -- which the whole MSC construct rests on --
    collapses.
    """

    def __init__(
        self,
        backbone: nn.Module,
        stage_channels: Sequence[int],
        num_classes: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList(
            [ExitHead(c, num_classes) for c in stage_channels]
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep the backbone in eval mode so BN statistics never move."""
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns K logit tensors, shallowest first."""
        feats = self.backbone.forward_features(x)   # backbone must expose this
        return [h(f) for h, f in zip(self.heads, feats)]


@torch.no_grad()
def sweep_axis(
    model: nn.Module,
    loader,
    n_configs: int,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Run a budget sweep and emit the per-sample arrays the oracle needs.

    Output feeds directly into msc_core.compute_msc.
    """
    model.eval()
    preds, top1p, top2p, labels = [], [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits_per_config = model(x)                # list of K tensors
        assert len(logits_per_config) == n_configs

        p = torch.stack([F.softmax(l, dim=1) for l in logits_per_config], dim=1)
        top2 = p.topk(2, dim=2)

        preds.append(top2.indices[:, :, 0].cpu().numpy())
        top1p.append(top2.values[:, :, 0].cpu().numpy())
        top2p.append(top2.values[:, :, 1].cpu().numpy())
        labels.append(y.numpy())

    return {
        "preds": np.concatenate(preds),
        "top1p": np.concatenate(top1p),
        "top2p": np.concatenate(top2p),
        "labels": np.concatenate(labels),
    }


# ---------------------------------------------------------------------------
# 2. Monotone ordinal sufficiency head
# ---------------------------------------------------------------------------

class OrdinalSufficiencyHead(nn.Module):
    """Predicts a sufficiency curve that is non-decreasing in budget, by design.

    Cumulative-link ordinal regression: a scalar u(x) plus K-1 ordered
    thresholds, with ordering enforced through a softplus cumulative sum:

        theta_1 = t_1,  theta_{k+1} = theta_k + softplus(delta_k)
        s_k(x)  = sigmoid(theta_k - u(x))

    Since theta is increasing, s_k is non-decreasing in k automatically.

    This replaces the auxiliary monotonicity penalty from the earlier plan.
    An architectural constraint beats a soft penalty on three counts: it
    cannot be violated, it adds no hyperparameter, and it cannot trade off
    against the other loss terms during optimisation.
    """

    def __init__(self, in_features: int, n_budgets: int, hidden: int = 128):
        super().__init__()
        self.n_budgets = n_budgets
        self.trunk = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.theta_0 = nn.Parameter(torch.zeros(1))
        self.deltas = nn.Parameter(torch.zeros(n_budgets - 1))

    def thresholds(self) -> torch.Tensor:
        steps = F.softplus(self.deltas) + 1e-4
        return torch.cat([self.theta_0, self.theta_0 + torch.cumsum(steps, 0)])

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns (B, K) PRE-SIGMOID scores `theta_k - u(x)`.

        The loss takes these, not probabilities: `F.binary_cross_entropy`
        refuses to run under AMP autocast, and the logit form is both
        autocast-safe and numerically stable. Monotonicity is unaffected --
        `thresholds()` is increasing and sigmoid is monotone. See D-21.
        """
        u = self.trunk(feat)                               # (B, 1)
        return self.thresholds().unsqueeze(0) - u

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns (B, K) sufficiency probabilities, non-decreasing along K."""
        return torch.sigmoid(self.logits(feat))

    @torch.no_grad()
    def route(self, feat: torch.Tensor, gamma: float) -> torch.Tensor:
        """Smallest budget index whose predicted sufficiency reaches gamma."""
        s = self.forward(feat)
        hit = s >= gamma
        # Monotone in k, so argmax finds the first True; fall back to full budget.
        return torch.where(
            hit.any(dim=1),
            hit.float().argmax(dim=1),
            torch.full((s.size(0),), self.n_budgets - 1, device=s.device),
        )


# ---------------------------------------------------------------------------
# 3. Objective
# ---------------------------------------------------------------------------

def sufficiency_targets(
    msc_teacher: torch.Tensor, rho: torch.Tensor
) -> torch.Tensor:
    """s_k = 1[rho_k >= MSC_T(x)] -- monotone by construction."""
    return (rho.unsqueeze(0) >= msc_teacher.unsqueeze(1)).float()


class MSCLoss(nn.Module):
    """L = L_CE + alpha * L_KD + beta * L_MSC

    Three terms, two weights. The earlier CEB-KD formulation had seven terms
    and six weights, which is unprovable at any realistic experiment budget
    and reads to a reviewer as "we tried everything". Feature, attention, and
    Pareto terms are deliberately absent.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 4.0,
        ignore_irreducible: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = temperature
        self.ignore_irreducible = ignore_irreducible

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        suff_logits: torch.Tensor,        # (B, K) PRE-SIGMOID, from .logits()
        suff_target: torch.Tensor,        # (B, K) from sufficiency_targets
        irreducible: torch.Tensor | None = None,   # (B,) bool
    ) -> tuple[torch.Tensor, dict[str, float]]:

        ce = F.cross_entropy(student_logits, labels)

        kd = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        ) * (self.T ** 2)

        # D-21: the logit form, not F.binary_cross_entropy on probabilities.
        # The latter raises under AMP autocast, and the `.clamp(1e-6, 1-1e-6)`
        # it needed was papering over a log(0) the fused kernel avoids.
        bce = F.binary_cross_entropy_with_logits(
            suff_logits, suff_target.to(suff_logits.dtype), reduction="none"
        ).mean(dim=1)

        if self.ignore_irreducible and irreducible is not None:
            keep = ~irreducible
            # Samples where the teacher itself is unconfident carry a
            # degenerate MSC == 1 target. Training on them teaches the router
            # "always spend everything" on exactly the inputs where the
            # teacher had no usable opinion.
            msc = bce[keep].mean() if keep.any() else bce.sum() * 0.0
        else:
            msc = bce.mean()

        total = ce + self.alpha * kd + self.beta * msc
        return total, {
            "loss": float(total.detach()),
            "ce": float(ce.detach()),
            "kd": float(kd.detach()),
            "msc": float(msc.detach()),
        }


# ---------------------------------------------------------------------------
# 4. Risk-controlled routing threshold
# ---------------------------------------------------------------------------

def learn_then_test_threshold(
    suff_pred: np.ndarray,        # (N, K) calibration-set sufficiency curves
    correct_at: np.ndarray,       # (N, K) bool: routed to k -> prediction correct?
    full_accuracy: float,
    epsilon: float = 0.01,
    delta: float = 0.05,
    grid: Sequence[float] | None = None,
) -> float:
    """Largest-savings gamma whose accuracy drop is provably below epsilon.

    Distribution-free: Learn-then-Test with a Hoeffding bound, testing
    candidate thresholds from most to least aggressive and stopping at the
    first that passes with fixed-sequence error control.

    This machinery is adopted, not claimed. Jazbec et al. (NeurIPS 2024)
    introduced risk control for early exit, and SAFE-KD (2602.03043) already
    pairs conformal risk control with early-exit distillation. Our
    differentiation is the supervision signal, not the calibration.
    """
    if grid is None:
        grid = np.linspace(0.99, 0.05, 60)

    n = suff_pred.shape[0]
    k_max = suff_pred.shape[1] - 1
    chosen = float(grid[0])

    # Fixed-sequence testing from conservative to aggressive: no multiplicity
    # correction needed, and we stop at the first failure.
    for gamma in grid:
        hit = suff_pred >= gamma
        route = np.where(hit.any(axis=1), hit.argmax(axis=1), k_max)
        acc = correct_at[np.arange(n), route].mean()

        # One-sided Hoeffding upper bound on the true accuracy drop.
        drop_hat = full_accuracy - acc
        slack = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
        if drop_hat + slack <= epsilon:
            chosen = float(gamma)
        else:
            break

    return chosen
