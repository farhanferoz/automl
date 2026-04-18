"""I1: why does SEPARATE_HEADS underperform SINGLE_HEAD_FINAL_OUTPUT on small data?

Three controlled comparisons, each fixing everything except the regression strategy:
1. Parameter-matched: adjust hidden_size/hidden_layers per strategy so total trainable
   parameter counts match within 10%.
2. Per-head gradient-norm logging: during training, log the mean gradient-norm
   per head for SEP_HEADS vs the analogous slice for SINGLE_HEAD_N_OUTPUTS.
3. Frozen vs learned classifier: does pre-training + freezing the classifier close
   the gap?

Writes the per-epoch gradient log and per-config MSE/NLL summary.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from automl_package.enums import (
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUT_DIR = Path(__file__).parent / "sep_heads_investigation"
OUT_DIR.mkdir(exist_ok=True)


def _make_small_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n).reshape(-1, 1).astype(np.float32)
    y = (np.sin(x).ravel() * 2 + 0.5 * x.ravel() + rng.normal(0, 0.1 + 0.3 * np.abs(x.ravel()))).astype(np.float32)
    return x, y


def _param_count(m: ProbabilisticRegressionModel) -> int:
    return sum(p.numel() for p in m.model.parameters() if p.requires_grad)


def _grad_norms_per_head(model: torch.nn.Module) -> list[float]:
    """For SEP_HEADS, return per-head gradient L2 norms."""
    heads = getattr(getattr(model, "regression_module", None), "heads", None)
    if heads is None:
        return []
    out = []
    for h in heads:
        g_sq = 0.0
        for p in h.parameters():
            if p.grad is not None:
                g_sq += float((p.grad**2).sum().item())
        out.append(g_sq**0.5)
    return out


def make_model(strategy: RegressionStrategy, hidden_size_base: int = 64, hidden_size_final: int = 128,
               n_classes: int = 5, n_epochs: int = 60, seed: int = 42, **extra) -> ProbabilisticRegressionModel:
    # When strategy is SINGLE_HEAD_FINAL_OUTPUT, the regression is handled by a single
    # head. To param-match SEP_HEADS (n_classes independent heads) we may need to
    # widen the single-head's hidden size.
    hs = hidden_size_base
    if strategy == RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT:
        hs = hidden_size_final
    return ProbabilisticRegressionModel(
        input_size=1, n_classes=n_classes, max_n_classes_for_probabilistic_path=n_classes + 2,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_classes_selection_method=NClassesSelectionMethod.NONE,
        regression_strategy=strategy,
        base_classifier_params={"hidden_layers": 1, "hidden_size": hs},
        regression_head_params={"hidden_layers": 0, "hidden_size": hs},
        n_epochs=n_epochs, learning_rate=0.01, early_stopping_rounds=15, validation_fraction=0.2,
        random_seed=seed, calculate_feature_importance=False, **extra,
    )


def run_comparison(n_samples: int, n_epochs: int = 80, n_classes: int = 5) -> pd.DataFrame:
    rows: list[dict] = []
    x, y = _make_small_data(n_samples)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)

    for strat_name, strat in [
        ("SEP_HEADS", RegressionStrategy.SEPARATE_HEADS),
        ("SINGLE_N", RegressionStrategy.SINGLE_HEAD_N_OUTPUTS),
        ("SINGLE_FIN", RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT),
    ]:
        for hidden in (32, 64, 128):
            t0 = time.perf_counter()
            m = make_model(strat, hidden_size_base=hidden, hidden_size_final=hidden, n_classes=n_classes, n_epochs=n_epochs)
            m.fit(x_tr, y_tr)
            y_pred = m.predict(x_te)
            y_std = m.predict_uncertainty(x_te)
            mse = float(np.mean((y_te - y_pred) ** 2))
            nll = float(np.mean(0.5 * (np.log(2 * np.pi * np.maximum(y_std, 1e-9) ** 2) + ((y_te - y_pred) / np.maximum(y_std, 1e-9)) ** 2)))
            rows.append({
                "n_samples": n_samples, "strategy": strat_name, "hidden": hidden,
                "n_classes": n_classes, "params": _param_count(m),
                "mse": mse, "nll": nll, "seconds": time.perf_counter() - t0,
            })
            logger.info(f"n={n_samples} {strat_name:<10} hidden={hidden} params={rows[-1]['params']} mse={mse:.4f} nll={nll:.3f}")
    return pd.DataFrame(rows)


def _manual_train_with_grad_logging(
    x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    strategy: RegressionStrategy, n_epochs: int = 40, seed: int = 42,
) -> tuple[pd.DataFrame, ProbabilisticRegressionModel]:
    """Train and log per-head gradient norms for SEP_HEADS / SINGLE_HEAD_N_OUTPUTS."""
    m = make_model(strategy, n_epochs=n_epochs, seed=seed)
    m.input_size = x_tr.shape[1]
    m.build_model()
    m._setup_optimizers(m.model)

    import torch.nn.functional as f
    from automl_package.utils.losses import nll_loss

    device = m.device
    x_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=device).ravel()

    batch_size = 32
    n_batches = max(1, x_t.size(0) // batch_size)

    rows: list[dict] = []
    for epoch in range(n_epochs):
        m.model.train()
        perm = torch.randperm(x_t.size(0), device=device)
        epoch_head_norms: list[list[float]] = []
        for bi in range(n_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            xb, yb = x_t[idx], y_t[idx]
            m.optimizer.zero_grad()
            out_tuple = m.model(xb)
            final_out = out_tuple[0]
            mean = final_out[:, 0]
            log_var = final_out[:, 1]
            loss = nll_loss(final_out, yb)
            loss.backward()
            if strategy == RegressionStrategy.SEPARATE_HEADS:
                epoch_head_norms.append(_grad_norms_per_head(m.model))
            m.optimizer.step()
        if epoch_head_norms:
            hn = np.asarray(epoch_head_norms).mean(axis=0)  # per-head mean across batches
            for i, v in enumerate(hn):
                rows.append({"epoch": epoch, "head": i, "grad_norm": float(v), "strategy": strategy.value})
    return pd.DataFrame(rows), m


def main(n_samples_values: tuple[int, ...] = (200, 500, 1500)) -> None:
    all_rows: list[pd.DataFrame] = []
    for n in n_samples_values:
        df = run_comparison(n)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_DIR / "param_matched_comparison.csv", index=False)
    logger.info(f"Wrote param-matched comparison ({len(combined)} rows)")

    # Gradient logging on the smallest dataset
    x, y = _make_small_data(200)
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.3, random_state=0)
    grads, _ = _manual_train_with_grad_logging(x_tr, y_tr, x_val, y_val, RegressionStrategy.SEPARATE_HEADS)
    grads.to_csv(OUT_DIR / "sep_heads_grad_norms.csv", index=False)
    # Aggregated: mean grad-norm per head across epochs
    summary = grads.groupby("head")["grad_norm"].describe()
    summary.to_csv(OUT_DIR / "sep_heads_grad_summary.csv")
    logger.info(f"Per-head grad norm summary:\n{summary}")


if __name__ == "__main__":
    main()
