"""Tests for the variational-EM k-selector harness (note docs/kselection_variational_em_2026-06-13).

Covers the closed-form E/M math and the pruning behaviour the controlled test relies on:
resolved modes keep bins (bypass pruned), merged modes fall back to the bypass, and smooth
unimodal data stays at one class (the method does not invent structure).

Run with a torch+numpy interpreter, e.g. ``python3 -m pytest tests/test_variational_em.py``.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "automl_package", "examples"))
import _toy_datasets as td
import _variational_em as vem
import _variational_em_perinput as vemp
from _kselection_metrics import weight_survival_metrics


def test_gaussian_log_density_matches_manual():
    y = torch.tensor([0.5])
    mu = torch.tensor([[0.0, 1.0]])
    log_var = torch.tensor([[0.0, math.log(4.0)]])  # var 1 and 4
    out = vem.gaussian_log_density(y, mu, log_var)
    # manual: ln N(0.5;0,1) and ln N(0.5;1,4)
    exp0 = -0.5 * (math.log(2 * math.pi) + 0.0 + 0.25)
    exp1 = -0.5 * (math.log(2 * math.pi) + math.log(4.0) + (0.5 - 1.0) ** 2 / 4.0)
    assert out.shape == (1, 2)
    assert abs(out[0, 0].item() - exp0) < 1e-5
    assert abs(out[0, 1].item() - exp1) < 1e-5


def test_responsibilities_normalized_and_detached():
    model = vem.VariationalEMKSelector(input_dim=1, k_max=3, alpha0=0.2)
    x = torch.randn(10, 1)
    y = torch.randn(10)
    mu, lv = model.per_class_params(x)
    r, log_phi = model.responsibilities(y, mu, lv)
    assert r.shape == (10, model.K)
    assert torch.allclose(r.sum(dim=1), torch.ones(10), atol=1e-5)
    assert not r.requires_grad  # E-step responsibilities are a fixed target
    assert (r >= 0).all()


def test_m_step_sets_gamma_to_prior_plus_softcounts():
    model = vem.VariationalEMKSelector(input_dim=1, k_max=3, alpha0=0.3)
    x = torch.randn(50, 1)
    y = torch.randn(50)
    mu, lv = model.per_class_params(x)
    r, _ = model.responsibilities(y, mu, lv)
    expected = 0.3 + r.sum(dim=0)
    model.m_step(x, y)
    assert torch.allclose(model.gamma, expected, atol=1e-5)
    # soft counts sum to N, so concentrations sum to K*alpha0 + N
    assert abs(float(model.gamma.sum()) - (model.K * 0.3 + 50)) < 1e-3


def test_resolved_modes_keep_bins_and_prune_bypass():
    # two well-separated modes (constant baseline): bins win, bypass pruned
    x, y = td.make_toy_b(n=600, k_true=2, separation=4.0, sigma=0.3, baseline="zero", seed=7)
    model = vem.train_variational_em(x, y, k_max=6, alpha0=0.1, n_epochs=400, lr=1e-2, seed=0)
    met = weight_survival_metrics(model.mean_weights())
    # Robust claim: resolved modes force MORE THAN ONE class and prune the bypass.
    # The exact count can over-shoot the true 2 (adjacent bins splitting a mode — the
    # VB over-counting the note's §10 caveats); that is reported as a median in the sweep.
    assert met["surviving_k"] >= 2, met
    assert met["bypass_weight"] < 0.3, met  # the single Gaussian cannot fake two peaks


def test_merged_modes_fall_back_to_bypass():
    # modes essentially on top of each other: one blob -> bypass carries it
    x, y = td.make_toy_b(n=600, k_true=2, separation=0.5, sigma=0.3, baseline="zero", seed=7)
    model = vem.train_variational_em(x, y, k_max=6, alpha0=0.1, n_epochs=400, lr=1e-2, seed=0)
    met = weight_survival_metrics(model.mean_weights())
    assert met["surviving_k"] <= 2, met
    assert met["bypass_weight"] > 0.6, met


def test_smooth_unimodal_negative_control():
    # smooth mean + Gaussian noise (x informative): bypass should take ~all the weight
    x, y = td.make_toy_a(n=600, sigma=0.2, seed=7)
    model = vem.train_variational_em(x, y, k_max=6, alpha0=0.1, n_epochs=400, lr=1e-2, seed=0)
    met = weight_survival_metrics(model.mean_weights())
    assert met["surviving_k"] == 1, met
    assert met["bypass_weight"] > 0.9, met


def test_toy_c_sep_schedule_crosses_two_at_x_star():
    # the bimodality boundary (means 2σ apart) must sit where sep_schedule(x) == 2
    sep_min, sep_max = 0.3, 4.0
    x_star = (2.0 - sep_min) / (sep_max - sep_min)
    assert abs(float(td.sep_schedule(np.array([x_star]), sep_min, sep_max)[0]) - 2.0) < 1e-6
    sched = td.sep_schedule(np.array([0.0, 1.0]), sep_min, sep_max)
    assert abs(sched[0] - sep_min) < 1e-6
    assert abs(sched[1] - sep_max) < 1e-6


def test_toy_c_and_broad_share_mean_and_variance_per_x():
    # the over-chopping trap is only fair if the bimodal and the single-mode twin match
    # in mean AND variance at every x (they may differ only in SHAPE)
    for xv in [0.0, 0.25, 0.46, 0.75, 1.0]:
        xs = np.full(60000, xv, dtype=np.float32)
        yc = td.sample_toy_c_given_x(xs, seed=11)
        yb = td.sample_toy_c_broad_given_x(xs, seed=12)
        assert abs(float(yc.mean()) - float(yb.mean())) < 0.02, xv
        assert abs(float(yc.var()) - float(yb.var())) < 0.02, xv
        # variance must follow the closed form σ²(1 + sep²/4)
        sep = float(td.sep_schedule(np.array([xv]))[0])
        assert abs(float(yc.var()) - 0.3**2 * (1.0 + sep**2 / 4.0)) < 0.02, xv


def test_toy_c_is_bimodal_when_resolved_unimodal_when_merged():
    # compare the two ends to EACH OTHER (robust, no fragile absolute threshold):
    # resolved (x=1) has a dip at zero and mass at ±offset; merged (x=0) is the reverse.
    offset = 0.5 * 4.0 * 0.3  # mode location at x=1: 0.6
    y_res = td.sample_toy_c_given_x(np.full(60000, 1.0, dtype=np.float32), seed=13)
    y_mer = td.sample_toy_c_given_x(np.full(60000, 0.0, dtype=np.float32), seed=14)
    center_mer = float(np.mean(np.abs(y_mer) < 0.15))
    center_res = float(np.mean(np.abs(y_res) < 0.15))
    mode_res = float(np.mean(np.abs(np.abs(y_res) - offset) < 0.15))
    mode_mer = float(np.mean(np.abs(np.abs(y_mer) - offset) < 0.15))
    assert center_mer > center_res, (center_mer, center_res)  # merged peaks at zero
    assert mode_res > mode_mer, (mode_res, mode_mer)  # resolved peaks at the modes


def test_toy_e_sep_hump_nonmonotone_crosses_two_twice():
    # the hump schedule peaks at x=0.5 and returns to sep_min at both ends, crossing the 2σ
    # bimodality boundary TWICE -> ground-truth count is non-monotone in x (1 -> 2 -> 1)
    sep_min, sep_max = 0.3, 4.0
    assert abs(float(td.sep_hump(np.array([0.0]), sep_min, sep_max)[0]) - sep_min) < 1e-6
    assert abs(float(td.sep_hump(np.array([0.5]), sep_min, sep_max)[0]) - sep_max) < 1e-6
    assert abs(float(td.sep_hump(np.array([1.0]), sep_min, sep_max)[0]) - sep_min) < 1e-6
    t = 1.0 - (2.0 - sep_min) / (sep_max - sep_min)  # |2x-1| at which sep_hump == 2
    x_lo, x_hi = (1.0 - t) / 2.0, (1.0 + t) / 2.0
    assert x_lo < 0.5 < x_hi
    assert abs(float(td.sep_hump(np.array([x_lo]), sep_min, sep_max)[0]) - 2.0) < 1e-6
    assert abs(float(td.sep_hump(np.array([x_hi]), sep_min, sep_max)[0]) - 2.0) < 1e-6


def test_toy_e_and_broad_share_mean_and_variance_per_x():
    # same fairness contract as toy C: the bimodal and its single-mode twin must match in mean
    # AND variance at every x (differing only in shape), now under the humped schedule
    for xv in [0.0, 0.23, 0.5, 0.77, 1.0]:
        xs = np.full(60000, xv, dtype=np.float32)
        ye = td.sample_toy_e_given_x(xs, seed=11)
        yb = td.sample_toy_e_broad_given_x(xs, seed=12)
        assert abs(float(ye.mean()) - float(yb.mean())) < 0.02, xv
        assert abs(float(ye.var()) - float(yb.var())) < 0.02, xv
        sep = float(td.sep_hump(np.array([xv]))[0])
        assert abs(float(ye.var()) - 0.3**2 * (1.0 + sep**2 / 4.0)) < 0.02, xv


def test_toy_e_bimodal_midrange_unimodal_at_ends():
    # at x=0.5 (resolved) the density dips at zero with mass at ±offset; at the ends (merged) it
    # peaks at zero — the non-monotone signature, compared middle-vs-end
    offset = 0.5 * 4.0 * 0.3  # mode location at x=0.5: 0.6
    y_mid = td.sample_toy_e_given_x(np.full(60000, 0.5, dtype=np.float32), seed=13)
    y_end = td.sample_toy_e_given_x(np.full(60000, 0.0, dtype=np.float32), seed=14)
    center_end = float(np.mean(np.abs(y_end) < 0.15))
    center_mid = float(np.mean(np.abs(y_mid) < 0.15))
    mode_mid = float(np.mean(np.abs(np.abs(y_mid) - offset) < 0.15))
    mode_end = float(np.mean(np.abs(np.abs(y_end) - offset) < 0.15))
    assert center_end > center_mid, (center_end, center_mid)  # ends peak at zero
    assert mode_mid > mode_end, (mode_mid, mode_end)  # middle peaks at the modes


def test_toy_d_staircase_count_steps_1_2_3():
    # ground-truth component count is 1, 2, 3 on the three thirds of x
    k = td._staircase_k(np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.99]))
    assert list(k) == [1, 1, 2, 2, 3, 3], k


def test_toy_d_variance_increases_across_thirds():
    # more modes (same spacing, same σ) -> larger marginal variance; mean stays ~0 in each third
    sig, sep = 0.3, 4.0
    v = []
    for xv in [0.1, 0.5, 0.9]:
        y = td.sample_toy_d_given_x(np.full(80000, xv, dtype=np.float32), sigma=sig, separation=sep, seed=5)
        assert abs(float(y.mean())) < 0.03, xv
        v.append(float(y.var()))
    assert v[0] < v[1] < v[2], v  # k=1 < k=2 < k=3
    # closed forms: var = σ² + Var(offsets); k=1 -> σ²; k=2 -> σ² + (sep·σ/2)²; k=3 -> σ² + (2/3)(sep·σ)²
    assert abs(v[0] - sig**2) < 0.02, v
    assert abs(v[1] - (sig**2 + (sep * sig / 2.0) ** 2)) < 0.03, v
    assert abs(v[2] - (sig**2 + (2.0 / 3.0) * (sep * sig) ** 2)) < 0.05, v


def test_dirichlet_kl_zero_at_prior_and_nonnegative():
    # KL(Dir(γ)‖Dir(α₀)) is 0 exactly at γ = α₀·1 and positive away from it
    g_prior = torch.full((3, 5), 0.1)
    assert torch.allclose(vemp.dirichlet_kl(g_prior, 0.1), torch.zeros(3), atol=1e-5)
    g_off = torch.tensor([[2.0, 0.1, 0.1, 0.1, 0.1]])
    assert float(vemp.dirichlet_kl(g_off, 0.1)) > 0.0


def test_perinput_weights_sum_to_one_and_concentrations_positive():
    model = vemp.PerInputVariationalEMKSelector(input_dim=1, k_max=4, alpha0=0.1)
    x = torch.randn(20, 1)
    g = model.concentrations(x)
    w = model.weights(x)
    assert g.shape == (20, model.K)
    assert (g > 0).all()
    assert torch.allclose(w.sum(dim=1), torch.ones(20), atol=1e-5)


def test_perinput_elbo_finite_and_differentiable():
    model = vemp.PerInputVariationalEMKSelector(input_dim=1, k_max=4, alpha0=0.1)
    x, y = torch.randn(32, 1), torch.randn(32)
    loss = model.elbo_loss(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_aggregate_sparsity_weights_adapt_with_input():
    # the repair: with the sparsity prior on the dataset-MEAN usage (one prior vs N points),
    # the per-input weights are driven by the mixture likelihood and DO vary with x — the
    # effective bucket count rises from the merged end to the resolved end of make_toy_c.
    x, y = td.make_toy_c(n=1500, seed=0)
    model = vemp.train_aggregate_sparsity(x, y, k_max=6, alpha0=0.1, n_epochs=800, lr=1e-2, seed=0)

    def eff_count(w):
        w = w.numpy().ravel()
        return float(np.exp(-(w * np.log(np.clip(w, 1e-12, None))).sum()))

    eff_lo = eff_count(model.weights(torch.zeros(1, 1)))   # merged: ~1 bucket
    eff_hi = eff_count(model.weights(torch.ones(1, 1)))    # resolved: ~2 buckets
    assert eff_lo < 1.3, eff_lo
    assert eff_hi > 1.6, eff_hi
    assert eff_hi - eff_lo > 0.4, (eff_lo, eff_hi)


def test_perinput_naive_elbo_collapses_weights_across_input():
    # FINDING (not a target): the per-input Dirichlet KL is charged in full against only ONE
    # point of reconstruction evidence per input, so the prior dominates early training and
    # pulls every input's weights to the SAME point — Basis B collapses to Basis A. The robust
    # signature is that the fitted weights are frozen across x. Flip this once the sparse prior
    # is moved onto the aggregate (dataset-mean) class usage instead of per input.
    x, y = td.make_toy_c(n=1200, seed=3)
    model = vemp.train_perinput_variational_em(x, y, k_max=6, alpha0=0.1, n_epochs=400, lr=1e-2, seed=0, adaptive_bin_means=False)
    w_lo, w_hi = model.weights(torch.zeros(1, 1)), model.weights(torch.ones(1, 1))
    assert float((w_lo - w_hi).abs().max()) < 0.02, (w_lo, w_hi)  # weights frozen across x


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
