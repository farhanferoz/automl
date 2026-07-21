"""Tests for the fixed-sigma mixture scorer (MASTER Decision 24; flexnn-package.md FP-12).

Covers: `fixed_sigma_mixture_log_likelihood` (the pure scorer), `NestedStrategy`'s new
per-component mixture readout it is built on, and the Decision-29 depth-selection retirement
guard (`automl_package/models/flexnn/strategies/layer.py`). The ProbReg-side Decision-29 guard
(n_classes_selection_method / n_classes_regularization / optimization_strategy / the P10
head-layout gate) is tested in `tests/test_phase3_dynamic_k.py`, this task's other test target.
"""

import math

import numpy as np
import pytest
import torch

from automl_package.enums import (
    LayerSelectionMethod,
    NClassesSelectionMethod,
    RegressionStrategy,
    UncertaintyMethod,
)
from automl_package.models.flexnn.depth.model import FlexibleHiddenLayersNN
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.utils.losses import fixed_sigma_mixture_log_likelihood

_ATOL = 1e-6
_MAX_K = 4  # matches _fit_nested's default max_n_classes_for_probabilistic_path
_MIN_MIXTURE_VS_COLLAPSED_GAP = 1e-3  # divergence floor proving the two scores are genuinely different objects
# Measured gap on the fixed hand-worked example below is ~0.693 nats; half that is a comfortable
# floor well clear of floating-point noise without being a tight/fragile bound.
_MIN_MOMENT_MATCHED_GAP_NATS = 0.3


class TestFixedSigmaMixtureLogLikelihoodKnownAnswer:
    """FP-12 verify (a)/(b): the exact contract the root's verify block checks."""

    def test_k1_reduces_to_plain_gaussian(self):
        """A single component at prob=1 is exactly a Gaussian log-density at fixed sigma."""
        y = torch.tensor([0.5])
        probs = torch.tensor([[1.0]])
        mus = torch.tensor([[0.0]])
        sigma = 2.0
        want = -0.5 * (math.log(2 * math.pi) + 2 * math.log(sigma) + (0.5**2) / sigma**2)
        got = float(fixed_sigma_mixture_log_likelihood(y, probs, mus, sigma=sigma)[0])
        assert abs(got - want) < _ATOL

    def test_sigma_is_required_no_default(self):
        """Calling without sigma must raise TypeError, not silently default (a shipped default
        is exactly how a per-arm sigma would silently reappear -- MASTER Decision 24)."""
        y = torch.tensor([0.5])
        probs = torch.tensor([[1.0]])
        mus = torch.tensor([[0.0]])
        with pytest.raises(TypeError):
            fixed_sigma_mixture_log_likelihood(y, probs, mus)

    def test_two_component_mixture_matches_manual_logsumexp(self):
        """A hand-computed 2-component mixture log-likelihood, checked against scipy-free
        manual log-sum-exp -- the scorer must be the genuine mixture density, not a moment-
        matched single Gaussian (the exact substitution error flexnn-package.md FP-12 warns
        against)."""
        y = torch.tensor([1.0])
        probs = torch.tensor([[0.3, 0.7]])
        mus = torch.tensor([[0.0, 2.0]])
        sigma = 1.0

        def _log_gauss(yv: float, mu: float, s: float) -> float:
            return -0.5 * (math.log(2 * math.pi) + 2 * math.log(s) + (yv - mu) ** 2 / s**2)

        want = math.log(0.3 * math.exp(_log_gauss(1.0, 0.0, sigma)) + 0.7 * math.exp(_log_gauss(1.0, 2.0, sigma)))
        got = float(fixed_sigma_mixture_log_likelihood(y, probs, mus, sigma=sigma)[0])
        assert abs(got - want) < _ATOL

    def test_moment_matched_single_gaussian_is_a_different_number(self):
        """The mixture density is NOT the same object as a moment-matched single Gaussian at the
        mixture mean -- this is exactly the substitution the spec (§4.3) warns is wrong. If this
        test ever starts passing with equality, the scorer regressed into the collapsed form."""
        y = torch.tensor([1.0])
        probs = torch.tensor([[0.5, 0.5]])
        mus = torch.tensor([[-2.0, 2.0]])
        sigma = 0.5

        mixture_ll = float(fixed_sigma_mixture_log_likelihood(y, probs, mus, sigma=sigma)[0])

        moment_matched_mean = 0.0  # 0.5*(-2) + 0.5*(2)
        collapsed_ll = -0.5 * (math.log(2 * math.pi) + 2 * math.log(sigma) + (1.0 - moment_matched_mean) ** 2 / sigma**2)

        assert abs(mixture_ll - collapsed_ll) > _MIN_MOMENT_MATCHED_GAP_NATS, "mixture and moment-matched-single-Gaussian scores should differ substantially on a bimodal case"

    def test_higher_is_better_direction(self):
        """A component mean closer to y must score higher (less negative) -- log-likelihood, not NLL."""
        y = torch.tensor([0.0])
        probs = torch.tensor([[1.0]])
        sigma = 1.0
        close = float(fixed_sigma_mixture_log_likelihood(y, probs, torch.tensor([[0.1]]), sigma=sigma)[0])
        far = float(fixed_sigma_mixture_log_likelihood(y, probs, torch.tensor([[5.0]]), sigma=sigma)[0])
        assert close > far, "a closer component mean must score HIGHER under fixed_sigma_mixture_log_likelihood"

    def test_batched_shape(self):
        """Per-sample output, not a scalar reduction."""
        y = torch.zeros(4)
        probs = torch.full((4, 3), 1.0 / 3)
        mus = torch.zeros(4, 3)
        out = fixed_sigma_mixture_log_likelihood(y, probs, mus, sigma=1.0)
        assert out.shape == (4,)


class TestNestedStrategyMixtureComponents:
    """The per-component mixture readout `all_rung_log_likelihood` is built on -- the part
    flexnn-package.md FP-12 says is NOT a one-line substitution."""

    def _fit_nested(self, x, y, max_k=4, n_epochs=25, seed=0):
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=max_k,
            uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED,
            regression_strategy=RegressionStrategy.SEPARATE_HEADS,
            n_epochs=n_epochs, learning_rate=0.01, random_seed=seed,
            calculate_feature_importance=False,
        )
        model.fit(x, y)
        return model

    def test_probs_sum_to_one_per_rung(self, multimodal_data):
        x, y = multimodal_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=10)
        x_t = torch.tensor(x[:16], dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            rungs = model.model.n_classes_strategy.all_rung_mixture_components(x_t)
        assert len(rungs) == _MAX_K
        for k, (probs, mus) in enumerate(rungs, start=1):
            assert probs.shape == (16, k)
            assert mus.shape == (16, k)
            np.testing.assert_allclose(probs.sum(dim=-1).cpu().numpy(), np.ones(16), atol=1e-5)

    def test_rung_one_is_the_bypass_head_as_a_trivial_mixture(self, multimodal_data):
        x, y = multimodal_data
        model = self._fit_nested(x, y, max_k=3, n_epochs=10)
        x_t = torch.tensor(x[:8], dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            probs, mus = model.model.n_classes_strategy.mixture_components_at_rung(x_t, 1)
            bypass_out = model.model.direct_regression_head(x_t)
        np.testing.assert_allclose(probs.cpu().numpy(), np.ones((8, 1)), atol=1e-6)
        np.testing.assert_allclose(mus.cpu().numpy(), bypass_out[:, 0:1].cpu().numpy(), atol=1e-6)

    def test_all_rung_log_likelihood_requires_sigma(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=3, n_epochs=5)
        x_t = torch.tensor(x[:8], dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y[:8], dtype=torch.float32).to(model.device)
        with pytest.raises(TypeError):
            model.model.n_classes_strategy.all_rung_log_likelihood(x_t, y_t)

    def test_all_rung_log_likelihood_shape_and_finite(self, heteroscedastic_data):
        x, y, _, _ = heteroscedastic_data
        model = self._fit_nested(x, y, max_k=4, n_epochs=10)
        x_t = torch.tensor(x[:20], dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y[:20], dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            ll = model.model.n_classes_strategy.all_rung_log_likelihood(x_t, y_t, sigma=0.3)
        assert ll.shape == (20, 4)
        assert torch.all(torch.isfinite(ll))

    def test_mixture_score_differs_from_collapsed_gaussian_score_on_multimodal_data(self, multimodal_data):
        """The regression FP-12 exists to fix: on genuinely multi-component data, the fixed-sigma
        MIXTURE score at a rung k>=2 must differ from a fixed-sigma single Gaussian scored at the
        SAME rung's collapsed (law-of-total-variance) mean -- if they agreed, the migration would
        have silently substituted a constant into the old collapsed form (the exact trap
        flexnn-package.md FP-12 warns is not a one-line substitution)."""
        x, y = multimodal_data
        model = self._fit_nested(x, y, max_k=3, n_epochs=40)
        x_t = torch.tensor(x[:32], dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y[:32], dtype=torch.float32).to(model.device)
        sigma = 0.3
        model.model.eval()
        with torch.no_grad():
            mixture_ll = model.model.n_classes_strategy.all_rung_log_likelihood(x_t, y_t, sigma=sigma)
            all_outputs, _ = model.model.n_classes_strategy.all_rung_outputs(x_t)
        collapsed_mean = all_outputs[:, :, 0]  # (batch, n_classes) -- the LTV-collapsed mean per rung
        y_col = y_t.view(-1, 1)
        collapsed_ll = -0.5 * (math.log(2 * math.pi) + 2 * math.log(sigma) + (y_col - collapsed_mean) ** 2 / sigma**2)

        # Rung k=1 (the bypass) has exactly one component either way -- must agree.
        np.testing.assert_allclose(mixture_ll[:, 0].cpu().numpy(), collapsed_ll[:, 0].cpu().numpy(), atol=1e-4)
        # Rung k>=2 genuinely mixes -- on bimodal data the two objects must diverge somewhere.
        diffs = (mixture_ll[:, 1:] - collapsed_ll[:, 1:]).abs()
        assert diffs.max().item() > _MIN_MIXTURE_VS_COLLAPSED_GAP, "mixture and collapsed-Gaussian scores should differ on multimodal data at k>=2"


class TestDepthRetiredSelectionStrategiesGuard:
    """MASTER Decision 29's depth-side guard (`layer.py`'s `_RetiredDepthSelectionStrategy`).

    ProbReg's identical guard (n_classes_selection_method / n_classes_regularization /
    optimization_strategy / the P10 head-layout gate) is tested in `test_phase3_dynamic_k.py`.
    """

    @pytest.mark.parametrize("method", [
        LayerSelectionMethod.GUMBEL_SOFTMAX,
        LayerSelectionMethod.SOFT_GATING,
        LayerSelectionMethod.STE,
        LayerSelectionMethod.REINFORCE,
    ])
    def test_retired_strategy_raises_at_construction(self, method):
        with pytest.raises(ValueError, match="RETIRED"):
            FlexibleHiddenLayersNN(input_size=1, output_size=1, layer_selection_method=method, n_predictor_layers=1)

    def test_default_construction_still_works(self):
        """The DEFAULT constructor must keep working, and must default to a surviving method.

        This test previously asserted the opposite -- that default construction RAISES -- because
        the class's default was `GUMBEL_SOFTMAX`, which MASTER Decision 29 retires, and that file
        sat outside FP-12's write set. **Asserting the breakage canonised a regression:** it made
        "a shipped model class cannot be instantiated with its own defaults" into expected
        behaviour, so no later run would flag it.

        Decision 29 retires a SELECTION MECHANISM; it does not license breaking the constructor.
        The defaults moved to `NONE` + `n_predictor_layers=0` (both survivors require 0), and this
        test now guards that invariant instead.
        """
        model = FlexibleHiddenLayersNN(input_size=1, output_size=1)
        assert model.layer_selection_method in (LayerSelectionMethod.NONE, LayerSelectionMethod.NESTED)
        assert model.n_predictor_layers == 0

    def test_search_space_offers_only_constructible_choices(self):
        """Every method the search space advertises must actually construct.

        Decision 29 names `get_hyperparameter_search_space` as the live exposure: leaving a retired
        member in `choices` means a tuning run samples it and the trial hard-crashes. The guard at
        construction alone is not enough.
        """
        model = FlexibleHiddenLayersNN(input_size=1, output_size=1)
        choices = model.get_hyperparameter_search_space()["layer_selection_method"]["choices"]
        assert choices, "search space must still offer at least one method"
        for choice in choices:
            FlexibleHiddenLayersNN(input_size=1, output_size=1, layer_selection_method=choice)
        for retired in (LayerSelectionMethod.GUMBEL_SOFTMAX, LayerSelectionMethod.STE,
                        LayerSelectionMethod.SOFT_GATING, LayerSelectionMethod.REINFORCE):
            assert retired not in choices

    @pytest.mark.parametrize("method", [
        LayerSelectionMethod.GUMBEL_SOFTMAX,
        LayerSelectionMethod.SOFT_GATING,
        LayerSelectionMethod.STE,
        LayerSelectionMethod.REINFORCE,
    ])
    def test_escape_hatch_allows_construction(self, method):
        model = FlexibleHiddenLayersNN(
            input_size=1, output_size=1, layer_selection_method=method, n_predictor_layers=1,
            allow_retired_capacity_selection=True,
        )
        assert model.strategy is not None

    @pytest.mark.parametrize("method", [LayerSelectionMethod.NONE, LayerSelectionMethod.NESTED])
    def test_survivor_strategies_construct_without_escape_hatch(self, method):
        model = FlexibleHiddenLayersNN(input_size=1, output_size=1, layer_selection_method=method, n_predictor_layers=0)
        assert model.strategy is not None
