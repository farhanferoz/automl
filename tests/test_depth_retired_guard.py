"""Tests for the Decision-29 depth-selection retirement guard (`automl_package/models/flexnn/strategies/layer.py`).

ProbReg's identical guard (n_classes_selection_method / n_classes_regularization / the P10
head-layout gate) is tested in `tests/test_phase3_dynamic_k.py`.
"""

import pytest

from automl_package.enums import LayerSelectionMethod
from automl_package.models.flexnn.depth.model import FlexibleHiddenLayersNN


class TestDepthRetiredSelectionStrategiesGuard:
    """MASTER Decision 29's depth-side guard (`layer.py`'s `_RetiredDepthSelectionStrategy`)."""

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
