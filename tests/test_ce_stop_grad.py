"""Tests for the cross-entropy training modes and CE_STOP_GRAD gradient routing (CT3).

CE_STOP_GRAD is the default `optimization_strategy` (owner ruling 2026-10-22): the classifier is
trained on cross-entropy against the y bins only, and its probabilities are detached before the
regression heads, so the regression loss cannot pull them off the true bin probabilities. All
cross-entropy modes construct without `allow_retired_capacity_selection`.
"""

import numpy as np
import pytest
import torch

from automl_package.enums import (
    BoundaryRegularizationMethod,
    CapacitySelection,
    ClassifierLink,
    HpoSampler,
    NClassesSelectionMethod,
    ProbabilisticRegressionOptimizationStrategy,
    RegressionStrategy,
    TaskType,
    TwoStagePhase,
    UncertaintyMethod,
)
from automl_package.models.base import _BASE_SETTINGS
from automl_package.models.common.losses import calculate_combined_loss
from automl_package.models.common.ordinal import MIN_CLASS_PROBABILITY, ProportionalOddsOutput
from automl_package.models.common.regression_heads import ConstantHead
from automl_package.models.neural_network import PyTorchNeuralNetwork
from automl_package.models.probabilistic_regression import ProbabilisticRegressionModel
from automl_package.optimizers.optuna_optimizer import OptunaOptimizer
from automl_package.models.pytorch_linear_regression import PyTorchLinearRegression
from automl_package.utils.convergence import ConvergenceResult
from automl_package.utils.losses import beta_nll_loss
from automl_package.utils.numerics import create_bins


def _make_tiny_dataset(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
    y = (x.ravel() + rng.normal(0, 0.3, n)).astype(np.float32)
    return x, y


def _make_model(strategy: ProbabilisticRegressionOptimizationStrategy | None = None, n_epochs: int = 3, **extra) -> ProbabilisticRegressionModel:
    """A tiny ProbReg; `strategy` left unset exercises the constructor default. TWO_STAGE is a one-pass strategy, so it is built with `refit_on_train_val=False`."""
    kwargs = {} if strategy is None else {"optimization_strategy": strategy}
    if strategy is ProbabilisticRegressionOptimizationStrategy.TWO_STAGE:
        kwargs["refit_on_train_val"] = False
    kwargs.update(extra)
    return ProbabilisticRegressionModel(
        input_size=1,
        n_classes=3,
        regression_strategy=RegressionStrategy.SEPARATE_HEADS,
        uncertainty_method=UncertaintyMethod.PROBABILISTIC,
        n_epochs=n_epochs,
        learning_rate=0.01,
        early_stopping_rounds=2,
        validation_fraction=0.2,
        random_seed=42,
        calculate_feature_importance=False,
        **kwargs,
    )


class TestCeModesConstructAndTrain:
    def test_defaults_follow_the_owner_rulings(self):
        """CE_STOP_GRAD supervision, mean-only heads (fixed spread), and the flat middle head only together with the monotonic constraints."""
        model = ProbabilisticRegressionModel(input_size=1, n_classes=3, calculate_feature_importance=False)
        assert model.optimization_strategy is ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD
        assert model.uncertainty_method is UncertaintyMethod.CONSTANT

        def has_constant_head(**kwargs) -> bool:
            built = ProbabilisticRegressionModel(input_size=1, n_classes=3, calculate_feature_importance=False, **kwargs)
            built.build_model()
            return any(isinstance(head, ConstantHead) for head in built.regression_heads)

        assert not has_constant_head()
        assert has_constant_head(use_monotonic_constraints=True)
        assert has_constant_head(constrain_middle_class=True)
        assert not has_constant_head(use_monotonic_constraints=True, constrain_middle_class=False)

        # Dropout is built whenever its rate is positive, under the default (CONSTANT) spread, for the plain network
        # and the ProbReg classifier; the default rate 0.0 builds none (I5).
        def n_dropout_layers(module: torch.nn.Module) -> int:
            return sum(isinstance(m, torch.nn.Dropout) for m in module.modules())

        plain = PyTorchNeuralNetwork(input_size=1, dropout_rate=0.5, calculate_feature_importance=False)
        plain.build_model()
        assert n_dropout_layers(plain.model) == 1
        with_dropout = ProbabilisticRegressionModel(input_size=1, n_classes=3, base_classifier_params={"dropout_rate": 0.5}, calculate_feature_importance=False)
        with_dropout.build_model()
        assert n_dropout_layers(with_dropout.model.classifier_layers) == 1
        without_dropout = ProbabilisticRegressionModel(input_size=1, n_classes=3, calculate_feature_importance=False)
        without_dropout.build_model()
        assert n_dropout_layers(without_dropout.model) == 0

    @pytest.mark.parametrize(
        "extra",
        [
            {"cv_folds": 2, "validation_fraction": None},
            {"optimize_hyperparameters": True, "n_trials": 2, "validation_fraction": 0.2},
            {"optimize_hyperparameters": True, "n_trials": 2, "cv_folds": 2, "validation_fraction": None},
            {"n_classes": 5, "n_classes_inf": 3, "validation_fraction": 0.2},
            {"refit_on_train_val": False, "validation_fraction": 0.2},
            {"refit_on_train_val": False, "optimize_hyperparameters": True, "n_trials": 2, "validation_fraction": 0.2},
            {"optimize_hyperparameters": True, "n_trials": 2, "validation_fraction": 0.2, "search_space_exclude": ["n_classes"]},
            {"optimize_hyperparameters": True, "n_trials": 2, "validation_fraction": 0.2, "hpo_sampler": HpoSampler.RANDOM, "hpo_seed": 3},
            {"use_anchored_heads": True, "validation_fraction": 0.2},
            {"boundary_regularization_method": BoundaryRegularizationMethod.BIN_LIMITS, "validation_fraction": 0.2},
            {"classifier_link": ClassifierLink.PROPORTIONAL_ODDS, "validation_fraction": 0.2},
        ],
        ids=["cv", "hpo", "hpo_cv", "direct_regression", "one_pass", "one_pass_hpo", "hpo_excluding_n_classes", "hpo_random_sampler", "anchored_heads", "bin_limits", "ordinal"],
    )
    def test_default_fits_under_cv_hpo_and_direct_regression(self, extra):
        """The CE_STOP_GRAD default must not crash where trials are scored on a clone (CV, search) or where no classes exist.

        The one-pass cases (`refit_on_train_val=False`) must also never refit on train + validation, keep the
        whole fitted model, and leave a convergence record; an explicit validation set is used as given, with no
        inner split and no test rows set aside.
        """
        x, y = _make_tiny_dataset()
        kwargs = {"input_size": 1, "n_classes": 3, "n_epochs": 2, "early_stopping_rounds": 2, "random_seed": 0, "calculate_feature_importance": False} | extra
        model = ProbabilisticRegressionModel(**kwargs)
        if extra.get("refit_on_train_val") is False:
            model._fit_final_model = lambda **_: pytest.fail("one-pass fit must not refit on train + validation")
        model.fit(x, y)
        assert np.all(np.isfinite(model.predict(x[:10])))
        if "search_space_exclude" in extra:
            # An excluded dimension is not searched (k stays as built), survives a clone, and an unknown name is an error that lists the valid ones.
            assert "n_classes" not in model._searched_space()
            assert model.n_classes == 3
            assert model._clone().search_space_exclude == ["n_classes"]
            with pytest.raises(ValueError, match="valid names"):
                ProbabilisticRegressionModel(**(kwargs | {"search_space_exclude": ["no_such_dimension"]}))._searched_space()
        if "hpo_sampler" in extra:
            # The sampler and its seed survive a clone and reach Optuna: past TPE's 10 random start-up trials, RANDOM draws a different sequence.
            clone = model._clone()
            assert (clone.hpo_sampler, clone.hpo_seed) == (HpoSampler.RANDOM, 3)

            def draws(sampler: HpoSampler) -> list[float]:
                return [t.params["x"] for t in OptunaOptimizer(n_trials=16, random_seed=3, sampler=sampler).optimize(lambda t: (t.suggest_float("x", 0, 1) - 0.3) ** 2).trials]

            assert draws(HpoSampler.RANDOM) != draws(HpoSampler.TPE)
        if extra.get("refit_on_train_val") is False:
            assert model.model is not None
            # The adopted best trial must not bring the clone's settings: no feature-importance pass ran (SHAP would leave its
            # x / y tensors on the activations), and the outer model keeps its own settings.
            assert model.calculate_feature_importance is False
            assert all(name.endswith(("weight", "bias")) for name, _ in model.model.named_parameters())
            assert isinstance(model.convergence_, ConvergenceResult)
            assert model.convergence_.converged != model.convergence_.hit_cap

            explicit = ProbabilisticRegressionModel(**kwargs)
            explicit.fit(x[:60], y[:60], x_val=x[60:], y_val=y[60:])
            assert (explicit.train_indices.size, explicit.val_indices.size, explicit.test_indices.size) == (60, 20, 0)
            assert np.all(np.isfinite(explicit.predict(x[:10])))

            # Every BaseModel setting reaches a clone, and adopting a trial model never takes one of its settings.
            assert not _BASE_SETTINGS - set(model.get_params())
            own_settings = (model.n_trials, model.calculate_feature_importance)
            trial = model._clone()
            trial.n_trials, trial.calculate_feature_importance = model.n_trials + 1, True
            model._adopt_fitted_state(trial)
            assert (model.n_trials, model.calculate_feature_importance) == own_settings

    @pytest.mark.parametrize("strategy", list(ProbabilisticRegressionOptimizationStrategy))
    def test_every_mode_constructs_without_escape_hatch_and_trains(self, strategy):
        """All four modes (including the cross-entropy ones) construct with no `allow_retired_capacity_selection`."""
        x, y = _make_tiny_dataset()
        model = _make_model(strategy)
        assert model.allow_retired_capacity_selection is False
        model.fit(x, y)
        preds = model.predict(x[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))


class TestProportionalOddsClassifier:
    """The ordinal classifier link: one score, ordered thresholds, class log-probabilities that every softmax consumer reads unchanged."""

    def test_probabilities_cross_entropy_and_ordered_thresholds(self):
        x, y = _make_tiny_dataset()
        model = _make_model(classifier_link=ClassifierLink.PROPORTIONAL_ODDS, n_epochs=5)
        model.fit(x, y)
        output = next(m for m in model.model.classifier_layers if isinstance(m, ProportionalOddsOutput))
        assert bool(torch.all(torch.diff(output.thresholds()) > 0))  # strictly increasing after training
        output.raw_gaps.data.fill_(-5.0)  # ordered by construction, whatever the raw gaps are
        assert bool(torch.all(torch.diff(output.thresholds()) > 0))
        output.raw_gaps.data.fill_(0.0)

        x_t = torch.tensor(x)
        model.model.eval()
        with torch.no_grad():
            log_p = model.model.classifier_layers(x_t)
            eta = model.model.classifier_layers[0](x_t)
            cumulative = torch.sigmoid(output.thresholds().unsqueeze(0) - eta)
            f_all = torch.cat([torch.zeros_like(eta), cumulative, torch.ones_like(eta)], dim=1)
            expected = f_all[:, 1:] - f_all[:, :-1]
            probabilities = model.get_classifier_predictions(x, y)[1]
            assert np.allclose(probabilities, expected.numpy(), atol=1e-6)
            assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            assert torch.allclose(torch.softmax(log_p, dim=1), expected.clamp_min(MIN_CLASS_PROBABILITY), atol=1e-6)
            # The classification term is the ordinary cross-entropy on the classifier output: -log p_y.
            y_t = torch.tensor(y)
            ce, _ = model._class_cross_entropy(log_p, y_t, torch.full((len(y),), model.n_classes))
            _, y_binned = create_bins(data=y, unique_bin_edges=model.precomputed_class_boundaries[model.n_classes])
            picked = expected.clamp_min(MIN_CLASS_PROBABILITY)[torch.arange(len(y)), torch.tensor(y_binned)]
            assert float(ce) == pytest.approx(float(-torch.log(picked).mean()), abs=1e-5)

    def test_frozen_classifier_freezes_the_thresholds(self):
        model = _make_model(ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP, classifier_link=ClassifierLink.PROPORTIONAL_ODDS)
        model.build_model()
        assert not any(p.requires_grad for p in model.model.classifier_layers.parameters())

    @pytest.mark.parametrize(
        "extra",
        [
            {"n_classes_selection_method": NClassesSelectionMethod.NESTED, "allow_unredesigned_nested": True},
            {"capacity_selection": CapacitySelection.PER_INPUT},
            {"capacity_selection": CapacitySelection.GLOBAL_CHEAP},
        ],
        ids=["nested", "per_input", "global_cheap"],
    )
    def test_refused_for_a_dynamic_k(self, extra):
        with pytest.raises(ValueError, match="PROPORTIONAL_ODDS"):
            _make_model(classifier_link=ClassifierLink.PROPORTIONAL_ODDS, **extra)

    def test_global_sweep_is_allowed(self):
        _make_model(classifier_link=ClassifierLink.PROPORTIONAL_ODDS, capacity_selection=CapacitySelection.GLOBAL_SWEEP)


class TestCeStopGradRouting:
    """Plan CT3: verify classifier sees CE gradient only; heads see regression only.

    Works by fitting to build the model, then running one manual forward+backward on a
    fresh copy and inspecting per-parameter gradients:
      - classifier_layers params: non-zero grad under CE_STOP_GRAD, zero grad under
        REGRESSION_ONLY-with-detached-probs (we simulate by zeroing CE contribution).
      - regression_module params: always get gradient from regression loss.
    """

    def _fit_tiny(self, strategy: ProbabilisticRegressionOptimizationStrategy | None = None) -> ProbabilisticRegressionModel:
        x, y = _make_tiny_dataset()
        model = _make_model(strategy, n_epochs=2)
        model.fit(x, y)
        return model

    @staticmethod
    def _zero_grads(net: torch.nn.Module) -> None:
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()

    @staticmethod
    def _total_abs_grad(params) -> float:
        return float(sum(p.grad.abs().sum().item() for p in params if p.grad is not None))

    def _run_backward(self, model: ProbabilisticRegressionModel) -> tuple[float, float]:
        """Run one forward+backward pass, return (classifier_grad_norm, heads_grad_norm)."""
        x, y = _make_tiny_dataset()
        x_t = torch.tensor(x, dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(model.device).unsqueeze(1)
        model.model.train()
        self._zero_grads(model.model)

        model_outputs = model.model(x_t)
        loss = model._calculate_custom_loss(model_outputs, y_t, include_boundary_loss=False)
        loss.backward()

        classifier_grad = self._total_abs_grad(model.model.classifier_layers.parameters())
        heads_grad = self._total_abs_grad(model.model.regression_module.parameters())
        return classifier_grad, heads_grad

    def test_default_classifier_gets_ce_gradient_and_regression_loss_sends_none(self):
        """Default model: the classifier is trained by cross-entropy alone; the regression loss never reaches it.

        The regression term alone must leave every classifier gradient at zero (probabilities are
        detached before the heads) while the heads still learn from it; the full loss must give the
        classifier a gradient, and that gradient can only come from the cross-entropy term.
        """
        model = self._fit_tiny()
        assert model.optimization_strategy is ProbabilisticRegressionOptimizationStrategy.CE_STOP_GRAD
        x, y = _make_tiny_dataset()
        x_t = torch.tensor(x, dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(model.device).unsqueeze(1)
        model.model.train()

        self._zero_grads(model.model)
        outputs = model.model(x_t)
        regression_loss = calculate_combined_loss(
            predictions=outputs[0], y_true=y_t, uncertainty_method=model.uncertainty_method,
            include_boundary_loss=False, loss_type=model.loss_type, beta=model.beta,
        )
        regression_loss.backward()
        assert self._total_abs_grad(model.model.classifier_layers.parameters()) == 0.0, "regression loss must send no gradient into the classifier"
        assert self._total_abs_grad(model.model.regression_module.parameters()) > 0.0, "regression heads must learn from the regression loss"

        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0, "classifier must receive gradient from the cross-entropy loss"
        assert heads_grad > 0.0

    def test_regression_only_no_ce_gradient(self):
        """REGRESSION_ONLY: no CE, so classifier gradient comes only from regression loss path.

        Verifies classifier still gets gradient (via non-detached probs) and heads get gradient.
        """
        model = self._fit_tiny(ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad > 0.0
        assert heads_grad > 0.0

    @pytest.mark.parametrize("l2_lambda", [0.0, 1e-2])
    def test_gradient_stop_classifier_gets_no_gradient_and_keeps_initial_weights(self, l2_lambda):
        """GRADIENT_STOP at fixed k is the "classifier learns nothing" control: no CE and probabilities detached.

        The classifier must receive zero gradient from the full loss, the heads must still learn, and after a
        short fit the classifier's weights must equal the ones it was built with, also under an L2 penalty
        (which would otherwise shrink them toward uniform probabilities).
        """
        x, y = _make_tiny_dataset()
        model = _make_model(ProbabilisticRegressionOptimizationStrategy.GRADIENT_STOP, n_epochs=3, l2_lambda=l2_lambda)
        initial: dict[str, torch.Tensor] = {}
        build = model.build_model

        def build_and_snapshot() -> None:
            build()
            initial.clear()
            initial.update({k: v.detach().clone() for k, v in model.model.classifier_layers.state_dict().items()})

        model.build_model = build_and_snapshot
        model.fit(x, y)
        classifier_grad, heads_grad = self._run_backward(model)
        assert classifier_grad == 0.0, "GRADIENT_STOP + fixed k must send no gradient into the classifier"
        assert heads_grad > 0.0
        assert initial, "build_model was never called"
        for name, tensor in model.model.classifier_layers.state_dict().items():
            assert torch.equal(tensor, initial[name]), f"classifier weight {name} changed under GRADIENT_STOP"


class TestTwoStage:
    def test_classifier_trained_alone_then_frozen_while_heads_train(self):
        """TWO_STAGE: stage 1 moves only the classifier and stops on held-out cross-entropy; stage 2 moves only the heads, on a classifier that no longer changes, and stops on the held-out fit term."""
        x, y = _make_tiny_dataset()
        model = _make_model(ProbabilisticRegressionOptimizationStrategy.TWO_STAGE, n_epochs=40, l2_lambda=1e-2)
        model.early_stopping_rounds = 3
        seen: list[tuple[TwoStagePhase, list[torch.Tensor], list[torch.Tensor]]] = []

        def record(epoch: int, model: ProbabilisticRegressionModel, val_loss: float | None) -> None:  # noqa: ARG001 -- epoch_callback's keyword contract
            classifier = [p.detach().clone() for p in model.model.classifier_layers.parameters()]
            heads = [p.detach().clone() for p in model.model.regression_module.parameters()]
            seen.append((model.two_stage_phase_, classifier, heads))

        model.epoch_callback = record
        model.fit(x[:60], y[:60], x_val=x[60:], y_val=y[60:])

        classifier_epochs = [(c, h) for phase, c, h in seen if phase is TwoStagePhase.CLASSIFIER]
        head_epochs = [(c, h) for phase, c, h in seen if phase is TwoStagePhase.HEADS]
        assert classifier_epochs
        assert head_epochs
        # Stage 1: the heads never move; the classifier does.
        assert all(torch.equal(h_a, h_b) for h_a, h_b in zip(classifier_epochs[0][1], classifier_epochs[-1][1], strict=True))
        assert not all(torch.equal(a, b) for a, b in zip(classifier_epochs[0][0], classifier_epochs[-1][0], strict=True))
        # Stage 2: the classifier is the same in every epoch and in the final model (stage 1's best epoch restored); the heads move.
        final_classifier = list(model.model.classifier_layers.parameters())
        for c, _ in head_epochs:
            assert all(torch.equal(a, b) for a, b in zip(c, final_classifier, strict=True))
        assert not all(torch.equal(a, b) for a, b in zip(head_epochs[0][1], head_epochs[-1][1], strict=True))

        # Each stage stopped on its own held-out score, best epoch restored: the restored model reproduces each record's best value.
        assert isinstance(model.classifier_convergence_, ConvergenceResult)
        assert isinstance(model.convergence_, ConvergenceResult)
        assert model.classifier_convergence_ is not model.convergence_
        x_val_t = torch.tensor(x[60:], dtype=torch.float32).to(model.device)
        y_val_t = torch.tensor(y[60:], dtype=torch.float32).to(model.device).unsqueeze(1)
        model.model.eval()
        with torch.no_grad():
            model.two_stage_phase_ = TwoStagePhase.CLASSIFIER
            cross_entropy = model._validation_loss(model.model(x_val_t), y_val_t)
            model.two_stage_phase_ = None
        assert cross_entropy == pytest.approx(model.classifier_convergence_.best_val, rel=1e-4)
        assert model._held_out_fit_score(x[60:], y[60:]) == pytest.approx(model.convergence_.best_val, rel=1e-4)

        with pytest.raises(ValueError, match="refit_on_train_val=False"):
            ProbabilisticRegressionModel(input_size=1, n_classes=3, optimization_strategy=ProbabilisticRegressionOptimizationStrategy.TWO_STAGE)


# ---------------------------------------------------------------------------
# Held-out FIT TERM (DECISIONS.md "Validation score = the held-out FIT TERM", owner 2026-11-01)
# ---------------------------------------------------------------------------

_HALF_LOG_2PI = 0.9189385332046727


def _gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (_HALF_LOG_2PI + 0.5 * log_var + 0.5 * (y - mean) ** 2 / torch.exp(log_var)).mean()


class TestHeldOutFitTerm:
    def test_probreg_early_stopping_and_search_score_are_the_gaussian_nll_without_cross_entropy(self):
        """Early stopping and the search/CV score read the Gaussian NLL of the final prediction; the classifier cross-entropy is excluded."""
        x, y = _make_tiny_dataset()
        x_val, y_val = x[:30], y[:30]
        model = _make_model(n_epochs=3)
        model.early_stopping_rounds = 5
        seen: list[tuple[float, float, float]] = []

        def _record(epoch: int, model: ProbabilisticRegressionModel, val_loss: float | None) -> None:  # noqa: ARG001 -- epoch_callback's keyword contract
            x_t = torch.tensor(x_val, dtype=torch.float32).to(model.device)
            y_t = torch.tensor(y_val, dtype=torch.float32).to(model.device)
            model.model.eval()
            with torch.no_grad():
                outputs = model.model(x_t)
                nll = _gaussian_nll(outputs[0][:, 0], outputs[0][:, 1], y_t).item()
                training_loss = model._calculate_custom_loss(outputs, y_t.unsqueeze(1)).item()
            seen.append((val_loss, nll, training_loss))
            model.model.train()

        model.epoch_callback = _record
        model._fit_single(x, y, x_val=x_val, y_val=y_val)

        assert len(seen) == 3
        for val_loss, nll, training_loss in seen:
            assert val_loss == pytest.approx(nll, rel=1e-5), "early stopping must read the Gaussian NLL of the final prediction"
            assert training_loss - nll > 0.1, "the training loss carries the classifier cross-entropy on top of the fit; the validation score must not"
        score = model._held_out_fit_score(x_val, y_val)
        assert score == pytest.approx(seen[-1][1], rel=1e-5), "search and CV score the same quantity early stopping reads"
        assert model._held_out_score_is_smaller_better()

        # The DEFAULT model (mean-only heads): the held-out fit is the mean squared error of the final prediction, cross-entropy excluded.
        default = ProbabilisticRegressionModel(input_size=1, n_classes=3, n_epochs=2, random_seed=0, calculate_feature_importance=False)
        default.fit(x, y)
        x_t = torch.tensor(x_val, dtype=torch.float32).to(default.device)
        y_t = torch.tensor(y_val, dtype=torch.float32).to(default.device)
        default.model.eval()
        with torch.no_grad():
            outputs = default.model(x_t)
            mse = ((outputs[0][:, 0] - y_t) ** 2).mean().item()
            training_loss = default._calculate_custom_loss(outputs, y_t.unsqueeze(1)).item()
        assert default._held_out_fit_score(x_val, y_val) == pytest.approx(mse, rel=1e-5)
        assert training_loss - mse > 0.1

        # The score follows the model's own fit term, not a fixed metric: under beta-NLL it is the beta-NLL.
        model.loss_type = "beta_nll"
        x_t = torch.tensor(x_val, dtype=torch.float32).to(model.device)
        model.model.eval()
        with torch.no_grad():
            expected_beta = beta_nll_loss(model.model(x_t)[0], torch.tensor(y_val, dtype=torch.float32).to(model.device), beta=model.beta).item()
        assert model._held_out_fit_score(x_val, y_val) == pytest.approx(expected_beta, rel=1e-5)
        assert expected_beta != pytest.approx(seen[-1][1], rel=1e-3)

    def test_best_checkpoint_is_restored_not_the_last_epoch(self):
        """Early stopping restores the weights of the best held-out epoch: a COPY taken then, not references to the live parameters."""
        rng = np.random.default_rng(0)
        x_train = rng.uniform(-1, 1, (12, 1)).astype(np.float32)
        y_train = rng.normal(0, 1, 12).astype(np.float32)  # pure noise: the net memorises it, so held-out error rises after the early epochs
        x_val = rng.uniform(-1, 1, (40, 1)).astype(np.float32)
        y_val = np.zeros(40, dtype=np.float32)
        model = PyTorchNeuralNetwork(
            input_size=1, task_type=TaskType.REGRESSION, hidden_layers=2, hidden_size=64, learning_rate=0.05, n_epochs=150, batch_size=12, early_stopping_rounds=150, random_seed=0
        )
        history: list[tuple[float, dict[str, torch.Tensor]]] = []

        def _record(epoch: int, model: PyTorchNeuralNetwork, val_loss: float | None) -> None:  # noqa: ARG001 -- epoch_callback's keyword contract
            history.append((val_loss, {k: v.clone() for k, v in model.model.state_dict().items()}))

        model.epoch_callback = _record
        model._fit_single(x_train, y_train, x_val=x_val, y_val=y_val)

        best = int(np.argmin([loss for loss, _ in history]))
        assert best < len(history) - 1, "fixture must make the held-out loss worse after its best epoch"
        restored = model.model.state_dict()
        assert all(torch.equal(restored[k], history[best][1][k]) for k in restored), "restored weights must be the best epoch's"
        assert not all(torch.equal(restored[k], history[-1][1][k]) for k in restored), "restored weights must differ from the last epoch's"

    def test_tuple_output_without_final_prediction_override_raises(self):
        """A model whose network returns a tuple must say which element is the final prediction; guessing is refused."""
        nn_model = PyTorchNeuralNetwork(input_size=2, task_type=TaskType.REGRESSION, uncertainty_method=UncertaintyMethod.CONSTANT)
        with pytest.raises(TypeError, match="_final_prediction"):
            nn_model._final_prediction((torch.zeros(4, 1), torch.zeros(4)))
        probreg = _make_model()
        assert probreg._final_prediction((torch.ones(4, 2), None)).shape == (4, 2)

    @pytest.mark.parametrize(
        ("model_cls", "kwargs", "output_size"),
        [
            (PyTorchNeuralNetwork, {"task_type": TaskType.REGRESSION, "uncertainty_method": UncertaintyMethod.PROBABILISTIC}, 2),
            (PyTorchNeuralNetwork, {"task_type": TaskType.REGRESSION, "uncertainty_method": UncertaintyMethod.CONSTANT}, 1),
            (PyTorchNeuralNetwork, {"task_type": TaskType.CLASSIFICATION}, 1),
            (PyTorchNeuralNetwork, {"task_type": TaskType.CLASSIFICATION}, 3),
            (PyTorchLinearRegression, {"uncertainty_method": UncertaintyMethod.PROBABILISTIC}, 2),
            (PyTorchLinearRegression, {"uncertainty_method": UncertaintyMethod.CONSTANT}, 1),
        ],
        ids=["nn_nll", "nn_mse", "nn_bce", "nn_ce", "linear_nll", "linear_mse"],
    )
    def test_plain_criterion_equals_derived_fit_term(self, model_cls, kwargs, output_size):
        """For a base-loop model whose training loss is only the fit, `criterion` and `_fit_loss` agree, and the per-sample form averages to it."""
        model = model_cls(input_size=3, output_size=output_size, **kwargs)
        model.build_model()
        gen = torch.Generator().manual_seed(0)
        outputs = torch.randn(16, output_size, generator=gen)
        y_np = np.random.default_rng(0).integers(0, max(output_size, 2), 16) if kwargs.get("task_type") == TaskType.CLASSIFICATION else np.random.default_rng(0).normal(size=16)
        y = model._target_tensor(y_np)
        fit = model._fit_loss(model._final_prediction(outputs), y)
        assert fit.item() == pytest.approx(model.criterion(outputs, y).item(), rel=1e-6)
        per_sample = model._fit_loss(outputs, y, reduction="none")
        assert per_sample.shape == (16,)
        assert per_sample.mean().item() == pytest.approx(fit.item(), rel=1e-5)

    def test_probreg_training_loss_fit_term_unchanged_and_beta_nll_per_sample(self):
        """`_calculate_custom_loss` with no regularisation equals the plain combined loss (no change to training); beta-NLL has a per-sample form."""
        model = _make_model(ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY)
        model.loss_type = "beta_nll"
        outputs = torch.randn(12, 2)
        y = torch.randn(12, 1)
        expected = calculate_combined_loss(outputs, y, UncertaintyMethod.PROBABILISTIC, loss_type="beta_nll", beta=model.beta)
        model.ordering_constraint_weight = 0.0
        loss = model._calculate_custom_loss((outputs, torch.zeros(12, 3), torch.full((12,), 3), None, None), y)
        assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
        assert model._fit_loss(outputs, y, reduction="none").mean().item() == pytest.approx(expected.item(), rel=1e-5)


class TestKSelectorsReadTheFitTable:
    @staticmethod
    def _toy(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        x = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
        return x, (np.sin(x.ravel()) + rng.normal(0, 0.2 + 0.2 * np.abs(x.ravel()))).astype(np.float32)

    def test_global_selector_curve_is_the_gaussian_nll_per_rung(self):
        """Under PROBABILISTIC, each rung's curve entry is that rung's Gaussian NLL (mean and spread), not its squared error."""
        x, y = self._toy()
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=3, uncertainty_method=UncertaintyMethod.PROBABILISTIC,
            n_classes_selection_method=NClassesSelectionMethod.NESTED, allow_unredesigned_nested=True,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            n_epochs=8, learning_rate=0.01, random_seed=0, calculate_feature_importance=False,
        )
        model.fit(x, y)
        model.fit_global_selector(x, y, n_bootstrap=50)
        curve = model.global_selector_curve_["mean_fit_loss"]
        x_t, y_t = torch.tensor(x), torch.tensor(y)
        model.model.eval()
        with torch.no_grad():
            for k in (1, 2, 3):
                out = model.model.forward_at_k(x_t, k)
                assert curve[k - 1] == pytest.approx(_gaussian_nll(out[:, 0], out[:, 1], y_t).item(), rel=1e-4)

    def test_sweep_selector_curve_is_mse_under_constant_heads(self):
        """Under CONSTANT (mean-only heads) the fit term is squared error, so the sweep's curve entry for the winner is its held-out MSE."""
        x, y = self._toy()
        model = ProbabilisticRegressionModel(
            input_size=1, n_classes=3, max_n_classes_for_probabilistic_path=3, uncertainty_method=UncertaintyMethod.CONSTANT,
            optimization_strategy=ProbabilisticRegressionOptimizationStrategy.REGRESSION_ONLY,
            n_epochs=8, learning_rate=0.01, random_seed=0, calculate_feature_importance=False,
        )
        k = model.fit_sweep_selector(x[:150], y[:150], x[150:], y[150:], n_bootstrap=50)
        winner = model._sweep_submodel()
        mse = float(np.mean((winner.predict(x[150:]) - y[150:]) ** 2))
        curve = model.global_selector_curve_
        assert curve["mean_fit_loss"][curve["capacity_grid"].index(k)] == pytest.approx(mse, rel=1e-4)
