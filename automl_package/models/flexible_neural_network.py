"""Flexible Neural Network model with dynamic hidden layers."""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from automl_package.enums import ActivationFunction, DepthRegularization, ExplainerType, LayerSelectionMethod, TaskType, UncertaintyMethod
from automl_package.logger import logger
from automl_package.models.base_pytorch import PyTorchModelBase
from automl_package.models.common.distilled_router import DEFAULT_TOLERANCE, DistilledCapacityRouter
from automl_package.models.selection_strategies.layer_selection_strategies import (
    GumbelSoftmaxStrategy,
    NestedStrategy,
    NoneStrategy,
    ReinforceStrategy,
    SoftGatingStrategy,
    SteStrategy,
)
from automl_package.utils.convergence import DEFAULT_PATIENCE, ConvergenceTracker
from automl_package.utils.losses import nll_loss
from automl_package.utils.pytorch_utils import get_activation_function_map, get_device

BINARY_CLASSIFICATION_THRESHOLD = 0.5


class FlexibleHiddenLayersNN(PyTorchModelBase):
    """A PyTorch Neural Network with a dynamic number of active hidden layers.

    It includes an internal feedforward network that predicts 'n' (1 to max_hidden_layers),
    where 'n' is the number of active hidden layers: the FIRST n hidden-layer blocks
    (blocks[0:n]) are executed, and the LAST (max_hidden_layers - n) blocks are skipped
    entirely for that sample -- they are not run as identity layers, they simply do not
    execute.
    Supports optional Batch Normalization and L1/L2 regularization.
    Supports constant, MC-Dropout, and probabilistic uncertainty estimation for regression.
    """

    _defaults: ClassVar[dict[str, Any]] = {
        "max_hidden_layers": 3,
        "hidden_size": 64,
        "activation": ActivationFunction.RELU,
        "gumbel_tau": 0.5,
        "n_predictor_layers": 1,
        "feature_scaler": None,
        "gumbel_tau_anneal_rate": 0.99,
        "n_predictor_learning_rate": 0.001,
        "layer_selection_method": LayerSelectionMethod.GUMBEL_SOFTMAX,
        "depth_regularization": DepthRegularization.NONE,
        "depth_penalty_weight": 0.01,
        "cost_aware_lambda": 1.0,
    }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the FlexibleHiddenLayersNN."""
        # Apply this class's defaults and then pass to the parent constructor
        for key, value in FlexibleHiddenLayersNN._defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)

        # Validation logic
        if self.layer_selection_method in (LayerSelectionMethod.NONE, LayerSelectionMethod.NESTED) and self.n_predictor_layers != 0:
            raise ValueError("n_predictor_layers must be 0 when layer_selection_method is NONE or NESTED.")
        if (
            self.layer_selection_method in [LayerSelectionMethod.GUMBEL_SOFTMAX, LayerSelectionMethod.STE, LayerSelectionMethod.SOFT_GATING, LayerSelectionMethod.REINFORCE]
            and self.n_predictor_layers <= 0
        ):
            raise ValueError("n_predictor_layers must be > 0 for GUMBEL_SOFTMAX, STE, SOFT_GATING or REINFORCE methods.")

        strategy_map = {
            LayerSelectionMethod.NONE: NoneStrategy,
            LayerSelectionMethod.GUMBEL_SOFTMAX: GumbelSoftmaxStrategy,
            LayerSelectionMethod.SOFT_GATING: SoftGatingStrategy,
            LayerSelectionMethod.STE: SteStrategy,
            LayerSelectionMethod.REINFORCE: ReinforceStrategy,
            LayerSelectionMethod.NESTED: NestedStrategy,
        }
        self.strategy = strategy_map[self.layer_selection_method](self)

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return "FlexibleNeuralNetwork"

    class FlexibleNNModule(nn.Module):
        """Internal PyTorch nn.Module for FlexibleHiddenLayersNN.

        For a sample selecting depth n, runs the FIRST n blocks of
        `hidden_layers_blocks` (blocks[0:n]); the LAST (max_hidden_layers - n)
        blocks are skipped entirely, not executed as identity layers.
        """

        def __init__(self, outer_instance: Any) -> None:
            """Initializes the _FlexibleNN module."""
            super().__init__()
            self.outer = outer_instance

            # Output size for the main network's output layer
            final_output_neurons = self.outer.output_size
            if self.outer.task_type == TaskType.REGRESSION and self.outer.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
                final_output_neurons = 2  # Mean and Log-Variance

            # n-predictor: Takes full input features and outputs logits for n (1 to max_hidden_layers)
            if self.outer.n_predictor_layers > 0:
                predictor_layers = []
                predictor_hidden_size = max(128, self.outer.hidden_size)
                in_features = self.outer.input_size
                predictor_layers.append(nn.Linear(in_features, predictor_hidden_size))
                predictor_layers.append(self.outer.activation())
                for _ in range(self.outer.n_predictor_layers - 1):
                    predictor_layers.append(nn.Linear(predictor_hidden_size, predictor_hidden_size))
                    predictor_layers.append(self.outer.activation())
                output_layer_predictor = nn.Linear(predictor_hidden_size, self.outer.max_hidden_layers)
                nn.init.normal_(output_layer_predictor.bias, mean=0.0, std=0.1)
                self.n_predictor = nn.Sequential(*predictor_layers, output_layer_predictor)
            else:
                self.n_predictor = None

            self.hidden_layers_blocks = nn.ModuleList()
            for i in range(self.outer.max_hidden_layers):
                block_layers = []
                in_features = self.outer.input_size if i == 0 else self.outer.hidden_size
                block_layers.append(nn.Linear(in_features, self.outer.hidden_size))
                if self.outer.use_batch_norm:
                    block_layers.append(nn.BatchNorm1d(self.outer.hidden_size))
                block_layers.append(self.outer.activation())
                if self.outer.task_type == TaskType.REGRESSION and self.outer.uncertainty_method == UncertaintyMethod.MC_DROPOUT and self.outer.dropout_rate > 0:
                    block_layers.append(nn.Dropout(self.outer.dropout_rate))
                self.hidden_layers_blocks.append(nn.Sequential(*block_layers))

            self.output_layer = nn.Linear(self.outer.hidden_size, final_output_neurons)

        def forward(self, x_input: torch.Tensor) -> Any:
            """Forward pass for the flexible neural network."""
            n_logits = self.n_predictor(x_input) if self.n_predictor else None
            return self.outer.strategy.forward(x_input, n_logits)

        def hard_forward(self, x_input: torch.Tensor) -> torch.Tensor:
            """Inference-only hard forward: runs only the argmax depth per sample.

            Genuine compute savings — samples with depth=1 only execute layer 0,
            samples with depth=k execute layers 0..k-1. Groups samples by depth
            for batched execution per depth bucket.
            """
            if self.n_predictor is None:
                # No depth selection, run all layers
                current = x_input
                for block in self.hidden_layers_blocks:
                    current = block(current)
                return self.output_layer(current)

            n_logits = self.n_predictor(x_input)
            n_actual = torch.argmax(n_logits, dim=1)  # 0-indexed: 0 = depth 1
            output_features = self.output_layer.out_features
            result = torch.zeros(x_input.size(0), output_features, device=x_input.device, dtype=x_input.dtype)

            for depth_idx in range(self.outer.max_hidden_layers):
                mask = (n_actual == depth_idx)
                if not torch.any(mask):
                    continue
                x_subset = x_input[mask]
                current = x_subset
                # depth_idx=0 means depth 1 → run blocks[0], so range(depth_idx + 1)
                for layer_i in range(depth_idx + 1):
                    current = self.hidden_layers_blocks[layer_i](current)
                out_subset = self.output_layer(current)
                result[mask] = out_subset

            return result

        def forward_at_depth(self, x_input: torch.Tensor, depth: int) -> torch.Tensor:
            """Forces every sample in the batch through exactly `depth` blocks (1-indexed).

            Bypasses `n_predictor`/the selection strategy entirely -- used by `fit_router` to
            build a per-depth held-out error table (capacity-programme Task F3).
            """
            max_depth = len(self.hidden_layers_blocks)
            if not (1 <= depth <= max_depth):
                raise ValueError(f"depth={depth} out of range [1, {max_depth}]")
            current = x_input
            for layer_i in range(depth):
                current = self.hidden_layers_blocks[layer_i](current)
            return self.output_layer(current)

        def forward_at_depths(self, x_input: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
            """Per-sample forced depth (e.g. routed by a `DistilledCapacityRouter`), bucketed by depth.

            `depths` is a `(N,)` LongTensor of 1-indexed depth values in `[1, max_hidden_layers]`
            -- NOT 0-indexed, unlike `hard_forward`'s internal `n_actual` (which is 0-indexed
            against `n_predictor`'s logits).
            """
            output_features = self.output_layer.out_features
            result = torch.zeros(x_input.size(0), output_features, device=x_input.device, dtype=x_input.dtype)
            for depth in torch.unique(depths).tolist():
                mask = depths == depth
                result[mask] = self.forward_at_depth(x_input[mask], depth)
            return result

    def build_model(self) -> None:
        """Builds the internal PyTorch nn.Module for the FlexibleHiddenLayersNN."""
        if isinstance(self.activation, ActivationFunction):
            activation_function_map = get_activation_function_map()
            self.activation = activation_function_map.get(self.activation)
            if self.activation is None:
                raise ValueError(f"Unsupported activation function: {self.activation}")

        self.model = self.FlexibleNNModule(self).to(self.device)

        if self.task_type == TaskType.REGRESSION:
            self.criterion = nll_loss if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else nn.MSELoss()
        elif self.task_type == TaskType.CLASSIFICATION:
            self.criterion = nn.BCEWithLogitsLoss() if self.output_size == 1 else nn.CrossEntropyLoss()
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

    def _fit_single(
        self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, forced_iterations: int | None = None
    ) -> tuple[int, list[float]]:
        """Fits a single model instance.

        Args:
            x_train (np.ndarray): The training features.
            y_train (np.ndarray): The training targets.
            x_val (np.ndarray | None): The validation features.
            y_val (np.ndarray | None): The validation targets.
            forced_iterations (int | None): If provided, train for this many iterations, ignoring early stopping.

        Returns:
            tuple[int, list[float]]: A tuple containing:
                - The number of iterations the model was trained for.
                - A list of the validation loss values for each epoch.
        """
        if self.input_size is None:
            self.input_size = 1 if x_train.ndim == 1 else x_train.shape[1]

        self.device = get_device()

        if self.learn_regularization_lambdas:
            if self.using_l1_regularization:
                self.l1_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))
            if self.using_l2_regularization:
                self.l2_log_lambda = nn.Parameter(torch.tensor(np.log(1e-4), dtype=torch.float32))

        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

        self.build_model()
        self._setup_optimizers(self.model)

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        if self.task_type == TaskType.CLASSIFICATION:
            y_train_tensor = y_train_tensor.long()
        if (self.task_type == TaskType.REGRESSION) or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
            y_train_tensor = y_train_tensor.unsqueeze(1)

        train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        x_val_tensor, y_val_tensor = None, None
        if x_val is not None:
            x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            if self.task_type == TaskType.CLASSIFICATION:
                y_val_tensor = y_val_tensor.long()
            if (self.task_type == TaskType.REGRESSION) or (self.task_type == TaskType.CLASSIFICATION and self.output_size == 1):
                y_val_tensor = y_val_tensor.unsqueeze(1)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        val_loss_history = []

        if self.layer_selection_method not in [LayerSelectionMethod.GUMBEL_SOFTMAX, LayerSelectionMethod.STE]:
            logger.info("Ignoring gumbel_tau and gumbel_tau_anneal_rate as a non-Gumbel layer selection method is used.")

        n_epochs = forced_iterations or self.n_epochs
        for epoch in range(int(n_epochs)):
            self.model.train()
            epoch_log_probs = []
            for _batch_x, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                if self.strategy.policy_optimizer:
                    self.strategy.policy_optimizer.zero_grad()
                if self.lambda_optimizer:
                    self.lambda_optimizer.zero_grad()

                # Strategy tuple order: (output, n_actual, _unused, n_probs, log_prob).
                final_output, _, _unused, n_probs, log_prob = self.model(_batch_x)
                loss = self.criterion(final_output, batch_y)
                loss = self._calculate_regularization_loss(loss, self.model)

                if self.depth_regularization == DepthRegularization.ELBO and n_probs is not None:
                    # Uniform prior over depth. The earlier linspace(3,1) prefer-shallow prior
                    # is REMOVED: M0 revalidation measured complete depth-collapse to that prior
                    # regardless of input (report_b_results/, flexnn-moe.md Done ledger M0) --
                    # the identical trap already removed from ProbReg's k-prior (see
                    # probabilistic_regression.py, NClassesRegularization.ELBO history).
                    depth_prior_logits = torch.zeros(self.max_hidden_layers, dtype=torch.float, device=_batch_x.device)
                    depth_prior = torch.distributions.Categorical(logits=depth_prior_logits)
                    q_depth = torch.distributions.Categorical(probs=n_probs + 1e-8)
                    kl_div = torch.distributions.kl_divergence(q_depth, depth_prior).mean()
                    loss = loss + kl_div
                elif self.depth_regularization == DepthRegularization.COST_AWARE_ELBO and n_probs is not None:
                    # Cost-aware prior: uniform base shifted by -lambda * normalized FLOPs(depth).
                    # FLOPs at depth d (d = 1..max_hidden_layers) are approx linear in d for a fixed
                    # hidden_size, so we normalise a linear cost to [0,1] and subtract from logits.
                    base = torch.zeros(self.max_hidden_layers, dtype=torch.float, device=_batch_x.device)
                    depth_idx = torch.arange(self.max_hidden_layers, dtype=torch.float, device=_batch_x.device)
                    cost = depth_idx / max(self.max_hidden_layers - 1, 1)
                    depth_prior_logits = base - float(self.cost_aware_lambda) * cost
                    depth_prior = torch.distributions.Categorical(logits=depth_prior_logits)
                    q_depth = torch.distributions.Categorical(probs=n_probs + 1e-8)
                    kl_div = torch.distributions.kl_divergence(q_depth, depth_prior).mean()
                    loss = loss + kl_div
                elif self.depth_regularization == DepthRegularization.DEPTH_PENALTY and n_probs is not None:
                    depth_indices = torch.arange(1, self.max_hidden_layers + 1, dtype=torch.float, device=_batch_x.device)
                    expected_depth = torch.sum(n_probs * depth_indices, dim=1)
                    loss = loss + self.depth_penalty_weight * expected_depth.mean()

                if self.layer_selection_method == LayerSelectionMethod.REINFORCE and log_prob is not None:
                    reward = -loss.detach()
                    policy_loss = -log_prob * reward
                    loss = loss + policy_loss.mean()

                loss.backward()

                self.optimizer.step()
                if self.strategy.policy_optimizer:
                    self.strategy.policy_optimizer.step()
                if self.lambda_optimizer:
                    self.lambda_optimizer.step()

            if x_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs, _, _, _, _ = self.model(x_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                val_loss_history.append(val_loss)

                self.strategy.on_epoch_end(validation_loss=val_loss, epoch_log_probs=epoch_log_probs)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                    best_epoch = epoch
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_rounds:
                    logger.info(f"Early stopping at epoch {best_epoch + 1}")
                    break
            else:
                best_epoch = epoch

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.CONSTANT:
            y_pred_train = self.predict(x_train, filter_data=False)
            self._train_residual_std = np.std(y_train - y_pred_train)
            if np.isnan(self._train_residual_std):
                self._train_residual_std = 0.0

        if self.learn_regularization_lambdas:
            if self.l1_log_lambda is not None:
                logger.info(f"Learned L1 Lambda: {torch.exp(self.l1_log_lambda).item():.6f}")
            if self.l2_log_lambda is not None:
                logger.info(f"Learned L2 Lambda: {torch.exp(self.l2_log_lambda).item():.6f}")

        patience = self.early_stopping_rounds if self.early_stopping_rounds is not None else DEFAULT_PATIENCE
        tracker = ConvergenceTracker(patience=patience, min_delta=0.0)  # min_delta=0.0 mirrors this loop's own `val_loss < best_val_loss` rule exactly
        for i, v in enumerate(val_loss_history, start=1):
            tracker.update(i, v)
        self.convergence_summary_ = tracker.result(final_epoch=len(val_loss_history)).summary()

        return best_epoch + 1, val_loss_history

    def predict(self, x: np.ndarray, filter_data: bool = True, inference_mode: str = "soft") -> np.ndarray:
        """Makes predictions with the FlexibleHiddenLayersNN.

        Args:
            x: Input features.
            filter_data: Whether to filter input columns to those seen at fit time.
            inference_mode: "soft" uses the trained selection strategy (full forward pass).
                "hard" runs only the argmax-selected depth per sample for compute savings.
                "routed" uses a `DistilledCapacityRouter` fitted via `fit_router()` to pick a
                per-sample depth post-hoc (capacity-programme Task F3; MASTER Decision 13).
        """
        if inference_mode not in ("soft", "hard", "routed"):
            raise ValueError(f"Unknown inference_mode: {inference_mode!r}")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if inference_mode == "routed" and getattr(self, "capacity_router_", None) is None:
            raise RuntimeError("No router fitted; call fit_router() before predict(inference_mode='routed').")
        if filter_data:
            x = self._filter_predict_data(x)
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)

        if self.is_regression_model and self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    final_output, _, _, _, _ = self.model(x_tensor)
                    mc_predictions.append(final_output.cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.mean(mc_predictions, axis=0)

        self.model.eval()
        with torch.no_grad():
            if inference_mode == "hard":
                final_output = self.model.hard_forward(x_tensor)
            elif inference_mode == "routed":
                depths_np = np.array([capacity[0] for capacity in self.capacity_router_.route(x_array)], dtype=np.int64)
                depths_tensor = torch.as_tensor(depths_np, dtype=torch.long, device=self.device)
                final_output = self.model.forward_at_depths(x_tensor, depths_tensor)
            else:
                final_output, _, _, _, _ = self.model(x_tensor)
            if self.task_type == TaskType.CLASSIFICATION:
                if self.output_size == 1:
                    predictions = (torch.sigmoid(final_output) > BINARY_CLASSIFICATION_THRESHOLD).cpu().numpy().astype(int)
                else:
                    predictions = torch.argmax(final_output, dim=1).cpu().numpy()
            else:
                predictions = final_output[:, 0].cpu().numpy() if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else final_output.cpu().numpy()
        return predictions.flatten()

    def fit_router(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        capacity_grid: list[tuple[int, ...]] | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
        cost_fn: Callable[[tuple[int, ...]], float] | None = None,
    ) -> DistilledCapacityRouter:
        """Fits a `DistilledCapacityRouter` post-hoc for `predict(..., inference_mode="routed")`.

        Decision 13 (distilled, never in-training): trains an MLP mapping raw `x_val` to a depth,
        entirely separate from the trained `n_predictor`/selection strategy. `eval_fn` forces every
        sample through a fixed depth via `FlexibleNNModule.forward_at_depth` (bypassing
        `n_predictor` for that call) and scores it with the same per-sample error this model's own
        `predict()` decision rule implies: squared error for regression, 0/1 error for
        classification.

        Args:
            x_val: held-out inputs.
            y_val: held-out targets.
            capacity_grid: candidate depths as 1-tuples, e.g. `[(1,), (2,), (3,)]`. Defaults to
                every depth `1..max_hidden_layers`.
            tolerance: cheapest-within-tolerance labeling tolerance (see `DistilledCapacityRouter`).
            cost_fn: `cost_fn(capacity) -> float`. Defaults to `executed_flops` from
                `automl_package/examples/capacity_accounting.py` (S2 accounting) at the routed
                depth. Imported inside this method, not at module top, to avoid a load-time
                circular import: `capacity_accounting.py` itself imports `FlexibleHiddenLayersNN`
                to register its own accounting for this class.

        Returns:
            The fitted `DistilledCapacityRouter` (also stored as `self.capacity_router_`).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if capacity_grid is None:
            capacity_grid = [(depth,) for depth in range(1, self.max_hidden_layers + 1)]

        y_val_arr = np.asarray(y_val)

        def eval_fn(x: np.ndarray, capacity: tuple[int, ...]) -> np.ndarray:
            depth = capacity[0]
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                raw_output = self.model.forward_at_depth(x_tensor, depth)
            if self.task_type == TaskType.CLASSIFICATION:
                if self.output_size == 1:
                    pred = (torch.sigmoid(raw_output) > BINARY_CLASSIFICATION_THRESHOLD).float().squeeze(1).cpu().numpy()
                else:
                    pred = torch.argmax(raw_output, dim=1).float().cpu().numpy()
                return (pred != y_val_arr).astype(np.float64)
            target = raw_output[:, 0].cpu().numpy() if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC else raw_output.squeeze(1).cpu().numpy()
            return (target - y_val_arr) ** 2

        if cost_fn is None:
            from automl_package.examples.capacity_accounting import executed_flops  # noqa: PLC0415, I001 -- avoids a load-time circular import (capacity_accounting.py imports this class)

            def cost_fn(capacity: tuple[int, ...]) -> float:
                return float(executed_flops(self.model, capacity[0]))

        router = DistilledCapacityRouter(device=self.device)
        router.fit(eval_fn=eval_fn, x_val=x_val, y_val=y_val, capacity_grid=capacity_grid, tolerance=tolerance, cost_fn=cost_fn)
        self.capacity_router_ = router
        return router

    def predict_uncertainty(self, x: np.ndarray, filter_data: bool = True, inference_mode: str = "soft") -> np.ndarray:
        """Estimates uncertainty for regression.

        Args:
            x: Input features.
            filter_data: Whether to filter input columns to those seen at fit time.
            inference_mode: Must match the `inference_mode` a paired `predict()` call used, so
                the returned spread comes from the same selected depth as the mean it accompanies
                (DD2: previously this always used the soft selection regardless of what mode
                `predict()` was called with, silently pairing a hard/routed mean with a
                soft-selected uncertainty). Only `uncertainty_method=PROBABILISTIC` varies its
                depth selection with this argument -- it reuses `hard_forward`/`forward_at_depths`
                so the per-sample depths are byte-identical to `predict(inference_mode=...)`.
                CONSTANT is a fixed training-time residual std with no per-call depth dependence,
                and MC_DROPOUT's `predict()` already ignores `inference_mode` and always uses the
                soft selection (a separate, pre-existing quirk, out of DD2's scope) -- both accept
                any value of this argument without effect, since neither ever varies by depth
                selection the way `predict()`'s mean does.
        """
        if not self.is_regression_model:
            raise ValueError("predict_uncertainty is only available for regression models.")
        if inference_mode not in ("soft", "hard", "routed"):
            raise ValueError(f"Unknown inference_mode: {inference_mode!r}")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if filter_data:
            x = self._filter_predict_data(x)
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)

        if self.uncertainty_method == UncertaintyMethod.CONSTANT:
            return np.full(x.shape[0], self._train_residual_std)
        if self.uncertainty_method == UncertaintyMethod.PROBABILISTIC:
            self.model.eval()  # Use eval mode for prediction
            with torch.no_grad():
                if inference_mode == "hard":
                    final_output = self.model.hard_forward(x_tensor)
                elif inference_mode == "routed":
                    if getattr(self, "capacity_router_", None) is None:
                        raise RuntimeError("No router fitted; call fit_router() before predict_uncertainty(inference_mode='routed').")
                    depths_np = np.array([capacity[0] for capacity in self.capacity_router_.route(x_array)], dtype=np.int64)
                    depths_tensor = torch.as_tensor(depths_np, dtype=torch.long, device=self.device)
                    final_output = self.model.forward_at_depths(x_tensor, depths_tensor)
                else:
                    final_output, _, _, _, _ = self.model(x_tensor)
                log_var = final_output[:, 1]
                uncertainty = torch.exp(0.5 * log_var).cpu().numpy()
            return uncertainty.flatten()
        if self.uncertainty_method == UncertaintyMethod.MC_DROPOUT:
            self.model.train()  # Enable dropout during inference for MC Dropout
            mc_predictions = []
            with torch.no_grad():
                for _ in range(self.n_mc_dropout_samples):
                    final_output, _, _, _, _ = self.model(x_tensor)
                    mc_predictions.append(final_output.cpu().numpy().flatten())
            self.model.eval()  # Set back to eval mode
            return np.std(mc_predictions, axis=0)  # Return std dev of MC samples
        raise ValueError(f"Unknown uncertainty_method: {self.uncertainty_method.value}")

    def predict_proba(self, x: np.ndarray, filter_data: bool = True) -> np.ndarray:
        """Predicts class probabilities for classification tasks."""
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba is only available for classification tasks.")
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if filter_data:
            x = self._filter_predict_data(x)
        self.model.eval()
        x_array = x.values if hasattr(x, "values") else x
        x_tensor = torch.tensor(x_array, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            final_output, _, _, _, _ = self.model(x_tensor)
            if self.output_size == 1:
                proba = torch.sigmoid(final_output).cpu().numpy().flatten()
                return np.vstack((1 - proba, proba)).T
            return torch.softmax(final_output, dim=1).cpu().numpy()

    def get_hyperparameter_search_space(self) -> dict[str, Any]:
        """Defines the hyperparameter search space for FlexibleHiddenLayersNN."""
        space = super().get_hyperparameter_search_space()
        space.update(
            {
                "max_hidden_layers": {"type": "int", "low": 1, "high": 6},
                "hidden_size": {"type": "int", "low": 32, "high": 128, "step": 32},
                "activation": {"type": "categorical", "choices": [e.value for e in ActivationFunction]},
                "gumbel_tau": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                "n_predictor_layers": {"type": "int", "low": 0, "high": 2},
                "n_predictor_learning_rate": {"type": "float", "low": 1e-8, "high": 1e-2, "log": True},
                "layer_selection_method": {
                    "type": "categorical",
                    "choices": [
                        LayerSelectionMethod.GUMBEL_SOFTMAX,
                        LayerSelectionMethod.STE,
                        LayerSelectionMethod.SOFT_GATING,
                        LayerSelectionMethod.REINFORCE,
                        LayerSelectionMethod.NONE,
                    ],
                },
            }
        )

        if self.early_stopping_rounds is None:
            space["n_epochs"] = {"type": "int", "low": 5, "high": 50, "step": 10}

        if "hidden_layers" in space:
            del space["hidden_layers"]

        if self.search_space_override:
            space.update(self.search_space_override)

        return space

    def _setup_optimizers(self, model: torch.nn.Module) -> None:
        """Sets up the optimizers for FlexibleHiddenLayersNN.

        REINFORCE uses a separate policy optimizer for the n_predictor (score-function
        gradient estimator requires an explicit policy update). All other strategies
        (SoftGating, GumbelSoftmax, STE, NoneStrategy) train the n_predictor via
        backprop through the relaxed selection, so its parameters must sit in the
        main optimizer.
        """
        uses_policy_optimizer = self.layer_selection_method == LayerSelectionMethod.REINFORCE
        if uses_policy_optimizer:
            main_params = [p for n, p in model.named_parameters() if "n_predictor" not in n]
        else:
            main_params = list(model.parameters())
        self.optimizer = torch.optim.Adam(main_params, lr=self.learning_rate)

        if uses_policy_optimizer and model.n_predictor:
            policy_params = model.n_predictor.parameters()
            self.strategy.setup_optimizers(policy_params)

    def get_params(self) -> dict[str, Any]:
        """Gets the parameters of the model."""
        params = super().get_params()
        for key in self._defaults:
            params[key] = getattr(self, key)
        params["trustworthy"] = getattr(self, "convergence_summary_", {}).get("trustworthy", False)
        return params

    def get_internal_model(self) -> Any:
        """Returns a wrapper around the internal model that returns only the prediction tensor.

        The underlying ``FlexibleNNModule.forward`` returns a 5-tuple
        ``(final_output, _, n_probs, n_logits, log_prob)`` because training needs
        the auxiliary tensors for ELBO/REINFORCE/penalty losses. ``shap.DeepExplainer``
        only supports modules that return a single tensor, so we wrap.
        """

        class _ShapModelWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x)[0]

        return _ShapModelWrapper(self.model)

    def get_shap_explainer_info(self) -> dict[str, Any]:
        """Gets the SHAP explainer type and the model to be explained."""
        return {"explainer_type": ExplainerType.DEEP, "model": self.get_internal_model()}
