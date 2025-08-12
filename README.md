# **AutoML Package**

A versatile Automated Machine Learning (AutoML) framework built in Python, designed to streamline the process of model selection, hyperparameter optimization, training, prediction, and explainability for both regression and classification tasks. This package integrates various popular machine learning models from PyTorch, JAX, XGBoost, LightGBM, CatBoost, and Scikit-learn, alongside advanced features like dynamic neural network architectures and comprehensive uncertainty quantification.

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Key Features](#2-key-features)
3. [Supported Models](#3-supported-models)
   * [Base Models](#base-models)
   * [Tree-Based Models](#tree-based-models)
   * [Neural Networks](#neural-networks)
   * [Common Neural Network Features](#common-neural-network-features)
   * [Dynamic Architecture: The `LayerSelectionMethod`](#dynamic-architecture-the-layerselectionmethod)
   * [Probabilistic Regression Models](#probabilistic-regression-models)
   * [Composite Models](#composite-models)
4. [Uncertainty Quantification](#4-uncertainty-quantification)
5. [Lambda Estimation (Automatic Regularization Learning)](#5-lambda-estimation-automatic-regularization-learning)
6. [Data Preprocessing](#6-data-preprocessing)
   * [Feature Scaling](#feature-scaling)
   * [Target Scaling](#target-scaling)
   * [Categorical Feature Handling](#categorical-feature-handling)
   * [Missing Value Imputation](#missing-value-imputation)
7. [Explainability & Feature Selection](#7-explainability--feature-selection)
8. [Metrics](#8-metrics)
   * [Regression Metrics](#regression-metrics)
   * [Classification Metrics](#classification-metrics)
   * [Example Metric Plots](#example-metric-plots)
9. [Early Stopping](#9-early-stopping)
10. [Hyperparameter Optimization (Optuna)](#10-hyperparameter-optimization-optuna)
    * [How Optuna Works](#how-optuna-works)
    * [Setting Optuna Parameters in AutoML](#setting-optuna-parameters-in-automl)
11. [Persistency](#11-persistency)
12. [Experiment Tracking (Weights & Biases)](#12-experiment-tracking-weights--biases)
13. [Leaderboard](#13-leaderboard)
14. [Installation](#14-installation)
15. [Usage](#15-usage)
    * [Basic AutoML Workflow](#basic-automl-workflow)
    * [Prediction](#prediction)
    * [Uncertainty Prediction](#uncertainty-prediction)
    * [Feature Importance](#feature-importance)
    * [Retraining with Selected Features](#retraining-with-selected-features)
    * [Remaining Usage Examples (Classification, Save/Load)](#remaining-usage-examples-classification-saveload)
16. [Project Structure](#16-project-structure)
17. [Contributing](#17-contributing)
18. [License](#18-license)

## **1. Introduction**

This AutoML package aims to automate significant portions of the machine learning pipeline, enabling users to quickly find optimal models and understand their predictions without extensive manual tuning. It's built with modularity and extensibility in mind, allowing easy integration of new models and techniques.

## **2. Key Features**

*   **Automated Model Selection:** Evaluates a diverse set of machine learning algorithms.
*   **Hyperparameter Optimization:** Utilizes Optuna for efficient and robust hyperparameter tuning with cross-validation.
*   **Early Stopping:** Prevents overfitting and reduces training time by stopping training when validation performance degrades.
*   **Lambda Estimation:** Automatically learns optimal L1/L2 regularization strengths for PyTorch-based models.
*   **Regression & Classification:** Supports both common machine learning task types.
*   **Pluggable Architecture:** Easy to add new custom models adhering to the `BaseModel` interface.
*   **Dynamic Neural Networks:** Includes a novel `FlexibleNeuralNetwork` that adapts its depth based on input using a variety of advanced techniques.
*   **Uncertainty Quantification:** Provides methods to estimate prediction uncertainty for regression tasks.
*   **Data Scaling Integration:** Seamlessly integrates feature and target scaling into the pipeline.
*   **Categorical Feature Transformers:** Provides `OrderedTargetEncoder` and `OneHotEncoder` for preprocessing categorical data.
*   **Model Explainability:** Leverages SHAP (SHapley Additive exPlanations) for interpreting model predictions and for feature selection.
*   **Feature Selection:** Automated feature selection based on SHAP importance.
*   **Persistency:** Save and load the entire AutoML state, including fitted models and scalers.
*   **Experiment Tracking (Weights & Biases):** Seamless integration with Weights & Biases for logging hyperparameters, metrics, and visualizing training runs.

## **3. Supported Models**

The package includes wrappers for various popular machine learning models, all conforming to a common `BaseModel` interface.

### **Base Models**

*   **JAXLinearRegression**: A custom Linear Regression implementation using JAX for gradient-based optimization, including L1/L2 regularization.
    *   **Mathematical Formulation (Loss Function)**:
        The objective function minimized during training is the Mean Squared Error (MSE) augmented with L1 and L2 regularization terms:
        $$
L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - (\mathbf{x}_i^T w + b))^2 + \lambda_1 \sum_{j=1}^{D} |w_j| + \lambda_2 \sum_{j=1}^{D} w_j^2
$$
        Where:
        *   $N$ is the number of samples.
        *   $D$ is the number of features.
        *   $y_i$ is the true target value for sample $i$.
        *   $\mathbf{x}_i$ is the feature vector for sample $i$.
        *   $w$ is the vector of weights.
        *   $b$ is the bias term.
        *   $\lambda_1$ is the L1 regularization strength.
        *   $\lambda_2$ is the L2 regularization strength.


*   **NormalEquationLinearRegression**: A direct implementation of Linear Regression using the normal equation, suitable for smaller datasets. This model inherently includes L2 regularization (Ridge Regression).
    *   **Mathematical Formulation (Normal Equation with Ridge)**:
        The weights $\hat{\beta}$ (including the bias term) are found by solving the following equation:
        $$\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y
$$
        Where:
        *   $X$ is the design matrix (features augmented with a column of ones for the bias).
        *   $y$ is the target vector.
        *   $\lambda$ is the L2 regularization strength (`l2_lambda`).
        *   $I$ is the identity matrix (with the last diagonal element, corresponding to the bias, set to 0 to avoid regularizing the bias).

    *   **Comparison with JAXLinearRegression**:
        *   **JAXLinearRegression (Iterative/Gradient-based)**:
            *   **Pros**:
                *   **Scalability**: More suitable for large datasets (many samples, many features) as it avoids direct matrix inversion, which can be computationally expensive ($O(D^3)$).
                *   **Flexibility**: Can easily incorporate various regularization types (L1, L2, Elastic Net) and non-linear transformations by modifying the loss function and optimization process.
                *   **Online Learning**: Can be adapted for online or mini-batch learning, where data arrives sequentially.
                *   **GPU/TPU Acceleration**: Leverages JAX's automatic differentiation and JIT compilation for efficient computation on accelerators.
            *   **Cons**:
                *   **Convergence**: Requires careful tuning of learning rate and number of iterations to ensure convergence to a good solution.
                *   **Local Minima**: For non-convex problems (though not an issue for linear regression's convex loss), iterative methods can get stuck in local minima.
                *   **Slower for Small Data**: Can be slower than the Normal Equation for very small datasets due to the iterative nature.

        *   **NormalEquationLinearRegression (Direct/Analytical)**:
            *   **Pros**:
                *   **Exact Solution**: Provides a direct, analytical solution in a single step, guaranteeing the global optimum for the given data (for convex problems).
                *   **No Hyperparameter Tuning (Learning Rate/Iterations)**: Does not require tuning of learning rate or number of iterations.
                *   **Faster for Small Data**: Can be faster than iterative methods for datasets with a small number of features.
            *   **Cons**:
                *   **Scalability Issues**: Involves computing the inverse of $X^T X$, which has a computational complexity of $O(D^3)$ (where $D$ is the number of features). This becomes computationally prohibitive and numerically unstable for a large number of features.
                *   **Memory Intensive**: Requires storing the entire $X^T X$ matrix, which can be memory-intensive for high-dimensional data.
                *   **Singularity**: $X^T X$ might be singular (non-invertible) if features are highly correlated or if $N < D$, requiring regularization (like Ridge) to make it invertible.

    *   **When to Prefer Which**:
        *   **Prefer JAXLinearRegression when**:
            *   You have a **large dataset** (many samples or many features).
            *   You need to leverage **GPU/TPU acceleration**.
            *   You want more **flexibility** in regularization or plan to extend to more complex models.
            *   You are comfortable with tuning learning rates and iteration counts.
        *   **Prefer NormalEquationLinearRegression when**:
            *   You have a **small to medium dataset** where the number of features $D$ is relatively small (e.g., $D < 10,000$).
            *   You need an **exact, analytical solution** and don't want to tune iterative optimization parameters.
            *   Computational resources (especially memory) are not a bottleneck for matrix operations.


*   **PyTorchLinearRegression**: A linear regression model implemented using PyTorch, allowing for GPU acceleration and integration with PyTorch's ecosystem.


*   **PyTorchLogisticRegression**: A logistic regression model implemented using PyTorch, suitable for binary classification.


*   **SKLearnLogisticRegression**: Scikit-learn's Logistic Regression for classification, supporting various penalties.

### **Tree-Based Models**

*   **XGBoostModel**: Wrapper for XGBoost (Extreme Gradient Boosting), configurable for regression and classification. 

* **LightGBMModel**: Wrapper for LightGBM (Light Gradient Boosting Machine), highly efficient for large datasets.
  
* **CatBoostModel**: Wrapper for CatBoost, known for its robust handling of categorical features.

### **Neural Networks**

This section details the Neural Network models available in the package, designed for various applications and complexities.

#### **PyTorchNeuralNetwork**

*   **Description**: A standard feedforward neural network (Multi-Layer Perceptron) built using PyTorch. It's highly configurable, allowing you to specify the number of hidden layers, their size, the activation function, and apply regularization (L1/L2) and batch normalization. It can be used for both regression (predicting continuous values) and classification (predicting discrete categories). **Supports automatic learning of L1/L2 regularization strengths (lambdas).**
*   **Core Idea**: Learns complex, non-linear relationships between inputs and outputs through multiple layers of interconnected neurons.
*   **Diagram (Basic Architecture)**:
    ```
    +--------------------------+
    | Input Layer (Features X) |
    +----------+---------------+
               |
               v
    +-------------------------+
    |  Hidden Layer 1         |
    |  (Linear + BatchNorm +  |
    |  ReLU)                  |
    +----------+--------------+
               |
               v
    +-------------------------+
    |  Hidden Layer 2         |
    |  (Linear + BatchNorm +  |
    |  ReLU)                  |
    +----------+--------------+
               |
               v
            ... (more hidden layers)
               |
               v
    +-------------------------+
    |  Output Layer (Linear)  |
    +----------+--------------+
               |
               v
    +-------------------------+
    | Predicted Value (Y_pred)|
    |   (or logits)           |
    +-------------------------+
    ```
*   **Activation Functions**: Supports common activation functions like ReLU, $\text{ReLU}(x) = \max(0, x)$, and Tanh, $\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.
*   **Regularization**: When `learn_regularization_lambdas` is `False`, fixed L1 and L2 regularization are applied directly to the loss function. See [Lambda Estimation](#5-lambda-estimation-automatic-regularization-learning) for details on the loss formulation.

#### **FlexibleNeuralNetwork**

*   **Description**: A novel and adaptive PyTorch-based neural network that dynamically adjusts its effective hidden layer depth for *each individual input*. Instead of a fixed number of layers performing computation, it has a `max_hidden_layers` and an internal `n_predictor` that determines how many of these layers should be "active" (perform computations) for a specific input. The remaining layers act as "identity layers," simply passing their input through unchanged. **Supports automatic learning of L1/L2 regularization strengths (lambdas).**
*   **Core Idea**: To achieve adaptive complexity. For simpler inputs, it can effectively use fewer layers, potentially leading to faster inference and better generalization by avoiding overfitting to noise. For more complex inputs, it can utilize its full depth.
*   **Mechanism**:
    1.  **n_predictor**: A small sub-network takes the input and predicts an integer $n$ (from 1 to `max_hidden_layers`). During training, it uses various strategies (e.g., Gumbel-Softmax) for differentiable selection of $n$.
    2.  **Dynamic Forward Pass**: In the main network's forward pass, each hidden layer block checks the predicted $n$. If the current layer's index is among the first (`max_hidden_layers - n`) layers, it bypasses computation (identity mapping). Otherwise, it performs its standard linear transformation, batch norm, and activation.
*   **Diagram (Conceptual Dynamic Depth)**:
    ```
                        ┌──────────────────┐
    Input Features X ───┤ 1. N-Predictor   ├─────┐
                        └──────────────────┘     │
                            Layer Selection      │
                            (e.g., n=3 or probs) │
                               │                 │
    +----------------------------------------------------------------+
    |                                                                |
    |   Main Network (Max Hidden Layers)                             |
    |   (Dynamically activated based on N-Predictor's output)        |
    |                                                                |
    |   Hidden Layer Block 1                                         |
    |   Hidden Layer Block 2                                         |
    |   ...                                                          |
    |   Hidden Layer Block max_hidden_layers                         |
    +--------------------------+-------------------------------------+
                               |
                               v
                      +--------------------+
                      |   Output Layer     |
                      +----------+---------+
                                 |
                                 v
                      +--------------------+
                      | Final Prediction   |
                      +--------------------+
    ```

### **Common Neural Network Features**

Neural networks, including `PyTorchNeuralNetwork` and `FlexibleNeuralNetwork`, often incorporate several common techniques to improve training stability, generalization, and performance.

*   **Batch Normalization (BatchNorm)**:
    *   **Description**: A technique to normalize the inputs of each layer, typically after the linear transformation and before the activation function. It normalizes the activations of the previous layer for each mini-batch, meaning that the mean activation is close to zero and the standard deviation is close to one.
    *   **Purpose**:
        *   **Stabilizes Training**: Reduces internal covariate shift, allowing for higher learning rates and faster convergence.
        *   **Regularization**: Adds a slight regularization effect, sometimes reducing the need for dropout.
        *   **Improved Gradient Flow**: Prevents vanishing or exploding gradients.
    *   **Mathematical Formulation**: For an input $x$ to a layer, Batch Normalization transforms it to $\hat{x}$ as follows:
        $$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
        Then, it scales and shifts the normalized value:
        $$y = \gamma \hat{x} + \beta
$$
        Where $\mu_B$ and $\sigma_B^2$ are the mean and variance of the mini-batch, $\epsilon$ is a small constant for numerical stability, and $\gamma$ and $\beta$ are learnable scale and shift parameters.

*   **Dropout**:
    *   **Description**: A regularization technique where randomly selected neurons are "dropped out" (i.e., their outputs are set to zero) during training. This prevents complex co-adaptations on training data.
    *   **Purpose**:
        *   **Reduces Overfitting**: Forces the network to learn more robust features that are not dependent on the presence of specific neurons.
        *   **Ensemble Effect**: Can be seen as training an ensemble of many thinned networks.
    *   **Mechanism**: During training, for each training example and each layer, each neuron has a probability $p$ of being dropped out. During inference, all neurons are active, but their weights are scaled by $p$ (or the outputs are scaled by $1/p$) to maintain the same expected output.

*   **Regularization (L1 and L2)**:
    *   **Description**: Techniques used to prevent overfitting by adding a penalty term to the loss function. This penalty discourages the model from assigning excessively large weights to features.
    *   **Purpose**:
        *   **Controls Model Complexity**: Penalizes large weights, leading to simpler models.
        *   **Improves Generalization**: Helps the model perform better on unseen data.
    *   **Types**:
        *   **L1 Regularization (Lasso)**: Adds a penalty proportional to the absolute value of the weights. It can lead to sparse models where some weights become exactly zero, effectively performing feature selection.
            $$L_{L1} = \lambda_1 \sum_{i} |w_i|
$$
        *   **L2 Regularization (Ridge or Weight Decay)**: Adds a penalty proportional to the square of the weights. It encourages smaller weights but does not force them to zero.
            $$L_{L2} = \lambda_2 \sum_{i} w_i^2
$$
        *   **Elastic Net Regularization**: A combination of L1 and L2 regularization.
    *   **Implementation in this package**: For PyTorch-based models, regularization can be applied with fixed lambdas or learned automatically (see [Lambda Estimation](#5-lambda-estimation-automatic-regularization-learning)).

### **Dynamic Architecture: The `LayerSelectionMethod`**

The `FlexibleNeuralNetwork`'s ability to dynamically select its architecture is governed by the `layer_selection_method` parameter. This parameter specifies the algorithm used by the internal `n_predictor` network to choose the number of active layers.

*   **`LayerSelectionMethod.NONE`**
    *   **Description:** This method disables dynamic architecture selection. The `n_predictor` is not used, and the `FlexibleNeuralNetwork` behaves like a standard `PyTorchNeuralNetwork`, always using all `max_hidden_layers`. All layers are active for every input.
    *   **Use Case:** Baseline comparison or when a fixed architecture is desired.
    *   **Constraint:** Requires `n_predictor_layers` to be set to `0`.


*   **`LayerSelectionMethod.SOFT_GATING`**
    *   **Description:** Uses a `softmax` function on the `n_predictor`'s logits to produce a probability distribution over the number of active layers. The final output is a weighted average of the outputs from all possible network depths, where weights are these probabilities. All layers contribute to the final prediction, but their contributions are scaled by the predicted probabilities.    
    *   **Use Case:** When a smooth, differentiable, and continuous combination of different network depths is desired. It allows the model to softly blend architectures, which can be beneficial for creating a stable and well-behaved loss landscape during training, especially when the optimal depth is not clear-cut.
    *   **Constraint:** Requires `n_predictor_layers > 0` as it relies on the `n_predictor` to generate probabilities.
    *   **Core Idea:** Creates a smooth, fully differentiable landscape, allowing the model to learn to favor certain depths by assigning them higher probabilities.    
    *   **Mathematical Formulation:** Given logits $z$ from the `n_predictor` for $K$ possible depths, the probability $p_i$ for depth $i$ is:        $$
p_i = \text{softmax}\left(z_i\right) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$        The final output is a weighted sum: $Y_{\text{pred}} = \sum_{i=1}^{K} p_i \cdot Y_i$, where $Y_i$ is the output of the network with depth $i$.


*   **`LayerSelectionMethod.GUMBEL_SOFTMAX`**    
    *   **Description:** An advanced version of `SOFT_GATING` that uses the Gumbel-Softmax trick. It introduces stochasticity during training to explore different architectures more effectively. Like `SOFT_GATING`, it produces a weighted average of all architecture outputs, meaning all layers contribute to the final prediction, but their influence is weighted by the Gumbel-Softmax probabilities.    
    *   **Use Case:** Ideal for scenarios where the model needs to explore a wider range of architectural configurations during training to find the most suitable depth. It's particularly useful when the optimal architecture is unknown or when dealing with complex datasets where a fixed architecture might limit performance.    
    *   **Constraint:** Requires `n_predictor_layers > 0` as it relies on the `n_predictor` to generate probabilities.
    *   **Core Idea:** Improves exploration in the architecture search space by adding noise in a structured, differentiable way.    
    *   **Mathematical Formulation:** Given logits $z$ and Gumbel noise $g \sim \text{Gumbel}(0, 1)$: $$p_i = \text{softmax}\left(\frac{z_i + g_i}{\tau}\right)$$        Here, $\tau$ is a temperature parameter that is annealed (gradually lowered) during training. A high $\tau$ encourages exploration (probabilities are more uniform), while a low $\tau$ encourages exploitation (probabilities become closer to a one-hot selection).


*   **`LayerSelectionMethod.STE` (Straight-Through Estimator)**
    *   **Description:** This method makes a "hard" decision in the forward pass but uses a "soft" gradient in the backward pass. It uses the Gumbel-Softmax trick with `hard=True`, which outputs a one-hot vector (e.g., `[0, 1, 0]`). This vector selects a *single* architecture (a specific number of active layers) to be used for the forward pass for each input. Only the selected layers perform computation.
    *   **Use Case:** When computational efficiency during inference is critical, as it allows for a single, fixed-depth forward pass per input. It's suitable for deployment scenarios where dynamic computation graphs are undesirable, but the benefits of learned architecture selection during training are still desired.
    *   **Constraint:** Requires `n_predictor_layers > 0` and the `n_predictor` to be trained to output a one-hot vector for hard selection.
    *   **Core Idea:** To get the efficiency of using a single architecture per input while still allowing the `n_predictor` to learn via a differentiable gradient.
    *   **Mechanism (The "Trick"):** In the backward pass, the gradient is calculated *as if* the soft, continuous Gumbel-Softmax probabilities had been used, effectively "going through" the non-differentiable `argmax` operation.


*   **`LayerSelectionMethod.REINFORCE`**
    *   **Description:** This method frames architecture selection as a reinforcement learning problem. The `n_predictor` acts as a "policy network" or "agent" that *samples* a discrete number of layers to use for each input. Only the sampled layers are active.
    *   **Use Case:** When the goal is to directly optimize a discrete architectural choice based on a specific performance metric (reward). This is particularly useful for tasks where the architectural decision has a clear, measurable impact on the final outcome, and a direct gradient-based approach is difficult due to the discrete nature of the choice.
    *   **Constraint:** Requires careful tuning of reinforcement learning hyperparameters (e.g., learning rate for the policy network, reward shaping) and can be more sensitive to hyperparameter choices than gradient-based methods.
    *   **Core Idea:** To use a classic RL algorithm to directly optimize the architecture selection policy based on a performance-based reward signal.
    *   **Mechanism**:
        1.  **State:** The input features $X$.
        2.  **Action:** The policy network observes $X$ and *samples* an action (the number of layers to use) from the probability distribution it produces.
        3.  **Reward:** At the end of each training epoch, the model's performance is measured on a validation set. The reward $R$ is the negative validation loss ($R = -\text{validation loss}$). A lower loss means a higher reward.
        4.  **Policy Update:** The policy network is updated using the REINFORCE algorithm, which adjusts its weights to make actions that led to high rewards more likely in the future. The policy gradient is estimated as:
            $$\nabla_\theta J(\theta) \approx R \cdot \nabla_\theta \log \pi(a|s; \theta)$$
            Where $J(\theta)$ is the policy objective, and $\pi(a|s; \theta)$ is the policy (the `n_predictor`).

### **Probabilistic Regression Models**

These models are designed for regression tasks where quantifying the uncertainty of a prediction is as important as the prediction itself. Instead of just predicting a single value, they predict an entire probability distribution.

*   **ProbabilisticRegressionModel (PyTorch-based)**:
    *   **Description**: This is a composite PyTorch model that tackles regression by internally employing a classification strategy. It first conceptually discretizes the continuous target variable into a fixed number of `n_classes` (bins). It then trains an internal classifier to predict which bin an input belongs to. The probabilities from this classifier are then fed into specialized regression "heads" (either separate heads per class or a single head with multiple outputs) that predict the actual continuous value. Crucially, it can be configured to learn the mean and variance of a Gaussian distribution for each prediction, directly modeling uncertainty. **Supports automatic learning of L1/L2 regularization strengths (lambdas).**
    *   **Core Idea**: Leverages the strengths of classification (assigning inputs to meaningful "categories" of the target variable) to improve continuous regression prediction and provide robust uncertainty estimates.
    *   **Mechanism (Regression Heads Strategies)**:
        1.  **Target Discretization (Conceptual)**: The continuous target $y$ is divided into `n_classes` bins (e.g., via percentiles).
        2.  **Classifier Branch**: An internal `PyTorchNeuralNetwork` (as the base classifier) is trained to predict the probability of an input falling into each of these `n_classes`.
        3.  **Regression Heads**: The predicted probabilities for each class are then fed into subsequent regression subnetworks. The strategy for these heads can vary:
            *   **`RegressionStrategy.SEPARATE_HEADS`**: Creates `n_classes` distinct regression subnetworks. Each subnetwork specializes in mapping the probability of its corresponding class to an expected regression value for that class. The final prediction is a weighted sum of outputs from all heads.
                ```
                 Predicted Class Probs (P0, P1, P2)
                       ┌────┬────┬────┐
                       │ P0 │ P1 │ P2 │
                       └─┬──┴─┬──┴─┬──┘
                         │    │    │
                         v    v    v
                 +--------------------------------------+
                 | Reg Head 0  | Reg Head 1 | Reg Head 2 |
                 | (Maps P0)   | (Maps P1)  | (Maps P2)  |
                 +-------------+------------+-----------+
                         │        │        │
                         v        v        v
                   Y_exp_0  Y_exp_1  Y_exp_2
                         │        │        │
                         └────────┼────────┘
                                  v
                 Final Prediction = (P0 * Y_exp_0) + (P1 * Y_exp_1) + (P2 * Y_exp_2)
                ```
            *   **`RegressionStrategy.SINGLE_HEAD_N_OUTPUTS`**: A single regression subnetwork outputs `n_classes` distinct regression values (or mean/log-variance pairs). The final prediction is then a weighted sum of these `n_classes` outputs, weighted by their respective class probabilities.
                ```
                 Predicted Class Probs (P0, P1, P2)
                       ┌──────────────────────┐
                       │ Combined Probs Input │
                       └───────────┬──────────┘
                                   │
                                   v
                 +------------------------------------------+
                 | Single Regression Head                   |
                 | (Outputs Y_exp_0, Y_exp_1, Y_exp_2)      |
                 +-----------------┬---------------┬--------+
                                   │               │
                                   v               v
                 Final Prediction = (P0 * Y_exp_0) + (P1 * Y_exp_1) + (P2 * Y_exp_2)
                ```
            *   **`RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT`**: A single regression subnetwork directly outputs the final combined regression prediction (or mean/log-variance pair).
                ```
                 Predicted Class Probs (P0, P1, P2)
                       ┌──────────────────────┐
                       │ Combined Probs Input │
                       └───────────┬──────────┘
                                   │
                                   v
                 +---------------------------------+
                 | Single Regression Head          |
                 | (Directly outputs final Y_pred) |
                 +---------------------------------+
                                   │
                                   v
                         Final Prediction (Y_pred)
                ```
                **Uncertainty Calculation for `SINGLE_HEAD_FINAL_OUTPUT`**:
                When `UncertaintyMethod.PROBABILISTIC` is used with this strategy, the single regression head directly outputs both the predicted mean $\mu(X)$ and the predicted log-variance $\log(\sigma^2(X))$ for the final prediction. The classifier's probabilities serve as input features to this head, influencing its direct prediction of the mean and log-variance.

                The uncertainty (standard deviation) is then simply the square root of the variance derived from this directly predicted log-variance:
                $$\text{Uncertainty}(X) = \sqrt{\exp(\log(\sigma^2(X)))} = \sigma(X)
$$
                This means the single head is responsible for capturing the overall aleatoric uncertainty based on the input features and the classifier's probabilistic context.
        4.  **Prediction Aggregation**: The outputs of these regression heads are combined (often via a weighted sum using the class probabilities) to form the final continuous prediction.
        5.  **Uncertainty**: The model can learn both the mean $\mu(X)$ and the log-variance $\log(\sigma^2(X))$ of a Gaussian distribution for its final prediction, allowing for **probabilistic uncertainty**.
    *   **Mathematical Formulation (Negative Log-Likelihood Loss)**:
        When `UncertaintyMethod.PROBABILISTIC` is used, the model is trained to minimize the Negative Log-Likelihood (NLL) of a Gaussian distribution. For a given input $X$, the model predicts a mean $\mu(X)$ and a log-variance $\log(\sigma^2(X))$. The NLL loss is:
        $$
L(\mu, \sigma^2) = \frac{1}{2} \log(2\pi) + \frac{1}{2} \log(\sigma^2(X)) + \frac{(y - \mu(X))^2}{2\sigma^2(X)}
$$
        The model minimizes the average NLL over the training batch.
    *   **Diagram (Probabilistic Regression Model Flow - Consolidated)**:
        ```
                           ┌─────────────────────────┐
                           │      Input Features X   |
                           └───────────┬─────────────┘
                                       |
                                       v
                       ┌─────────────────────────┐
                       │   Classifier Branch     |
                       |   (PyTorchNeuralNetwork)|
                       └───────────┬─────────────┘
                                   |
                                   v
                       ┌────────────────────────────┐
                       │ Predicted Class Probs      |
                       |  (P_0, P_1, ..., P_N-1)    |
                       └───────────┬────────────────┘
                                   |
            +----------------------------------------------------+
            |          Regression Heads Branch                   |
            |   (Controlled by regression_strategy)              |
            |                                                    |
            |   - SEPARATE_HEADS: N individual heads             |
            |   - SINGLE_HEAD_N_OUTPUTS: One head, N outputs     |
            |   - SINGLE_HEAD_FINAL_OUTPUT: One head, 1 output   |
            +---------------------┬------------------------------+
                                  |
                                  v
                       ┌───────────────────────────────────┐
                       │ Final Predicted Mean μ(X)         |
                       | Final Predicted LogVar log(σ²(X)) |
                       └───────────────────────────────────┘
        ```

      *   **Dynamic `n_classes` Estimation**:
          The `ProbabilisticRegressionModel` can dynamically estimate the optimal number of classes (`n_classes`) for discretizing the target variable, rather than requiring it to be a fixed hyperparameter. This allows the model to adapt its internal classification granularity to the complexity of the regression task and the data itself. It also seamlessly integrates a direct regression path, allowing the model to choose between a probabilistic, class-based approach and a simpler direct regression, based on the input.

          **Mechanism**:
          1.  **`estimate_n_classes` Parameter**: A new boolean parameter `estimate_n_classes` in `ProbabilisticRegressionModel` controls this feature. If `True`, the model will dynamically determine `n_classes`; otherwise, it uses the `n_classes` provided by the user.
          2.  **`n_classes_predictor` Sub-network**: When `estimate_n_classes` is `True`, an internal `n_classes_predictor` (a small neural network) is introduced. This sub-network takes the input features and outputs logits for a predefined range of possible `n_classes` values (e.g., from 2 up to `max_n_classes_for_probabilistic_path`), plus an additional logit for a 'direct regression' mode.
          3.  **`n_classes_selection_method`**: The `n_classes_selection_method` parameter governs how the `n_classes_predictor` chooses the number of classes or the direct regression path. The available methods are detailed in the following section.
          4.  **Dynamic Logit Masking and Softmax**: The main classifier branch always outputs raw logits for `max_n_classes_for_probabilistic_path` classes. Based on the `k` selected by the `n_classes_predictor` (if the probabilistic path is chosen), logits for unselected classes are masked (set to $-\infty$) before applying softmax. This ensures that only the `k` selected probabilities are non-zero and sum to 1.
              *   **Mathematical Formulation (Masked Softmax)**:
                  Given raw classifier logits $L = [l_0, l_1, \dots, l_{\text{max N}-1}]$ and a selected number of classes $k$, the masked logits $L'$ are:
                  $
L'_i = \begin{cases} l_i & \text{if } i < k \ -
\infty & \text{if } i \ge k \end{cases}$
                  The probabilities are then calculated as: $
 P_i = \text{softmax}(L')_i = \frac{e^{L'_i}}{\sum_{j=0}^{\text{max_N}-1} e^{L'_j}}$

                  This ensures $\sum_{i=0}^{k-1} P_i = 1$ and $P_i = 0$ for $i \ge k$.
          5.  **Integrated Direct Regression Path**: If the `n_classes_predictor` selects the 'direct regression' mode, the input bypasses the classifier and class-specific regression heads, and is instead passed through a dedicated direct regression head. This ensures gradient flow to the `n_classes_predictor` even when it chooses a non-probabilistic path.
          6.  **Pre-calculated Class Boundaries**: To maintain efficiency, the class boundaries for target discretization are pre-calculated for all possible `k` values (from 2 to `max_n_classes_for_probabilistic_path`) using the entire training dataset. During training, the appropriate pre-calculated boundaries are selected based on the `k` determined by the `n_classes_predictor`.
          7.  **`n_classes_inf` Constraint**: The `max_n_classes_for_probabilistic_path` must always be less than `n_classes_inf`. If the `n_classes_predictor` selects a value equal to or greater than `n_classes_inf`, it defaults to the direct regression path, ensuring consistent behavior with the original `n_classes_inf` logic.

          **Diagram (Dynamic `n_classes` Estimation Flow)**:
          ```
                                         ┌─────────────────────────┐
                                         │      Input Features X   │
                                         └───────────┬─────────────┘
                                                     |
                                                     v
                                         ┌─────────────────────────┐
                                         │   N-Classes Predictor   |
                                         │   (Outputs Logits for   |
                                         │   k=2...max_k, Direct)  |
                                         └───────────┬─────────────┘
                                                     | (Selection Method)
                                                     v
                                         ┌─────────────────────────┐
                                         │   Selected Mode (k or   |
                                         │   'Direct Regression')  |
                                         └───────────┬─────────────┘
                                                     |
                          ┌──────────────────────────┴──────────────────────────┐
                          |                                                     |
                          v                                                     v
            ┌─────────────────────────┐                           ┌─────────────────────────┐
            |   Probabilistic Path    |                           |   Direct Regression Path|
            | (if k < n_classes_inf)  |                           | (if 'Direct Regression')|
            └───────────┬─────────────┘                           └───────────┬─────────────┘
                        |
                        v
            ┌─────────────────────────┐                           ┌─────────────────────────┐
            |   Classifier Branch     |                           |   Direct Reg. Head      |
            | (Outputs max_k Logits)  |                           | (Outputs Mean, LogVar)  |
            └───────────┬─────────────┘                           └───────────┬─────────────┘
                        | (Dynamic Masking & Softmax)                         |
                        v                                                     v
            ┌────────────────────────────┐                           ┌─────────────────────────┐
            | Predicted Class Probs      |                           | Final Predicted Mean μ(X)|
            | (k non-zero, sum to 1)     |                           | Final Predicted LogVar log(σ²(X))|
            └───────────┬────────────────┘                           └─────────────────────────┘
                        |
                        v
            ┌────────────────────────────┐
            |   Regression Heads Branch  |
            | (Weighted by Probs)        |
            └───────────┬────────────────┘
                        |
                        v
            ┌────────────────────────────┐
            | Final Predicted Mean μ(X)  |
            | Final Predicted LogVar log(σ²(X))|
            └────────────────────────────┘
        ```
*   **Dynamic Granularity: The `NClassesSelectionMethod`**:
    The `ProbabilisticRegressionModel`'s ability to dynamically select its internal granularity (number of classes) is governed by the `n_classes_selection_method` parameter. This parameter specifies the algorithm used by the internal `n_classes_predictor` network to choose the number of classes for the probabilistic path or to opt for a direct regression path.

    *   **`NClassesSelectionMethod.NONE`**
        *   **Description:** This method disables dynamic `n_classes` selection. The `n_classes_predictor` is not used, and the model relies on the fixed `n_classes` hyperparameter.
        *   **Use Case:** When a fixed number of classes is desired or for baseline comparisons.
        *   **Constraint:** Requires `n_classes_predictor_layers` to be set to `0`.

    *   **`NClassesSelectionMethod.SOFT_GATING`**
        *   **Description:** Uses a `softmax` function on the `n_classes_predictor`'s logits to produce a probability distribution over the possible number of classes and the direct regression path. The final output is a weighted average of the outputs from all possible configurations (different `n_classes` and direct regression), where weights are these probabilities.
        *   **Use Case:** When a smooth, differentiable, and continuous combination of different model structures is desired.
        *   **Constraint:** Requires `n_classes_predictor_layers > 0`.
        *   **Mathematical Formulation:** Given logits $z$ from the `n_classes_predictor` for $K$ possible configurations, the probability $p_i$ for configuration $i$ is:
            $
 p_i = \text{softmax}\left(z_i\right) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$
            The final output is a weighted sum: $Y_{\text{pred}} = \sum_{i=1}^{K} p_i \cdot Y_i$, where $Y_i$ is the output of the model with configuration $i$.

    *   **`NClassesSelectionMethod.GUMBEL_SOFTMAX`**
        *   **Description:** An advanced version of `SOFT_GATING` that uses the Gumbel-Softmax trick. It introduces stochasticity during training to explore different configurations more effectively. Like `SOFT_GATING`, it produces a weighted average of all outputs.
        *   **Use Case:** Ideal for exploring a wider range of configurations during training to find the most suitable `n_classes` or to decide if direct regression is better.
        *   **Constraint:** Requires `n_classes_predictor_layers > 0`.
        *   **Mathematical Formulation**: Given logits $z$ and Gumbel noise $g \sim \text{Gumbel}(0, 1)$:
            $
 p_i = \text{softmax}\left(\frac{z_i + g_i}{\tau}\right)$
            Here, $\tau$ is a temperature parameter that is annealed (gradually lowered) during training. A high $\tau$ encourages exploration (probabilities are more uniform), while a low $\tau$ encourages exploitation (probabilities become closer to a one-hot selection).

    *   **`NClassesSelectionMethod.STE` (Straight-Through Estimator)**
        *   **Description:** This method makes a "hard" decision in the forward pass but uses a "soft" gradient in the backward pass. It uses the Gumbel-Softmax trick with `hard=True`, which outputs a one-hot vector. This vector selects a *single* configuration (a specific `n_classes` or direct regression) to be used for the forward pass for each input.
        *   **Use Case:** When computational efficiency is critical, as it allows for a single forward pass per input.
        *   **Constraint:** Requires `n_classes_predictor_layers > 0`.
        *   **Mechanism (The "Trick"):** In the backward pass, the gradient is calculated *as if* the soft, continuous Gumbel-Softmax probabilities had been used, effectively "going through" the non-differentiable `argmax` operation.
        *   **Mathematical Formulation**: Given logits $z$ and Gumbel noise $g \sim \text{Gumbel}(0, 1)$, the one-hot selection $s$ is obtained as:
            $
 s = \text{one_hot}(\text{argmax}(\frac{z + g}{\tau}))$
            During the backward pass, gradients are approximated as if the soft (non-hard) Gumbel-Softmax probabilities were used.

    *   **`NClassesSelectionMethod.REINFORCE`**
        *   **Description:** This method frames the selection as a reinforcement learning problem. The `n_classes_predictor` acts as a "policy network" that *samples* a discrete action (a specific `n_classes` or direct regression) for each input.
        *   **Use Case:** When the goal is to directly optimize a discrete structural choice based on a specific performance metric (reward).
        *   **Constraint:** Can be more sensitive to hyperparameter choices than gradient-based methods.
        *   **Mechanism**:
            1.  **State:** The input features $X$.
            2.  **Action:** The policy network samples an action (the `n_classes` to use or direct regression).
            3.  **Reward:** The reward $R$ is the negative validation loss ($R = -\text{validation loss}$). 
            4.  **Policy Update:** The policy network is updated using the REINFORCE algorithm to make actions that led to high rewards more likely. The policy gradient is estimated as:
                $
\nabla_\theta J(\theta) \approx R \cdot \nabla_\theta \log \pi(a|s; \theta)$
                Where $J(\theta)$ is the policy objective, and $\pi(a|s; \theta)$ is the policy (the `n_classes_predictor`).


### **Composite Models**

This section covers a meta-model that combines different modeling paradigms to solve complex problems.

*   **ClassifierRegressionModel**:
    *   **Description**: This is a **meta-model** (a model that uses other models internally) designed to solve **regression problems** by framing them as classification tasks. It works by discretizing the continuous target variable into a fixed number of bins (classes), training a *classifier* (which can be any `BaseModel` classifier like `XGBoostModel`, `PyTorchNeuralNetwork`, etc.) on these discrete classes, and then mapping the classifier's predicted probabilities back to a continuous regression output.
    *   **Core Idea**: It's particularly useful when the relationship between features and the regression target is complex or non-monotonic, or when traditional regression models struggle. By breaking the problem into classification, it can simplify the learning task.
    *   **Mechanism**:
        1.  **Target Discretization**: The continuous target variable ($y$) from the training data is divided into `n_classes` (e.g., 5-10 bins based on percentiles). Each original $y$ value is assigned a discrete class label.
        2.  **Base Classifier Training**: A chosen `BaseModel` classifier (e.g., `XGBoostModel`, `PyTorchNeuralNetwork`) is trained to predict these discrete class labels from the input features ($X$).
        3.  **Class Probability Mapper**: For new predictions, the trained base classifier outputs probabilities for each class. The `ClassProbabilityMapper` then takes these probabilities for a specific class and maps them to an *expected original regression value* for that class. This mapping is learned during fitting, often through methods like:
            *   **`MapperType.LINEAR`**: A simple linear regression fit between the predicted probability for a class and the actual target values associated with that class in the training data.
                $$
Y_{\text{exp},c} = m_c \cdot P_c + b_c
$$
                Where $P_c$ is the probability of class $c$, and $m_c, b_c$ are the learned slope and intercept for class $c$.
            *   **`MapperType.LOOKUP_MEAN / LOOKUP_MEDIAN`**: Divides the probability space into bins and calculates the mean (or median) of the original target values that fall into each bin. When a new probability comes, it looks up the corresponding mean/median.
                ```
                 +-------------------------+
                 | Predicted Probability   |
                 | for Class C (P_C)       |
                 +----------+--------------+
                            |
                            v
                 +-----------------------------+
                 |   Binning / Lookup Table    |
                 |   (e.g., 0-0.1 -> Mean_Y1)  |
                 |   (0.1-0.2 -> Mean_Y2)    |
                 +-----------------------------+
                            |
                            v
                 +-------------------------+
                 | Expected Y for Class C  |
                 | (Y_exp_C)               |
                 +-------------------------+
                ```
            *   **`MapperType.SPLINE`**: Fits a smooth spline interpolation function between the predicted probabilities and the actual target values for the class. This can capture non-linear relationships in the mapping more flexibly.
                $$
Y_{\text{exp},c} = S_c(P_c)
$$
                Where $S_c$ is the fitted spline function for class $c$.
        4.  **Final Prediction**: The final regression prediction for an input is the expected value derived from the probabilities and the learned continuous mapping (e.g., a weighted sum of the expected values from each class, weighted by their probabilities):
            $$
Y_{\text{pred}} = \sum_{c=0}^{N-1} P_c \cdot Y_{\text{exp},c}(P_c)
$$
            Where $P_c$ is the probability of input $X$ belonging to class $c$, and $Y_{\text{exp},c}(P_c)$ is the expected regression value mapped from $P_c$ for class $c$.

## **4. Uncertainty Quantification**

For regression models, the package supports quantifying prediction uncertainty using various methods:

*   **`UncertaintyMethod.CONSTANT`**: The simplest method. It calculates the standard deviation of residuals from the training data, applying this constant uncertainty to all new predictions.
    $$\sigma_{\text{constant}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$
    Where $N$ is the number of training samples, $y_i$ are true training targets, and $\hat{y}_i$ are predictions on training data.

*   **`UncertaintyMethod.PROBABILISTIC`**: The model is designed to directly learn both the mean ($\mu(X)$) and the variance ($\sigma^2(X)$) of the target variable's distribution. This captures **aleatoric uncertainty** (inherent noise in the data). The model is trained with a Negative Log-Likelihood (NLL) loss (see [Probabilistic Regression Models](#probabilistic-regression-models)). The uncertainty returned is the predicted standard deviation $\sigma(X)$.

*   **`UncertaintyMethod.MC_DROPOUT`**: This technique approximates **epistemic uncertainty** (model's uncertainty due to limited data/knowledge). For models with dropout layers, multiple predictions are made for the same input while dropout is active (model in `train()` mode). The standard deviation of these multiple predictions estimates the uncertainty.
    Let $\hat{y}_t(X)$ be the prediction for input $X$ at MC dropout pass $t$, for $T$ passes.
    $$\text{Uncertainty}(X) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (\hat{y}_t(X) - \bar{y}(X))^2}
$$
    Where $\bar{y}(X) = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t(X)$ is the mean prediction.

*   **Probabilistic Regression Model Uncertainty**
    *   **Description**: For the `ProbabilisticRegressionModel`, uncertainty is intrinsically linked to its architecture, which combines a classifier and regression heads. The model predicts both the mean and variance of the target distribution. The final prediction's mean and variance are derived by weighting the outputs of the regression heads (which themselves predict mean and variance) by the probabilities generated by the internal classifier. This approach captures both the uncertainty in classifying the input into a "bin" and the inherent aleatoric uncertainty within each bin.
    *   **Mechanism**:
        1.  **Classifier Probabilities**: The internal classifier outputs probabilities $P_c(X)$ for each class $c$ given input $X$.
        2.  **Regression Head Outputs**: Each regression head (or the single head for `SINGLE_HEAD_N_OUTPUTS` and `SINGLE_HEAD_FINAL_OUTPUT` strategies) outputs a mean $\mu_c(X)$ and a log-variance $\log(\sigma^2_c(X))$ for its corresponding class or output.
        3.  **Weighted Aggregation**: The final predicted mean $\mu(X)$ and variance $\sigma^2(X)$ are calculated. For `SEPARATE_HEADS` and `SINGLE_HEAD_N_OUTPUTS` strategies, this involves a weighted sum of the individual means and variances from the regression heads, where the weights are the classifier probabilities. For `SINGLE_HEAD_FINAL_OUTPUT`, the single regression head directly outputs the final mean and log-variance, implicitly incorporating the classifier probabilities in its prediction.
    *   **Mathematical Formulation**:
        For a given input $X$, let $P_c(X)$ be the probability of class $c$ from the classifier, and let $\mu_c(X)$ and $\sigma^2_c(X)$ be the mean and variance predicted by the regression head associated with class $c$.

        The final predicted mean $\mu(X)$ is:
        $$\mu(X) = \sum_{c=0}^{N-1} P_c(X) \cdot \mu_c(X)
$$

        The final predicted variance $\sigma^2(X)$ is derived from the log-variance outputs:
        $$\sigma^2(X) = \sum_{c=0}^{N-1} P_c(X) \cdot \exp(\log(\sigma^2_c(X)))
$$

        The uncertainty returned is the standard deviation:
        $$\text{Uncertainty}(X) = \sqrt{\sigma^2(X)}
$$

        This formulation ensures that the overall uncertainty reflects both the confidence of the classifier in assigning an input to a particular class and the inherent spread of values within that class as modeled by the regression heads.

*   **ClassifierRegressionModel Uncertainty Quantification**:
    *   **Description**: For `ClassifierRegressionModel`, uncertainty is estimated by considering the variance of the original target values within each probability bin, as learned by the `ClassProbabilityMapper`. This model treats the regression problem as a series of classification probabilities, and the uncertainty arises from the spread of true values within the ranges associated with those probabilities.
    *   **Mechanism**: For a new prediction, the model aggregates the variance contributions from each class's mapper, weighted by the square of their predicted probabilities. This approach effectively translates the uncertainty in classifying into a bin, and the inherent variability within each bin, into a total prediction uncertainty. The `predict_uncertainty` method returns the standard deviation derived from this aggregated variance.
    *   **Mathematical Formula**:
        The total variance for a given input $X$ is calculated as the sum of squared probabilities multiplied by the variance contribution from each class mapper:
        $$\text{TotalVariance}(X) = \sum_{c=0}^{N-1} P_c(X)^2 \cdot \text{VarianceContribution}_c(P_c(X))
$$
        Where:
        *   $P_c(X)$ is the probability of input $X$ belonging to class $c$, predicted by the base classifier.
        *   $\text{VarianceContribution}_c(P_c(X))$ is the variance predicted by the `ClassProbabilityMapper` for class $c$, given its probability $P_c(X)$. This variance represents the inherent spread of original $y$ values that mapped to that probability range for class $c$.

        The final uncertainty (standard deviation) is the square root of this total variance:
        $$\text{Uncertainty}(X) = \sqrt{\text{TotalVariance}(X)}
$$

## **5. Lambda Estimation (Automatic Regularization Learning)**

For PyTorch-based neural networks (`PyTorchNeuralNetwork`, `FlexibleNeuralNetwork`, and `ProbabilisticRegressionModel`), the package supports the automatic learning of regularization strengths (L1 and L2 lambdas).

Traditionally, regularization lambdas are fixed hyperparameters that need to be manually tuned. This feature allows the model to learn these values during training, potentially leading to better generalization and reducing the burden of hyperparameter optimization.

**Mechanism**:

1.  **Learnable Parameters**: The L1 and L2 regularization lambdas are treated as learnable parameters within the model. They are initialized to small values (e.g., `1e-4`).
2.  **Separate Optimizer**: A dedicated optimizer (e.g., Adam) is used to update these lambda parameters. Crucially, a *separate and typically much smaller learning rate* is used for the lambdas compared to the main model weights. This prevents the lambdas from exploding to very large values and destabilizing the training.
3.  **Loss Integration**: The regularization terms (L1 and L2 penalties) are integrated directly into the model's loss function. The gradients of this combined loss are then used to update both the model weights and the lambda parameters. Let $L_0$ be the base loss (e.g., MSE or NLL), $D$ be the number of trainable parameters (excluding bias if specified), $\lambda_1$ and $\lambda_2$ be the learned L1 and L2 regularization strengths respectively, $\sum |w_i|$ be the sum of absolute values of weights, and $\sum w_i^2$ be the sum of squared weights.

    *   **L1 Regularization Only (`LearnedRegularizationType.L1_ONLY`)**:
        $$
L = L_0 - D \log\left(\frac{\lambda_1}{2}\right) + \lambda_1 \sum_{i} |w_i|
$$

    *   **L2 Regularization Only (`LearnedRegularizationType.L2_ONLY`)**:
        $$
L = L_0 - \frac{D}{2} \log\left(\frac{\lambda_2}{\pi}\right) + \lambda_2 \sum_{i} w_i^2
$$

    *   **Elastic Net Regularization (`LearnedRegularizationType.L1_L2`)**:
        $$
L = L_0 + D \left[ \frac{1}{2} \log\left(\frac{\pi}{\lambda_2}\right) + \frac{\lambda_1^2}{4\lambda_2} + \log\left(\text{erfc}\left(\frac{\lambda_1}{2\sqrt{\lambda_2}}\right)\right) \right] + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2
$$
        The $\log(\text{erfc}(x))$ term is implemented using `log_erfc` for numerical stability.

4.  **Early Stopping Alignment**: The early stopping criterion is modified to consider the prediction loss on the validation set. By default, regularization loss is *not* included in this validation loss calculation for early stopping. It can be optionally included by setting `include_reg_loss_in_val_loss` to `True`. This ensures that the training stops when the model's generalization performance (on the validation set) is no longer improving, allowing the lambdas to converge to optimal values.

This approach enables the models to dynamically adjust their regularization strength based on the data, potentially leading to more robust and better-performing models.

## **6. Data Preprocessing**

The AutoML package handles essential data preprocessing steps to ensure models receive data in an optimal format.

### **Feature Scaling**
Automatically applies scaling to input features (X) using a user-defined `feature_scaler` (e.g., `StandardScaler`, `MinMaxScaler`).
*   **Why it's Important**: Scaling ensures that features with larger numerical ranges do not disproportionately influence the model's learning process. It helps optimization algorithms (like gradient descent) converge faster and more stably, leading to improved model performance and preventing issues where features with larger values dominate the objective function. It also ensures that regularization techniques (L1/L2) are applied fairly across all features.
*   **Models that Benefit Most**: 
    *   **Distance-based models**: K-Nearest Neighbors (KNN), Support Vector Machines (SVMs) with RBF kernels, K-Means clustering.
    *   **Gradient-descent based models**: Neural Networks (`PyTorchNeuralNetwork`, `FlexibleNeuralNetwork`, `ProbabilisticRegressionModel`), Linear Regression (`JAXLinearRegression`, `PyTorchLinearRegression`, `PyTorchLogisticRegression`, `SKLearnLogisticRegression`).
    *   Models where regularization strength is sensitive to feature scales.

### **Target Scaling**
For regression tasks, the target variable (y) can also be scaled using a `target_scaler`.
*   **Why it's Important**: Scaling the target variable can stabilize training, especially for neural networks, by bringing the target values into a more manageable range (e.g., 0-1 or -1 to 1). This can prevent issues like exploding or vanishing gradients and improve the overall convergence of the model. Predictions are automatically inverse-transformed back to the original scale for interpretability.
*   **Models that Benefit Most**: 
    *   **Neural Networks**: (`PyTorchNeuralNetwork`, `FlexibleNeuralNetwork`, `ProbabilisticRegressionModel`).
    *   Models that are sensitive to the magnitude of the output, particularly when using certain activation functions in the output layer or specific loss functions.

### **Categorical Feature Handling**
The package provides transformers for handling categorical features, which should be applied *before* passing data to the AutoML pipeline (e.g., as part of a `sklearn.pipeline.Pipeline`). While tree-based models like CatBoost have built-in capabilities for categorical data, other models require explicit encoding.
*   **`OneHotEncoder`**: A wrapper around Scikit-learn's `OneHotEncoder`. It converts categorical variables into a binary (0/1) format across multiple new columns, making them suitable for most machine learning algorithms. This is a standard and effective method for low-cardinality features.
*   **`OrderedTargetEncoder`**: A more advanced, leakage-free target encoder, particularly useful for high-cardinality categorical features. Instead of creating many new columns, it encodes each category based on its relationship with the target variable.
    *   **Core Idea**: It replaces each category with a smoothed version of the target variable's mean for that category. To prevent target leakage (where information from the target variable of a sample is used to encode a feature for that same sample), the calculation is performed on a randomly shuffled version of the data.
    *   **Mathematical Formulation**: For each row `i` in the shuffled training data, the encoded value for a specific category is calculated as:
        $$\text{EncodedValue}_i = \frac{\text{cumsum}_i + \mu_{\text{global}} \cdot s}{\text{cumcount}_i + s}
$$
        Where:
        *   $\text{cumsum}_i$: The cumulative sum of the target values for all *previous* rows belonging to the same category.
        *   $\text{cumcount}_i$: The cumulative count of *previous* rows belonging to the same category.
        *   $\mu_{\text{global}}$: The overall mean of the target variable across the entire training set.
        *   $s$: A `smoothing` factor that regularizes the encoding by pulling the estimate towards the global mean. This is especially important for rare categories to prevent overfitting.
    *   **Transformation**: For new data (in the `transform` step), each category is mapped to the overall mean of the target for that category, as learned from the full training set.


### **Missing Value Imputation**
The package does not include built-in missing value imputation. It is assumed that missing values are handled *before* data is passed to the AutoML pipeline. Users should apply appropriate imputation strategies (e.g., mean, median, mode imputation, or more advanced methods) as a preprocessing step.

## **7. Explainability & Feature Selection**

Understanding why a model makes certain predictions is as important as the predictions themselves. This package integrates powerful tools for model explainability and leverages them for automated feature selection.

*   **SHAP (SHapley Additive exPlanations)**: The package uses SHAP values to explain the output of any model. SHAP values provide a consistent and theoretically sound way to attribute the prediction of a model to its input features.
    *   **Global Interpretability**: By aggregating SHAP values across many samples, you can understand which features are most important overall for the model's predictions.
    *   **Local Interpretability**: For a single prediction, SHAP values show how each feature contributes to pushing the prediction from the base value (average prediction) to the actual predicted value.
*   **Automated Feature Selection**: Based on SHAP importance, the AutoML pipeline can automatically select a subset of the most influential features.
    *   **Mechanism**: During the `retrain_with_selected_features` process, SHAP values are computed for the best-performing model. Features are then ranked by their average absolute SHAP value (mean(|SHAP|)).
    *   **`shap_threshold`**: This parameter allows you to control the stringency of feature selection. For example, `shap_threshold=0.95` means that features contributing to 95% of the cumulative SHAP importance will be retained, and the rest will be discarded.
    *   **Benefits**:
        *   **Reduced Model Complexity**: Training with fewer features can lead to simpler, more interpretable models.
        *   **Faster Training/Inference**: Fewer features mean less data to process, speeding up both training and prediction times.
        *   **Improved Generalization**: Removing irrelevant or noisy features can sometimes prevent overfitting and improve the model's performance on unseen data.
        *   **Data Collection Efficiency**: Identifies features that are truly impactful, potentially reducing the cost and effort of data collection in the future.
*   **`get_feature_importance` Method**: This method provides a summary of feature importance, typically as a dictionary mapping feature names to their SHAP importance scores. This allows users to inspect the relative importance of features and gain insights into the model's decision-making process.

## **8. Metrics**

This AutoML package provides a comprehensive set of metrics for evaluating model performance across both regression and classification tasks. These metrics are used during hyperparameter optimization (by Optuna) and for final model evaluation.

### **Regression Metrics**

For regression tasks, the package focuses on metrics that quantify the difference between predicted and actual continuous values.

*   **Mean Absolute Error (MAE)**:
    *   **Description**: The average of the absolute differences between predictions and actual observations. It measures the average magnitude of the errors in a set of predictions, without considering their direction.
    *   **Formula**:
        $$
MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$
        Where $N$ is the number of samples, $y_i$ is the actual value, and $\hat{y}_i$ is the predicted value.
    *   **Interpretation**: MAE is robust to outliers and is in the same unit as the target variable. A lower MAE indicates better model performance.

*   **Mean Squared Error (MSE)**:
    *   **Description**: The average of the squared differences between predictions and actual observations. It penalizes larger errors more heavily than MAE due to the squaring.
    *   **Formula**:
        $$
MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$
    *   **Interpretation**: MSE is sensitive to outliers. A lower MSE indicates better model performance. Its square root, RMSE, is often preferred for interpretability as it's in the same units as the target.

*   **Root Mean Squared Error (RMSE)**:
    *   **Description**: The square root of the Mean Squared Error. It represents the standard deviation of the residuals (prediction errors).
    *   **Formula**:
        $$
RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$
    *   **Interpretation**: RMSE is widely used and is in the same units as the target variable, making it more interpretable than MSE. A lower RMSE indicates better model performance.

*   **R-squared ($R^2$) Score**:
    *   **Description**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It indicates how well the model fits the observed data.
    *   **Formula**:
        $$
R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
$$
        Where $\bar{y}$ is the mean of the actual values.
    *   **Interpretation**: $R^2$ ranges from 0 to 1 (or can be negative for very poor fits). A higher $R^2$ indicates a better fit.

### **Classification Metrics**

For classification tasks, the package provides metrics to assess the model's ability to correctly classify instances into discrete categories.

*   **Accuracy**:
    *   **Description**: The proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances.
    *   **Formula**:
        $$
Accuracy = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$
    *   **Interpretation**: Simple and intuitive, but can be misleading for imbalanced datasets.

*   **Precision**:
    *   **Description**: The proportion of true positive predictions among all positive predictions. It answers: "Of all instances predicted as positive, how many were actually positive?"
    *   **Formula**:
        $$
Precision = \frac{TP}{TP + FP}
$$
        Where TP is True Positives and FP is False Positives.
    *   **Interpretation**: High precision indicates a low false positive rate.

*   **Recall (Sensitivity)**:
    *   **Description**: The proportion of true positive predictions among all actual positive instances. It answers: "Of all actual positive instances, how many were correctly identified?"
    *   **Formula**:
        $$
Recall = \frac{TP}{TP + FN}
$$
        Where FN is False Negatives.
    *   **Interpretation**: High recall indicates a low false negative rate.

*   **F1-Score**:
    *   **Description**: The harmonic mean of Precision and Recall. It provides a single score that balances both precision and recall, useful for imbalanced datasets.
    *   **Formula**:
        $$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$
    *   **Interpretation**: A higher F1-score indicates better balance between precision and recall.

*   **ROC AUC (Receiver Operating Characteristic - Area Under Curve)**:
    *   **Description**: Measures the performance of a binary classifier to distinguish between classes. It's the area under the ROC curve, which plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.
    *   **Interpretation**: Ranges from 0 to 1. An AUC of 0.5 suggests no discrimination (random guessing), while 1.0 indicates perfect discrimination. Useful for evaluating models across all possible classification thresholds.

*   **Log Loss (Cross-Entropy Loss)**:
    *   **Description**: Measures the performance of a classification model where the prediction input is a probability value between 0 and 1. Log loss increases as the predicted probability diverges from the actual label.
    *   **Formula (Binary Classification)**:
        $$
L_{log} = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]
$$
        Where $y_i$ is the true label (0 or 1) and $p_i$ is the predicted probability for class 1.
    *   **Interpretation**: A lower log loss indicates better model performance. It heavily penalizes confident but incorrect predictions.

### **Example Metric Plots**

The `Metrics` utility class automatically generates and saves various plots to help visualize model performance. These plots are saved to the `metrics` directory within your experiment run.

*   **Regression Plots**:
    *   **Predicted vs. Actual Values**: A scatter plot showing predicted values against true values. A perfect model would show points lying on a 45-degree line.
        ![Predicted vs. Actual Values Example](https://via.placeholder.com/400x300?text=Predicted+vs.+Actual+Plot)
    *   **Residuals vs. Predicted Values**: A scatter plot of residuals (errors) against predicted values. Ideally, residuals should be randomly scattered around zero, indicating no systematic errors.
        ![Residuals vs. Predicted Values Example](https://via.placeholder.com/400x300?text=Residuals+vs.+Predicted+Plot)

*   **Classification Plots**:
    *   **Confusion Matrix**: A table showing the counts of true positives, true negatives, false positives, and false negatives.
        ![Confusion Matrix Example](https://via.placeholder.com/400x300?text=Confusion+Matrix)
    *   **ROC Curve**: Plots True Positive Rate vs. False Positive Rate.
        ![ROC Curve Example](https://via.placeholder.com/400x300?text=ROC+Curve)
    *   **Precision-Recall Curve**: Plots Precision vs. Recall. Useful for imbalanced datasets.
        ![Precision-Recall Curve Example](https://via.placeholder.com/400x300?text=Precision-Recall+Curve)
    *   **Predicted vs. True Classification Rate**: Shows how well the predicted probabilities align with the actual classification rate.
        ![Predicted vs. True Classification Rate Example](https://via.placeholder.com/400x300?text=Predicted+vs.+True+Classification+Rate)
    *   **Completeness vs. Purity**: Visualizes the trade-off between identifying all relevant instances (completeness) and ensuring identified instances are correct (purity).
        ![Completeness vs. Purity Example](https://via.placeholder.com/400x300?text=Completeness+vs.+Purity+Plot)



## **9. Early Stopping**

Early stopping is a regularization technique used to prevent overfitting when training iterative models, such as neural networks or gradient boosting models. It works by monitoring the model's performance on a validation set during training and stopping the training process when the performance on the validation set starts to degrade, even if the performance on the training set is still improving.

*   **Mechanism**:
    1.  **Monitoring Metric**: A specific metric (e.g., validation loss, RMSE, accuracy) is chosen to monitor the model's performance on a separate validation set.
    2.  **Patience**: Training continues as long as the monitored metric improves. If the metric does not improve for a specified number of consecutive epochs (the "patience" parameter), training is stopped.
    3.  **Best Model Restoration**: The model weights corresponding to the best performance on the validation set (before degradation began) are typically restored.

*   **Early Stopping with Cross-Validation**:
    When combined with cross-validation, early stopping is applied independently within each fold. For each fold:
    1.  The model is trained on the training subset of that fold, and its performance is monitored on the validation subset of that fold.
    2.  Early stopping criteria (e.g., patience) are applied to stop training for that specific fold.
    3.  The optimal number of training iterations (epochs, boosting rounds) determined by early stopping for each fold is recorded.
    4.  After all folds are complete, the average of these optimal iteration counts is often used to train the final model on the entire dataset. This ensures that the final model is trained for an appropriate duration, balancing performance and generalization.

*   **Benefits**:
    *   **Prevents Overfitting**: Stops training before the model starts to memorize the training data and lose its ability to generalize to unseen data.
    *   **Reduces Training Time**: Avoids unnecessary training iterations once the model's performance on new data plateaus or worsens.
    *   **Implicit Regularization**: Acts as a form of regularization by limiting the model's capacity to overfit.

## **10. Hyperparameter Optimization (Optuna)**

Hyperparameter optimization is a critical step in machine learning that involves finding the best set of hyperparameters for a given model and dataset. This AutoML package leverages [Optuna](https://optuna.org/), an open-source hyperparameter optimization framework, to efficiently search for optimal model configurations.

### **How Optuna Works**

Optuna automates the trial-and-error process of hyperparameter tuning by intelligently exploring the search space. Its core components are:

*   **Study**: An optimization session that manages multiple trials. It stores the history of all trials and their results.
*   **Trial**: A single execution of the objective function with a specific set of hyperparameters. Optuna proposes new hyperparameters for each trial.
*   **Objective Function**: A user-defined function that Optuna aims to minimize or maximize. In this AutoML package, the objective function trains a model with a given set of hyperparameters (proposed by Optuna) and returns a performance metric (e.g., RMSE, Accuracy) from cross-validation.
*   **Samplers**: Algorithms that propose new hyperparameter values for each trial. Optuna supports various samplers, including:
    *   **Tree-structured Parzen Estimator (TPE)**: A Bayesian optimization algorithm that builds a probabilistic model of the objective function and uses it to propose promising new trials. This is the default sampler used in this package.
    *   **Random Search**: Explores the search space randomly.
*   **Pruners**: Algorithms that automatically stop unpromising trials early during cross-validation. This saves computational resources by avoiding the completion of trials that are unlikely to yield good results. This package uses `MedianPruner`.

### **Setting Optuna Parameters in AutoML**

The `AutoML` class exposes several parameters that control Optuna's behavior:

*   **`n_trials`**:
    *   **Description**: The total number of optimization trials (i.e., hyperparameter combinations) that Optuna will evaluate. Each trial involves training and evaluating a model using cross-validation.
    *   **How to Set**: Passed directly to the `AutoML` constructor (e.g., `n_trials=50`). A higher number of trials generally increases the chance of finding better hyperparameters but also increases computation time.
*   **`metric`**:
    *   **Description**: The performance metric that Optuna will optimize (minimize or maximize). This metric is calculated during the cross-validation process for each trial.
    *   **How to Set**: Passed to the `AutoML` constructor (e.g., `metric=Metric.RMSE` for regression, `metric=Metric.ACCURACY` for classification). The `Metric` enum defines available metrics.
*   **`direction`**:
    *   **Description**: Determines whether Optuna should `minimize` or `maximize` the `metric`. This is automatically inferred by the `AutoML` class based on the chosen `metric` (e.g., RMSE is minimized, Accuracy is maximized).
*   **`n_splits`**:
    *   **Description**: The number of folds for K-Fold cross-validation performed within each Optuna trial. Cross-validation provides a more robust estimate of a model's performance and helps prevent overfitting to a single validation set.
    *   **How to Set**: Passed to the `AutoML` constructor (e.g., `n_splits=5`).
*   **`random_state`**:
    *   **Description**: A seed for the random number generator, ensuring reproducibility of the Optuna optimization process.
    *   **How to Set**: Passed to the `AutoML` constructor (e.g., `random_state=42`).

By configuring these parameters, users can control the thoroughness and reproducibility of the hyperparameter optimization process, allowing the AutoML system to efficiently discover high-performing models.



## **11. Persistency**

The AutoML package provides robust functionality to save and load the entire state of an AutoML run, including the best-trained model, feature scalers, and target scalers. This ensures that you can resume work, deploy models, or share results without needing to retrain from scratch.

*   **Saving State**: The `save_automl_state(file_path: str)` method serializes the `AutoML` object to a specified file path (e.g., using `joblib`). This captures the best model found, its hyperparameters, and any fitted data transformers.
*   **Loading State**: The static method `AutoML.load_automl_state(file_path: str)` allows you to deserialize a previously saved AutoML object. Once loaded, you can use the `predict()`, `predict_proba()`, or `predict_uncertainty()` methods directly without retraining.

**Example:**
```python
# Save the trained AutoML state
automl_reg.save_automl_state("my_automl_regression_model.joblib")

# Load the AutoML state later
loaded_automl = AutoML.load_automl_state("my_automl_regression_model.joblib")

if loaded_automl:
    # Use the loaded model for predictions
    new_predictions = loaded_automl.predict(new_X_data)
    print(f"Predictions from loaded model: {new_predictions[:5]}")
```

## **12. Experiment Tracking (Weights & Biases)**

Effective experiment tracking is crucial for managing machine learning workflows, comparing model performance, and reproducing results. This AutoML package seamlessly integrates with [Weights & Biases (W&B)](https://wandb.ai/), a powerful platform for MLOps.

When `use_wandb=True` is set during the `AutoML` initialization, the following information is automatically logged to your W&B project:

*   **Hyperparameters**: All hyperparameters explored by Optuna for each trial, including the model type and its specific configuration.
*   **Metrics**: Performance metrics (e.g., RMSE, accuracy, F1-score) for each cross-validation fold and the aggregated average metric for each trial.
*   **System Metrics**: CPU/GPU utilization, memory usage, and other system-level information during training.
*   **Visualizations**: W&B provides interactive dashboards to visualize training curves, hyperparameter importance, and model comparisons.

**Benefits of W&B Integration:**

*   **Reproducibility**: Easily track and reproduce any experiment run.
*   **Comparison**: Compare different models and hyperparameter configurations side-by-side.
*   **Collaboration**: Share your experiment results with team members.
*   **Insights**: Gain deeper insights into model behavior and training dynamics.

To use W&B, ensure you have an account and are logged in (`wandb login`) in your environment. You can specify your W&B project and entity during `AutoML` initialization:

```python
automl = AutoML(
    task_type=TaskType.REGRESSION,
    metric=Metric.RMSE,
    n_trials=50,
    use_wandb=True,
    wandb_project="my_automl_project",
    wandb_entity="my_wandb_username"
)
```

## **13. Leaderboard**

The AutoML process inherently generates a "leaderboard" of models by evaluating and ranking them based on the chosen optimization metric during hyperparameter tuning. The `AutoML` class keeps track of the best-performing model and its configuration.

*   **Automatic Ranking**: During the `train()` method, Optuna explores various models and their hyperparameters. It identifies the trial that yields the best performance according to the specified `metric`.
*   **Accessing Best Model Info**: You can retrieve details about the overall best model found by the AutoML process using the `get_best_model_info()` method. This returns a dictionary containing the best model's name, its optimal hyperparameters, and the achieved metric value.

**Example:**
```python
# After training the AutoML instance
best_model_info = automl_reg.get_best_model_info()

if best_model_info:
    print(f"Overall Best Model: {best_model_info['name']}")
    print(f"Best Hyperparameters: {best_model_info['hyperparameters']}")
    print(f"Achieved Metric ({automl_reg.metric.value}): {best_model_info['metric_value']:.4f}")
```

This leaderboard functionality allows you to quickly identify the most promising models and configurations from your automated experiments.

## **14. Installation**

To use this package in your own projects, you can install it directly from the source code.

### **1. Clone the Repository**

First, clone this repository to your local machine:

```bash
git clone https://github.com/jordanelridge31/automl.git
cd automl
```

### **2. Install the Package**

It is recommended to install the package in an editable, or "developer," mode. This allows you to make changes to the source code and have them immediately reflected in your environment without needing to reinstall.

From the root directory of the project (where `setup.py` is located), run:

```bash
pip install -e .
```

This command will install the `` and all its dependencies listed in `setup.py`.

### **3. Verify the Installation**

After installation, you should be able to import the `AutoML` class and other components from any Python script or notebook in your environment:

```python
from automl_package.automl import AutoML
from automl_package.enums import TaskType

# If this runs without an ImportError, the installation was successful.
print("AutoML package imported successfully!")
```


## **15. Usage**

Refer to the `automl_package/examples` directory for detailed scripts:

*   `run_automl.py`: Demonstrates the core AutoML functionality for both regression and classification tasks.
*   `noisy_data_example.py`: Provides a use case for comparing advanced regression models on noisy data.

### **Basic AutoML Workflow**

Here's a basic example demonstrating how to train and evaluate models using the AutoML orchestrator:

```python
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score # Import metrics

# Assuming automl_package is installed and its modules are importable
from automl_package.automl import AutoML
from automl_package.enums import TaskType, ModelName, UncertaintyMethod # Import necessary enums

import logging
import json # Import json for pretty printing

logger = logging.getLogger('automl_package')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

print("===== Starting AutoML Package Demonstration =====")

# --- Regression Example ---
logger.info("\n--- Running Regression Example with Scaling and Uncertainty ---")
X_reg, y_reg = make_regression(n_samples=200, n_features=10, noise=10.0, random_state=42)
y_reg = y_reg + abs(y_reg.min()) + 1 # Ensure y is positive for certain percentile calcs

# Split data for initial training (for AutoML's CV) and final test evaluation
X_full_reg, X_test_full_reg, y_full_reg, y_test_full_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_initial_reg, y_train_initial_reg = X_full_reg, y_full_reg

feature_names_reg = [f'feature_{i}' for i in range(X_reg.shape[1])]

# Instantiate scalers
X_scaler_reg = StandardScaler()
y_scaler_reg = StandardScaler()

automl_reg = AutoML(task_type=TaskType.REGRESSION, metric='rmse', n_trials=3, n_splits=2, random_state=42,
                     feature_scaler=X_scaler_reg, target_scaler=y_scaler_reg, use_wandb=False) # Set use_wandb to False for local demo

# Specify which models to consider for this regression run
# Including the new FlexibleNeuralNetwork
models_for_reg = [
    ModelName.JAX_LINEAR_REGRESSION,
    ModelName.PYTORCH_NEURAL_NETWORK,
    ModelName.FLEXIBLE_NEURAL_NETWORK, # NEW MODEL INCLUDED
    ModelName.XGBOOST,
    ModelName.LIGHTGBM,
    ModelName.CATBOOST,
    ModelName.CLASSIFIER_REGRESSION, # Only for regression where classification informs regression
    ModelName.PROBABILISTIC_REGRESSION
]
# Filter out models if their dependencies are not installed
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:
    models_for_reg = [m for m in models_for_reg if m not in [ModelName.XGBOOST, ModelName.LIGHTGBM, ModelName.CATBOOST]]

automl_reg.train(X_train_initial_reg, y_train_initial_reg, models_to_consider=models_for_reg)
```

### **Prediction**

```python
if automl_reg.best_model_name:
    logger.info(f"\n--- Making Predictions with Best Regression Model ({automl_reg.best_model_name.value}) ---")
    y_pred_test = automl_reg.predict(X_test_full_reg)
    test_rmse = np.sqrt(mean_squared_error(y_test_full_reg, y_pred_test))
    logger.info(f"Best model test RMSE (original scale): {test_rmse:.4f}")
    logger.info(f"Sample predictions (first 5, original scale): {y_pred_test[:5].round(2)}")
```

### **Uncertainty Prediction**

```python
    logger.info(f"\n--- Predicting Uncertainty with Best Regression Model ({automl_reg.best_model_name.value}) ---")
    try:
        uncertainty_values = automl_reg.predict_uncertainty(X_test_full_reg)
        logger.info(f"Mean uncertainty estimate (original scale): {np.mean(uncertainty_values):.4f}")
        logger.info(f"Uncertainty estimates (first 5, original scale): {uncertainty_values[:5].round(2)}")
    except ValueError as e:
        logger.error(f"Could not get uncertainty estimates for model {automl_reg.best_model_name.value}: {e}")
    except NotImplementedError as e:
        logger.error(f"Uncertainty prediction not implemented for {automl_reg.best_model_name.value}: {e}")
```

### **Feature Importance**

```python
    logger.info(f"\n--- Getting Feature Importance for Best Regression Model ({automl_reg.best_model_name.value}) ---")
    feature_importance_summary = automl_reg.get_feature_importance(X_test_full_reg, feature_names=feature_names_reg)
    if "error" not in feature_importance_summary:
        logger.info(f"Top 5 Features by SHAP Importance:\n{json.dumps(dict(list(feature_importance_summary.items())[:5]), indent=2)}")
```

### **Retraining with Selected Features**

```python
    logger.info("\n--- Retraining with Selected Features (Regression) ---")
    retrained_results_reg = automl_reg.retrain_with_selected_features(
        X_full_train=X_full_reg, y_full_train=y_full_reg,
        X_full_test=X_test_full_reg, y_full_test=y_test_full_reg,
        feature_names=feature_names_reg, shap_threshold=0.95
    )
    logger.info(f"Retrained model test RMSE (with selected features, original scale): {retrained_results_reg['retrained_metric_value']:.4f}")
    logger.info(f"Selected features: {retrained_results_reg['selected_feature_names']}")
```

**Remaining Usage Examples (Classification, Save/Load)**

```python
# --- Classification Example ---
logger.info("\n\n===== Running Classification AutoML Example =====")
X_clf, y_clf = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

X_full_clf, X_test_full_clf, y_full_clf, y_test_full_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
X_train_initial_clf, y_train_initial_clf = X_full_clf, y_full_clf

feature_names_clf = [f'feature_{i}' for i in range(X_clf.shape[1])]
X_scaler_clf = StandardScaler()

automl_clf = AutoML(task_type=TaskType.CLASSIFICATION, metric='accuracy', n_trials=3, n_splits=2, random_state=42,
                     feature_scaler=X_scaler_clf, use_wandb=False) # Set use_wandb to False for local demo

models_for_clf = [
    ModelName.SKLEARN_LOGISTIC_REGRESSION,
    ModelName.PYTORCH_NEURAL_NETWORK,
    ModelName.FLEXIBLE_NEURAL_NETWORK, # NEW MODEL INCLUDED
    ModelName.XGBOOST,
    ModelName.LIGHTGBM,
    ModelName.CATBOOST,
]
# Filter out models if their dependencies are not installed
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:
    models_for_clf = [m for m in models_for_clf if m not in [ModelName.XGBOOST, ModelName.LIGHTGBM, ModelName.CATBOOST]]

automl_clf.train(X_train_initial_clf, y_train_initial_clf, models_to_consider=models_for_clf)

logger.info(f"\n--- Making Predictions with Best Classification Model ({automl_clf.best_model_name.value}) ---")
y_pred_test_clf = automl_clf.predict(X_test_full_clf)
test_accuracy_clf = accuracy_score(y_test_full_clf, y_pred_test_clf)
logger.info(f"Best model test accuracy: {test_accuracy_clf:.4f}")
logger.info(f"Sample predictions (first 5): {y_pred_test_clf[:5].round(2)}")

logger.info(f"\n--- Getting Feature Importance for Best Classification Model ({automl_clf.best_model_name.value}) ---")
feature_importance_summary_clf = automl_clf.get_feature_importance(X_test_full_clf, feature_names=feature_names_clf)
if "error" not in feature_importance_summary_clf:
    logger.info(f"Top 5 Features by SHAP Importance:\n{json.dumps(dict(list(feature_importance_summary_clf.items())[:5]), indent=2)}")

logger.info("\n--- Retraining with Selected Features (Classification) ---")
retrained_results_clf = automl_clf.retrain_with_selected_features(
    X_full_train=X_full_clf, y_full_train=y_full_clf,
    X_full_test=X_test_full_clf, y_full_test=y_test_full_clf,
    feature_names=feature_names_clf, shap_threshold=0.95
)
logger.info(f"Retrained model test accuracy (with selected features): {retrained_results_clf['retrained_metric_value']:.4f}")
logger.info(f"Selected features: {retrained_results_clf['selected_feature_names']}")

# --- Saving and Loading AutoML state example ---
logger.info("\n--- Saving and Loading AutoML State ---")
save_path_reg_automl = "automl_reg_state.joblib"
automl_reg.save_automl_state(save_path_reg_automl)
loaded_automl_reg = AutoML.load_automl_state(save_path_reg_automl)

if loaded_automl_reg:
    logger.info(f"Loaded AutoML best regression model: {loaded_automl_reg.get_best_model_info()['name']}")
    # You can use loaded_automl_reg to predict on new data.
    # loaded_predictions_reg = loaded_automl_reg.predict(X_test_full_reg[:5])
    # logger.info(f"Predictions from loaded regression model (first 5): {loaded_predictions_reg.flatten().round(2)}")

save_path_clf_automl = "automl_clf_state.joblib"
automl_clf.save_automl_state(save_path_clf_automl)
loaded_automl_clf = AutoML.load_automl_state(save_path_clf_automl)

if loaded_automl_clf:
    logger.info(f"Loaded AutoML best classification model: {loaded_automl_clf.get_best_model_info()['name']}")
    # You can use loaded_automl_clf to predict on new data.
    # loaded_predictions_clf = loaded_automl_clf.predict(X_test_full_clf[:5])
    # logger.info(f"Predictions from loaded classification model (first 5): {loaded_predictions_clf.flatten().round(2)}")

logger.info("\n===== AutoML Package Demonstration Complete =====")
```

## **16. Project Structure**
The project is organized as follows:

*   `automl_package/`: The core package containing the AutoML framework.
    -   `examples/`: Contains example scripts demonstrating the package's functionalities.
    -   `automl.py`: Contains the main `AutoML` class that orchestrates the model selection, training, and evaluation pipeline.
    -   `enums.py`: Defines enumerations used throughout the package.
    -   `logger.py`: Configures a centralized logger for the package.
    -   `preprocessing.py`: Contains preprocessing utilities like `OrderedTargetEncoder` and `OneHotEncoder`.
    -   `models/`: Directory for all the machine learning model implementations.
        -   `base.py`: Defines the abstract `BaseModel` class.
        -   `base_pytorch.py`: Provides a base class for PyTorch-based models, handling common training loops and regularization.
        -   `flexible_neural_network.py`: Contains the novel `FlexibleNeuralNetwork`.
        -   `neural_network.py`: Implements the standard `PyTorchNeuralNetwork`.
        -   `probabilistic_regression.py`: Implements the `ProbabilisticRegressionModel`.
        -   `classifier_regression.py`: Implements the `ClassifierRegressionModel`.
        -   `linear_regression.py`: Implements `JAXLinearRegression`.
        -   `normal_equation_linear_regression.py`: Implements `NormalEquationLinearRegression`.
        -   `pytorch_linear_regression.py`: Implements `PyTorchLinearRegression`.
        -   `pytorch_logistic_regression.py`: Implements `PyTorchLogisticRegression`.
        -   `sklearn_logistic_regression.py`: Implements `SKLearnLogisticRegression`.
        -   `xgboost_model.py`: Implements `XGBoostModel`.
        -   `lightgbm_model.py`: Implements `LightGBMModel`.
        -   `catboost_model.py`: Implements `CatBoostModel`.
        -   `selection_strategies/`: Contains generic and specific selection strategy implementations.
            -   `base_selection_strategy.py`: Defines the abstract base class for all selection strategies.
            -   `n_classes_strategies.py`: Contains strategy classes for dynamic n-classes selection.
            -   `layer_selection_strategies.py`: Contains strategy classes for dynamic layer selection.
    -   `optimizers/`: Directory for hyperparameter optimization logic (e.g., `optuna_optimizer.py`).
    -   `explainers/`: Directory for model explainability features (e.g., `feature_explainer.py`).
    -   `utils/`: Directory for utility functions (e.g., `metrics.py`, `numerics.py`, `probability_mapper.py`).
*   `docs/`: Contains documentation files.
*   `requirements.txt`: Lists the Python packages required to run the project.
*   `README.md`: The main README file for the project.

## **17. Contributing**

Contributions are welcome! Please feel free to open issues or submit pull requests.

## **18. License**

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
