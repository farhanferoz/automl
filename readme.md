# **AutoML Package**

A versatile Automated Machine Learning (AutoML) framework built in Python, designed to streamline the process of model selection, hyperparameter optimization, training, prediction, and explainability for both regression and classification tasks. This package integrates various popular machine learning models from PyTorch, JAX, XGBoost, LightGBM, CatBoost, and Scikit-learn, alongside advanced features like dynamic neural network architectures and comprehensive uncertainty quantification.

## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Key Features](#2-key-features)
3. [Supported Models](#3-supported-models)
   * [Base Models](#base-models)
   * [Tree-Based Models](#tree-based-models)
   * [Neural Networks](#neural-networks)
   * [Probabilistic Regression Models](#probabilistic-regression-models)
   * [Composite Models](#composite-models)
4. [Uncertainty Quantification](#4-uncertainty-quantification)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Explainability & Feature Selection](#6-explainability--feature-selection)
7. [Installation](#7-installation)
8. [Usage](#8-usage)
   * [Basic AutoML Workflow](#basic-automl-workflow)
   * [Prediction](#prediction)
   * [Uncertainty Prediction](#uncertainty-prediction)
   * [Feature Importance](#feature-importance)
   * [Retraining with Selected Features](#retraining-with-selected-features)
   * [Saving and Loading AutoML State](#saving-and-loading-automl-state)
   * [Leaderboard](#leaderboard)
9. [Project Structure](#9-project-structure)
10. [Contributing](#10-contributing)
11. [License](#11-license)

## **1. Introduction**

This AutoML package aims to automate significant portions of the machine learning pipeline, enabling users to quickly find optimal models and understand their predictions without extensive manual tuning. It's built with modularity and extensibility in mind, allowing easy integration of new models and techniques.

## **2. Key Features**

* **Automated Model Selection:** Evaluates a diverse set of machine learning algorithms.
* **Hyperparameter Optimization:** Utilizes Optuna for efficient and robust hyperparameter tuning with cross-validation.
* **Regression & Classification:** Supports both common machine learning task types.
* **Pluggable Architecture:** Easy to add new custom models adhering to the BaseModel interface.
* **Dynamic Neural Networks:** Includes a novel FlexibleNeuralNetwork that adapts its depth based on input.
* **Uncertainty Quantification:** Provides methods to estimate prediction uncertainty for regression tasks.
* **Data Scaling Integration:** Seamlessly integrates feature and target scaling into the pipeline.
* **Model Explainability:** Leverages SHAP (SHapley Additive exPlanations) for interpreting model predictions and for feature selection.
* **Feature Selection:** Automated feature selection based on SHAP importance.
* **Persistency:** Save and load the entire AutoML state, including fitted models and scalers.

## **3. Supported Models**

The package includes wrappers for various popular machine learning models, all conforming to a common BaseModel interface.

### **Base Models**

* **JAXLinearRegression**: A custom Linear Regression implementation using JAX for gradient-based optimization, including L1/L2 regularization.
* **SKLearnLogisticRegression**: Scikit-learn's Logistic Regression for classification, supporting various penalties.

### **Tree-Based Models**

* **XGBoostModel**: Wrapper for XGBoost (Extreme Gradient Boosting), configurable for regression and classification.
* **LightGBMModel**: Wrapper for LightGBM (Light Gradient Boosting Machine), highly efficient for large datasets.
* **CatBoostModel**: Wrapper for CatBoost, known for its robust handling of categorical features.

### **Neural Networks**

This section details the Neural Network models available in the package, designed for various applications and complexities.

* **PyTorchNeuralNetwork**:
  * **Description**: A standard feedforward neural network (Multi-Layer Perceptron) built using PyTorch. It's highly configurable, allowing you to specify the number of hidden layers, their size, the activation function, and apply regularization (L1/L2) and batch normalization. It can be used for both regression (predicting continuous values) and classification (predicting discrete categories).
  * **Core Idea**: Learns complex, non-linear relationships between inputs and outputs through multiple layers of interconnected neurons.
  * **Diagram (Basic Architecture)**:
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

* **FlexibleNeuralNetwork**:
  * **Description**: A novel and adaptive PyTorch-based neural network that dynamically adjusts its effective hidden layer depth for *each individual input*. Instead of a fixed number of layers performing computation, it has a max_hidden_layers and an internal n_predictor that determines how many of these layers should be "active" (perform computations) for a specific input. The remaining layers act as "identity layers," simply passing their input through unchanged.
  * **Core Idea**: To achieve adaptive complexity. For simpler inputs, it can effectively use fewer layers, potentially leading to faster inference and better generalization by avoiding overfitting to noise. For more complex inputs, it can utilize its full depth.
  * **Mechanism**:
    1. **n_predictor**: A small sub-network takes the input and predicts an integer n (from 1 to max_hidden_layers). During training, it uses Gumbel-Softmax for differentiable selection of n.
    2. **Dynamic Forward Pass**: In the main network's forward pass, each hidden layer block checks the predicted n. If the current layer's index is among the first (max_hidden_layers - n) layers, it bypasses computation (identity mapping). Otherwise, it performs its standard linear transformation, batch norm, and activation.
  * **Diagram (Dynamic Depth with Identity Layers)**:
    ```
                        ┌──────────────────┐
    Input Features X ───┤ 1. N-Predictor   ├─────┐
                        └──────────────────┘     │
                            Predicted 'n'
                             (e.g., n=3)
                               │                 │
    +----------------------------------------------------------------+
    |                                                                |
    |   Hidden Layer Block 1 (Linear, BatchNorm, ReLU, Dropout)      |
    |      (If n=3, this block is IDENTITY for this input)           |
    +----------------------------------------------------------------+
    |                                                                |
    |   Hidden Layer Block 2 (Linear, BatchNorm, ReLU, Dropout)      |
    |      (If n=3, this block is IDENTITY for this input)           |
    +----------------------------------------------------------------+
    |                                                                |
    |   Hidden Layer Block 3 (Linear, BatchNorm, ReLU, Dropout)      |
    |      (This block is ACTIVE for this input)                     |
    +----------------------------------------------------------------+
    |                                                                |
    |   Hidden Layer Block 4 (Linear, BatchNorm, ReLU, Dropout)      |
    |      (This block is ACTIVE for this input)                     |
    +----------------------------------------------------------------+
    |                                                                |
    |   Hidden Layer Block 5 (Linear, BatchNorm, ReLU, Dropout)      |
    |      (This block is ACTIVE for this input)                     |
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

### **Probabilistic Regression Models**

These models are designed for regression tasks where quantifying the uncertainty of a prediction is as important as the prediction itself. Instead of just predicting a single value, they predict an entire probability distribution.

* **ProbabilisticRegressionModel (PyTorch-based)**:
  * **Description**: This is a composite PyTorch model that tackles regression by internally employing a classification strategy. It first conceptually discretizes the continuous target variable into a fixed number of n_classes (bins). It then trains an internal classifier to predict which bin an input belongs to. The probabilities from this classifier are then fed into specialized regression "heads" (either separate heads per class or a single head with multiple outputs) that predict the actual continuous value. Crucially, it can be configured to learn the mean and variance of a Gaussian distribution for each prediction, directly modeling uncertainty.
  * **Core Idea**: Leverages the strengths of classification (assigning inputs to meaningful "categories" of the target variable) to improve continuous regression prediction and provide robust uncertainty estimates.
  * **Mechanism (Regression Heads Strategies)**:
    1. **Target Discretization (Conceptual)**: The continuous target y is divided into n_classes bins (e.g., via percentiles).
    2. **Classifier Branch**: An internal PyTorchNeuralNetwork (as the base classifier) is trained to predict the probability of an input falling into each of these n_classes.
    3. **Regression Heads**: The predicted probabilities for each class are then fed into subsequent regression subnetworks. The strategy for these heads can vary:
       * **RegressionStrategy.SEPARATE_HEADS**: Creates n_classes distinct regression subnetworks. Each subnetwork specializes in mapping the probability of its corresponding class to an expected regression value for that class. The final prediction is a weighted sum of outputs from all heads.
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

       * **RegressionStrategy.SINGLE_HEAD_N_OUTPUTS**: A single regression subnetwork outputs n_classes distinct regression values (or mean/log-variance pairs). The final prediction is then a weighted sum of these n_classes outputs, weighted by their respective class probabilities.
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

       * **RegressionStrategy.SINGLE_HEAD_FINAL_OUTPUT**: A single regression subnetwork directly outputs the final combined regression prediction (or mean/log-variance pair).
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
                           |
                           v
                 Final Prediction (Y_pred)
        ```

    4. **Prediction Aggregation**: The outputs of these regression heads are combined (often via a weighted sum using the class probabilities) to form the final continuous prediction.
    5. **Uncertainty**: The model can learn both the mean and the log-variance of a Gaussian distribution for its final prediction, allowing for **probabilistic uncertainty**.
  * **Diagram (Probabilistic Regression Model Flow - Consolidated)**:
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

### **Composite Models**

This section covers a meta-model that combines different modeling paradigms to solve complex problems.

* **ClassifierRegressionModel**:
  * **Description**: This is a **meta-model** (a model that uses other models internally) designed to solve **regression problems** by framing them as classification tasks. It works by discretizing the continuous target variable into a fixed number of bins (classes), training a *classifier* (which can be any BaseModel classifier like XGBoostModel, PyTorchNeuralNetwork, etc.) on these discrete classes, and then mapping the classifier's predicted probabilities back to a continuous regression output.
  * **Core Idea**: It's particularly useful when the relationship between features and the regression target is complex or non-monotonic, or when traditional regression models struggle. By breaking the problem into classification, it can simplify the learning task.
  * **Mechanism**:
    1. **Target Discretization**: The continuous target variable (y) from the training data is divided into n_classes (e.g., 5-10 bins based on percentiles). Each original y value is assigned a discrete class label.
    2. **Base Classifier Training**: A chosen BaseModel classifier (e.g., XGBoostModel, PyTorchNeuralNetwork) is trained to predict these discrete class labels from the input features (X).
    3. **Class Probability Mapper**: For new predictions, the trained base classifier outputs probabilities for each class. The ClassProbabilityMapper then takes these probabilities for a specific class and maps them to an *expected original regression value* for that class. This mapping is learned during fitting, often through methods like:
       * **MapperType.LINEAR**: A simple linear regression fit between the predicted probability for a class and the actual target values associated with that class in the training data.
        ```
         +-------------------------+      +----------------------------+      +---------------------------+
         | Predicted Probability   |      | Linear Regression Model    |      | Expected Y for Class C    |
         | for Class C (P_C)       +-----> (Trained on P_C vs Y_orig) +---> (Y_exp_C)               |
         +-------------------------+      +----------------------------+      +---------------------------+
         (on training data where y is in Class C)
        ```

       * **MapperType.LOOKUP_MEAN / LOOKUP_MEDIAN**: Divides the probability space into bins and calculates the mean (or median) of the original target values that fall into each bin. When a new probability comes, it looks up the corresponding mean/median.
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

       * **MapperType.SPLINE**: Fits a smooth spline interpolation function between the predicted probabilities and the actual target values for the class. This can capture non-linear relationships in the mapping more flexibly.
        ```
         +-------------------------+      +----------------------------+      +---------------------------+
         | Predicted Probability   |      | Spline Interpolation       |      | Expected Y for Class C    |
         | for Class C (P_C)       +-----> (Fitted to P_C vs Y_orig) +---> (Y_exp_C)               |
         +-------------------------+      +----------------------------+      +---------------------------+
         (on training data where y is in Class C)
        ```

    4. **Final Prediction**: The final regression prediction for an input is the expected value derived from the probabilities and the learned continuous mapping (e.g., a weighted sum of the expected values from each class, weighted by their probabilities).

## **4. Uncertainty Quantification**

For regression models, the package supports quantifying prediction uncertainty using various methods:

* **UncertaintyMethod.CONSTANT**: The simplest method. It calculates the standard deviation of residuals from the training data, applying this constant uncertainty to all new predictions.
  * **Supported by:** JAXLinearRegression, XGBoostModel, LightGBMModel, CatBoostModel, SKLearnLogisticRegression (can be configured to use this for general confidence for classification if treated as regression context), PyTorchNeuralNetwork, FlexibleNeuralNetwork.
* **UncertaintyMethod.PROBABILISTIC**: The model is designed to directly learn both the mean (μ(x)) and the variance (σ²(x)) of the target variable's distribution. This captures **aleatoric uncertainty** (inherent noise in the data). The model is trained with a Negative Log-Likelihood (NLL) loss.
  * **Supported by:** PyTorchNeuralNetwork, FlexibleNeuralNetwork, ProbabilisticRegressionModel.
* **UncertaintyMethod.MC_DROPOUT**: This technique approximates **epistemic uncertainty** (model's uncertainty due to limited data/knowledge). For models with dropout layers, multiple predictions are made for the same input while dropout is active (model in train() mode). The standard deviation of these multiple predictions estimates the uncertainty.
  * **Supported by:** PyTorchNeuralNetwork, FlexibleNeuralNetwork, ProbabilisticRegressionModel.
* **ClassifierRegressionModel Uncertainty Quantification**:
  * **Description**: For ClassifierRegressionModel, uncertainty is estimated by considering the variance of the original target values within each probability bin, as learned by the ClassProbabilityMapper. This model treats the regression problem as a series of classification probabilities, and the uncertainty arises from the spread of true values within the ranges associated with those probabilities.
  * **Mechanism**: For a new prediction, the model aggregates the variance contributions from each class's mapper, weighted by the square of their predicted probabilities. This approach effectively translates the uncertainty in classifying into a bin, and the inherent variability within each bin, into a total prediction uncertainty. The predict_uncertainty method returns the standard deviation derived from this aggregated variance.
  * **Mathematical Formula**:
    The total variance for a given input X is calculated as the sum of squared probabilities multiplied by the variance contribution from each class mapper:
    <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BTotalVariance%7D(X)%20%3D%20%5Csum_%7Bc%3D0%7D%5E%7Bn-1%7D%20P_c(X)%5E2%20%5Ccdot%20%5Ctext%7BVarianceContribution%7D_c(P_c(X))" alt="TotalVariance(X) = \sum_{c=0}^{n-1} P_c(X)^2 \cdot \text{VarianceContribution}_c(P_c(X))"/>
    Where:
    * $P_c(X)$ is the probability of input X belonging to class c, predicted by the base classifier.
    * $\text{VarianceContribution}_c(P_c(X))$ is the variance predicted by the ClassProbabilityMapper for class c, given its probability $P_c(X)$. This variance represents the inherent spread of original y values that mapped to that probability range for class c.

  The final uncertainty (standard deviation) is the square root of this total variance:
    <img src="https://latex.codecogs.com/svg.latex?%5Ctext%7BUncertainty%7D(X)%20%3D%20%5Csqrt%7B%5Ctext%7BTotalVariance%7D(X)%7D" alt="Uncertainty(X) = \sqrt{\text{TotalVariance}(X)}"/>

  * **Diagram (Classifier Regression Model Flow - Consolidated with Uncertainty)**:
    ```
                       ┌─────────────────────────┐
                       │      Input Features X   |
                       └───────────┬─────────────┘
                                   |
                                   v
                   ┌──────────────────────────────────────┐
                   │   1. Base Classifier (e.g., XGBoost) |
                   |      (Trained on discretized y)      |
                   └───────────┬──────────────────────────┘
                               |
                               v
                   ┌────────────────────────────┐
                   │  Predicted Class Probs     |
                   |  (P_0, P_1, ..., P_N-1)    |
                   └───────────┬────────────────┘
                               |
                               v
                   ┌───────────────────────────────────────────────┐
                   │ 2. Class Probability                          |
                   |    Mapper (for each P_i)                      |
                   |    (Maps P_i to Y_i_expected)                 |
                   |    (and maps P_i to VarianceContribution_i)  |
                   └───────────┬───────────────────────────────────┘
                               |
                               v
                   ┌─────────────────────────┐
                   │  Final Regression Value |
                   |  (and Uncertainty)      |
                   └─────────────────────────┘
    ```

## **5. Data Preprocessing**

The AutoML class seamlessly integrates common preprocessing steps:

* **Feature Scaling (feature_scaler):** An optional scikit-learn compatible scaler (e.g., StandardScaler) can be provided during AutoML initialization. If present, features will be automatically scaled before training and prediction.
* **Target Scaling (target_scaler):** For regression tasks, an optional target scaler (e.g., StandardScaler) can be provided. This scales the target variable before training and inverse-transforms predictions back to the original scale. This helps models like neural networks converge better.

## **6. Explainability & Feature Selection**

The package utilizes [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/index.html) to provide insights into model predictions:

* **FeatureExplainer:** A wrapper class that adapts SHAP explainers (TreeExplainer, DeepExplainer, KernelExplainer) to different model types within the AutoML pipeline.
* **Feature Importance:** Calculates global feature importance based on the mean absolute SHAP values, showing which features contribute most to model predictions on average.
* **Automated Feature Selection:** The select_features_by_cumulative_importance method can identify a subset of most important features based on a cumulative SHAP importance threshold, allowing for model retraining on reduced feature sets.

### **How Most Important Features are Selected (using SHAP)**

The AutoML package uses SHAP (SHapley Additive exPlanations) to identify and select the most important features based on a user-defined threshold. This process helps simplify models, improve interpretability, and potentially boost performance by reducing noise.

1. **Calculate Raw Feature Importances (SHAP Values)**:
   * The AutoML.get_feature_importance(X_test, feature_names) method is called.
   * Internally, a FeatureExplainer is initialized for the currently best-performing model. This explainer (e.g., shap.TreeExplainer, shap.DeepExplainer, shap.KernelExplainer) is chosen based on the model type.
   * SHAP values are computed for a subset of the test data (X_test). SHAP values represent the contribution of each feature to the model's output for each prediction.
   * For each feature, the *mean absolute SHAP value* across all explained samples is calculated. This value represents the global importance of that feature to the model's predictions.
2. **Normalize Importances**:
   * The mean absolute SHAP values are then normalized by dividing each feature's importance by the sum of all feature importances. This gives a percentage-like contribution for each feature, summing up to 1 (or 100%).
3. **Cumulative Importance Thresholding**:
   * Features are sorted in descending order based on their normalized importance.
   * The algorithm then iterates through these sorted features, accumulating their normalized importance.
   * It continues to select features until the cumulative_importance reaches or exceeds a user-defined threshold (e.g., 0.95 for 95% of the total importance). All features contributing to this cumulative sum are selected.

**Diagram: Feature Selection by Cumulative SHAP Importance**
```
Raw Features (X)
+-------------------+
| F1, F2, F3, F4, F5|
+-------------------+
        |
        v
+-----------------+
|  Trained Model  |
+-----------------+
        |
        v
+------------------------+
| SHAP Explainer         |
| (calculates SHAP values)|
+------------------------+
        |
        v
+----------------------------------------------+
| Mean Absolute SHAP Values (Raw Importance)   |
| F1: 0.5, F2: 0.3, F3: 0.1, F4: 0.05, F5: 0.05|
+----------------------------------------------+
        |
        v
+--------------------------------------------+
| Normalized & Sorted Importance (Cumulative)|
| F1: 0.50 (0.50)                            |  <-- Select (Threshold 0.95)
| F2: 0.30 (0.80)                            |  <-- Select
| F3: 0.10 (0.90)                            |  <-- Select
| F4: 0.05 (0.95)                            |  <-- Select
| F5: 0.05 (1.00)                            |  <-- Stop here if next adds over 0.95
+--------------------------------------------+
        |
        v
+-------------------------+
| Selected Features:      |
| [F1, F2, F3, F4]        |
+-------------------------+
```

### **Retraining with Selected Features**

After identifying a subset of the most important features, the AutoML package provides a method to retrain the best-performing model using only these selected features. This process can lead to:

* **Simpler Models**: Fewer features mean a less complex model.
* **Improved Generalization**: By removing irrelevant or noisy features, the model might generalize better to unseen data.
* **Faster Training and Inference**: Reduced dimensionality can speed up both training and prediction times.
* **Enhanced Interpretability**: Focusing on a smaller set of key features makes the model's decisions easier to understand.

**Mechanism**:

1. **Feature Selection**: The select_features_by_cumulative_importance method (as explained above) is called to determine the names of the most important features based on a SHAP threshold.
2. **Dataset Filtering**: The original training and testing datasets are filtered to include only the columns corresponding to the selected_feature_names.
3. **Scaler Application**: If a feature_scaler was used in the initial AutoML training, it is *re-used* (already fitted) to transform these filtered feature sets. If a target_scaler was used, the target data is also scaled (or inverse-transformed for predictions) accordingly.
4. **Model Re-instantiation**: The best-performing model from the initial AutoML run is re-instantiated with its previously optimized hyperparameters. Crucially, its input_size (for neural networks) is adjusted to match the number of selected features.
5. **Retraining**: The re-instantiated model is then trained from scratch using the filtered and scaled training data.
6. **Re-evaluation**: The retrained model's performance is evaluated on the filtered and scaled test data (with predictions denormalized if target scaling was applied), providing a new metric score for the reduced feature set.

This retraining step ensures that the final deployed model is optimized not just for its hyperparameters but also for its input feature set.

## **7. Installation**

To use the AutoML package, you'll need Python 3.8+ and install the necessary dependencies:

```bash
pip install numpy pandas scikit-learn optuna jax jaxlib[cuda] # or jaxlib for CPU only
pip install torch torchvision torchaudio # Install PyTorch based on your CUDA version: https://pytorch.org/get-started/locally/
pip install xgboost lightgbm catboost flax optax shap
```

**Note**:

* `jaxlib[cuda]` will install CUDA-enabled JAX, otherwise `pip install jaxlib` for CPU.
* Install PyTorch from their official website (pytorch.org/get-started/locally/) for the version compatible with your CUDA setup. `pip install torch` is a CPU-only version.
* `flax` and `optax` are for JAX models.
* `shap` is for explainability.

**Using a Virtual Environment (Recommended)**

It's highly recommended to use a virtual environment to manage your project's dependencies and avoid conflicts with other Python projects.

1. **Create a requirements.txt file:** In the root of your project directory, create a file named `requirements.txt` and paste the following content into it:
   ```
   numpy
   pandas
   scikit-learn
   optuna
   # For JAX (choose one based on your hardware)
   jax
   # jaxlib==0.4.26+cuda12.cudnn89 # Example for specific CUDA version
   # jaxlib==0.4.26 # For CPU only

   # For PyTorch (choose one based on your CUDA version)
   # torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 # Example for specific CUDA version
   # torch==2.3.1 # For CPU only
   torchvision
   torchaudio

   xgboost
   lightgbm
   catboost
   flax
   optax
   shap
   ```

   * **Important:** Uncomment and adjust the `jaxlib` and `torch` lines based on your specific CUDA version or if you're using a CPU-only setup. Refer to the official JAX and PyTorch installation guides for exact commands.
2. **Create a virtual environment**:
   ```bash
   python -m venv venv_automl
   ```

   (You can replace `venv_automl` with your preferred environment name.)
3. **Activate the virtual environment**:
   * **On Windows**:
     ```bash
     .\venv_automl\Scripts\activate
     ```

   * **On macOS/Linux**:
     ```bash
     source venv_automl/bin/activate
     ```

4. **Install dependencies from requirements.txt**:
   Once your virtual environment is active, install all libraries:
   ```bash
   pip install -r requirements.txt
   ```

## **8. Usage**

### **Basic AutoML Workflow**

Here's a basic example demonstrating how to train and evaluate models using the AutoML orchestrator:

```python
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming automl_package is installed and its modules are importable
from automl_package.automl import AutoML
from automl_package.enums import TaskType, ModelName, UncertaintyMethod # Import necessary enums

import logging
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
                     feature_scaler=X_scaler_reg, target_scaler=y_scaler_reg)

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

**Part 3 of 3: Remaining Usage Examples (Classification, Save/Load)**

```python
# --- Classification Example ---
logger.info("\n\n===== Running Classification AutoML Example =====")
X_clf, y_clf = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

X_full_clf, X_test_full_clf, y_full_clf, y_test_full_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
X_train_initial_clf, y_train_initial_clf = X_full_clf, y_full_clf

feature_names_clf = [f'feature_{i}' for i in range(X_clf.shape[1])]
X_scaler_clf = StandardScaler()

automl_clf = AutoML(task_type=TaskType.CLASSIFICATION, metric='accuracy', n_trials=3, n_splits=2, random_state=42,
                     feature_scaler=X_scaler_clf)

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

# Classification uncertainty: Predict_uncertainty is designed for regression std_dev, will raise error.
# You would typically inspect predict_proba for classification uncertainty.
# try:
#     uncertainty_clf = automl_clf.predict_uncertainty(X_test_full_clf)
# #     logger.info(f"Mean uncertainty estimate for classification: {np.mean(uncertainty_clf):.4f}")
# # except ValueError as e:
# #     logger.warning(f"As expected, cannot get uncertainty for classification task directly: {e}")
# # except NotImplementedError as e:
# #     logger.error(f"Uncertainty prediction not implemented for {automl_clf.best_model_name.value}: {e}")

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
    # You can now use loaded_automl_reg to predict on new data.
    # loaded_predictions_reg = loaded_automl_reg.predict(X_test_full_reg[:5])
    # logger.info(f"Predictions from loaded regression model (first 5): {loaded_predictions_reg.flatten().round(2)}")

save_path_clf_automl = "automl_clf_state.joblib"
automl_clf.save_automl_state(save_path_clf_automl)
loaded_automl_clf = AutoML.load_automl_state(save_path_clf_automl)

if loaded_automl_clf:
    logger.info(f"Loaded AutoML best classification model: {loaded_automl_clf.get_best_model_info()['name']}")
    # You can now use loaded_automl_clf to predict on new data.
    # loaded_predictions_clf = loaded_automl_clf.predict(X_test_full_clf[:5])
    # logger.info(f"Predictions from loaded classification model (first 5): {loaded_predictions_clf.flatten().round(2)}")

logger.info("\n===== AutoML Package Demonstration Complete =====")
   ```

## **9. Project Structure**
The project is organized as follows:

-   `run_automl.py`: The main script to run the AutoML pipeline for both regression and classification tasks. It demonstrates the key functionalities of the package.
-   `automl_package/`: The core package containing the AutoML framework.
    -   `automl.py`: Contains the main `AutoML` class that orchestrates the model selection, training, and evaluation pipeline.
    -   `enums.py`: Defines enumerations used throughout the package, such as `TaskType`, `ModelName`, `UncertaintyMethod`, etc., for clear and robust type-setting.
    -   `logger.py`: Configures a centralized logger for the package.
    -   `models/`: Directory for all the machine learning model implementations.
        -   `base.py`: Defines the abstract `BaseModel` class, which all models inherit from, ensuring a consistent interface.
        -   `linear_regression.py`: Implements `JAXLinearRegression`, a linear regression model using JAX.
        -   `neural_network.py`: Contains `PyTorchNeuralNetwork` and the novel `FlexibleNeuralNetwork`.
        -   `probabilistic_regression.py`: Implements `ProbabilisticRegressionModel`, a PyTorch-based model for regression with uncertainty.
        -   `classifier_regression.py`: Implements `ClassifierRegressionModel`, a meta-model that uses a classifier for regression tasks.
        -   `xgboost_lgbm.py`: Wrappers for `XGBoost` and `LightGBM` models.
        -   `sklearn_logistic_regression.py`: Wrapper for scikit-learn's `LogisticRegression`.
        -   `catboost_model.py`: Wrapper for the `CatBoost` model.
    -   `optimizers/`: Directory for hyperparameter optimization logic.
        -   `optuna_optimizer.py`: Contains the `OptunaOptimizer` class, which uses Optuna for hyperparameter tuning.
    -   `explainers/`: Directory for model explainability features.
        -   `feature_explainer.py`: Implements the `FeatureExplainer` class, which uses SHAP to explain model predictions.
    -   `utils/`: Directory for utility functions.
        -   `probability_mapper.py`: Contains the `ClassProbabilityMapper`, used by the `ClassifierRegressionModel`.

## **10. Contributing**

## **11. License**

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
