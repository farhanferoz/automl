# **AutoML Package**

A versatile Automated Machine Learning (AutoML) framework built in Python, designed to streamline the process of model selection, hyperparameter optimization, training, prediction, and explainability for both regression and classification tasks. This package integrates various popular machine learning models from PyTorch, JAX, XGBoost, LightGBM, CatBoost, and Scikit-learn, alongside advanced features like dynamic neural network architectures and comprehensive uncertainty quantification.

## **Table of Contents**

1. [Introduction](#bookmark=id.jwskq0decd2x)  
2. [Key Features](#bookmark=id.oqwp9pcbnxur)  
3. [Supported Models](#bookmark=id.pbjlg9gujmok)  
   * [Base Models](#bookmark=id.6954v8lwz7ki)  
   * [Tree-Based Models](#bookmark=id.w3x2pjgurh0l)  
   * [Neural Networks](#bookmark=id.dzmvx9uvw17q)  
   * [Probabilistic Regression Models](#bookmark=id.rq115nyhnb2u)  
   * [Composite Models](#bookmark=id.42nwc6hmfyv4)  
4. [Uncertainty Quantification](#bookmark=id.709l8eg82wxh)  
5. [Data Preprocessing](#bookmark=id.r2zen06q1by8)  
6. [Explainability & Feature Selection](#bookmark=id.m235ejck4naz)  
7. [Installation](#bookmark=id.388p904x0ypy)  
8. [Usage](#bookmark=id.ynuqosnrh5lx)  
   * [Basic AutoML Workflow](#bookmark=id.rodimut9zagj)  
   * [Prediction](#bookmark=id.8o128d1v1y85)  
   * [Uncertainty Prediction](#bookmark=id.mhb0f910oyg4)  
   * [Feature Importance](#bookmark=id.tv2xtikgv1h)  
   * [Retraining with Selected Features](#bookmark=id.w1nhstw0j2lk)  
   * [Saving and Loading AutoML State](#bookmark=id.a9vrdgu5wff3)  
   * [Leaderboard](#bookmark=id.q4ccbjaigut2)  
9. [Project Structure](#bookmark=id.sl1kf75cuecq)  
10. [Contributing](#bookmark=id.x3m2ururht5l)  
11. [License](#bookmark=id.yw0ezvbnx27t)

## **1\. Introduction**

This AutoML package aims to automate significant portions of the machine learning pipeline, enabling users to quickly find optimal models and understand their predictions without extensive manual tuning. It's built with modularity and extensibility in mind, allowing easy integration of new models and techniques.

## **2\. Key Features**

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

## **3\. Supported Models**

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
    \+---------------------------+  
    | Input Layer (Features X)   |  
    \+----------+----------------+  
               |  
               V  
    \+-------------------------+  
    |  Hidden Layer 1          |  
    |  (Linear \+ BatchNorm \+ |  
    |  ReLU)                   |  
    \+----------+--------------+  
               |  
               V  
    \+-------------------------+  
    |  Hidden Layer 2          |  
    |  (Linear \+ BatchNorm \+ |  
    |  ReLU)                   |  
    \+----------+--------------+  
               |  
               V  
            ... (more hidden layers)  
               |  
               V  
    \+-------------------------+  
    |  Output Layer (Linear)   |  
    \+----------+--------------+  
               |  
               V  
    \+-------------------------+  
    | Predicted Value (Y\_pred)|  
    |   (or logits)            |  
    \+-------------------------+

* **FlexibleNeuralNetwork**:  
  * **Description**: A novel and adaptive PyTorch-based neural network that dynamically adjusts its effective hidden layer depth for *each individual input*. Instead of a fixed number of layers performing computation, it has a max\_hidden\_layers and an internal n\_predictor that determines how many of these layers should be "active" (perform computations) for a specific input. The remaining layers act as "identity layers," simply passing their input through unchanged.  
  * **Core Idea**: To achieve adaptive complexity. For simpler inputs, it can effectively use fewer layers, potentially leading to faster inference and better generalization by avoiding overfitting to noise. For more complex inputs, it can utilize its full depth.  
  * **Mechanism**:  
    1. **n\_predictor**: A small sub-network takes the input and predicts an integer n (from 1 to max\_hidden\_layers). During training, it uses Gumbel-Softmax for differentiable selection of n.  
    2. **Dynamic Forward Pass**: In the main network's forward pass, each hidden layer block checks the predicted n. If the current layer's index is among the first (max\_hidden\_layers \- n) layers, it bypasses computation (identity mapping). Otherwise, it performs its standard linear transformation, batch norm, and activation.  
  * **Diagram (Dynamic Depth with Identity Layers)**:  
                        ┌──────────────────┐  
    Input Features X ───┤ 1\. N-Predictor  ├─────┐  
                        └──────────────────┘     │  
                            Predicted 'n'        │  
                             (e.g., n=3)         │  
                               │                 │  
    \+----------------------------------------------------------------+  
    |                                                                 |  
    |   Hidden Layer Block 1 (Linear, BatchNorm, ReLU, Dropout)       |  
    |      (If n=3, this block is IDENTITY for this input)            |  
    \+----------------------------------------------------------------+  
    |                                                                 |  
    |   Hidden Layer Block 2 (Linear, BatchNorm, ReLU, Dropout)       |  
    |      (If n=3, this block is IDENTITY for this input)            |  
    \+----------------------------------------------------------------+  
    |                                                                 |  
    |   Hidden Layer Block 3 (Linear, BatchNorm, ReLU, Dropout)       |  
    |      (This block is ACTIVE for this input)                      |  
    \+----------------------------------------------------------------+  
    |                                                                 |  
    |   Hidden Layer Block 4 (Linear, BatchNorm, ReLU, Dropout)       |  
    |      (This block is ACTIVE for this input)                      |  
    \+----------------------------------------------------------------+  
    |                                                                 |  
    |   Hidden Layer Block 5 (Linear, BatchNorm, ReLU, Dropout)       |  
    |      (This block is ACTIVE for this input)                      |  
    \+--------------------------+-------------------------------------+  
                               |  
                               V  
                      \+--------------------+  
                      |   Output Layer      |  
                      \+----------+---------+  
                                 |  
                                 V  
                      \+--------------------+  
                      | Final Prediction    |  
                      \+--------------------+

### **Probabilistic Regression Models**

These models are designed for regression tasks where quantifying the uncertainty of a prediction is as important as the prediction itself. Instead of just predicting a single value, they predict an entire probability distribution.

* **ProbabilisticRegressionModel (PyTorch-based)**:  
  * **Description**: This is a composite PyTorch model that tackles regression by internally employing a classification strategy. It first conceptually discretizes the continuous target variable into a fixed number of n\_classes (bins). It then trains an internal classifier to predict which bin an input belongs to. The probabilities from this classifier are then fed into specialized regression "heads" (either separate heads per class or a single head with multiple outputs) that predict the actual continuous value. Crucially, it can be configured to learn the mean and variance of a Gaussian distribution for each prediction, directly modeling uncertainty.  
  * **Core Idea**: Leverages the strengths of classification (assigning inputs to meaningful "categories" of the target variable) to improve continuous regression prediction and provide robust uncertainty estimates.  
  * **Mechanism (Regression Heads Strategies)**:  
    1. **Target Discretization (Conceptual)**: The continuous target y is divided into n\_classes bins (e.g., via percentiles).  
    2. **Classifier Branch**: An internal PyTorchNeuralNetwork (as the base classifier) is trained to predict the probability of an input falling into each of these n\_classes.  
    3. **Regression Heads**: The predicted probabilities for each class are then fed into subsequent regression subnetworks. The strategy for these heads can vary:  
       * **RegressionStrategy.SEPARATE\_HEADS**: Creates n\_classes distinct regression subnetworks. Each subnetwork specializes in mapping the probability of its corresponding class to an expected regression value for that class. The final prediction is a weighted sum of outputs from all heads.  
         Predicted Class Probs (P0, P1, P2)  
               ┌────┬────┬────┐  
               │ P0 │ P1 │ P2 │  
               └─┬──┴─┬──┴─┬──┘  
                 │    │    │  
                 V    V    V  
         \+--------------------------------------+  
         | Reg Head 0  | Reg Head 1 | Reg Head 2 |  
         | (Maps P0)   | (Maps P1)  | (Maps P2)  |  
         \+-------------+------------+-----------+  
                 │        │        │  
                 V        V        V  
           Y\_exp\_0  Y\_exp\_1  Y\_exp\_2  
                 │        │        │  
                 └────────┼────────┘  
                          V  
         Final Prediction \= (P0 \* Y\_exp\_0) \+ (P1 \* Y\_exp\_1) \+ (P2 \* Y\_exp\_2)

       * **RegressionStrategy.SINGLE\_HEAD\_N\_OUTPUTS**: A single regression subnetwork outputs n\_classes distinct regression values (or mean/log-variance pairs). The final prediction is then a weighted sum of these n\_classes outputs, weighted by their respective class probabilities.  
         Predicted Class Probs (P0, P1, P2)  
               ┌──────────────────────┐  
               │ Combined Probs Input │  
               └───────────┬──────────┘  
                           │  
                           V  
         \+------------------------------------------+  
         | Single Regression Head                    |  
         | (Outputs Y\_exp\_0, Y\_exp\_1, Y\_exp\_2) |  
         \+-----------------┬---------------┬--------+  
                            │               │  
                            V               V  
         Final Prediction \= (P0 \* Y\_exp\_0) \+ (P1 \* Y\_exp\_1) \+ (P2 \* Y\_exp\_2)

       * **RegressionStrategy.SINGLE\_HEAD\_FINAL\_OUTPUT**: A single regression subnetwork directly outputs the final combined regression prediction (or mean/log-variance pair).  
         Predicted Class Probs (P0, P1, P2)  
               ┌──────────────────────┐  
               │ Combined Probs Input │  
               └───────────┬──────────┘  
                           │  
                           V  
         \+---------------------------------+  
         | Single Regression Head           |  
         | (Directly outputs final Y\_pred) |  
         \+---------------------------------+  
                           │  
                           V  
                 Final Prediction (Y\_pred)

    4. **Prediction Aggregation**: The outputs of these regression heads are combined (often via a weighted sum using the class probabilities) to form the final continuous prediction.  
    5. **Uncertainty**: The model can learn both the mean and the log-variance of a Gaussian distribution for its final prediction, allowing for **probabilistic uncertainty**.  
  * **Diagram (Probabilistic Regression Model Flow \- Consolidated)**:  
                       ┌─────────────────────────┐  
                       │      Input Features X   │  
                       └───────────┬─────────────┘  
                                   │  
                                   V  
                   ┌─────────────────────────┐  
                   │   Classifier Branch     │  
                   │   (PyTorchNeuralNetwork)│  
                   └───────────┬─────────────┘  
                               │  
                               V  
                   ┌────────────────────────────┐  
                   │ Predicted Class Probs      │  
                   │  (P\_0, P\_1, ..., P\_N-1) │  
                   └───────────┬────────────────┘  
                               │  
        \+----------------------------------------------------+  
        |          Regression Heads Branch                    |  
        |   (Controlled by regression\_strategy)              |  
        |                                                     |  
        |   \- SEPARATE\_HEADS: N individual heads            |  
        |   \- SINGLE\_HEAD\_N\_OUTPUTS: One head, N outputs  |  
        |   \- SINGLE\_HEAD\_FINAL\_OUTPUT: One head, 1 output|  
        \+---------------------┬------------------------------+  
                               │  
                               V  
                   ┌───────────────────────────────────┐  
                   │ Final Predicted Mean µ(X)         │  
                   │ Final Predicted LogVar log(σ²(X)) │  
                   └───────────────────────────────────┘

* **JAXProbabilisticRegressionModel**:  
  * **Description**: This model mirrors the functionality of ProbabilisticRegressionModel but is built entirely using JAX and Flax. It offers similar capabilities in internal classification and probabilistic regression heads, benefiting from JAX's high-performance numerical computation and automatic differentiation.  
  * **Core Idea**: Same as ProbabilisticRegressionModel but leveraging the JAX ecosystem for potentially faster training and inference on accelerators.

### **Composite Models**

This section covers a meta-model that combines different modeling paradigms to solve complex problems.

* **ClassifierRegressionModel**:  
  * **Description**: This is a **meta-model** (a model that uses other models internally) designed to solve **regression problems** by framing them as classification tasks. It works by discretizing the continuous target variable into a fixed number of bins (classes), training a *classifier* (which can be any BaseModel classifier like XGBoostModel, PyTorchNeuralNetwork, etc.) on these discrete classes, and then mapping the classifier's predicted probabilities back to a continuous regression output.  
  * **Core Idea**: It's particularly useful when the relationship between features and the regression target is complex or non-monotonic, or when traditional regression models struggle. By breaking the problem into classification, it can simplify the learning task.  
  * **Mechanism**:  
    1. **Target Discretization**: The continuous target variable (y) from the training data is divided into n\_classes (e.g., 5-10 bins based on percentiles). Each original y value is assigned a discrete class label.  
    2. **Base Classifier Training**: A chosen BaseModel classifier (e.g., XGBoostModel, PyTorchNeuralNetwork) is trained to predict these discrete class labels from the input features (X).  
    3. **Class Probability Mapper**: For new predictions, the trained base classifier outputs probabilities for each class. The ClassProbabilityMapper then takes these probabilities for a specific class and maps them to an *expected original regression value* for that class. This mapping is learned during fitting, often through methods like:  
       * **MapperType.LINEAR**: A simple linear regression fit between the predicted probability for a class and the actual target values associated with that class in the training data.  
         \+-------------------------+      \+----------------------------+      \+---------------------------+  
         | Predicted Probability    |      | Linear Regression Model     |      | Expected Y for Class C     |  
         | for Class C (P\_C)       \+-----\> (Trained on P\_C vs Y\_orig) \+---\> (Y\_exp\_C)               |  
         \+-------------------------+      \+----------------------------+      \+---------------------------+  
         (on training data where y is in Class C)

       * **MapperType.LOOKUP\_MEAN / LOOKUP\_MEDIAN**: Divides the probability space into bins and calculates the mean (or median) of the original target values that fall into each bin. When a new probability comes, it looks up the corresponding mean/median.  
         \+-------------------------+  
         | Predicted Probability    |  
         | for Class C (P\_C)       |  
         \+----------+--------------+  
                    |  
                    V  
         \+-----------------------------+  
         |   Binning / Lookup Table     |  
         |   (e.g., 0-0.1 \-\> Mean\_Y1)|  
         |   (0.1-0.2 \-\> Mean\_Y2)    |  
         \+-----------------------------+  
                    |  
                    V  
         \+-------------------------+  
         | Expected Y for Class C   |  
         | (Y\_exp\_C)              |  
         \+-------------------------+

       * **MapperType.SPLINE**: Fits a smooth spline interpolation function between the predicted probabilities and the actual target values for the class. This can capture non-linear relationships in the mapping more flexibly.  
         \+-------------------------+      \+----------------------------+      \+---------------------------+  
         | Predicted Probability    |      | Spline Interpolation        |      | Expected Y for Class C     |  
         | for Class C (P\_C)       \+-----\> (Fitted to P\_C vs Y\_orig) \+----\> (Y\_exp\_C)               |  
         \+-------------------------+      \+----------------------------+      \+---------------------------+  
         (on training data where y is in Class C)

    4. **Final Prediction**: The final regression prediction for an input is the expected value derived from the probabilities and the learned continuous mapping (e.g., a weighted sum of the expected values from each class, weighted by their probabilities).

## **4\. Uncertainty Quantification**

For regression models, the package supports quantifying prediction uncertainty using various methods:

* **UncertaintyMethod.CONSTANT**: The simplest method. It calculates the standard deviation of residuals from the training data, applying this constant uncertainty to all new predictions.  
  * **Supported by:** JAXLinearRegression, XGBoostModel, LightGBMModel, CatBoostModel, SKLearnLogisticRegression (can be configured to use this for general confidence for classification if treated as regression context), PyTorchNeuralNetwork, FlexibleNeuralNetwork.  
* **UncertaintyMethod.PROBABILISTIC**: The model is designed to directly learn both the mean (μ(x)) and the variance (σ2(x)) of the target variable's distribution. This captures **aleatoric uncertainty** (inherent noise in the data). The model is trained with a Negative Log-Likelihood (NLL) loss.  
  * **Supported by:** PyTorchNeuralNetwork, FlexibleNeuralNetwork, ProbabilisticRegressionModel, JAXProbabilisticRegressionModel.  
* **UncertaintyMethod.MC\_DROPOUT**: This technique approximates **epistemic uncertainty** (model's uncertainty due to limited data/knowledge). For models with dropout layers, multiple predictions are made for the same input while dropout is active (model in train() mode). The standard deviation of these multiple predictions estimates the uncertainty.  
  * **Supported by:** PyTorchNeuralNetwork, FlexibleNeuralNetwork, ProbabilisticRegressionModel, JAXProbabilisticRegressionModel.  
* **ClassifierRegressionModel Uncertainty Quantification**:  
  * **Description**: For ClassifierRegressionModel, uncertainty is estimated by considering the variance of the original target values within each probability bin, as learned by the ClassProbabilityMapper. This model treats the regression problem as a series of classification probabilities, and the uncertainty arises from the spread of true values within the ranges associated with those probabilities.  
  * **Mechanism**: For a new prediction, the model aggregates the variance contributions from each class's mapper, weighted by the square of their predicted probabilities. This approach effectively translates the uncertainty in classifying into a bin, and the inherent variability within each bin, into a total prediction uncertainty. The predict\_uncertainty method returns the standard deviation derived from this aggregated variance.  
  * Mathematical Formula:  
    The total variance for a given input X is calculated as the sum of squared probabilities multiplied by the variance contribution from each class mapper:TotalVariance(X)=c=0∑n\_classes−1​Pc​(X)2⋅VarianceContributionc​(Pc​(X))  
    Where:  
    * Pc​(X) is the probability of input X belonging to class c, predicted by the base classifier.  
    * VarianceContributionc​(Pc​(X)) is the variance predicted by the ClassProbabilityMapper for class c, given its probability Pc​(X). This variance represents the inherent spread of original y values that mapped to that probability range for class c.

  The final uncertainty (standard deviation) is the square root of this total variance:Uncertainty(X)=TotalVariance(X)​

  * **Diagram (Classifier Regression Model Flow \- Consolidated with Uncertainty)**:  
                       ┌─────────────────────────┐  
                       │      Input Features X   │  
                       └───────────┬─────────────┘  
                                   │  
                                   V  
                   ┌──────────────────────────────────────┐  
                   │   1\. Base Classifier (e.g., XGBoost)│  
                   │      (Trained on discretized y)      │  
                   └───────────┬──────────────────────────┘  
                               │  
                               V  
                   ┌────────────────────────────┐  
                   │  Predicted Class Probs     │  
                   │  (P\_0, P\_1, ..., P\_N-1) │  
                   └───────────┬────────────────┘  
                               │  
                               V  
                   ┌───────────────────────────────────────────────┐  
                   │ 2\. Class Probability                         │  
                   │    Mapper (for each P\_i)                     │  
                   │    (Maps P\_i to Y\_i\_expected)              │  
                   │    (and maps P\_i to VarianceContribution\_i) │  
                   └───────────┬───────────────────────────────────┘  
                               │  
                               V  
                   ┌─────────────────────────┐  
                   │  Final Regression Value │  
                   │  (and Uncertainty)      │  
                   └─────────────────────────┘

## **5\. Data Preprocessing**

The AutoML class seamlessly integrates common preprocessing steps:

* **Feature Scaling (feature\_scaler):** An optional scikit-learn compatible scaler (e.g., StandardScaler) can be provided during AutoML initialization. If present, features will be automatically scaled before training and prediction.  
* **Target Scaling (target\_scaler):** For regression tasks, an optional target scaler (e.g., StandardScaler) can be provided. This scales the target variable before training and inverse-transforms predictions back to the original scale. This helps models like neural networks converge better.

## **6\. Explainability & Feature Selection**

The package utilizes [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/index.html) to provide insights into model predictions:

* **FeatureExplainer:** A wrapper class that adapts SHAP explainers (TreeExplainer, DeepExplainer, KernelExplainer) to different model types within the AutoML pipeline.  
* **Feature Importance:** Calculates global feature importance based on the mean absolute SHAP values, showing which features contribute most to model predictions on average.  
* **Automated Feature Selection:** The select\_features\_by\_cumulative\_importance method can identify a subset of most important features based on a cumulative SHAP importance threshold, allowing for model retraining on reduced feature sets.

### **How Most Important Features are Selected (using SHAP)**

The AutoML package uses SHAP (SHapley Additive exPlanations) to identify and select the most important features based on a user-defined threshold. This process helps simplify models, improve interpretability, and potentially boost performance by reducing noise.

1. **Calculate Raw Feature Importances (SHAP Values)**:  
   * The AutoML.get\_feature\_importance(X\_test, feature\_names) method is called.  
   * Internally, a FeatureExplainer is initialized for the currently best-performing model. This explainer (e.g., shap.TreeExplainer, shap.DeepExplainer, shap.KernelExplainer) is chosen based on the model type.  
   * SHAP values are computed for a subset of the test data (X\_test). SHAP values represent the contribution of each feature to the model's output for each prediction.  
   * For each feature, the *mean absolute SHAP value* across all explained samples is calculated. This value represents the global importance of that feature to the model's predictions.  
2. **Normalize Importances**:  
   * The mean absolute SHAP values are then normalized by dividing each feature's importance by the sum of all feature importances. This gives a percentage-like contribution for each feature, summing up to 1 (or 100%).  
3. **Cumulative Importance Thresholding**:  
   * Features are sorted in descending order based on their normalized importance.  
   * The algorithm then iterates through these sorted features, accumulating their normalized importance.  
   * It continues to select features until the cumulative\_importance reaches or exceeds a user-defined threshold (e.g., 0.95 for 95% of the total importance). All features contributing to this cumulative sum are selected.

**Diagram: Feature Selection by Cumulative SHAP Importance**Raw Features (X)  
\+-------------------+  
| F1, F2, F3, F4, F5 |  
\+-------------------+  
        │  
        V  
\+-----------------+  
|  Trained Model   |  
\+-----------------+  
        │  
        V  
\+------------------------+  
| SHAP Explainer          |  
| (calculates SHAP values)|  
\+------------------------+  
        │  
        V  
\+----------------------------------------------+  
| Mean Absolute SHAP Values (Raw Importance)    |  
| F1: 0.5, F2: 0.3, F3: 0.1, F4: 0.05, F5: 0.05 |  
\+----------------------------------------------+  
        │  
        V  
\+--------------------------------------------+  
| Normalized & Sorted Importance (Cumulative) |  
| F1: 0.50 (0.50)                             |  \<-- Select (Threshold 0.95)  
| F2: 0.30 (0.80)                             |  \<-- Select  
| F3: 0.10 (0.90)                             |  \<-- Select  
| F4: 0.05 (0.95)                             |  \<-- Select  
| F5: 0.05 (1.00)                             |  \<-- Stop here if next adds over 0.95  
\+--------------------------------------------+  
        │  
        V  
\+-------------------------+  
| Selected Features:       |  
| \[F1, F2, F3, F4\]       |  
\+-------------------------+

### **Retraining with Selected Features**

After identifying a subset of the most important features, the AutoML package provides a method to retrain the best-performing model using only these selected features. This process can lead to:

* **Simpler Models**: Fewer features mean a less complex model.  
* **Improved Generalization**: By removing irrelevant or noisy features, the model might generalize better to unseen data.  
* **Faster Training and Inference**: Reduced dimensionality can speed up both training and prediction times.  
* **Enhanced Interpretability**: Focusing on a smaller set of key features makes the model's decisions easier to understand.

**Mechanism**:

1. **Feature Selection**: The select\_features\_by\_cumulative\_importance method (as explained above) is called to determine the names of the most important features based on a SHAP threshold.  
2. **Dataset Filtering**: The original training and testing datasets are filtered to include only the columns corresponding to the selected\_feature\_names.  
3. **Scaler Application**: If a feature\_scaler was used in the initial AutoML training, it is *re-used* (already fitted) to transform these filtered feature sets. If a target\_scaler was used, the target data is also scaled (or inverse-transformed for predictions) accordingly.  
4. **Model Re-instantiation**: The best-performing model from the initial AutoML run is re-instantiated with its previously optimized hyperparameters. Crucially, its input\_size (for neural networks) is adjusted to match the number of selected features.  
5. **Retraining**: The re-instantiated model is then trained from scratch using the filtered and scaled training data.  
6. **Re-evaluation**: The retrained model's performance is evaluated on the filtered and scaled test data (with predictions denormalized if target scaling was applied), providing a new metric score for the reduced feature set.

This retraining step ensures that the final deployed model is optimized not just for its hyperparameters but also for its input feature set.

## **7\. Installation**

To use the AutoML package, you'll need Python 3.8+ and install the necessary dependencies:

pip install numpy pandas scikit-learn optuna jax jaxlib\[cuda\] \# or jaxlib for CPU only  
pip install torch torchvision torchaudio \# Install PyTorch based on your CUDA version: https://pytorch.org/get-started/locally/  
pip install xgboost lightgbm catboost flax optax shap

**Note:**

* jaxlib\[cuda\] will install CUDA-enabled JAX, otherwise pip install jaxlib for CPU.  
* Install PyTorch from their official website (pytorch.org/get-started/locally/) for the version compatible with your CUDA setup. pip install torch is a CPU-only version.  
* flax and optax are for JAX models.  
* shap is for explainability.

**Using a Virtual Environment (Recommended)**

It's highly recommended to use a virtual environment to manage your project's dependencies and avoid conflicts with other Python projects.

1. **Create a requirements.txt file:** In the root of your project directory, create a file named requirements.txt and paste the following content into it:  
   numpy  
   pandas  
   scikit-learn  
   optuna  
   \# For JAX (choose one based on your hardware)  
   jax  
   \# jaxlib==0.4.26+cuda12.cudnn89 \# Example for specific CUDA version  
   \# jaxlib==0.4.26 \# For CPU only

   \# For PyTorch (choose one based on your CUDA version)  
   \# torch==2.3.1+cu121 \--index-url https://download.pytorch.org/whl/cu121 \# Example for specific CUDA version  
   \# torch==2.3.1 \# For CPU only  
   torchvision  
   torchaudio

   xgboost  
   lightgbm  
   catboost  
   flax  
   optax  
   shap

   * **Important:** Uncomment and adjust the jaxlib and torch lines based on your specific CUDA version or if you're using a CPU-only setup. Refer to the official JAX and PyTorch installation guides for exact commands.  
2. **Create a virtual environment:**  
   python \-m venv venv\_automl

   (You can replace venv\_automl with your preferred environment name.)  
3. **Activate the virtual environment:**  
   * **On Windows:**  
     .\\venv\_automl\\Scripts\\activate

   * **On macOS/Linux:**  
     source venv\_automl/bin/activate

4. Install dependencies from requirements.txt:  
   Once your virtual environment is active, install all libraries:  
   pip install \-r requirements.txt

## **8\. Usage**

### **Basic AutoML Workflow**

Here's a basic example demonstrating how to train and evaluate models using the AutoML orchestrator:

import numpy as np  
from sklearn.datasets import make\_regression, make\_classification  
from sklearn.model\_selection import train\_test\_split  
from sklearn.preprocessing import StandardScaler

\# Assuming automl\_package is installed and its modules are importable  
from automl\_package.automl import AutoML  
from automl\_package.enums import TaskType, ModelName, UncertaintyMethod \# Import necessary enums

import logging  
logger \= logging.getLogger('automl\_package')  
logger.setLevel(logging.INFO)  
if not logger.handlers:  
    ch \= logging.StreamHandler()  
    formatter \= logging.Formatter('%(asctime)s \- %(name)s \- %(levelname)s \- %(message)s')  
    ch.setFormatter(formatter)  
    logger.addHandler(ch)

print("===== Starting AutoML Package Demonstration \=====")

\# \--- Regression Example \---  
logger.info("\\n--- Running Regression Example with Scaling and Uncertainty \---")  
X\_reg, y\_reg \= make\_regression(n\_samples=200, n\_features=10, noise=10.0, random\_state=42)  
y\_reg \= y\_reg \+ abs(y\_reg.min()) \+ 1 \# Ensure y is positive for certain percentile calcs

\# Split data for initial training (for AutoML's CV) and final test evaluation  
X\_full\_reg, X\_test\_full\_reg, y\_full\_reg, y\_test\_full\_reg \= train\_test\_split(X\_reg, y\_reg, test\_size=0.2, random\_state=42)  
X\_train\_initial\_reg, y\_train\_initial\_reg \= X\_full\_reg, y\_full\_reg

feature\_names\_reg \= \[f'feature\_{i}' for i in range(X\_reg.shape\[1\])\]

\# Instantiate scalers  
X\_scaler\_reg \= StandardScaler()  
y\_scaler\_reg \= StandardScaler()

automl\_reg \= AutoML(task\_type=TaskType.REGRESSION, metric='rmse', n\_trials=3, n\_splits=2, random\_state=42,  
                     feature\_scaler=X\_scaler\_reg, target\_scaler=y\_scaler\_reg)

\# Specify which models to consider for this regression run  
\# Including the new FlexibleNeuralNetwork  
models\_for\_reg \= \[  
    ModelName.JAX\_LINEAR\_REGRESSION,  
    ModelName.PYTORCH\_NEURAL\_NETWORK,  
    ModelName.FLEXIBLE\_NEURAL\_NETWORK, \# NEW MODEL INCLUDED  
    ModelName.XGBOOST,  
    ModelName.LIGHTGBM,  
    ModelName.CATBOOST,  
    ModelName.CLASSIFIER\_REGRESSION, \# Only for regression where classification informs regression  
    ModelName.PROBABILISTIC\_REGRESSION,  
    ModelName.JAX\_PROBABILISTIC\_REGRESSION  
\]  
\# Filter out models if their dependencies are not installed  
try:  
    import xgboost as xgb  
    import lightgbm as lgb  
    from catboost import CatBoostRegressor, CatBoostClassifier  
except ImportError:  
    models\_for\_reg \= \[m for m in models\_for\_reg if m not in \[ModelName.XGBOOST, ModelName.LIGHTGBM, ModelName.CATBOOST\]\]

automl\_reg.train(X\_train\_initial\_reg, y\_train\_initial\_reg, models\_to\_consider=models\_for\_reg)

### **Prediction**

if automl\_reg.best\_model\_name:  
    logger.info(f"\\n--- Making Predictions with Best Regression Model ({automl\_reg.best\_model\_name.value}) \---")  
    y\_pred\_test \= automl\_reg.predict(X\_test\_full\_reg)  
    test\_rmse \= np.sqrt(mean\_squared\_error(y\_test\_full\_reg, y\_pred\_test))  
    logger.info(f"Best model test RMSE (original scale): {test\_rmse:.4f}")  
    logger.info(f"Sample predictions (first 5, original scale): {y\_pred\_test\[:5\].round(2)}")

### **Uncertainty Prediction**

    logger.info(f"\\n--- Predicting Uncertainty with Best Regression Model ({automl\_reg.best\_model\_name.value}) \---")  
    try:  
        uncertainty\_values \= automl\_reg.predict\_uncertainty(X\_test\_full\_reg)  
        logger.info(f"Mean uncertainty estimate (original scale): {np.mean(uncertainty\_values):.4f}")  
        logger.info(f"Uncertainty estimates (first 5, original scale): {uncertainty\_values\[:5\].round(2)}")  
    except ValueError as e:  
        logger.error(f"Could not get uncertainty estimates for model {automl\_reg.best\_model\_name.value}: {e}")  
    except NotImplementedError as e:  
        logger.error(f"Uncertainty prediction not implemented for {automl\_reg.best\_model\_name.value}: {e}")

### **Feature Importance**

    logger.info(f"\\n--- Getting Feature Importance for Best Regression Model ({automl\_reg.best\_model\_name.value}) \---")  
    feature\_importance\_summary \= automl\_reg.get\_feature\_importance(X\_test\_full\_reg, feature\_names=feature\_names\_reg)  
    if "error" not in feature\_importance\_summary:  
        logger.info(f"Top 5 Features by SHAP Importance:\\n{json.dumps(dict(list(feature\_importance\_summary.items())\[:5\]), indent=2)}")

### **Retraining with Selected Features**

    logger.info("\\n--- Retraining with Selected Features (Regression) \---")  
    retrained\_results\_reg \= automl\_reg.retrain\_with\_selected\_features(  
        X\_full\_train=X\_full\_reg, y\_full\_train=y\_full\_reg,  
        X\_full\_test=X\_test\_full\_reg, y\_full\_test=y\_test\_full\_reg,  
        feature\_names=feature\_names\_reg, shap\_threshold=0.95  
    )  
    logger.info(f"Retrained model test RMSE (with selected features, original scale): {retrained\_results\_reg\['retrained\_metric\_value'\]:.4f}")  
    logger.info(f"Selected features: {retrained\_results\_reg\['selected\_feature\_names'\]}")

**Part 3 of 3: Remaining Usage Examples (Classification, Save/Load)**

\# \--- Classification Example \---  
logger.info("\\n\\n===== Running Classification AutoML Example \=====")  
X\_clf, y\_clf \= make\_classification(n\_samples=500, n\_features=10, n\_classes=2, random\_state=42)

X\_full\_clf, X\_test\_full\_clf, y\_full\_clf, y\_test\_full\_clf \= train\_test\_split(X\_clf, y\_clf, test\_size=0.2, random\_state=42)  
X\_train\_initial\_clf, y\_train\_initial\_clf \= X\_full\_clf, y\_full\_clf

feature\_names\_clf \= \[f'feature\_{i}' for i in range(X\_clf.shape\[1\])\]  
X\_scaler\_clf \= StandardScaler()

automl\_clf \= AutoML(task\_type=TaskType.CLASSIFICATION, metric='accuracy', n\_trials=3, n\_splits=2, random\_state=42,  
                     feature\_scaler=X\_scaler\_clf)

models\_for\_clf \= \[  
    ModelName.SKLEARN\_LOGISTIC\_REGRESSION,  
    ModelName.PYTORCH\_NEURAL\_NETWORK,  
    ModelName.FLEXIBLE\_NEURAL\_NETWORK, \# NEW MODEL INCLUDED  
    ModelName.XGBOOST,  
    ModelName.LIGHTGBM,  
    ModelName.CATBOOST,  
\]  
\# Filter out models if their dependencies are not installed  
try:  
    import xgboost as xgb  
    import lightgbm as lgb  
    from catboost import CatBoostRegressor, CatBoostClassifier  
except ImportError:  
    models\_for\_clf \= \[m for m in models\_for\_clf if m not in \[ModelName.XGBOOST, ModelName.LIGHTGBM, ModelName.CATBOOST\]\]

automl\_clf.train(X\_train\_initial\_clf, y\_train\_initial\_clf, models\_to\_consider=models\_for\_clf)

logger.info(f"\\n--- Making Predictions with Best Classification Model ({automl\_clf.best\_model\_name.value}) \---")  
y\_pred\_test\_clf \= automl\_clf.predict(X\_test\_full\_clf)  
test\_accuracy\_clf \= accuracy\_score(y\_test\_full\_clf, y\_pred\_test\_clf)  
logger.info(f"Best model test accuracy: {test\_accuracy\_clf:.4f}")  
logger.info(f"Sample predictions (first 5): {y\_pred\_test\_clf\[:5\].round(2)}")

\# Classification uncertainty: Predict\_uncertainty is designed for regression std\_dev, will raise error.  
\# You would typically inspect predict\_proba for classification uncertainty.  
\# try:  
\#     uncertainty\_clf \= automl\_clf.predict\_uncertainty(X\_test\_full\_clf)  
\# \#     logger.info(f"Mean uncertainty estimate for classification: {np.mean(uncertainty\_clf):.4f}")  
\# \# except ValueError as e:  
\# \#     logger.warning(f"As expected, cannot get uncertainty for classification task directly: {e}")  
\# \# except NotImplementedError as e:  
\# \#     logger.error(f"Uncertainty prediction not implemented for {automl\_clf.best\_model\_name.value}: {e}")

logger.info(f"\\n--- Getting Feature Importance for Best Classification Model ({automl\_clf.best\_model\_name.value}) \---")  
feature\_importance\_summary\_clf \= automl\_clf.get\_feature\_importance(X\_test\_full\_clf, feature\_names=feature\_names\_clf)  
if "error" not in feature\_importance\_summary\_clf:  
    logger.info(f"Top 5 Features by SHAP Importance:\\n{json.dumps(dict(list(feature\_importance\_summary\_clf.items())\[:5\]), indent=2)}")

logger.info("\\n--- Retraining with Selected Features (Classification) \---")  
retrained\_results\_clf \= automl\_clf.retrain\_with\_selected\_features(  
    X\_full\_train=X\_full\_clf, y\_full\_train=y\_full\_clf,  
    X\_full\_test=X\_test\_full\_clf, y\_full\_test=y\_test\_full\_clf,  
    feature\_names=feature\_names\_clf, shap\_threshold=0.95  
)  
logger.info(f"Retrained model test accuracy (with selected features): {retrained\_results\_clf\['retrained\_metric\_value'\]:.4f}")  
logger.info(f"Selected features: {retrained\_results\_clf\['selected\_feature\_names'\]}")

\# \--- Saving and Loading AutoML state example \---  
logger.info("\\n--- Saving and Loading AutoML State \---")  
save\_path\_reg\_automl \= "automl\_reg\_state.joblib"  
automl\_reg.save\_automl\_state(save\_path\_reg\_automl)  
loaded\_automl\_reg \= AutoML.load\_automl\_state(save\_path\_reg\_automl)

if loaded\_automl\_reg:  
    logger.info(f"Loaded AutoML best regression model: {loaded\_automl\_reg.get\_best\_model\_info()\['name'\]}")  
    \# You can now use loaded\_automl\_reg to predict on new data.  
    \# loaded\_predictions\_reg \= loaded\_automl\_reg.predict(X\_test\_full\_reg\[:5\])  
    \# logger.info(f"Predictions from loaded regression model (first 5): {loaded\_predictions\_reg.flatten().round(2)}")

save\_path\_clf\_automl \= "automl\_clf\_state.joblib"  
automl\_clf.save\_automl\_state(save\_path\_clf\_automl)  
loaded\_automl\_clf \= AutoML.load\_automl\_state(save\_path\_clf\_automl)

if loaded\_automl\_clf:  
    logger.info(f"Loaded AutoML best classification model: {loaded\_automl\_clf.get\_best\_model\_info()\['name'\]}")  
    \# You can now use loaded\_automl\_clf to predict on new data.  
    \# loaded\_predictions\_clf \= loaded\_automl\_clf.predict(X\_test\_full\_clf\[:5\])  
    \# logger.info(f"Predictions from loaded classification model (first 5): {loaded\_predictions\_clf.flatten().round(2)}")

logger.info("\\n===== AutoML Package Demonstration Complete \=====")

