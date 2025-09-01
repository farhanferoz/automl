"""Setup script for the automl_package."""

from setuptools import find_packages, setup

setup(
    name="automl_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==2.2.6",
        "pandas",
        "scikit-learn",
        "optuna==4.5.0",
        "joblib",
        "torch==2.6.0+cu124",
        "matplotlib==3.10.5",
        "seaborn",
        "jax==0.7.1",
        "jaxlib==0.7.1",
        "catboost",
        "xgboost",
        "lightgbm",
        "scipy",
        "shap",
        "bokeh",
        "wandb==0.21.2",
    ],
    python_requires=">=3.9",
)
