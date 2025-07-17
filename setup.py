from setuptools import setup, find_packages

setup(
    name='automl_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'optuna',
        'joblib',
        'torch',
        'matplotlib',
        'seaborn',
        'jax',
        'jaxlib',
        'catboost',
        'xgboost',
        'lightgbm',
        'scipy',
        'shap',
        'bokeh',
        'wandb'
    ],
    python_requires='>=3.9',
)
