# Deep Bayesian Active Learning for Image Regression

This repository contains code for experiments performed during the making of the report "Deep Bayesian Active Learning for Image Regression" conducted as part of the Uncertainty in Deep Learning final exam.

# How to use

- Each `experimentX_notebook.ipynb` notebook is accompanied by an `experimentX.py` script. Since the available compute for this exam was on Google Colab, only the notebooks were continuously tested on, and used to obtain the final results.
- The notebooks are self-contained and do not call helper functions and do not require any installs on the Colab environment. Experimients were conducted on A100 GPUs. Feasibility of running experiments on CPU was not checked.
- `experimentX.py` scripts call helper functions from the `replication`, `min_extension` or `novel_extension` folders.
- `pyproject.toml` is the configuration file needed by uv to create an environment. It is human-readable and can be used to manually set up an environment for this repository
- `plots` folder has a copy of the plots in the report.

# Experiment Descriptions

1. Experiment 1: Replication of Section 5.1 from the Deep Bayesian Active Learning for Image Data paper.
2. Experiment 2: Replication of Section 5.2 from the Deep Bayesian Active Learning for Image Data paper.
3. Experiment 3: Minimal extension - Extend the framework towards regression on MNIST, with one-hot encoded labels as target
4. Experiment 4: Novel extension - Use EfficientNet-B0 as a backbone to evaluate framework effectiveness on a real-world dataset - the Biwi head pose dataset.
