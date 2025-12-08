# Tensor Decomposition Algorithms

Implementation and comparison of three CP (CANDECOMP/PARAFAC) tensor decomposition algorithms: Jennrich's algorithm, CP-ALS, and Ortho-ALS.

## Algorithms

- **Jennrich's Algorithm**: Spectral method using generalized eigenvalue decomposition
- **CP-ALS**: Standard Alternating Least Squares
- **Ortho-ALS**: ALS with QR orthogonalization after each update

## Usage

See [run.ipynb](run.ipynb) for example usage.

## Experiments

- **Experiment 1**: Bad conditioning test (varying `rho`)
- **Experiment 2**: Noise sensitivity (varying `sigma`)
- **Experiment 3**: Initialization sensitivity

Run with `experiment1()`, `experiment2()`, `experiment3()` from `experiments.py`.
