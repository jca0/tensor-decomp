import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import reconstruct_tensor, fro_norm, factor_recovery_error

def generate_data_cp(d, k, sigma, rho=0):
    """
    Generate tensor data from CP decomposition.
    """
    A = np.random.randn(d, k)
    B = np.random.randn(d, k)
    if rho == 0:
        C = np.random.randn(d, k)
    else:
        Sigma = rho * np.ones((k, k)) + (1 - rho) * np.eye(k)
        L = np.linalg.cholesky(Sigma)
        C = np.random.randn(d, k) @ L
    norm = lambda X: np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-10)
    A = A / norm(A)
    B = B / norm(B)
    C = C / norm(C)
    T = reconstruct_tensor(A, B, C)
    E = np.random.randn(d, d, d)
    if fro_norm(T) > 0:
        E = E / fro_norm(E) * fro_norm(T) * sigma
    T_noisy = T + E
    return T, T_noisy, A, B, C

def generate_data_gmm(d=20, k=3, sigma=0.1, rho=0, n_samples=10000, var_scale=0.01):
    """
    Generate tensor data from Gaussian Mixture Model (GMM) decomposition.
    """
    if rho == 0:
        means = np.random.randn(d, k)
    else:
        Sigma = rho * np.ones((k, k)) + (1 - rho) * np.eye(k)
        L = np.linalg.cholesky(Sigma)
        means = np.random.randn(d, k) @ L
    norm = lambda X: np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-10)
    means = means / norm(means)
    
    weights = np.ones(k) / k
    covariances = [var_scale * np.eye(d) for _ in range(k)]
    
    z = np.random.choice(k, size=n_samples, p=weights)
    X = np.zeros((n_samples, d))
    for i in range(n_samples):
        X[i] = np.random.multivariate_normal(means[:, z[i]], covariances[z[i]])
    
    T_noisy = np.zeros((d, d, d))
    for i in range(n_samples):
        x = X[i]
        T_noisy += np.einsum('p,q,r->pqr', x, x, x) / n_samples
    
    T = np.zeros((d, d, d))
    for j in range(k):
        mu = means[:, j]
        T += weights[j] * np.einsum('p,q,r->pqr', mu, mu, mu)
    
    E = np.random.randn(d, d, d)
    if fro_norm(T) > 0:
        E = E / fro_norm(E) * fro_norm(T) * sigma
    T_noisy += E
    
    return T, T_noisy, means, means, means


def generate_data_hmm(d=20, k=3, sigma=0.1, rho=0.5, n_steps=100000, var_scale=0.01):
    """
    Generate tensor data from Hidden Markov Model (HMM) decomposition.
    """
    means = np.random.randn(d, k)
    norm = lambda X: np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-10)
    means = means / norm(means)
    
    off_diag = (1 - rho) / (k - 1) if k > 1 else 0
    trans = off_diag * np.ones((k, k))
    np.fill_diagonal(trans, rho)
    
    init_prob = np.ones(k) / k
    covariances = [var_scale * np.eye(d) for _ in range(k)]
    
    states = np.zeros(n_steps, dtype=int)
    states[0] = np.random.choice(k, p=init_prob)
    for t in range(1, n_steps):
        states[t] = np.random.choice(k, p=trans[states[t-1]])
    
    X = np.zeros((n_steps, d))
    for t in range(n_steps):
        X[t] = np.random.multivariate_normal(means[:, states[t]], covariances[states[t]])
    
    T_noisy = np.zeros((d, d, d))
    for t in range(n_steps - 2):
        T_noisy += np.einsum('p,q,r->pqr', X[t], X[t+1], X[t+2]) / (n_steps - 2)
    
    eigvals, eigvecs = np.linalg.eig(trans.T)
    stationary = eigvecs[:, np.argmin(np.abs(eigvals - 1))].real
    pi = stationary / stationary.sum()
    
    T = np.zeros((d, d, d))
    for i in range(k):
        for j in range(k):
            for l in range(k):
                contrib = pi[i] * trans[i, j] * trans[j, l]
                T += contrib * np.einsum('p,q,r->pqr', means[:, i], means[:, j], means[:, l])
    
    E = np.random.randn(d, d, d)
    if fro_norm(T) > 0:
        E = E / fro_norm(E) * fro_norm(T) * sigma
    T_noisy += E
    
    return T, T_noisy, means, means, means