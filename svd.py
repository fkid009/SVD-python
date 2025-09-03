import numpy as np
from  typing import Tuple

from eigen import *


def compute_singular_values_and_V(
    A: np.ndarray, 
    max_iter: int = 1000, 
    tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute singular values and V matrix by finding eigenvalues/eigenvectors of A^T * A.
    
    Args:
        A: Input matrix (m x n)
        max_iterations: Maximum iterations for eigenvalue computation
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (singular_values, V_matrix)
    """
    # Step 1: Compute A^T * A
    AtA = np.dot(A.T, A)  # n x n matrix
    
    # Step 2: Find eigenvalues and eigenvectors of A^T * A
    eigenvalues, V = compute_eigenvalues_eigenvectors(AtA, max_iter, tol)
    
    # Step 3: Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    # Step 4: Compute singular values (σ = √λ)
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))  # Avoid negative due to numerical errors
    
    return singular_values, V

def compute_U_matrix(
    A: np.ndarray, 
    V: np.ndarray, 
    singular_values: np.ndarray, 
    tol: float = 1e-10
) -> np.ndarray:
    """
    Compute U matrix using the relationship: u_i = A * v_i / σ_i
    
    Args:
        A: Input matrix (m x n)
        V: Right singular vectors matrix
        singular_values: Singular values
        tolerance: Tolerance for numerical zero
        
    Returns:
        U matrix (left singular vectors)
    """
    m, n = A.shape
    U = np.zeros((m, min(m, n)))
    
    # Find rank (number of non-zero singular values)
    rank = np.sum(singular_values > tol)
    
    # Compute U columns for non-zero singular values
    for i in range(rank):
        if singular_values[i] > tol:
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]
    
    # Fill remaining columns of U with orthogonal vectors (if needed)
    if m > rank:
        existing_columns = [U[:, j] for j in range(rank)]
        for i in range(rank, min(m, n)):
            # Generate random vector and orthogonalize
            vec = np.random.randn(m)
            vec = gram_schmidt_orthogonalize(vec, existing_columns)
            U[:, i] = vec
            existing_columns.append(vec)
    
    return U

def create_sigma_matrix(
    singular_values: np.ndarray, 
    m: int,
    n: int
) -> np.ndarray:
    """
    Create the diagonal Sigma matrix from singular values.
    
    Args:
        singular_values: Array of singular values
        m: Number of rows in original matrix
        n: Number of columns in original matrix
        
    Returns:
        Sigma matrix (m x n)
    """
    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        if i < len(singular_values):
            Sigma[i, i] = singular_values[i]
    return Sigma

def svd_decomposition(
    A: np.ndarray, 
    max_iter: int = 1000, 
    tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SVD: A = U * Σ * V^T
    
    Args:
        A: Input matrix (m x n)
        max_iterations: Maximum iterations for eigenvalue computation
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (U, Σ, V^T)
    """
    m, n = A.shape
    
    # Step 1: Compute singular values and V matrix
    singular_values, V = compute_singular_values_and_V(A, max_iter, tol)
    
    # Step 2: Compute U matrix
    U = compute_U_matrix(A, V, singular_values, tol)
    
    # Step 3: Create Sigma matrix
    Sigma = create_sigma_matrix(singular_values, m, n)
    
    return U, Sigma, V.T