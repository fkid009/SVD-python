import numpy as np
from typing import Tuple

def power_iteration(
    A: np.ndarray,
    num_iter: int = 100,
    tol: float = 1e-10
) -> Tuple[float, np.ndarray]:
    """
    Power iteration method to find the dominant eigenvalue and eigenvector.
    
    Args:
        A: Square matrix
        num_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (dominant eigenvalue, dominant eigenvector)
    """

    n = A.shape[0]
    # Random initialization
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(num_iter):
        # A * x
        y = np.dot(A, x)
        # Normalize
        x_new = y / np.linalg.norm(y)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
    
    # Calculate eigenvalue: λ = x^T * A * x
    eigenvalue = np.dot(x, np.dot(A, x))
    return eigenvalue, x

def matrix_deflation(
    A: np.ndarray, 
    eigenvalue: float, 
    eigenvector: np.ndarray
) -> np.ndarray:
    """
    Matrix deflation: Remove the found eigenvalue/eigenvector from the matrix.
    
    Args:
        A: Original matrix
        eigenvalue: Found eigenvalue
        eigenvector: Corresponding eigenvector
        
    Returns:
        Deflated matrix
    """
    # A' = A - λ * v * v^T (for symmetric matrices)
    outer_product = np.outer(eigenvector, eigenvector)
    return A - eigenvalue * outer_product

def gram_schmidt_orthogonalize(
    vector: np.ndarray, 
    existing_vectors: list
) -> np.ndarray:
    """
    Orthogonalize a vector against a list of existing orthogonal vectors.
    
    Args:
        vector: Vector to orthogonalize
        existing_vectors: List of existing orthogonal vectors
        
    Returns:
        Orthogonalized and normalized vector
    """
    orthogonal_vec = vector.copy()
    
    # Make orthogonal to existing vectors
    for existing_vec in existing_vectors:
        projection = np.dot(orthogonal_vec, existing_vec) * existing_vec
        orthogonal_vec = orthogonal_vec - projection
    
    # Normalize
    norm = np.linalg.norm(orthogonal_vec)
    if norm > 1e-10:
        return orthogonal_vec / norm
    else:
        return np.zeros_like(orthogonal_vec)
    
def compute_eigenvalues_eigenvectors(
    M: np.ndarray, 
    max_iter: int = 1000, 
    tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix using power iteration.
    
    Args:
        M: Symmetric matrix
        max_iterations: Maximum iterations for power iteration
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    n = M.shape[0]
    eigenvalues = []
    eigenvectors = []
    
    # Copy matrix for deflation
    A = M.copy()
    
    for i in range(n):
        try:
            # Find dominant eigenvalue and eigenvector
            eigenval, eigenvec = power_iteration(A, max_iter, tol)
            
            # Skip if eigenvalue is too small (numerical zero)
            if abs(eigenval) < tol:
                break
            
            eigenvalues.append(eigenval)
            eigenvectors.append(eigenvec)
            
            # Deflate matrix to find next eigenvalue
            A = matrix_deflation(A, eigenval, eigenvec)
            
        except:
            break
    
    # Handle remaining small eigenvalues by filling with zeros
    while len(eigenvalues) < n:
        eigenvalues.append(0.0)
        # Create orthogonal vector for zero eigenvalue
        vec = np.random.randn(n)
        vec = gram_schmidt_orthogonalize(vec, eigenvectors)
        eigenvectors.append(vec)
    
    return np.array(eigenvalues), np.column_stack(eigenvectors)