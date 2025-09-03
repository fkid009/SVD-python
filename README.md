# Low-Level SVD Implementation with NumPy

A pure NumPy implementation of Singular Value Decomposition (SVD) from scratch, built for educational purposes to understand the mathematical foundations of SVD and eigenvalue decomposition.

## Overview

This project implements SVD using only basic NumPy operations, avoiding high-level linear algebra functions like `numpy.linalg.svd` or `numpy.linalg.eigh`. The implementation follows the mathematical derivation step-by-step:

**SVD: A = UΣV^T**

Where:
- **U**: Left singular vectors (m × min(m,n))
- **Σ**: Diagonal matrix of singular values (min(m,n) × min(m,n))  
- **V^T**: Right singular vectors transposed (n × n)

## Mathematical Approach

### Step 1: Find V (Right Singular Vectors) and Singular Values
1. **Compute A^T A**: This creates a symmetric positive semi-definite matrix
2. **Find eigenvalues and eigenvectors of A^T A**:
   - Eigenvalues of A^T A = σᵢ² (squared singular values)
   - Eigenvectors of A^T A = columns of V matrix
3. **Sort by descending eigenvalue magnitude**

### Step 2: Find U (Left Singular Vectors)  
1. **Use relationship**: uᵢ = Avᵢ/σᵢ for each non-zero singular value
2. **Orthogonalize remaining vectors** using Gram-Schmidt process

### Step 3: Assemble Final SVD
- Construct U, Σ, V^T matrices from computed components

## Implementation Features

### Eigenvalue/Eigenvector Computation
- **Power Iteration**: Finds dominant eigenvalue and eigenvector iteratively
- **Matrix Deflation**: Removes found eigenvalues to discover subsequent ones
- **Gram-Schmidt Orthogonalization**: Ensures orthogonal basis vectors

### SVD Functions
```python
# Main SVD function
U, Sigma, Vt = svd_decomposition(A)

# Individual components  
eigenvals, eigenvecs = compute_eigenvalues_eigenvectors(symmetric_matrix)
singular_values, V = compute_singular_values_and_V(A)
U = compute_U_matrix(A, V, singular_values)
```

## Usage

```python
import numpy as np
from svd_ import svd_decomposition

# Example matrix
A = np.array([[1, 2, 3],
              [4, 5, 6], 
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

# Perform SVD decomposition
U, Sigma, Vt = svd_decomposition(A)

# Verify reconstruction
A_reconstructed = U @ Sigma @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")
```

## Function Reference

### Core SVD Functions

| Function | Description |
|----------|-------------|
| `svd_decomposition(A)` | Main SVD function returning U, Σ, V^T |
| `compute_singular_values_and_V(A)` | Computes singular values and V matrix from A^T A |
| `compute_U_matrix(A, V, singular_values)` | Computes U matrix using uᵢ = Avᵢ/σᵢ |
| `create_sigma_matrix(singular_values, m, n)` | Creates diagonal Σ matrix |

### Eigenvalue Functions

| Function | Description |
|----------|-------------|
| `compute_eigenvalues_eigenvectors(M)` | Full eigendecomposition of symmetric matrix |
| `power_iteration(A)` | Finds dominant eigenvalue/eigenvector |
| `matrix_deflation(A, eigenval, eigenvec)` | Removes eigenvalue from matrix |
| `gram_schmidt_orthogonalize(vector, existing_vectors)` | Orthogonalizes vector against existing ones |

### Parameters

- `max_iter`: Maximum iterations for power iteration (default: 1000)
- `tol`: Convergence tolerance (default: 1e-10)


---

**Note**: This implementation is designed for learning and understanding SVD fundamentals. For production use, prefer optimized libraries like NumPy, SciPy, or specialized linear algebra libraries.