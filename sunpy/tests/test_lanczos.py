"""
SUNPy A Python Library for solving SU(N) Heisenberg models
Copyright (C) 2026  Samuel Gozel, GNU GPLv3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pytest

import numpy as np
import scipy.sparse
import math

from sunpy.lanczos import lanczos


prec = 1.0e-14


def exact_spectrum_laplacian_1d_dirichlet_bc(n):
    """
    Exact spectrum of 1D Laplacian with Dirichlet BCs
    """
    energy = lambda j : -4.0*math.sin(math.pi * (n-j) / (2.0*(n+1)))**2
    return energy(0)

def laplacian_1d_dirichlet_bc(n):
    """
    Matrix of 1D Laplacian with Dirichlet BCs
    """
    data = np.full(shape=(n,), fill_value=-2.0, dtype=float)
    row_ind = np.arange(n)
    col_ind = np.arange(n)
    H = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    data = np.full(shape=(n-1,), fill_value=1.0, dtype=float)
    row_ind = np.arange(n-1)
    col_ind = np.arange(1, n)
    H += scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    H += scipy.sparse.csr_matrix((data, (col_ind, row_ind)), shape=(n, n))
    return H

@pytest.mark.parametrize("n", [
    (int(10)),
    (int(16)),
])
def test_lanczos_1d_laplacian(n):
    # arrange
    H = laplacian_1d_dirichlet_bc(n)
    multiply = lambda v : H @ v    
    rng = np.random.default_rng(seed=42)
    v_init = rng.uniform(low=0.0, high=1.0, size=n)
    v_init = v_init/np.linalg.norm(v_init)

    expected = exact_spectrum_laplacian_1d_dirichlet_bc(n)
    
    # act
    energy, eigvec = lanczos.lanczos(multiply, v_init, min_iter=int(4), max_iter=int(n+1),
                                     tol_residual=1.0e-13,
                                     tol_ritz=1.0e-13)
    
    # assert
    assert(abs(energy - expected)<prec)
    assert(np.sum(abs(multiply(eigvec) - energy*eigvec))/n<prec)
    