"""
SUN4Py A Python Library for solving SU(N) Heisenberg models
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

import numpy as np
import scipy.sparse

from sun4py.sun import sun



def set_overall_phase(COEFF_REF):
    """
    Set the overall phase of a series of SDCs
    
    Parameters
    ----------
    COEFF_REF : numpy array
        set of SDCs (stored in the columns)
    
    Returns
    -------
    COEFF_REF : numpy array
        rotated SDCs with correct phase convention
    IND_REF : list
        positions of non-zero elements for each sequence of SDCs in COEFF_REF
        IND_REF is a list of length COEFF_REF.shape[1]
        each element in IND_REF is a numpy array of indices for non-zero elements
        in output COEFF_REF
    """
    tol = 1.0e-12
    
    nu1nu2nu = COEFF_REF.shape[1]
    
    if (nu1nu2nu == 1):
        # The coefficient of the first (in increasing order of the LLOS) non-zero
        # term must be positive
        ind_sdc = np.argwhere(abs(COEFF_REF[:, 0]) > tol).flatten()
        if (COEFF_REF[ind_sdc[0], 0] < 0):
            COEFF_REF *= -1.0
        IND_REF = [ind_sdc]
    else:
        COEFF_REF = canonicalize_basis(COEFF_REF)
        IND_REF = []
        for i in range(nu1nu2nu):
            ind = np.argwhere( abs(COEFF_REF[:, i]) > tol ).flatten()
            IND_REF.append(ind)
        
    return COEFF_REF, IND_REF


def canonicalize_basis(COEFF_REF):
    """
    Generate a canonical orthonormal basis of the subspace spanned by the 
    K columns of COEFF_REF.
    
    Parameters
    ----------
    COEFF_REF : numpy array shape (N, K)
        K orthonormal vectors (stored in the columns)
    
    Returns
    -------
    W : numpy array shape (N, K)
        K orthogonal vectors (stored in the columns) spaning a K-dimensional 
        subspace

    """
    tol = 1.0e-12
    
    N, K = COEFF_REF.shape
    
    # build diagonal of projection matrix onto the spanned subspace
    Pi_diag = np.sum(COEFF_REF ** 2, axis=1)
    
    W = np.zeros(shape=(N, K), dtype=float)
    used_axes = []
    
    # residual projection matrix diagonal - updated at each step
    Pi_residual_diag = np.copy(Pi_diag)
    
    for step in range(K):
        candidates = np.arange(N)
        mask = np.ones(N, dtype=bool)
        for ax in used_axes:
            mask[ax] = False
        candidates = candidates[mask]
        
        k_pivot = candidates[np.argmax(Pi_residual_diag[candidates])]
        used_axes.append(k_pivot)
        
        # project e_{k_pivot} onto the subspace
        coords = COEFF_REF[k_pivot, :]
        p = coords @ COEFF_REF.T
        
        # orthogonalize
        for j in range(step):
            p -= np.dot(p, W[:, j]) * W[:, j]
        
        # normalize
        norm = np.linalg.norm(p)
        if norm < tol:
            raise RuntimeError(f'Step {step}: projected vector has near-zero norm.')
        w = p/norm
        
        # fix sign
        ind_first_nonzero = np.argwhere(np.abs(w) > tol).flatten()[0]
        if (w[ind_first_nonzero] < 0):
            w *= -1
        
        W[:, step] = w
        
        # update residual projection diagonal
        Pi_residual_diag -= w**2
    
    return W


def casimir_canonical_chain(nu, y):
    """
    Get the eigenvalues of the quadratic Casimir operator for the canonical 
    group chain of nu defined by SYT y.
    
    Parameters
    ----------
    nu : numpy array
        irrep
    y : numpy array
        SYT
    
    Returns
    -------
    casimir : numpy array
        eigenvalues of CSCO-II of y of nu
    
    References
    ----------
    [1]         Group Representation Theory for Physicists
                Jin-Quan Chen, Hialun Ping and Fan Wang
                World Scientific, 2nd edition, (2002)
    """
    
    n = np.sum(nu)
    nup = np.copy(nu)
    casimir = np.zeros(shape=(n,), dtype=float)
    for j in range(n-1, -1, -1):
        casimir[j] = sun.casimir_quadratic(nup)
        nup[y[j]] -= 1
    
    return casimir


def build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, casimir, ordering='dmrg'):
    """
    Build projection operator of the CSCO-II associated with input casimir values
    """
    
    M = scipy.sparse.csr_matrix((NY, NY))
    
    if ordering=='math':
        f = lambda j, k : (n1+k, n1+j)
    elif ordering=='dmrg':
        f = lambda j, k : (n-1-j, n-1-k)
    
    for j in range(1, n-n1):
        Mj = scipy.sparse.csr_matrix((NY, NY))    
        for k in range(0, j):
            transpo = np.array(f(j, k))
            sigma = sun.transposition_to_adjacent_transpositions(transpo)
            Mtemp = scipy.sparse.eye(NY)    
            for atr in sigma:
                Mtemp = Mtemp @ MatAdjaTranspo[atr-n1]
            Mj += Mtemp
        val = -(casimir[j] - casimir[j-1])
        Mj = Mj + val*scipy.sparse.eye(NY)
        M += (Mj @ Mj)
    
    return M
