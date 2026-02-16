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

import sys
import time
import numpy as np



class TMatrix:
    """
    Class to represent the tridiagonal Hessenberg matrix generated during the
    Lanczos algorithm
    """
    
    def __init__(self):
        """
        Constructor of empty TMatrix
        """
        self._alpha = np.array([], dtype=float)
        self._beta = np.array([], dtype=float)
        self._n = int(0)
    
    def size(self):
        return self._n
    
    def get_alpha(self, i):
        assert(i<self._n)
        return self._alpha[i]
    
    def get_beta(self, i):
        assert(i<self._n)
        return self._beta[i]
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta
    
    def copy(self):
        tmatcopy = TMatrix()
        tmatcopy._n = self._n
        tmatcopy._alpha = np.zeros(shape=tmatcopy._n)
        tmatcopy._beta = np.zeros(shape=tmatcopy._n)
        tmatcopy._alpha = self._alpha
        tmatcopy._beta = self._beta
        return tmatcopy
    
    def push_back(self, alpha, beta):
        self._alpha = np.hstack((self._alpha, alpha))
        self._beta = np.hstack((self._beta, beta))
        self._n += 1
    
    def pop_back(self):
        self._alpha = self._alpha[:-1]
        self._beta = self._beta[:-1]
        self._n -= 1
    
    def tmatrix(self):
        if (self._n==1):
            tmat = np.array([[self._alpha[0]]])
        else:
            tmat = np.diag(self._alpha) + np.diag(self._beta[:self._n-1], 1) + np.diag(self._beta[:self._n-1], -1)
        return tmat
    
    def eigenvalues(self):
        if (self._n==1):
            eigvals = np.zeros(shape=(self._n,), dtype=float)
            eigvals[0] = self._alpha[0]
        else:
            tmat = self.tmatrix()
            eigvals = np.linalg.eigvalsh(tmat)
        return eigvals

    def eigenpairs(self):
        eigvals = np.zeros(shape=(self._n, ), dtype=float)
        eigvecs = np.zeros(shape=(self._n, self._n), dtype=float)
        if (self._n==1):
            eigvals[0] = self._alpha[0]
            eigvecs[0, 0] = 1.0
        else:
            tmat = self.tmatrix()
            eigvals, eigvecs = np.linalg.eigh(tmat)
        return eigvals, eigvecs


def ritz_value_stabilization(tmat, k):
    """
    Compute relative change in k first eigenvalues of T-matrix between the two 
    last Lanczos iterations
    """
    eigvals = tmat.eigenvalues()
    tmat_prev = tmat.copy()
    tmat_prev.pop_back()
    eigvals_prev = tmat_prev.eigenvalues()
    assert(tmat_prev.size()>=k+1)
    delta = abs(eigvals[:k] - eigvals_prev[:k]) / abs(eigvals[:k])
    return delta


def residual(tmat, k):
    """
    Compute residual norms on k smallest eigenpairs
    """
    tmat_prev = tmat.copy()
    tmat_prev.pop_back()
    _, eigvecs = tmat_prev.eigenpairs()
    residuals = np.abs(tmat.get_beta(-1) * eigvecs[-1, :k]).flatten()
    return residuals


def convergence(tmat, k, tol_residual, tol_ritz):
    """
    Assess convergence of the k-th first eigenvalues
    """
    isConverged = False
    n = tmat.size()
    if (n>k+1):
        residuals = residual(tmat, k)
        deltas = ritz_value_stabilization(tmat, k)
        if (all(residuals<tol_residual) | (all(deltas<tol_ritz))):
            isConverged = True
    return isConverged


def lanczos(multiply, v_init, **kwargs):
    """
    Lanczos algorithm to obtain the lowest eigenpair, using 3 Lanczos vectors on
    top of the starting initialization vector v_init
    
    Parameters
    ----------
    multiply : function
        function which returns w <--- H*v as w = multiply(v)
    v_init : numpy array
        normalized starting vector
    num_eigenvalues : int [optional][default: 1][]
        number of eigenvalues to extract
    max_iter : int [optional][default: 200]
        maximum number of Lanczos iterations
    min_iter : int [optional][default: 40]
        minimum number of Lanczos iterations
    tol_residual : float [optional][default: 1.0e-14]
        tolerance on residual norm of eigenpairs
    tol_ritz : float [optional][default: 1.0e-14]
        relative accuracy on eigenvalues
   
    Returns
    -------
    energy : float
        eigenvalue of lowest algebraic value
    v_init : numpy array
        associated eigenvector
    
    Remarks
    -------
    - v_init must be normalized to 1
    - on output, v_init contains the eigenvector of the Hamiltonian
    - even if num_eigenvalues>1, only the first eigenpair will be extracted, to
      reduce memory usage
    """
    
    print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print('Start Lanczos ...')
    tstart = time.time()
    
    max_iter = int(kwargs.get('max_iter', 200))
    min_iter = int(kwargs.get('min_iter', 40))
    
    if (min_iter>max_iter):
        print('WARNING : lanczos.lanczos : max_iter<min_iter. Setting max_iter to min_iter + 10')
        max_iter = min_iter + 10
    
    tol_residual = kwargs.get('tol_residual', 1.0e-14)
    tol_ritz = kwargs.get('tol_ritz', 1.0e-14)
    
    num_eigenvalues = int(kwargs.get('num_eigenvalues', 1))
    if num_eigenvalues>1:
        print('WARNING: lanczos.lanczos : num_eigenvalue reset to 1.')
        num_eigenvalues = int(1)
    
    tmat = TMatrix()
    dimension = v_init.shape[0]
    v = np.copy(v_init)
    u = np.zeros(shape=dimension)
    w = np.zeros(shape=dimension)
    
    min_iter = min(min_iter, dimension-1)
    
    # Compute eigenvalue
    
    cpt = int(0)
    isConverged = False
    alpha = 0.0
    beta = 0.0
    
    while ((cpt<min_iter) | ((isConverged==False) & (cpt<max_iter))):
        u, v, w, alpha, beta = lanczos_step(u, v, w, beta, multiply)
        tmat.push_back(alpha, beta)
        eigvals = tmat.eigenvalues()
        print('e0 = ', eigvals[0])
        if (cpt>=(min_iter-1)):
            isConverged = convergence(tmat, 
                                      k=num_eigenvalues, 
                                      tol_residual=tol_residual, 
                                      tol_ritz=tol_ritz)
        cpt += 1
    
    if ((cpt==max_iter) & (isConverged==False)):
        print('WARNING: lanczos.lanczos : reached maximum number of iterations without convergence')
        print('n = ', tmat.size())
        print('alpha = ')
        print(tmat.alpha)
        print('beta = ')
        print(tmat.beta)
        sys.exit()
    
    eigvals, eigvecs = tmat.eigenpairs()
    # eigvals = eigvals[:num_eigenvalues]
    energy = eigvals[0]
    
    # Compute eigenvector
    gs = np.asarray(eigvecs[:, 0]).flatten()
    
    v = np.copy(v_init)
    w = np.zeros(shape=dimension)
    u = np.zeros(shape=dimension)
    v_init *= gs[0]
    
    for i in range(1, cpt):
        u, v, w, alpha, beta = lanczos_step(u, v, w, beta, multiply)
        assert(abs(alpha-tmat.get_alpha(i-1))<1.0e-14)
        assert(abs(beta-tmat.get_beta(i-1))<1.0e-14)
        v_init += gs[i]*v
    
    v_init = v_init/np.linalg.norm(v_init)
    
    tend = time.time()
    print('Number of Lanczos iterations: ', cpt)
    print('Time = ', (tend-tstart), 's')
    print('done.')
    
    return energy, v_init


def lanczos_step(u, v, w, beta, multiply):
    """
    Perform one Lanczos iteration, using 3 Lanczos vectors
    """
    w = multiply(v) # w <--- H*v
    alpha = np.dot(v, w)
    w -= alpha*v
    w -= beta*u
    u = v
    v = w
    beta = np.linalg.norm(v)
    v = v/beta
    return u, v, w, alpha, beta
