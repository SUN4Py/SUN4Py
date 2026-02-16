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
import numpy as np
#from numpy.typing import NDArray # numpy>=1.20
import scipy.sparse
from abc import ABC, abstractmethod

from sun4py.ed.lattice import Lattice
from sun4py.lanczos import lanczos


class EDSolver(ABC):
    """
    Abstract class to represent an exact diagonalization solver for SU(N) 
    Heisenberg model
    """
    
    def __init__(self, N: int, Ns: int, alpha, lattice: Lattice, **kwargs):
        """
        Contructor
        
        Parameters
        ----------
        N : int
            SU(N)
        Ns : int
            number of sites
        alpha : numpy array
            irrep
        lattice : Lattice
            lattice of bonds
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
        """
        self._N = int(N)
        self._Ns = int(Ns)
        self._alpha = np.copy(alpha)
        self._n = np.sum(self._alpha)
        self._lattice = lattice
        
        if not len(alpha)<=self._N:
            sys.exit('ERROR : EDEngine : __init__ : mismatch between N and number of rows in alpha.')
        
        if not self._lattice.Ns==self._Ns:
            sys.exit('ERROR : EDEngine : __init__ : lattice object does not have the correct number of sites.')
        
        self._basis_computed = False
        self._H_computed = False
        
        self._threshold_eigfull = int(1000)
        self._threshold_eigsh = int(16000)
        
        self._lanczos_params = {}
        self._lanczos_params['num_eigenvalues'] = int(kwargs.get('num_eigenvalues', 1))
        self._lanczos_params['max_iter'] = int(kwargs.get('max_iter', 200))
        self._lanczos_params['min_iter'] = int(kwargs.get('min_iter', 40))
        self._lanczos_params['tol_residual'] = kwargs.get('tol_residual', 1.0e-14)
        self._lanczos_params['tol_ritz'] = kwargs.get('tol_ritz', 1.0e-14)
        
        return 
    
    def diagonalize(self, **kwargs):
        """
        Diagonalize the Hamiltonian for the smallest eigenpairs
        
        Optional Parameters
        -------------------
        num_eigenvalues : int [optional][default: 1][]
            number of eigenvalues to extract
        
        For Lanczos based diagonalization:
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
        eigvals : numpy array
            eigenvalues
        eigvecs : numpy array
            eigenvectors (stored in the columns)
        """
        
        self._lanczos_params['num_eigenvalues'] = int(kwargs.get('num_eigenvalues', self._lanczos_params['num_eigenvalues']))
        self._lanczos_params['max_iter'] = int(kwargs.get('max_iter', self._lanczos_params['max_iter']))
        self._lanczos_params['min_iter'] = int(kwargs.get('min_iter', self._lanczos_params['min_iter']))
        self._lanczos_params['tol_residual'] = kwargs.get('tol_residual', self._lanczos_params['tol_residual'])
        self._lanczos_params['tol_ritz'] = kwargs.get('tol_ritz', self._lanczos_params['tol_ritz'])
        
        if ((self._H_computed) & (self._NY<=self._threshold_eigfull)):
            
            eigvals, eigvecs = np.linalg.eigh(self._H.todense())
            eigvals = eigvals[:self._lanczos_params['num_eigenvalues']]
            eigvecs = eigvecs[:, :self._lanczos_params['num_eigenvalues']]
                
        elif ((self._H_computed) & (self._NY<=self._threshold_eigsh)):
                
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(self._H, k=self._lanczos_params['num_eigenvalues'], which='SA')
            eigvals = np.asarray(eigvals)
            eigvecs = np.asarray(eigvecs)
            
        else:
            # use custom Lanczos algorithm
            rng = np.random.default_rng(seed=42)
            v_init = rng.uniform(low=-1.0, high=1.0, size=self._NY)
            v_init = v_init/np.linalg.norm(v_init)
            
            multiply = lambda v : self.multiply(v)
            
            eigvals, eigvecs = lanczos.lanczos(multiply, v_init, **self._lanczos_params)
            # currently, lanczos.lanczos returns only the first eigenpair
            eigvals = np.array([eigvals])
            eigvecs = np.reshape(eigvecs, (self._NY, 1))
        
        return eigvals, eigvecs
    
    def _add_H_link(self, i, Hbond):
        """
        Add all bond Hamiltonians associated to a link to the Hamiltonian matrix
        
        Parameters
        ----------
        i : int
            index of link under consideration
        
        Details
        -------
        Hbond is the bilinear Heisenberg interaction on the i-th link
        Several bonds can live on the same link, for instance:
            (1, 2) HB  J=1.0
            (1, 2) HB2 J=-0.3
            (1, 2) HB3 J=0.1
        denotes 3 bonds living on the same link. The second and third bond 
        (namely, the biquadratic and bicubic Heisenberg interaction, respectively)
        can be obtained from the bilinear term by exponentiating.
        """
        order_next = self._lattice.bond_orders[i][0]
        for t in range(1, order_next):
            Hbond = Hbond @ Hbond
        order_prev = order_next
        
        self._H += self._lattice.bonds[self._lattice.indices[i][0]].J * Hbond
        
        for j in range(1, len(self._lattice.indices[i])):
            order_next = self._lattice.bond_orders[i][j]
            for t in range(order_prev, order_next):
                Hbond = Hbond @ Hbond
            order_prev = order_next
            self._H += self._lattice.bonds[self._lattice.indices[i][j]].J * Hbond
        return
    
    @abstractmethod
    def get_basis(self):
        """
        Compute the collection of SYTs
        """
        raise NotImplementedError()
    
    def multiply(self, v):
        """
        Compute w <--- H @ v
        """
        if self._H_computed:
            return self._H @ v
        else:
            raise NotImplementedError()
    
    @abstractmethod
    def sun_hamiltonian(self):
        """
        Compute Hamiltonian matrix
        """
        raise NotImplementedError()
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def NY(self):
        return self._NY
    
    @property
    def Y(self):
        return self._Y
    
    @property
    def CY(self):
        return self._CY
    
    @property
    def H(self):
        return self._H
    
    @property
    def basis_computed(self):
        return self._basis_computed
    
    @property
    def H_computed(self):
        return self._H_computed
