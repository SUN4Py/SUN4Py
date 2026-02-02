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

import sys
import time
import numpy as np
# from numpy.typing import NDArray # numpy>=1.20
import scipy.sparse

from sunpy.ed.edbase import EDSolver
from sunpy.ed.lattice import Lattice
from sunpy.sun import sun


class EDSolverFund(EDSolver):
    """
    Class to represent an exact diagonalization solver for SU(N) Heisenberg 
    models with 1 particle per site (fundamental irrep on each site)
    """
    
    def __init__(self, N: int, Ns: int, alpha, lattice: Lattice, basisOrder='iLLOS', **kwargs):
        """
        Constructor of ED engine for 1 particle per site
        
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
        basisOrder : str [optional][default: 'iLLOS']
            'iLLOS' or 'LLOS' : order of the basis
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
        
        super().__init__(N, Ns, alpha, lattice, **kwargs)
        
        if not self._n==self._Ns:
            sys.exit('ERROR : EDEngineFund : __init__ : number of boxes in alpha must match Ns')
        
        self._NY = sun.multiplicity(self._alpha)
        self._basisOrder = basisOrder
        
        return
    
    def get_basis(self):
        """
        Compute the collection of SYTs
        """
        tstart = time.perf_counter()
        self._Y = sun.get_SYT(self._alpha, order=self._basisOrder)
        self._CY = sun.get_column(self._Y)
        self._basis_computed = True
        tend = time.perf_counter()
        print('Elapsed time Basis construction = {}s'.format((tend - tstart)))
        return
    
    def sun_hamiltonian(self):
        """
        Compute the Hamiltonian
        """
        
        if (self._basis_computed==False):
            self.get_basis()
        
        tstart = time.perf_counter()
        
        P = sun.get_adjacent_transposition_matrices(self._alpha, self._Y, self._CY)
        
        self._H = scipy.sparse.csr_matrix((self._NY, self._NY))
        
        # We illustrate the very generic case, although for the fundamental irrep
        # at each site, only bilinear terms are relevant
        for i, link in enumerate(self._lattice.links):            
            adja_transpos = sun.transposition_to_adjacent_transpositions(link)
            Hbond = scipy.sparse.eye(self._NY)
            for k in adja_transpos:
                Hbond = P[k] @ Hbond
            
            
            self._add_H_link(i, Hbond)
        
        self._H_computed = True
        
        tend = time.perf_counter()
        print('Elapsed time Hamiltonian construction = {}s'.format((tend - tstart)))
        
        return self._H
    
    def multiply(self, v):
        """
        Compute w <--- H @ v
        """
        assert(len(v)==self._NY)
        
        if (self._H_computed==True):
            return self._H @ v
        else:
            # matrix free implementation of w <--- H @ v
            w = np.zeros(self._NY)
            
            # serial loop over all bonds in the lattice
            for bond in self._lattice.bonds:
                
                if (bond.bond_order>1):
                    raise ValueError('ERROR : SUNFundamental : multiply : bonds should be order-1 '
                                     'bonds (bilinear Heisenberg exchange is the only non-trivial interaction '
                                     'for 1 particle per site).')
                
                # transform transposition (bond.site1, bond.site2) into product of adjacent transpositions
                adja_transpos = sun.transposition_to_adjacent_transpositions(bond.bond)
                
                # parallel loop over all states in the Hilbert space
                for i in range(0, self._NY):
                    
                    if (self._basis_computed==True):
                        y = self._Y[i]
                        cy = self._CY[i]
                    else:
                        y = sun.index_to_SYT(i, self._alpha, order=self._basisOrder)
                        cy = sun.get_column(y)
                    
                    ydev = np.array([y])
                    cydev = np.array([cy])
                    coeff = np.array([1.0])
                    
                    # serial loop over all adjacent transpositions
                    for k in adja_transpos:
                        ydev, cydev, coeff = sun.develop_consecutive_number(self._alpha, ydev, cydev, coeff, k)
                    
                    del cydev
                    
                    for j in range(0, len(coeff)):
                        # find index of ydev[j] in the basis
                        if (self._basis_computed==True):
                            indj = sun.binary_search_SYT(ydev[j], self._Y)
                        else:
                            indj = sun.SYT_to_index(ydev[j], self._alpha, self._basisOrder)
                        #w[indj] += bond.J * coeff[j] * v[i] # this is prone to concurrent writes when the loop on i is parallel
                        w[i] += bond.J * coeff[j] * v[indj] # trick to avoid concurrent writes on w
            
            return w
    