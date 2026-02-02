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



class EDSolverAntiSymm(EDSolver):
    """
    Class to represent an exact diagonalization solver for SU(N) Heisenberg 
    models with m particles per site in the antisymmetric m-box representation
    """
    
    def __init__(self, N: int, Ns: int, alpha, m: int, lattice: Lattice):
        """
        Constructor of ED engine for m particles per site in the antisymmetric irrep
        
        Parameters
        ----------
        N : int
            SU(N)
        Ns : int
            number of sites
        alpha : numpy array
            irrep
        m : int
            number of particles per site
        lattice : Lattice
            lattice of bonds
        """
        
        super().__init__(N, Ns, alpha, lattice)
        self._m = int(m)
        
        if not self._n==self._Ns*self._m:
            sys.exit('ERROR : EDEngineAntiSymm : __init__ : number of boxes in alpha must match Ns*m')
        
        if self._m>2:
            sys.exit('ERROR : EDEngineAntiSymm : __init__ : Code not yet implemented for m>2')
        
        self._NY = sun.multiplicity_symm(sun.transpose_shape(self._alpha), self._m) # sun.multiplicity_antisymm(self._alpha, self._m)
        
        return
        
    def get_basis(self):
        """
        Compute the collection of SYTs
        """
        tstart = time.perf_counter()
        self._Y = sun.get_SYT_antisymm(self._alpha, self._m)
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
        
        self._H = scipy.sparse.csr_matrix((self._NY, self._NY))
        
        for i, link in enumerate(self._lattice.links):
            
            Hbond = scipy.sparse.csr_matrix((self._NY, self._NY))
            
            for j in range(0, self._NY):
                ydev, cydev, coeffdev = sun.developp_antisymmetric(
                                            self._alpha, 
                                            self._Y[j], 
                                            self._CY[j], 
                                            self._m, 
                                            link[0], 
                                            link[1])
                ndev = len(coeffdev)
                row = np.full(shape=(ndev,), fill_value=j, dtype=int)
                col = np.zeros(shape=(ndev,), dtype=int)
                for t in range(0, ndev):
                    index = np.argwhere(np.sum(abs(self._Y - ydev[t]), axis=1)==0).flatten()[0]
                    col[t] = index
                Hbond += scipy.sparse.csr_matrix( (coeffdev, (row, col)), shape=(self._NY, self._NY))
            self._add_H_link(i, Hbond)
            
        self._H_computed = True
        
        tend = time.perf_counter()
        print('Elapsed time Hamiltonian construction = {}s'.format((tend - tstart)))
        
        return self._H
