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
import numpy as np
import scipy.sparse

from sunpy.sun import sun



class SUNAntiSymmetric:
    """
    Class to represent SU(N) Heisenberg models with m particle per site in the 
    antisymmetric m-box representation
    """
    
    def __init__(self, Ns, N, m, alpha, lattice):
        """
        Constructor of ED engine for m particles per site in the antisymmetric irrep
        
        Parameters
        ----------
        Ns : int
            number of sites
        N : int
            SU(N)
        m : int
            number of particles per site
        alpha : numpy array
            irrep
        lattice : Lattice
            lattice of bonds
        """
        
        if not np.sum(alpha)==int(Ns)*int(m):
            sys.exit('Problem: number of boxes in alpha must match Ns*m')
        
        if not lattice.Ns==int(Ns):
            sys.exit('lattice object does not have the correct number of sites.')
        
        if m>2:
            sys.exit('Code not yet implemented for m>2.')
        
        self._N = int(N)
        self._Ns = int(Ns)
        self._m = int(m)
        self._alpha = np.copy(alpha)
        self._n = np.sum(self._alpha)
        self._lattice = lattice
        
        self._Y = None
        self._CY = None
        self._NY = None # sun.multiplicity_antisymm(self._alpha, self._m)
        self._basis_computed = False
        
        self._H = None
        self._H_computed = False
        
        return
        
    def get_basis(self):
        """
        Compute the collection of SYTs
        """
        self._Y = sun.get_SYT_antisymm(self._alpha, self._m)
        self._CY = sun.get_column(self._Y)
        self._NY = self._Y.shape[0]
        self._basis_computed = True
        return
    
    def sun_hamiltonian(self):
        """
        Compute the Hamiltonian
        """
        
        if (self._basis_computed==False):
            self.get_basis()
        
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
        
        self._H_computed = True
        
        return self._H
