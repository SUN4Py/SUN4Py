# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np
import scipy.sparse

from sunpy.sun import sun
from sunpy.lanczos import lanczos


class SUNFundamental:
    """
    Class to represent SU(N) Heisenberg models with 1 particle per site (fundamental 
    irrep on each site)
    """
    
    def __init__(self, Ns, N, alpha, lattice, basisOrder='iLLOS'):
        """
        Constructor of ED engine for 1 particle per site
        
        Parameters
        ----------
        Ns : int
            number of sites
        N : int
            SU(N)
        alpha : numpy array
            irrep
        lattice : Lattice
            lattice of bonds
        basisOrder : str [optional][default: 'iLLOS']
            'iLLOS' or 'LLOS' : order of the basis
        
        """
        
        if not np.sum(alpha)==Ns:
            sys.exit('Problem: number of boxes in alpha must match Ns')
        
        if not lattice.Ns==Ns:
            sys.exit('lattice object does not have the correct number of sites.')
        
        self._N = N
        self._Ns = Ns
        self._alpha = np.copy(alpha)
        self._n = np.sum(self._alpha)
        self._falpha = sun.multiplicity(self._alpha)
        self._lattice = lattice
        self._basisOrder = basisOrder
        
        self._Y = None
        self._CY = None
        self._basis_computed = False
        
        self._H = None
        self._H_computed = False
        
        self.threshold_eigfull = int(1000)
        self.threshold_eigsh = int(16000)
        
        return
    
    @property
    def falpha(self):
        return self._falpha
    
    @property
    def alpha(self):
        return self._alpha
    
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
    
    def get_basis(self):
        """
        Compute the collection of SYTs
        """
        self._Y = sun.get_SYT(self._alpha, order=self._basisOrder)
        self._CY = sun.get_column(self._Y)
        self._basis_computed = True
        return
    
    def sun_hamiltonian(self):
        """
        Compute the Hamiltonian
        """
        
        if (self._basis_computed==False):
            self.get_basis()
        
        P = sun.get_adjacent_transposition_matrices(self._alpha, self._Y, self._CY)
        
        self._H = scipy.sparse.csr_matrix((self._falpha, self._falpha))
        
        # We illustrate the very generic case, although for the fundamental irrep
        # at each site, only bilinear terms are relevant
        for i, link in enumerate(self._lattice.links):            
            adja_transpos = sun.transposition_to_adjacent_transpositions(link)
            Hbond = scipy.sparse.eye(self._falpha)
            for k in adja_transpos:
                Hbond = P[k] @ Hbond
            
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
    
    def multiply(self, v):
        """
        Compute w <--- H @ v
        """
        assert(len(v)==self._falpha)
        
        if (self._H_computed==True):
            return self._H @ v
        else:
            # matrix free implementation of w <--- H @ v
            w = np.zeros(self._falpha)
            
            # serial loop over all bonds in the lattice
            for bond in self._lattice.bonds:
                
                if (bond.bond_order>1):
                    raise ValueError('ERROR : SUNFundamental : multiply : bonds should be order-1 '
                                     'bonds (bilinear Heisenberg exchange is the only non-trivial interaction '
                                     'for 1 particle per site).')
                
                # transform transposition (bond.site1, bond.site2) into product of adjacent transpositions
                adja_transpos = sun.transposition_to_adjacent_transpositions(bond.bond)
                
                # parallel loop over all states in the Hilbert space
                for i in range(0, self._falpha):
                    
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
    
    def diagonalize(self):
        """
        Diagonalize the Hamiltonian for the smallest eigenpairs
        """
        
        if ((self._H_computed) & (self._falpha<=self.threshold_eigfull)):
            
            eigvals, eigvecs = np.linalg.eigh(self._H.todense())
                
        elif ((self._H_computed) & (self._falpha<=self.threshold_eigsh)):
                
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(self._H, k=3, which='SA')
            eigvecs = np.asarray(eigvecs)
                
        else:
            # use custom Lanczos algorithm
            rng = np.random.default_rng(seed=42)
            v_init = rng.uniform(low=-1.0, high=1.0, size=self._falpha)
            v_init = v_init/np.linalg.norm(v_init)
            
            multiply = lambda v : self.multiply(v)
            
            eigvals, eigvecs = lanczos.lanczos(multiply, v_init)
        
        return eigvals, eigvecs
    