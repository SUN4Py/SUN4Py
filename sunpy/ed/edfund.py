# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np
import scipy.sparse

from sunpy.sun import sun



class SUNFundamental:
    
    def __init__(self, Ns, N, alpha, lattice, basisOrder='iLLOS'):
        
        if not np.sum(alpha)==Ns:
            sys.exit('Problem: number of boxes in alpha must match Ns')
        
        if not lattice.Ns==Ns:
            sys.exit('lattice object does not have the correct number of sites.')
        
        self.N = N
        self.Ns = Ns
        self.alpha = np.copy(alpha)
        self.n = np.sum(self.alpha)
        self.falpha = sun.multiplicity(self.alpha)
        self.lattice = lattice
        self.Y = sun.get_SYT(self.alpha, order=basisOrder)
        self.CY = sun.get_column(self.Y)
    
    
    def sun_hamiltonian(self):
        """
        Compute the Hamiltonian
        """
        
        listtranspositions, nbtranspositions = sun.get_transpositions(self.lattice.links)
        
        P = sun.get_adjacent_transposition_matrices(self.alpha, self.Y, self.CY)
        
        H = scipy.sparse.csr_matrix((self.falpha, self.falpha))
        
        for j in range(0, self.lattice.nlinks):
            Hj = scipy.sparse.eye(self.falpha)
            for l in range(nbtranspositions[j]-1, -1, -1):
                k = listtranspositions[j][l]
                Hj = P[k] @ Hj
            H += Hj
        
        #H = 0.5*(H + H.transpose())
        
        return H

