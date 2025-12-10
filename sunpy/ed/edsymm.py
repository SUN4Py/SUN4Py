# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np
import scipy.sparse

from sunpy.sun import sun



class SUNSymmetric:
    
    def __init__(self, Ns, N, m, alpha, lattice):
        
        if not np.sum(alpha)==Ns*m:
            sys.exit('Problem: number of boxes in alpha must match Ns*m')
        
        if not lattice.Ns==Ns:
            sys.exit('lattice object does not have the correct number of sites.')
        
        if m>3:
            sys.exit('Code not yet implemented for m>3')
        
        self.N = N
        self.Ns = Ns
        self.m = m
        self.alpha = np.copy(alpha)
        self.n = np.sum(self.alpha)
        self.lattice = lattice
        self.Y = sun.get_SYT_symm(self.alpha, self.m)
        self.CY = sun.get_column(self.Y)
        self.NY = self.Y.shape[0]
    
    
    def sun_hamiltonian(self):
        
        H = scipy.sparse.csr_matrix((self.NY, self.NY))
        
        for link in self.lattice.links:
            
            Hbond = scipy.sparse.csr_matrix((self.NY, self.NY))
            
            for i in range(0, self.NY):
                
                ydev, cydev, coeffdev = sun.developp_symmetric(
                                            self.alpha, 
                                            self.Y[i], 
                                            self.CY[i], 
                                            self.m, 
                                            link[0], 
                                            link[1])
                ndev = len(coeffdev)                
                row = np.full(shape=(ndev,), fill_value=i, dtype=int)
                col = np.zeros(shape=(ndev,), dtype=int)
                
                for t in range(0, ndev):
                    index = np.argwhere(np.sum(abs(self.Y - ydev[t]), axis=1) < 1e-13).flatten()[0]                    
                    col[t] = index
                
                Hbond += scipy.sparse.csr_matrix( (coeffdev, (row, col)), shape=(self.NY, self.NY))
            
            H += Hbond
        
        #H = 0.5 * (H + H.transpose())
        
        return H
