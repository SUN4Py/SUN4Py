#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np



class Lattice:
    
    def __init__(self, Ns, typeLattice, isPBC):
        
        self.Ns = Ns
        
        if typeLattice=='chain':
            
            if isPBC==False:
                links = [None] * (self.Ns-1)
                for i in range(0, Ns-1):
                    links[i] = np.array([i, i+1], dtype=int)
                siteorder = np.arange(0, Ns)
            else:
                links = [None] * self.Ns
                links[0] = np.array([0, 1], dtype=int)
                for i in range(0, Ns-1):
                    links[i+1] = np.array([i, min(i+2, self.Ns-1)], dtype=int)
                siteorder = np.zeros((Ns, ), dtype=int)
                siteorder[1:(np.floor(Ns/2.0+0.1).astype(int)+1)] = np.arange(1, Ns, 2)
                siteorder[np.floor(Ns/2+0.1).astype(int)+1:] = np.flipud(np.arange(2, Ns, 2))
        else:
            sys.exit('Lattice undefined')
        
        
        self.links = links
        self.siteorder = siteorder
        self.nlinks = len(links)
