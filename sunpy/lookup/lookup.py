# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np

from sunpy.common import partitions
import sunpy.common.math
from sunpy.sun import sun



class PartialLookupTool:
    """
    Class to deal with a partial lookup in a (very) large collection of 
    ordered SYTs
    """
    
    
    def __init__(self, N, alpha, nlookupboxes):
        
        if nlookupboxes==0:
            sys.exit('Problem: for nlookupboxes==0, no lookup. Use dedicated functions.')
        elif nlookupboxes==1:
            print('Warning: nlookupboxes=1 is equivalent to no lookup.')
        elif nlookupboxes>=np.sum(alpha):
            sys.exit('Problem: for partial lookup, nlookupboxes must be stricly smaller than the total number of boxes.')
        
        self.alpha = alpha
        self.N = N
        self.n = np.sum(alpha)
        self.nlookupboxes = nlookupboxes
        self.n0 = nlookupboxes
        self.n1 = self.n - self.n0
        
        return
    
    
    
    def init_lookup(self):
        # Generate all lookup tables
        # 
        
        # generate all subshapes of alpha with <self.n0> boxes
        pstarnm = partitions.pstarnm(self.n0, self.N) # number of partitions of <self.n0> in at most <self.N> parts
        
        alpha_all = sun.get_all_irreps(self.N, self.n)
        
        # restrict to potential shapes
        ind = np.argwhere( (np.sum(alpha_all, axis=1)<=self.n0) & ((np.sum(alpha_all, axis=1)%self.N)==(self.n0%self.N)) ).flatten()
        alpha_all = alpha_all[ind]
        
        if not alpha_all.shape[0]==pstarnm:
            sys.exit('Problem: the number of generated shapes does not match the number of parititions.')
        
        # add columns with N boxes
        for i in range(0, alpha_all.shape[0]):
            r = (self.n0 - np.sum(alpha_all[i]))//self.N
            alpha_all[i] += np.full(shape=alpha_all[i].shape, fill_value=r, dtype=int)
        
        # remove shapes which are not subshapes
        ind = np.zeros(shape=(0,), dtype=int)
        for i in range(0, alpha_all.shape[0]):
            issubshape = True
            for l in range(0, self.N):
                if alpha_all[i][l]>self.alpha[l]:
                    issubshape = False
                    break
            if issubshape==True:
                ind = np.hstack([ind, i])
        
        self.alpha0 = np.copy(alpha_all[ind])
        
        ###########################################################################
        
        # reorder alpha0 so as to correspond to the iLLOS
        # irreps must be sorted according to the length of the rows
        
        self.alpha0, _ = sunpy.common.math.sortrows(self.alpha0)
        self.Na = self.alpha0.shape[0] # total number of subshapes
        
        ###########################################################################
        
        # for each subshape of alpha, generate all SYTs
        
        self.Y0 = [None] * self.Na
        self.dims0 = np.zeros((self.Na,), dtype=int)
        
        for i in range(0, self.Na):
            self.Y0[i] = sun.get_SYT(self.alpha0[i], order='iLLOS')
            self.dims0[i] = self.Y0[i].shape[0]
        
        ###########################################################################
        
        # Generate all remainder shapes
        
        self.alpha1 = self.alpha - self.alpha0
        
        ###########################################################################
        
        # for each remainder shape, generate all SYTs
        
        self.Y1 = [None] * self.Na
        self.dims1 = np.zeros((self.Na,), dtype=int)
        
        for i in range(0, self.Na):
            self.Y1[i] = sun.get_subSYT(self.alpha, self.alpha0[i])
            self.dims1[i] = self.Y1[i].shape[0]
        
        ###########################################################################
        
        # Verify dimension
        
        self.dim = int(0)
        for i in range(0, self.Na):
            self.dim += (self.dims0[i] * self.dims1[i])
        
        if not self.dim==sun.multiplicity(self.alpha):
            sys.exit('Problem: the sum of the product of the dimensions does not match the multiplicity.')
        
        ###########################################################################
        # GENERATE THE ORDERING TO MATCH THE ILLOS
        ###########################################################################
        
        # note that sorting the SYTs is not strictly necessary, as one can simply
        # redefine the basis, as in each "block", the SYTs are ordered in the iLLOS
        # by construction. However, the iLLOS is a convenient order.
        
        self.orderingIrrep = np.zeros(shape=(np.sum(self.dims1),), dtype=int)
        self.orderingState = np.zeros(shape=(np.sum(self.dims1),), dtype=int)
        
        self.lookup = [None] * self.Na
        for i in range(0, self.Na):
            self.lookup[i] = np.zeros(shape=(self.dims1[i],), dtype=int)
        
        self.lookup[0][0] = int(0)
        
        start_index = np.zeros(shape=(self.Na,), dtype=int)
        start_index[0] = int(1)
        
        cpt = int(1)
        while cpt<np.sum(self.dims1):
            
            Ytmp = np.zeros(shape=(0, self.n1), dtype=int)
            ar = []
            for i in range(0, self.Na):
                if start_index[i]<self.dims1[i]:
                    ar.append(i)
                    Ytmp = np.vstack([Ytmp, self.Y1[i][start_index[i]]])
            ar = np.array(ar, dtype=int)
            _, ind = sunpy.common.math.sortrows(np.fliplr(Ytmp))
            
            k = ar[ind[0]]
            
            self.orderingIrrep[cpt] = k
            self.orderingState[cpt] = start_index[k]
            self.lookup[k][start_index[k]] = cpt
            
            start_index[ar[ind[0]]] += 1
            cpt += 1
        
        ###########################################################################
        # GENERATE A USEFUL LOOKUP TO GET SYT FROM INDEX
        ###########################################################################
        
        self.acc_dims = np.zeros(shape=(np.sum(self.dims1)+1,), dtype=int)
        
        for j in range(1, np.sum(self.dims1)+1):
            ind = self.orderingIrrep[j-1]
            self.acc_dims[j] = self.acc_dims[j-1] + self.dims0[ind]
        
        return
    
    
        
    def get_SYT(self, i):
        # Get the <i>-th SYT.
        # 
        # Input index <i> is in the iLLOS: i=0 ---> largest SYT.
        # 
        
        ind = np.argwhere(self.acc_dims<=i).flatten()[-1]
        
        k = self.orderingIrrep[ind] # index of irrep
        i0 = i - self.acc_dims[ind] # index of low part
        i1 = self.orderingState[ind] # index of high part
        
        y0 = np.copy(self.Y0[k][i0])
        y1 = np.copy(self.Y1[k][i1])
        
        y = np.hstack([y0, y1])
        
        return y
    
    
    
    def get_index(self, y):
        # get the index corresponding to SYT <y> in the ordered (wrt iLLOS) 
        # list of all SYTs.
        # 
        
        y0 = np.copy(y[0:self.n0]) # low part
        y1 = np.copy(y[self.n0:]) # high part
        
        # generate irrep for the low part
        a0 = np.zeros(shape=(self.N,), dtype=int)
        for i in range(0, self.n0):
            a0[y0[i]] += 1
        
        # search index of a0 in self.alpha0 [here linear search]
        k = np.argwhere( np.sum(abs(self.alpha0-a0), axis=1)==0 ).flatten()[0]
        
        # search index of state for low part [here linear search]
        #i0 = np.argwhere( np.sum(abs(self.Y0[k]-y0), axis=1)==0 ).flatten()[0]
        # [here binary search]
        i0 = sun.binary_search_SYT(y0, self.Y0[k])
        
        # search index of state for high part [here linear search]
        # i1 = np.argwhere( np.sum(abs(self.Y1[k]-y1), axis=1)==0 ).flatten()[0]
        # [here binary search]
        i1 = sun.binary_search_SYT(y1, self.Y1[k])
        
        # find index from high part
        # indhigh = np.argwhere( (self.orderingIrrep==k) & (self.orderingState==i1) ).flatten()[0]
        # use <lookup> technique: - slightly faster than argwhere, but requires self.lookup
        indhigh = self.lookup[k][i1]
        
        # compute index
        i = int(0)
        for j in range(0, indhigh):
            i += self.dims0[self.orderingIrrep[j]]
        i += i0
        
        return i




