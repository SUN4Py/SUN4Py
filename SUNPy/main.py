#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:39:13 2023

@author: sgozel
"""
# Copyright 2023 Samuel GOZEL, GNU GPLv3

import numpy as np
import scipy.sparse

import sun
import Lattice




N = int(3)
m = int(1)
alpha = np.array([4, 3, 2], dtype=int)
Ns = np.sum(alpha)//m

# get the lattice
lattice = Lattice.Lattice(Ns=Ns, typeLattice='chain', isPBC=False)

# build SU(N) engine
if m==1:
    Engine = sun.SUNFundamental(Ns=Ns, N=N, alpha=alpha, lattice=lattice)
else:
    Engine = sun.SUNSymmetric(Ns=Ns, N=N, m=m, alpha=alpha, lattice=lattice)

# construct matrix of Hamiltonian
H = Engine.sun_hamiltonian()

# diagonalize
E, PSI = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
EGS = E[0]
print('GS Energy: ', E[0])



'''
alpha = np.array([3, 2, 2], dtype=int)
alpha0 = np.array([2, 1, 1], dtype=int)
alpha1 = alpha - alpha0

Y1 = sun.get_subSYT(alpha, alpha0)
# Y2 = sun.fill_subSYT(Y1, alpha, fill_type='largest')

for i in range(0, Y1.shape[0]):
    sun.print_subSYT_to_latex(Y1[i], alpha, highcol='col2')
'''


'''
N = int(3)
alpha = np.array([3, 2, 2], dtype=int)
NY = sun.multiplicity(alpha)
n = np.sum(alpha)
nlookupboxes = int(4)
Yalpha = sun.get_SYT(alpha, order='iLLOS')

lkptool = sun.PartialLookupTool(N, alpha, nlookupboxes)
lkptool.init_lookup()

for i in range(0, Yalpha.shape[0]):
    ii = lkptool.get_index(Yalpha[i])
for i in range(0, NY):
    y = lkptool.get_SYT(i)
'''
