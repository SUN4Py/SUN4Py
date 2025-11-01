#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:39:13 2023

@author: sgozel
"""
# Copyright 2023 Samuel GOZEL, GNU GPLv3

import numpy as np
import scipy.sparse
import time

import sun
import Lattice


# Example of creation and diagonalization of Heisenberg Hamiltonian with local
# adjoint irrep on each site
N = int(3)
Ns = int(5)
alpha = np.array([6, 5, 4], dtype=int) # global target sector
beta_loc = np.array([2, 1, 0], dtype=int) # local irrep
beta = np.matlib.repmat(beta_loc, Ns, 1) # same local irrep on each site

'''
# using different irreps at the edges
beta_loc_edge = np.array([2, 1, 0], dtype=int)
beta_loc_bulk = np.array([3, 0, 0], dtype=int)
beta = np.matlib.repmat(beta_loc, Ns, 1)
beta[0] = beta_loc_edge
beta[-1] = beta_loc_edge
'''

lattice = Lattice.Lattice(Ns=Ns, typeLattice='chain', isPBC=False)

start = time.perf_counter()
engine = sun.SUNGeneral(alpha, beta, N, lattice)
end = time.perf_counter()
print("Elapsed init general = {}s".format((end - start)))

start = time.perf_counter()
H = engine.sun_hamiltonian()
end = time.perf_counter()
print("Elapsed Hamiltonian general = {}s".format((end - start)))

if H.shape[0]==1:
    EGS = H[0,0]
elif H.shape[0]<100:
    H = H.todense()
    E, _ = np.linalg.eigh(H)
    EGS = E[0]
else:
    E, PSI = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    EGS = E[0]

print('GS Energy: ', EGS)
print('GS Energy per site: ', EGS/Ns)



'''
# Example for symmetric local constraints with m particles per site
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

'''
# This shows how to use get_subSYT, fill_subSYT and how to print SYTs to text
# for LaTeX
# 
#         x x o                 x x               o
# alpha = x o     ---> alphaB = x    ; alphaP = o
#         x o                   x               o
# 
alpha = np.array([3, 2, 2], dtype=int)
alphaB = np.array([2, 1, 1], dtype=int)
alphaP = alpha - alphaB

Y1 = sun.get_subSYT(alpha, alphaB)
for i in range(0, Y1.shape[0]):
    sun.print_subSYT_to_latex(Y1[i], alpha, highcol='col2')

Y2 = sun.fill_subSYT(Y1, alpha, fill_type='largest')
cols = {0: 'blue', 1: 'blue', 2: 'blue', 3: 'red', 4: 'green'}
for i in range(0, Y2.shape[0]):
    sun.print_to_latex(Y2[i], colors=cols)
'''


'''
# This shows that for symmetric local irreps on each site, the general code 
# for computing the SYTs is much slower than the dedicated symmetric routine

N = int(3)
Ns = int(8)
alpha = np.array([Ns, Ns, Ns], dtype=int)
beta_loc = np.array([3, 0, 0], dtype=int)
beta = np.matlib.repmat(beta_loc, Ns, 1)

start = time.perf_counter()
Ysymm = sun.get_SYT_symm(alpha, 3, order='iLLOS')
CYsymm = sun.get_column(Ysymm)
end = time.perf_counter()
print("Elapsed symm = {}s".format((end - start)))

start = time.perf_counter()
CYsymm = sun.get_column(Ysymm)
end = time.perf_counter()
print("Elapsed cols = {}s".format((end - start)))

NY = sun.multiplicity_irrep_mixed(alpha, beta, N)
start = time.perf_counter()
Ygen, CYgen = sun.get_SYT_general(alpha, beta, N)
end = time.perf_counter()
print("Elapsed gen = {}s".format((end - start)))

assert np.linalg.norm(Ysymm - Ygen)==0
'''
