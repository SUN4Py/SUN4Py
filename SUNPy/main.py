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


#sigma=np.array([3,2,5,0,1,4,6,8,7], dtype=int)
#at = sun.permutation_to_adjacent_transpositions(sigma)

'''
alpha = np.array([5, 3, 2], dtype=int)
Y = sun.get_SYT(alpha)
CY = sun.get_column(Y)

y = Y[44] # check nb 3
cy = CY[44]
sun.print_to_latex(y)
particles = np.array([4, 5, 6])
alphaTot, alphaB, alphaP, offset = sun.get_subshape(y, cy, particles)

subY = sun.get_subSYT(alphaTot, alphaB)
'''

N = int(3)
Ns = int(6)
alpha = np.array([Ns, Ns, Ns], dtype=int)
#alpha = np.array([5, 2, 2], dtype=int)
beta_loc = np.array([[2, 1, 0]], dtype=int)
#beta_loc = np.array([[3, 0, 0]], dtype=int)
beta = np.repeat(beta_loc, Ns, axis=0)
beta_loc = beta_loc.flatten()

'''
Y, CY = sun.get_SYT_general(alpha, beta)
ortho = sun.OrthogonalUnits(beta_loc, only00='True')
y = Y[3]
cy = CY[3]
particles = np.array([6, 7, 8], dtype=int)
particles = np.array([9, 10, 11], dtype=int)
alphaM, alphaB, alphaP, offset = sun.get_subshape(y, cy, particles)
YM = sun.get_subSYT(alphaM, alphaB, order='LLOS')
YMc = sun.fill_subSYT(YM, alphaM, fill_type='smallest')
CYMc = sun.get_column(YMc)
particles_shifted = particles - particles[0] + np.sum(alphaB)
P = [None] * (len(particles_shifted) -  1)
for i in range(0, len(particles_shifted)-1):
    P[i] = sun.get_adjacent_transposition_matrix(alphaM, YMc, CYMc, k=particles_shifted[i])

Proj = sun.get_projector(beta_loc, y, cy, particles)

V = sun.get_local_states(y, cy, beta, site=3)
'''

#genBasis = sun.GeneralBasis(alpha, beta, N)

#ls = genBasis.get_states_of_class(ec=int(1), sites=np.array([2, 3], dtype=int))




lattice = Lattice.Lattice(Ns=Ns, typeLattice='chain', isPBC=False)

'''
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
symmEngine = sun.SUNSymmetric(Ns, N, 3, alpha, lattice)
Hsymm = symmEngine.sun_hamiltonian()
Esymm, PSIsymm = scipy.sparse.linalg.eigsh(Hsymm, k=1, which='SA')
EGSsymm = Esymm[0]
print('SYMM GS Energy: ', EGSsymm)
print('SYMM GS Energy per site: ', EGSsymm/Ns)
print('------------')
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
'''

engine = sun.SUNGeneral(alpha, beta, N, lattice)
H = engine.sun_hamiltonian()

if H.shape[0]==1:
    H = H.todense()
    EGS = H[0,0]
else:
    E, PSI = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    EGS = E[0]

print('GS Energy: ', EGS)
print('GS Energy per site: ', EGS/Ns)

#######################

'''
sigmaMat, sigmaMatinverse, sigmaMatinverse2 = sun.get_orthogonal_units(alpha)
for i in range(0, len(sigmaMat)):
    sigmaMat[i] = sigmaMat[i].toarray()
    sigmaMatinverse[i] = sigmaMatinverse[i].toarray()
    sigmaMatinverse2[i] = sigmaMatinverse2[i].toarray()
'''

'''
for i in range(0, len(sigmaMat)):
    res = np.linalg.norm(sigmaMatinverse[i]-sigmaMatinverse2[i])
    if res>1.0e-14:
        print(i)
'''

'''
ortho = sun.OrthogonalUnits(alpha)

for i in range(ortho.nn):
    print('-------------')
    print(ortho.permutations[i])
    print(ortho.adjaTranspo[i])
    #print( np.linalg.norm( (ortho.sigmaMat[i] - ortho.sigmaMatv2[i]).toarray() ) )
    #print(ortho.coeffs[0,0][i])
'''

'''
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


'''
import time

N = int(3)
alpha=np.array([8, 8, 8], dtype=int)
beta=np.array([[3,0,0],[3,0,0],[3,0,0],[3,0,0],[3,0,0],[3,0,0],[3,0,0],[3,0,0]],dtype=int)


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
'''
