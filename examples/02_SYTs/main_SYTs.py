"""
SUN4Py A Python Library for solving SU(N) Heisenberg models
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

import numpy as np
import scipy.sparse

from sun4py.sun import sun

# In SUN4Py, Standard Young Tableaux (SYTs) associated to a n-box irrep of S_n 
# (and/or, equivalently, of a n-box irrep of SU(N)) are stored as n-dimensional 
# numpy arrays of integers. The i-th element of the array (of the SYT) is the 
# row of number i in the Young diagram associated to the irrep.
# 
# On top of SYTs, which contain the row positions of each particle in the Young
# diagram, it is useful to keep track of the column positions of these particles. 
# This is equivalently the row positions in the transpose shape of the irrep, 
# namely the irrep of S_n obtained by transforming each row into a colum.

#:::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::: S_6 :::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::

# Let's take an irrep of S_6 (which corresponds to the singlet of SU(3) with 
# 6 particles)
alpha = np.array([2, 2, 2], dtype=int)

n = np.sum(alpha) # number of particles in the irrep alpha

# Let's compute the number of SYTs associated to this irrep
falpha = sun.multiplicity(alpha)

# Let's generate all SYTs associated to this irrep, ordered in the ascending 
# order of the Last Letter Order Sequence (LLOS)
Y = sun.get_SYT(alpha, order='LLOS') # use order='iLLOS' for the descending (Inverse) order of the LLOS

# SYTs are stored in the rows of Y. Y[0] is the first (namely, lowest) SYT in 
# the Last Letter Order Sequence. Y[falpha-1] is the last one.
assert(Y.shape[0]==falpha)
assert(Y.shape[1]==n)

print(f'Irrep of S_{n}: {alpha} : {falpha} SYTs')

# Let's compute the transpose shape of alpha
alphaT = sun.transpose_shape(alpha)

print(f'alpha = {alpha} : transpose shape = {alphaT}')
print('='*20)

# Let's compute the column positions
CY = sun.get_column(Y)

# Verify that the column positions of the SYTs for alpha correspond to the row
# positions of the transpose shape (CAUTION: The order is reversed)
Yt = sun.get_SYT(alphaT, order='iLLOS')
assert(np.sum(abs(CY-Yt))==0)

# Similarly, verify that getting the column positions of column positions gives
# the row positions, namely sun.get_column is involutional
assert(np.sum(abs(Y - sun.get_column(CY)))==0)

# Let's compute the axial distance from 1 to 4 (in zero-based ordering) in each
# SYT, and print each SYT with the result
# 
# For a definition of the axial distance from i to j, see for instance:
# Nataf & Mila, Phys. Rev. Lett. 113, 127204 (2014)
# https://doi.org/10.1103/PhysRevLett.113.127204
for y, cy in zip(Y, CY):
    ax = sun.get_axial_distance(y, cy, 1, 4)
    print('The axial distance from 1 to 4 in:')
    sun.print_to_latex(y, zeroBased=True)
    print(f'is: {ax}')
    print('-'*6)
print('='*20)

# Compute the eigenvalue of the quadratic Casimir operator for the irrep alpha, 
# in the language of permutations
casimir = sun.casimir_quadratic(alpha)

# get the matrices of the n-1 adjacent transpositions in the basis of SYTs
P = sun.get_adjacent_transposition_matrices(alpha, Y, CY)

# Let's build the quadratic Casimir operator in the basis of SYTs
# 
# See for example the following Reference for a definition:
#   Equation (S26) of:
# Supplemental Material of Gozel et al., Phys. Rev. Lett. 125, 057202 (2020)
# https://doi.org/10.1103/PhysRevLett.125.057202
C = scipy.sparse.csr_matrix((falpha, falpha)) # square matrix of dimension falpha, 0 everywhere
for i in range(0, n-1):
    for j in range(i+1, n):
        sigma = np.array([i, j]) # transposition (i, j)
        # transform the transposition sigma into a product of adjacent transpositions
        atr = sun.transposition_to_adjacent_transpositions(sigma)
        # build the operator representing sigma in the basis of SYTs, using the
        # matrices of the adjacent transpositions
        Csigma = scipy.sparse.eye(falpha) # identity matrix of dimension falpha
        for k in atr:
            Csigma = Csigma @ P[k]
        C += Csigma

# Let's verify that the eigenvalue of the Casimir operator corresponds to the 
# eigenvalue obtained from the shape of the irrep alpha
assert(np.linalg.norm(C.todense() - casimir*np.eye(falpha))<1.0e-13)
print(f'The eigenvalue of the permutational Casimir operator for {alpha} is: {casimir}')
