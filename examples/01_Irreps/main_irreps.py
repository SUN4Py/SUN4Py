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

from sun4py.sun import sun

# In SUN4Py, an irreducible representation of the permutation group S_n is 
# represented as a numpy array of non-ascending positive integers where the sum 
# of all the elements in the array equals n. The i-th element of the array 
# represents the length of the i-th (zero-based indexing) rows of the Young 
# diagram associated to the irrep.
# 
# From a purely mathematical point of view, and irrep of SU(N) is parametrized 
# by N-1 integers. For instance, irreps of SU(2) are parametrized by a single
# integer, 2*S, where S is the spin (S=0, 1/2, 1, 3/2, ...). These N-1 
# (non-ascending) integers can be seen as N-1 row lengths in the Young diagram 
# of the irrep. In fact, having N rows in the Young diagram of an SU(N) irrep 
# can be seen as keeping track of particle number conservation when building
# the n-boxes irrep from the tensor product of fundamental irreps, keeping in 
# mind that a fully antisymmetric state made of N particles is a singlet state.
# As such, it is invaluable to preserve the number of boxes (particles), and to 
# "over-parametrize" irreps of SU(N) by adding an N-th row to the Young diagram.
# Any column with N boxes of a Young diagram of an irrep of SU(N) can thus be
# removed to extract SU(N)-related quantities, as these boxes cancel out, as 
# they form a singlet, but it is crucial to preserve them when using technologies
# inherited from the permutation group S_n.
# 
# As a consequence, it is good practice in SUN4Py to directly define an irrep of
# SU(N) as N-dimensional array, even if some rows have length zero.

#:::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::: SU(2) ::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::

N = int(2)

# An SU(2) irrep in SUN4Py is a (N=2)-dimensional array of non-descending 
# integers. Since each column of (N=2) boxes can be removed as forming a 
# singlet, the difference between the two row-lengths corresponds to twice the 
# spin

fspin = lambda nu : (nu[0]-nu[1])/2 # get the spin value of a Young diagram (irrep)

# Let us build the fundamental irrep of SU(2), having a single box in its Young
# diagram, corresponding to spin S=1/2
alpha_fund = np.array([1, 0], dtype=int)

# Compute the dimension of this irrep with SUN4Py
dimension = sun.dim_irrep_sun(alpha_fund, N)

# the fundamental irrep of SU(N) has dimension N
assert(dimension==N)
# Recall the SU(2) formula: dimension(S) = 2*S+1
assert(dimension==2*fspin(alpha_fund)+1)

# Let's compute the eigenvalue of the quadratic Casimir operator of SU(2)
casimir = sun.casimir_quadratic_TT(alpha_fund, N)

# Recall that the eigenvalue for SU(2) is S*(S+1)
assert(casimir==fspin(alpha_fund)*(fspin(alpha_fund)+1))

print(f'SU({N}): irrep = {alpha_fund} : Spin = {fspin(alpha_fund)} : dimension = {dimension} : Casimir = {casimir}')

#------------------------

# Build the spin-1 Young diagram - spin 1 states are fully symmetric in their
# two spin-1/2 constituents, thus 2 boxes in the first row
alpha_spin1 = np.array([2, 0], dtype=int)
spin1 = fspin(alpha_spin1)
assert(spin1==1)
dimension = sun.dim_irrep_sun(alpha_spin1, N)
assert(dimension==2*spin1+1)
casimir = sun.casimir_quadratic_TT(alpha_spin1, N)
assert(casimir==spin1*(spin1+1))

print(f'SU({N}): irrep = {alpha_spin1} : Spin = {fspin(alpha_spin1)} : dimension = {dimension} : Casimir = {casimir}')

#------------------------

# add one column with (N=2) boxes to the spin-1 irrep of SU(2)
alpha_spin1 = np.array([3, 1], dtype=int)
spin1 = fspin(alpha_spin1)
assert(spin1==1)
dimension = sun.dim_irrep_sun(alpha_spin1, N)
assert(dimension==2*spin1+1)
casimir = sun.casimir_quadratic_TT(alpha_spin1, N)
assert(casimir==spin1*(spin1+1))

print(f'SU({N}): irrep = {alpha_spin1} : Spin = {fspin(alpha_spin1)} : dimension = {dimension} : Casimir = {casimir}')
print('-'*30)

#:::::::::::::::::::::::::::::::::::::::::::
#:::::::::::::::::: SU(3) ::::::::::::::::::
#:::::::::::::::::::::::::::::::::::::::::::

N = int(3)

# Let's build the fundamental irrep of SU(3)
alpha = np.array([1, 0, 0], dtype=int)

# get the dimension
dimension = sun.dim_irrep_sun(alpha, N)
assert(dimension==N) # the fundamental irrep of SU(N) has dimension N

# and its Casimir
casimir = sun.casimir_quadratic_TT(alpha, N)

print(f'SU({N}): irrep = {alpha} : dimension = {dimension} : Casimir = {casimir}')

#------------------------

# Let's build the adjoint irrep of SU(3)
alpha = np.array([2, 1, 0], dtype=int)

# get the dimension
dimension = sun.dim_irrep_sun(alpha, N)

# the adjoint irrep of SU(N) has dimension N**2-1
assert(dimension==N*N-1)

# and its Casimir
casimir = sun.casimir_quadratic_TT(alpha, N)

# the adjoint irrep of SU(N) has Casimir eigenvalue N
assert(casimir==N)

print(f'SU({N}): irrep = {alpha} : dimension = {dimension} : Casimir = {casimir}')
print('-'*30)

#------------------------

# Let's generate the genealogy of the adjoint irrep of SU(3)

# Start with the ascendants shapes:
alpha_asc, _ = sun.get_direct_ascendants(alpha, N)

for asc in alpha_asc:
    print(f'SU({N}): irrep {asc} is an ascendant of {alpha}')
print('-'*30)

# Now compute the descendants:
alpha_desc, _ = sun.get_direct_descendants(alpha, N)
for desc in alpha_desc:
    print(f'SU({N}): irrep {desc} is a descendant of {alpha}')

print('-'*30)

#------------------------

# Let's compute the tensor product of two adjoint irreps of SU(3)

alpha_cross = sun.tensor_product_irrep(alpha, alpha, N)

dimension_tot = int(0)

for res in alpha_cross:
    dimension_res = sun.dim_irrep_sun(res, N)
    dimension_tot += dimension_res
    print(f'SU({N}): {alpha} x {alpha} ---> {res} (dimension: {dimension_res})')

assert(sun.dim_irrep_sun(alpha, N)**2==dimension_tot)
print(f'Dimension({alpha} x {alpha}) = {sun.dim_irrep_sun(alpha, N)**2} : sum of dimensions in decomposition: {dimension_tot}')
