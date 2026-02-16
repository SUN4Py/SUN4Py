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

#from sun4py.sun import sun
from sun4py.ed import edfund
from sun4py.ed import edsymm
from sun4py.ed import edgeneral
import sun4py.ed.lattice

###############################################################################
# Example for ED with fundamental irrep at each site
###############################################################################

print(':::::::::::::::::::::::::::::::')
print('EXAMPLE FUNDAMENTAL IRREPS')
print(':::::::::::::::::::::::::::::::')

N = int(3) # SU(N)
alpha = np.array([4, 4, 2], dtype=int) # global target irrep
Ns = np.sum(alpha) # number of sites
latt0 = sun4py.ed.lattice.chainLattice(Ns=Ns, isPBC=True)
latt0.plot()

engine = edfund.EDSolverFund(N, Ns, alpha, latt0) # light initialization
engine.sun_hamiltonian() # build Hamiltonian matrix
EGS, _ = engine.diagonalize(num_eigenvalues=1)
print('Target irrep: ', alpha)
print('GS Energy: ', EGS[0])


###############################################################################
# Example for ED with m-box symmetric irrep at each site
###############################################################################

print(':::::::::::::::::::::::::::::::')
print('EXAMPLE SYMMETRIC IRREPS')
print(':::::::::::::::::::::::::::::::')

N = int(3) # SU(N)
m = int(3) # number of particles per site
alpha = np.array([4, 4, 4], dtype=int) # global target irrep
Ns = np.sum(alpha)//m # number of sites
isPBC = False
latt1 = sun4py.ed.lattice.chainLattice(Ns=Ns, isPBC=isPBC)

# Let's add the biquadratic couplings which make it the AKLT model of 
# Greiter & Rachel, expressed in the language of permutations
# See Eq. (S30) in the Supplemental Material of
# Gozel, Nataf, Mila, Physical Review Letters 125, 057202 (2020)
# https://doi.org/10.1103/PhysRevLett.125.057202
# And see https://doi.org/10.1103/PhysRevB.75.184441 for the original formulation
# by Greiter & Rachel
J2 = 1.0/4.0 # biquadratic coupling
for i in range(Ns-1):
    latt1.add_bond(i, i+1, 'HB2', J2)
latt1.plot()

engine = edsymm.EDSolverSymm(N, Ns, alpha, m, latt1)
engine.sun_hamiltonian()
EGS, _ = engine.diagonalize(num_eigenvalues=1)
# Let's add the constant energy shift which was omitted so far
EGS += (Ns-1+int(isPBC)) * 3.0/4.0
print('Target irrep: ', alpha)
print('GS Energy of AKLT model: ', EGS[0])
assert(abs(EGS[0])<1.0e-13)


###############################################################################
# Example for ED with general irrep at each site (here, adjoint)
###############################################################################

print(':::::::::::::::::::::::::::::::')
print('EXAMPLE GENERAL LOCAL IRREPS')
print(':::::::::::::::::::::::::::::::')

N = int(3) # SU(N)
Ns = int(5) # number of sites
alpha = np.array([6, 5, 4], dtype=int) # global target sector
beta_loc = np.array([2, 1, 0], dtype=int) # local irrep
beta = np.tile(beta_loc, (Ns, 1)) # same local irrep on each site

'''
# using different irreps at the edges
beta_loc_edge = np.array([2, 1, 0], dtype=int)
beta_loc_bulk = np.array([3, 0, 0], dtype=int)
<<<<<<< a0bd5eca243a2117f8a6d332d2adde8161610b6c
beta = np.tile(beta_loc, (Ns, 1))
=======
beta = np.tile(beta_loc, (Ns, 1)) # np.matlib.repmat(beta_loc, Ns, 1)
>>>>>>> Replace numpy.matlib.repmat by numpy.tile
beta[0] = beta_loc_edge
beta[-1] = beta_loc_edge
'''

latt2 = sun4py.ed.lattice.chainLattice(Ns=Ns, isPBC=False)
latt2.plot()

engine = edgeneral.EDSolverGeneral(N, Ns, alpha, beta, latt2)
engine.get_basis()
engine.sun_hamiltonian()
EGS, _ = engine.diagonalize(num_eigenvalues=1)
print('Target irrep: ', alpha)
print('GS Energy: ', EGS[0])



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

N = int(3) # SU(N)
Ns = int(8) # numbr of sites
alpha = np.array([Ns, Ns, Ns], dtype=int) # global target sector
beta_loc = np.array([3, 0, 0], dtype=int) # local irrep ---> symmetric irrep
<<<<<<< a0bd5eca243a2117f8a6d332d2adde8161610b6c
beta = np.tile(beta_loc, (Ns, 1))
=======
beta = np.tile(beta_loc, (Ns, 1)) # np.matlib.repmat(beta_loc, Ns, 1)
>>>>>>> Replace numpy.matlib.repmat by numpy.tile

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
