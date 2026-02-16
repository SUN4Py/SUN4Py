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

from sun4py.ed.edsymm import EDSolverSymm
from sun4py.ed.lattice import chainLattice

# In this example, we look at the Affleck-Kennedy-Lieb-Tasaki (AKLT) model for
# the 3-box symmetric irrep of dimension 10 (made of a Young diagram with a 
# single row with 3 boxes) of SU(3) at each site.
# The AKLT Hamiltonian was derived in Ref. [1], and the AKLT wave-function was
# pictured, but the investigation of this wave-function was omitted.
# Reference [2] studies the properties of this wave-function and reframes the 
# construction of the Hamiltonian in a general setting. Reference [3] rewrites
# this Hamiltonian in the permutation language, to be studied with DMRG with 
# full SU(3) symmetry.
# 
# The goal of this example, in particular, is to assess the nature of the edge
# states on an open chain.
# 
# References
# [1] Greiter & Rachel, Phys. Rev. B 75, 184441 (2007), https://doi.org/10.1103/PhysRevB.75.184441
# [2] Gozel et al., Nucl. Phys. B 945, 114663 (2019), https://doi.org/10.1016/j.nuclphysb.2019.114663
# [3] Gozel et al., Phys. Rev. Lett. 125, 057202 (2020), https://doi.org/10.1103/PhysRevLett.125.057202
# 

# Let's define the model
N = int(3) # SU(N)
m = int(3) # number of particles per site (here, in the symmetric irrep)

# We will look at a 4-sites chain (but feel free to increase this)
Ns = int(4)

# we postpone the creation of the target sector

# Let's build a chain lattice with open boundary conditions
isPBC = True
lattice_pbc = chainLattice(Ns=Ns, isPBC=isPBC)

# At this point, the lattice_pbc object describes a pure Heisenberg (namely, 
# only nearest-neightbor bilinear terms)

# Let's add the biquadratic couplings which makes it the AKLT model of 
# Greiter & Rachel, expressed in the language of permutations
# See Eq. (S30) in the Supplemental Material of Ref [3]

# biquadratic coupling
J2 = 1.0/4.0

links = lattice_pbc.links
for link in links:
    lattice_pbc.add_bond(link[0], link[1], 'HB2', J2)
lattice_pbc.plot()

# The Hamiltonian does not contain the constant energy shift which makes it
# positive definite. We will add the constant energy shift to the eigenvalues
# after diagonalization
Eshift = (Ns-1+int(isPBC)) * 3.0/4.0

# Let's define some target sectors we are interested in:
alpha_000 = np.array([Ns, Ns, Ns], dtype=int) # singlet sector
alpha_210 = np.array([Ns+1, Ns, Ns-1], dtype=int) # adjoint sector
alpha_300 = np.array([Ns+2, Ns-1, Ns-1], dtype=int) # 3-box symmetric sector
alpha_330 = np.array([Ns+1, Ns+1, Ns-2], dtype=int) # conjugate of 3-box symmetric sector
alpha_420 = np.array([Ns+2, Ns, Ns-2], dtype=int) # [4, 2, 0] sector

targets = [alpha_000, alpha_210, alpha_300, alpha_330, alpha_420]

for alpha in targets:
    print('='*20)
    
    engine = EDSolverSymm(N, Ns, alpha, m, lattice_pbc)
    engine.sun_hamiltonian()
    energy, _ = engine.diagonalize(num_eigenvalues=3)
    
    # add the energy shift
    energy += Eshift
    
    print(f'SU({N}): AKLT model : isPBC={isPBC} : {alpha} : dimension = {engine.H.shape}')
    print(f'Energy = {energy}')
    if (np.sum(abs(alpha -  np.array([Ns, Ns, Ns], dtype=int)))==0):
        assert(abs(energy[0])<1.0e-13)
    
# We make the following observations:
# - The GS of the chain (accross all sectors) is a singlet, as expected from the
#   the construction of the Hamiltonian as a projector onto singlets on each bond.
# - The energy of the GS is 0, as expected from the Hamiltonian definition.
# - All other eigenstates have strictly positive energy, namely the ground state
#   is unique.


# Now let's look at an open chain
isPBC = False
lattice_obc = chainLattice(Ns=Ns, isPBC=isPBC)
links = lattice_obc.links
for link in links:
    lattice_obc.add_bond(link[0], link[1], 'HB2', J2)
lattice_obc.plot()

# The energy shift is updated, because there is one less link
Eshift = (Ns-1+int(isPBC)) * 3.0/4.0

for alpha in targets:
    print('='*20)
    
    engine = EDSolverSymm(N, Ns, alpha, m, lattice_obc)
    engine.sun_hamiltonian()
    energy, _ = engine.diagonalize(num_eigenvalues=3)
    
    # add the energy shift
    energy += Eshift
    
    print(f'SU({N}): AKLT model : isPBC={isPBC} : {alpha} : dimension = {engine.H.shape}')
    print(f'Energy = {energy}')
    if (np.sum(abs(alpha -  np.array([Ns, Ns, Ns], dtype=int)))==0):
        assert(abs(energy[0])<1.0e-13)


# We make the following observations
# - The spectrum is semi-positive definite. There are multiple 0-energy states.
#   The zero-energy states are distributed as follows:
#       - 1 state in the singlet sector
#       - 2 states in the adjoint sector
#       - 1 state in the 3-box symmetric sector
#       - 1 state in its conjugate sector ([3 3 0])
#       - 1 state in the 27-dimensional irrep ([4 2 0])
# - Further analysis: the degenerate ground states thus appear in the irreps of
#   SU(3) which appear in the tensor product of two adjoint irreps !
#   Indeed, we have
#       [2, 1, 0] x [2, 1, 0] = [2, 2, 2] + 2x[3, 2, 1] + [4, 1, 1] + [3, 3, 0] + [4, 2, 0]
#   or equivalently, in dimensions:
#       8 x 8 = 1 + 2x8 + 10 + 10 + 27 # where the second "10" is a \bar{10}, the conjugate of 10
#   This tensor product can be obtained in SUNPy as:
#       from sun4py.sun import sun
#       adjoint = np.array([2, 1, 0], dtype=int)
#       alpha_cross = sun.tensor_product_irrep(adjoint, adjoint, int(3))
# - The zero-energy manifold thus hints at the presence of edge states belonging
#   to the adjoint irrep. This is in fact a confirmation of the predictions 
#   elaborated in Ref. [2], based on the generic construction of the AKLT model,
#   and of the study of the Matrix-Product-State form of the AKLT wave-function.
# 
# In the next example, we will screen the edge states by adding adjoint edge 
# irreps at both ends of the open chain, and show that the ground state becomes
# unique again.
