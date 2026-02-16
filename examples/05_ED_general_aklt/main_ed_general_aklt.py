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
from sun4py.ed.edgeneral import EDSolverGeneral
from sun4py.ed.lattice import chainLattice


# In this example, we continue the study of the AKLT model of Greiter & Rachel, 
# Ref. [1]. In the previous example, we have shown that:
# - For periodic boundary conditions, the ground state is unique.
# - For open boundary conditions, the ground state is 64-dimensional, with states
#   belonging to all irreps appearing in the tensor product of two 8-dimensional
#   adjoint irreps.
# 
# Here, we will add adjoint edge irreps at both ends of the open chain, to attempt
# to screen the edge states and recover a unique ground state.
# 
# References
# [1] Greiter & Rachel, Phys. Rev. B 75, 184441 (2007), https://doi.org/10.1103/PhysRevB.75.184441
# [2] Gozel et al., Nucl. Phys. B 945, 114663 (2019), https://doi.org/10.1016/j.nuclphysb.2019.114663
# [3] Gozel et al., Phys. Rev. Lett. 125, 057202 (2020), https://doi.org/10.1103/PhysRevLett.125.057202
# 

N = int(3) # SU(N)
Ns = int(6) # number of sites

# Let's define the 3-box symmetric irrep (of dimension 10) of SU(3), which will
# be used in the "bulk" of the chain
bulk = np.array([3, 0, 0], dtype=int)

# Let's define the adjoint irrep of SU(3), which we will use at both ends of 
# the chain
adjoint = np.array([2, 1, 0], dtype=int)

# Since the irreps will not be all the same in the system, one needs to use the 
# general ED solver for this system.
# This solver requires the user to provide the irrep used at each site.

# We create a numpy array of irreps, stored in the rows.
beta = np.tile(bulk, (Ns, 1))
# update the first and last irrep to correspond to the adjoint edge irrep
beta[0] = adjoint
beta[-1] = adjoint

# build a chain lattice with open boundary conditions
lattice = chainLattice(Ns=Ns, isPBC=False)

# biquadratic coupling
J2 = 1.0/4.0

# let's add the biquadratic coupling for all bulk links
for i in range(1, Ns-2):
    lattice.add_bond(i, i+1, 'HB2', J2)

# and plot the lattice
lattice.plot()

# Let's figure out what is the constant energy shift to be added to the 
# Hamiltonian to obtain a semi-positive definite spectrum.
# 
# First, there is the energy shift coming from the projector definition, namely
# the energy shift already encountered in the previous example. Here however, 
# we are considering a Ns-sites chain, and there is (Ns-2) symmetric irreps, 
# since the first and last sites have adjoint irreps. There are thus Ns-3 links
# in the bulk, which carry an AKLT term with a bilinear and a biquadratic 
# interaction. As a consequence, the first energy shift to apply is
Eshift = (Ns-3) * 3.0/4.0

# Now, the edge irreps are adjoint. From our guess understanding, the AKLT 
# wave-function with open boundary conditions leaves adjoint edge states. The 
# latter will couple with the added adjoint edge irreps. Let's figure out the
# energy of this links.
# 
# singlet irrep with 2 columns of N boxes, will be the lowest energy sector of 
# an antiferromagnetic Heisenberg interaction bewteen two adjoint irreps 
singlet_222 = np.array([2, 2, 2], dtype=int)
# Use the Casimir eigenvalues to determine the energy of the interaction bond
Eshift_edges = -2*(sun.casimir_quadratic(singlet_222) - 2*sun.casimir_quadratic(adjoint))
# the first factor of 2 accounts for both ends of the chain

# Let's define some global target sectors we are interested in:
alpha_000 = np.array([Ns, Ns, Ns], dtype=int) # singlet sector
alpha_210 = np.array([Ns+1, Ns, Ns-1], dtype=int) # adjoint sector
alpha_300 = np.array([Ns+2, Ns-1, Ns-1], dtype=int) # 3-box symmetric sector
alpha_330 = np.array([Ns+1, Ns+1, Ns-2], dtype=int) # conjugate of 3-box symmetric sector
alpha_420 = np.array([Ns+2, Ns, Ns-2], dtype=int) # [4, 2, 0] sector

targets = [alpha_000, alpha_210, alpha_300, alpha_330, alpha_420]

# diagonalize the Hamiltonian in each sector
for alpha in targets:
    print('='*20)
    
    engine = EDSolverGeneral(N, Ns, alpha, beta, lattice)
    engine.sun_hamiltonian()
    energy, _ = engine.diagonalize(num_eigenvalues=3)
    
    # add the energy shift
    energy += Eshift + Eshift_edges
    
    print(f'SU({N}): AKLT model with screening : {alpha} : dimension = {engine.H.shape}')
    print(f'Energy = {energy}')
    if (np.sum(abs(alpha -  np.array([Ns, Ns, Ns], dtype=int)))==0):
        assert(abs(energy[0])<1.0e-13)

# Conclusion
# - We have successfully screened the edge states ! Indeed, the ground state is
#   now unique again. This means that we have lifted 63 zero-energy states from
#   the isotropic open chain !
# - The spectrum is semi-positive definite ! This means that we have rightfully
#   determined the constant energy shift in the Hamiltonian. This shows the 
#   usefullness of SUN4Py to address simple physical considerations !
# 
# Post Conclusion
# - The reasonning presented in this example and the previous one was the key
#   to successfully study the 3-box symmetric Heisenberg model, and in particular
#   its bulk gap, using Density Matrix Renormalization Group (DMRG) [3].
# 
