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

from sun4py.sun.sun import multiplicity
from sun4py.ed.edfund import EDSolverFund
from sun4py.ed.lattice import chainLattice

# In SUN4Py, performing an Exact Diagonalization (ED) calculation in a specific
# SU(N) sector is particularly simple. One needs to define a Lattice object, 
# which is essentially an abstraction of a geometric lattice seen from a 
# Hamiltonian point of view. This means that in SUN4Py, a Lattice object does 
# not carry a geometrical meaning per se, but rather defines a set of interaction
# bonds. The number of sites in the lattice must match the number of boxes in 
# the Young diagram of the target SU(N) sector under investigation.
# 
# As a side remark, we note that for the fundamental irrep of SU(N) at each site,
# only the bilinar interaction between two sites is relevant (when considering 
# two-sites interactions).

# Let's diagonalize the 12-sites SU(3) Heisenberg model on the chain lattice, 
# with fundamental irrep on each site.
# Note that the full Hilbert space dimension is N**Ns where Ns is the number of
# sites. Thus here the full Hilbert space dimension is 3**10 = 531441.
# Using the full SU(3) symmetry reduces this Hilbert space to falpha=1320, the 
# number of SYTs for the target irrep under consideration. This is two orders 
# of magnitude smaller !

N = int(3) # SU(N)

# Let's define the global target irrep. We will diagonalize the Heisenberg 
# Hamiltonian in this sector
alpha = np.array([5, 5, 2], dtype=int)

# The number of sites is inferred from the Young diagram
Ns = np.sum(alpha)

# Let's compute the Hilbert space dimension, namely the size of the Hamiltonian
# in the selected symmetry sector
falpha = multiplicity(alpha)

print(f'SU({N}) : alpha = {alpha} : Ns = {Ns} : NH = {N**Ns} : falpha = {falpha}')

# Create a chain lattice with periodic boundary conditions for Ns sites, and plot it
lattice = chainLattice(Ns=Ns, isPBC=True)
lattice.plot()

# by default, all bonds are bilinear Heisenberg with coupling constant J=1

# list all the bonds in the lattice
lattice.print()

# Generate an ED solver with a light initialization (neither the basis nor the 
# Hamiltonian is computed at this step)
engine = EDSolverFund(N, Ns, alpha, lattice)

# build the Hamiltonian matrix (This generates the basis first)
# Note that the Hamiltonian matrix is also returned by the call to sun_hamiltonian()
engine.sun_hamiltonian()

print(f'The Hamiltonian has dimension: {engine.H.shape}')

# Perform diagonalization of the Heisenberg Hamiltonian
EGS, _ = engine.diagonalize(num_eigenvalues=1)

print(f'GS Energy: {EGS[0]}')
