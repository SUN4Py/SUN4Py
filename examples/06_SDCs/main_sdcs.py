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
from sun4py.sdc.math import sdcmath

# In this example, we illustrate how to compute Subduction Coefficients (SDCs)
# of the SU(N) group using SUN4Py.
# 
# This example will reproduce Eq. (G.8), (G.14), (G.19) of:
#    Asymptotic freedom, Haldane gap and edge states of SU(N) spin chains
#    Samuel Gozel
#    EPFL, 2020
#    https://doi.org/10.5075/epfl-thesis-8417
# 

N = int(3) # SU(N)
nu = np.array([4, 4, 3], dtype=int)
nu1 = np.array([3, 3, 1], dtype=int)
nu2 = np.array([2, 1, 1], dtype=int)

Y, _, sdcs = sdcmath.get_SDC(N, nu, nu1, nu2, ref2firstLLOS=True, ref1firstLLOS=True)

# The SDCs are stored in a rank-3 numpy array with the following dimension:
#   fnu2 x fhatnu x {nu1nu2nu}
# where:
#   - fnu2 is the number of SYTs for the irrep nu2.
#   - fhatnu is the number of SYTs for the irrep nu with n1 first particles 
#     located according to nu1 (n1 is the number of boxes in nu1), and according 
#     to the selected ordering in ref1firstLLOS.
#   - {nu1nu2nu} is the multiplicity of irrep nu in the tensor product of nu1
#     with nu2
#   

for tau in range(sdcs.shape[2]):
    print('='*30)
    print(f'tau = {tau}')
    print('='*30)
    for m2 in range(sdcs.shape[0]):
        print('-'*10)
        print(f'm2 = {m2}')
        print('-'*10)
        for j in range(sdcs.shape[1]):
            if abs(sdcs[m2, j, tau])>1.0e-13:
                print(f'{sdcs[m2, j, tau]} x')
                sun.print_to_latex(Y[j])
