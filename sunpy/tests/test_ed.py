"""
SUNPy A Python Library for solving SU(N) Heisenberg models
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

import pytest

import sys
import numpy as np

from sunpy.ed import lattice
from sunpy.ed import edfund
from sunpy.ed import edsymm
from sunpy.ed import edantisymm
from sunpy.ed import edgeneral


prec = 1.0e-13



@pytest.mark.parametrize("alpha, m, symmetry, N, isPBC, expected", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # FUNDAMENTAL
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=2, m=1
    (np.array([2,0], dtype=int), int(1), 'fundamental', int(2), False, 1.0),
    (np.array([3,0], dtype=int), int(1), 'fundamental', int(2), False, 2.0),
    (np.array([4,0], dtype=int), int(1), 'fundamental', int(2), False, 3.0),
    (np.array([2,1], dtype=int), int(1), 'fundamental', int(2), False, -1.0),
    (np.array([2,2], dtype=int), int(1), 'fundamental', int(2), False, -1.732050807568877),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - OBC - Singlet
    (np.array([3,3,3], dtype=int), int(1), 'fundamental', int(3), False, -6.162794759771583),
    (np.array([4,4,4], dtype=int), int(1), 'fundamental', int(3), False, -8.263188004910717),
    (np.array([5,5,5], dtype=int), int(1), 'fundamental', int(3), False, -10.366907518918115),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - PBC - Singlet
    (np.array([3,3,3], dtype=int), int(1), 'fundamental', int(3), True, -6.579435759086468),
    (np.array([4,4,4], dtype=int), int(1), 'fundamental', int(3), True, -8.624526804645157),
    # (np.array([5,5,5], dtype=int), int(1), 'fundamental', int(3), True, -10.696221973842384),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - OBC - Adjoint
    (np.array([4,3,2], dtype=int), int(1), 'fundamental', int(3), False, -5.713816495016518),
    # (np.array([5,4,3], dtype=int), int(1), 'fundamental', int(3), False, -7.908329500628006),
    # (np.array([6,5,4], dtype=int), int(1), 'fundamental', int(3), False, -10.072858996269495), # remove this test to save time
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - PBC - Adjoint
    (np.array([4,3,2], dtype=int), int(1), 'fundamental', int(3), True, -5.647199911888293),
    # (np.array([5,4,3], dtype=int), int(1), 'fundamental', int(3), True, -7.924221842740132),
    # (np.array([6,5,4], dtype=int), int(1), 'fundamental', int(3), True, -10.134975270128132), # remove this test to save time
    #:::::::::::::::::::::::::::::::::::::::::::
    # SYMMETRIC
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=2, m=2
    (np.array([2,2], dtype=int), int(2), 'symmetric', int(2), False, -2.0),
    (np.array([3,1], dtype=int), int(2), 'symmetric', int(2), False, 0.0),
    (np.array([4,0], dtype=int), int(2), 'symmetric', int(2), False, 4.0),
    (np.array([3,3], dtype=int), int(2), 'symmetric', int(2), False, 0.0),
    (np.array([4,2], dtype=int), int(2), 'symmetric', int(2), False, -2.0),
    (np.array([5,1], dtype=int), int(2), 'symmetric', int(2), False, 2.0),
    (np.array([6,0], dtype=int), int(2), 'symmetric', int(2), False, 8.0),
    (np.array([4,4], dtype=int), int(2), 'symmetric', int(2), False, -3.291502622129181),
    (np.array([5,3], dtype=int), int(2), 'symmetric', int(2), False, -2.273163042697025),
    (np.array([6,2], dtype=int), int(2), 'symmetric', int(2), False, 0.417424305044159),
    # (np.array([7,1], dtype=int), int(2), 'symmetric', int(2), False, 5.171572875253810),
    # (np.array([8,0], dtype=int), int(2), 'symmetric', int(2), False, 12.0),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=2
    (np.array([2,2,2], dtype=int), int(2), 'symmetric', int(3), False, -4.0),
    (np.array([3,2,1], dtype=int), int(2), 'symmetric', int(3), False, -3.0),
    (np.array([4,1,1], dtype=int), int(2), 'symmetric', int(3), False, 0.0),
    (np.array([4,2,0], dtype=int), int(2), 'symmetric', int(3), False, -2.0),
    (np.array([5,1,0], dtype=int), int(2), 'symmetric', int(3), False, 2.0),
    (np.array([6,0,0], dtype=int), int(2), 'symmetric', int(3), False, 8.0),
    (np.array([3,3,2], dtype=int), int(2), 'symmetric', int(3), False, -3.414213562373094),
    (np.array([4,3,1], dtype=int), int(2), 'symmetric', int(3), False, -4.130648586880582),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=2, m=3
    (np.array([3,3], dtype=int), int(3), 'symmetric', int(2), False, -3.0),
    (np.array([4,2], dtype=int), int(3), 'symmetric', int(2), False, -1.0),
    (np.array([5,1], dtype=int), int(3), 'symmetric', int(2), False, 3.0),
    (np.array([6,0], dtype=int), int(3), 'symmetric', int(2), False, 9.0),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=3
    (np.array([3,3,3], dtype=int), int(3), 'symmetric', int(3), False, -6.0),
    (np.array([4,3,2], dtype=int), int(3), 'symmetric', int(3), False, -5.0),
    (np.array([5,3,1], dtype=int), int(3), 'symmetric', int(3), False, -4.0),
    (np.array([4,4,4], dtype=int), int(3), 'symmetric', int(3), False, -3.0),
    (np.array([5,4,3], dtype=int), int(3), 'symmetric', int(3), False, -5.758895915294917),
    # (np.array([5,5,5], dtype=int), int(3), 'symmetric', int(3), False, -6.567134109811708),
    # (np.array([6,5,4], dtype=int), int(3), 'symmetric', int(3), False, -8.683291472392161),
    #:::::::::::::::::::::::::::::::::::::::::::
    # ANTISYMMETRIC
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=2
    (np.array([2,1,1], dtype=int), int(2), 'antisymmetric', int(3), False, 0.0),
    (np.array([2,2,0], dtype=int), int(2), 'antisymmetric', int(3), False, 2.0),
    (np.array([2,2,2], dtype=int), int(2), 'antisymmetric', int(3), False, 0.0),
    (np.array([3,2,1], dtype=int), int(2), 'antisymmetric', int(3), False, 1.0),
    (np.array([3,3,0], dtype=int), int(2), 'antisymmetric', int(3), False, 4.0),
    (np.array([3,3,2], dtype=int), int(2), 'antisymmetric', int(3), False, 0.585786437626905),
    (np.array([4,3,1], dtype=int), int(2), 'antisymmetric', int(3), False, 2.585786437626907),
    # (np.array([4,4,0], dtype=int), int(2), 'antisymmetric', int(3), False, 6.0),
    # (np.array([4,4,2], dtype=int), int(2), 'antisymmetric', int(3), False, 1.763932022500210),
    # (np.array([5,4,1], dtype=int), int(2), 'antisymmetric', int(3), False, 4.381966011250105),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=4, m=2
    # (np.array([3,3,3,3], dtype=int), int(2), 'antisymmetric', int(4), False, -10.598765818237256),
    # (np.array([4,3,3,2], dtype=int), int(2), 'antisymmetric', int(4), False, -7.895930963124478),
    #:::::::::::::::::::::::::::::::::::::::::::
    # GENERAL (ADJOINT)
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, Ns=2
    (np.array([2,2,2], dtype=int), int(3), 'general::adjoint', int(3), False, -3.0),
    (np.array([3,2,1], dtype=int), int(3), 'general::adjoint', int(3), False, 0.0),
    (np.array([3,3,0], dtype=int), int(3), 'general::adjoint', int(3), False, 3.0),
    (np.array([4,1,1], dtype=int), int(3), 'general::adjoint', int(3), False, 3.0),
    (np.array([4,2,0], dtype=int), int(3), 'general::adjoint', int(3), False, 5.0),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, Ns=3
    (np.array([3,3,3], dtype=int), int(3), 'general::adjoint', int(3), False, 0.0),
    (np.array([4,3,2], dtype=int), int(3), 'general::adjoint', int(3), False, -2.0),
    (np.array([4,4,1], dtype=int), int(3), 'general::adjoint', int(3), False, 1.0),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, Ns=4
    (np.array([4,4,4], dtype=int), int(3), 'general::adjoint', int(3), False, -3.928203230275505),
    (np.array([5,4,3], dtype=int), int(3), 'general::adjoint', int(3), False, -2.088218984750024),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, Ns=5
    (np.array([5,5,5], dtype=int), int(3), 'general::adjoint', int(3), False, -2.3409737774146677),
    (np.array([6,5,4], dtype=int), int(3), 'general::adjoint', int(3), False, -3.4623044036159474),
    #:::::::::::::::::::::::::::::::::::::::::::
    # GENERAL (SYMM)
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, Ns=4, m=3
    (np.array([4,4,4], dtype=int), int(3), 'general::symmetric', int(3), False, -3.0),
    (np.array([5,4,3], dtype=int), int(3), 'general::symmetric', int(3), False, -5.758895915294917),
    #:::::::::::::::::::::::::::::::::::::::::::
    # GENERAL (ANTISYMM)
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, Ns=4, m=2
    (np.array([3,3,2], dtype=int), int(2), 'general::antisymmetric', int(3), False, 0.585786437626905),
    (np.array([4,3,1], dtype=int), int(2), 'general::antisymmetric', int(3), False, 2.585786437626907),
])
def test_energy(alpha, m, symmetry, N, isPBC, expected):
    Ns = np.sum(alpha)//m
    latt = lattice.chainLattice(Ns=Ns, isPBC=isPBC)
    if m==1:
        Engine = edfund.EDSolverFund(N, Ns, alpha, latt)
    else:        
        if symmetry=='symmetric':
            Engine = edsymm.EDSolverSymm(N, Ns, alpha, m, latt)
        elif symmetry=='antisymmetric':
            Engine = edantisymm.EDSolverAntiSymm(N, Ns, alpha, m, latt)
        elif 'general' in symmetry:
            if 'adjoint' in symmetry:
                beta_loc = np.zeros(shape=N, dtype=int)
                beta_loc[0] = int(2)
                beta_loc[1:-1] = int(1)
            elif '::symmetric' in symmetry:
                beta_loc = np.zeros(shape=N, dtype=int)
                beta_loc[0] = m
            elif '::antisymmetric' in symmetry:
                beta_loc = np.zeros(shape=N, dtype=int)
                beta_loc[:m] = int(1)
            beta = np.tile(beta_loc, (Ns, 1))
            Engine = edgeneral.EDSolverGeneral(N, Ns, alpha, beta, latt)
        else:
            sys.exit('symmetry undefined.')
    
    Engine.sun_hamiltonian()
    E, _ = Engine.diagonalize(num_eigenvalues=1)
    E = E[0]
    assert(abs(E-expected)<prec)


@pytest.mark.parametrize("Ns, expected", [
    (int(4), 0.0), 
    (int(5), 0.0), 
    (int(6), 0.0), 
])
def test_aklt_su3_obc(Ns, expected):
    # arrange
    N = int(3)
    m = int(3)
    alpha_000 = np.array([Ns, Ns, Ns], dtype=int)
    alpha_210 = np.array([Ns+1, Ns, Ns-1], dtype=int)
    alpha_300 = np.array([Ns+2, Ns-1, Ns-1], dtype=int)
    alpha_330 = np.array([Ns+1, Ns+1, Ns-2], dtype=int)
    alpha_420 = np.array([Ns+2, Ns, Ns-2], dtype=int)
    latt = lattice.chainLattice(Ns=Ns, isPBC=False)
    J2 = 1.0/4.0 # biquadratic coupling
    for i in range(Ns-1):
        latt.add_bond(i, i+1, 'HB2', J2)
    # act
    Engine_000 = edsymm.EDSolverSymm(N, Ns, alpha_000, m, latt)
    Engine_000.sun_hamiltonian()
    E_000, _ = Engine_000.diagonalize(num_eigenvalues=1)
    E_000 = E_000[0] + (Ns-1) * 3.0/4.0
    
    Engine_210 = edsymm.EDSolverSymm(N, Ns, alpha_210, m, latt)
    Engine_210.sun_hamiltonian()
    E_210, _ = Engine_210.diagonalize(num_eigenvalues=2)
    E_210_0 = E_210[0] + (Ns-1) * 3.0/4.0
    E_210_1 = E_210[1] + (Ns-1) * 3.0/4.0
    
    Engine_300 = edsymm.EDSolverSymm(N, Ns, alpha_300, m, latt)
    Engine_300.sun_hamiltonian()
    E_300, _ = Engine_300.diagonalize(num_eigenvalues=1)
    E_300 = E_300[0] + (Ns-1) * 3.0/4.0
    
    Engine_330 = edsymm.EDSolverSymm(N, Ns, alpha_330, m, latt)
    Engine_330.sun_hamiltonian()
    E_330, _ = Engine_330.diagonalize(num_eigenvalues=1)
    E_330 = E_330[0] + (Ns-1) * 3.0/4.0
    
    Engine_420 = edsymm.EDSolverSymm(N, Ns, alpha_420, m, latt)
    Engine_420.sun_hamiltonian()
    E_420, _ = Engine_420.diagonalize(num_eigenvalues=1)
    E_420 = E_420[0] + (Ns-1) * 3.0/4.0
    
    # assert
    assert(abs(E_000-expected)<prec)
    assert(abs(E_210_0-expected)<prec)
    assert(abs(E_210_1-expected)<prec)
    assert(abs(E_300-expected)<prec)
    assert(abs(E_420-expected)<prec)
