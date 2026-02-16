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

import os
import numpy as np

from sun4py.ed import edfund
from sun4py.ed import lattice
from sun4py.dmrg.dmrgfund import DMRGSolverFund
from . import TESTS_PATH

prec = 1.0e-13


def energy_from_ed(N, Ns):
    """
    Extract GS energy of Ns-sites SU(N) chain with ED
    """
    ncols = Ns//N
    r = Ns % N
    alpha = np.full(shape=(N,), fill_value=ncols, dtype=int)
    for i in range(r):
        alpha[i] += 1
    latt = lattice.chainLattice(Ns, isPBC=False)
    ed_engine = edfund.EDSolverFund(N, Ns, alpha, latt)
    ed_engine.sun_hamiltonian()
    E, _ = ed_engine.diagonalize()
    return E[0]

def test_dmrg_su3():
    # arrange
    N = int(3)
    Ns = int(12)
    Ns_min = int(2)
    num_irreps = int(16)
    max_num_states = int(120)
    target = 'GS'
    tech = 'shortcut_cols'
    rme_filename = f'testdata_rmefund_SU{N}_numirreps{num_irreps}_{target}_{tech}.pickle'
    rme_filename = os.path.join(TESTS_PATH, 'testdata', rme_filename)
    
    expected = np.zeros(shape=(Ns//2+1,), dtype=float)
    for L in range(Ns_min, Ns//2+1):
        expected[L] = energy_from_ed(N, 2*L)
    
    # act
    dmrg_engine = DMRGSolverFund(N=N, 
                                 Ns=Ns, 
                                 num_irreps=num_irreps, 
                                 max_num_states=max_num_states, 
                                 target=target,
                                 Ns_min=Ns_min, 
                                 rme_filename=rme_filename)
    dmrg_engine.idmrg()
    
    # assert
    assert(all(abs(dmrg_engine.idmrg_energy - expected)<prec))
