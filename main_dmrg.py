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

import os
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from sunpy.dmrg import dmrg
from sunpy.ed import edfund
from sunpy.ed import lattice


def bethe_infinite(N):
    """
    Compute energy per site of infinite-chain (Bethe Ansatz) for SU(N)
    This is in permutational language where H = \sum_i P_{i, i+1}
    """
    M = int(100)
    Eadd = np.zeros(M)
    for k in range(2, M):
        Eadd[k] = 2*(-1)**k*scipy.special.zeta(k)/(N**k)
    E = -1.0 + np.sum(Eadd)
    return E


N = int(5) # SU(N)
Ns = int(30) # total chain length at the end of iDMRG
Ns_min = int(2) # starting half-chain length
num_irreps = int(18) # number of irreps to consider
max_num_states = int(200) # total number of "states" (SYTs) to keep
target = 'GS'

###############################################################################
# Preparation
###############################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))

# Check that the list of RMEs is already on disk
tech = 'shortcut_cols'
rme_filename = f'RME_fund_SU{N}_GS_numirreps{num_irreps}_{tech}.pickle'
foldername = os.path.join(current_dir, 'sunpy', 'rme_coefficients')
rme_filename = os.path.join(foldername, rme_filename)

if not os.path.exists(rme_filename):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    # perform calculation of RMEs
    from sunpy.dmrg.rme import rmefund
    rme_engine = rmefund.RMEEngineFund(N, num_irreps, 
                                   target=target, 
                                   tech=tech, 
                                   restarting=True)
    rme_engine.run()

###############################################################################
# RUN DMRG
###############################################################################

dmrg_engine = dmrg.DMRG(N=N, 
                        Ns=Ns, 
                        num_irreps=num_irreps, 
                        max_num_states=max_num_states, 
                        Ns_min=Ns_min, 
                        target=target)
dmrg_engine.idmrg()

###############################################################################

e_bethe = bethe_infinite(N)

###############################################################################

print('::::::::::::::::::::::')
print('Check with ED')
print('::::::::::::::::::::::')

ed_lengths = np.array([4, 6, 8, 10, 12], dtype=int)
ed_energies = np.zeros(shape=ed_lengths.shape, dtype=float)

for k, edNs in enumerate(ed_lengths):
    latticeedNs = lattice.chainLattice(Ns=edNs, isPBC=False)
    nc = edNs//N
    r = edNs % N
    alphaGS = np.full(shape=(N,), fill_value=nc, dtype=int)
    for i in range(0, r):
        alphaGS[i] += 1
    edfundenginefullchain = edfund.EDSolverFund(N, edNs, alphaGS, latticeedNs)
    HedNs = edfundenginefullchain.sun_hamiltonian()
    HedNs = HedNs.todense()
    EedNs, _ = np.linalg.eigh(HedNs)
    ed_energies[k] = EedNs[0]

for k, ed_Ns in enumerate(ed_lengths):
    print('Chain length: ', ed_Ns)
    print('ED energy  : ', ed_energies[k])
    print('DMRG energy: ', dmrg_engine.idmrg_energy[ed_Ns//2])
    print('---------------------')

###############################################################################

fig0, ax0 = plt.subplots(1, 1, figsize=(10, 6))
ax0.set_title(f'Energy SU({N})', size=16)
chain_lengths = np.arange(2*Ns_min, Ns+1, 2)
#------------------------
ind_0modN = np.argwhere(chain_lengths%N==0).flatten()
ax0.plot(1.0/np.log(chain_lengths[ind_0modN])**3, 
         dmrg_engine.idmrg_energy[Ns_min:][ind_0modN]/chain_lengths[ind_0modN],
         marker='.', color='g', label='singlet')
#------------------------
if N==3:    
    ind_1modN = np.argwhere(chain_lengths%N==1).flatten()
    ind_2modN = np.argwhere(chain_lengths%N==2).flatten()
    
    ax0.plot(1.0/np.log(chain_lengths[ind_1modN])**3, 
             dmrg_engine.idmrg_energy[Ns_min:][ind_1modN]/chain_lengths[ind_1modN],
             marker='.', color='b', label='[1]')
    
    ax0.plot(1.0/np.log(chain_lengths[ind_2modN])**3, 
             dmrg_engine.idmrg_energy[Ns_min:][ind_2modN]/chain_lengths[ind_2modN],
             marker='.', color='r', label='[1, 1]')
    
elif N==4:
    ind_2modN = np.argwhere(chain_lengths%N==2).flatten()
    
    ax0.plot(1.0/np.log(chain_lengths[ind_2modN])**3, 
             dmrg_engine.idmrg_energy[Ns_min:][ind_2modN]/chain_lengths[ind_2modN],
             marker='.', color='r', label='[1, 1]')
    
elif N==5:
    ind_1modN = np.argwhere(chain_lengths%N==1).flatten()
    ind_2modN = np.argwhere(chain_lengths%N==2).flatten()
    ind_3modN = np.argwhere(chain_lengths%N==3).flatten()
    ind_4modN = np.argwhere(chain_lengths%N==4).flatten()
    
    ax0.plot(1.0/np.log(chain_lengths[ind_1modN])**3, 
             dmrg_engine.idmrg_energy[Ns_min:][ind_1modN]/chain_lengths[ind_1modN],
             marker='.', color='b', label='[1]')
    
    ax0.plot(1.0/np.log(chain_lengths[ind_2modN])**3, 
             dmrg_engine.idmrg_energy[Ns_min:][ind_2modN]/chain_lengths[ind_2modN],
             marker='.', color='r', label='[1, 1]')
    
    ax0.plot(1.0/np.log(chain_lengths[ind_3modN])**3, 
             dmrg_engine.idmrg_energy[Ns_min:][ind_3modN]/chain_lengths[ind_3modN],
             marker='.', color='m', label='[1, 1, 1]')
    
    ax0.plot(1.0/np.log(chain_lengths[ind_4modN])**3, 
             dmrg_engine.idmrg_energy[Ns_min:][ind_4modN]/chain_lengths[ind_4modN],
             marker='.', color='brown', label='[1, 1, 1, 1]')
#------------------------
ax0.plot(0.0, e_bethe, marker='s', color='g', linestyle='none', label='Bethe ansatz')
ax0.plot(1.0/np.log(ed_lengths)**3, ed_energies/ed_lengths, marker='o', markerfacecolor='none', color='k', linestyle='none', label='ED')
ax0.set_xlabel('$1/log(N_s)^3$')
ax0.set_ylabel('$E/N_s$')
ax0.legend()

###############################################################################
