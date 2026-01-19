# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from sunpy.dmrg import dmrg
from sunpy.ed import edfund
from sunpy.ed import lattice


def bethe_infinite(N):
    """
    Compute energy per site of infinite-chain (Bethe Ansatz) for SU(N)
    """
    M = int(100)
    Eadd = np.zeros(M)
    for k in range(2, M):
        Eadd[k] = 2*(-1)**k*scipy.special.zeta(k)/(N**k)
    E = -1.0 + np.sum(Eadd)
    return E


N = int(3)
Ns = int(30) # total chain length at the end of iDMRG
Ns_min = int(2) # starting half-chain length
num_irreps = int(85)
max_num_states = int(200)

engine = dmrg.DMRG(N=N, Ns=Ns, 
                   num_irreps=num_irreps, 
                   max_num_states=max_num_states, 
                   Ns_min=Ns_min, 
                   target='GS', )
engine.idmrg()

###############################################################################

e_bethe = bethe_infinite(N)

###############################################################################

print('::::::::::::::::::::::')
print('Check with ED')
print('::::::::::::::::::::::')

ed_lengths = np.array([4, 6, 8, 10, 12], dtype=int)
ed_energies = np.zeros(shape=ed_lengths.shape, dtype=float)

for k, edNs in enumerate(ed_lengths):
    latticeedNs = lattice.Lattice(edNs, 'chain', isPBC=False)
    nc = edNs//N
    r = edNs % N
    alphaGS = np.full(shape=(N,), fill_value=nc, dtype=int)
    for i in range(0, r):
        alphaGS[i] += 1
    edfundenginefullchain = edfund.SUNFundamental(edNs, N, alphaGS, latticeedNs)
    HedNs = edfundenginefullchain.sun_hamiltonian()
    HedNs = HedNs.todense()
    EedNs, _ = np.linalg.eigh(HedNs)
    ed_energies[k] = EedNs[0]

for k, ed_Ns in enumerate(ed_lengths):
    print('Chain length: ', ed_Ns)
    print('ED energy  : ', ed_energies[k])
    print('DMRG energy: ', engine.idmrg_energy[ed_Ns//2])
    print('---------------------')

###############################################################################

fig0, ax0 = plt.subplots(1, 1, figsize=(10, 6))
ax0.set_title('Energy', size=16)
chain_lengths = np.arange(2*Ns_min, Ns+1, 2)
ind_0modN = np.argwhere(chain_lengths%N==0).flatten()
ind_1modN = np.argwhere(chain_lengths%N==1).flatten()
ind_2modN = np.argwhere(chain_lengths%N==2).flatten()
ax0.plot(1.0/chain_lengths[ind_0modN], 
         engine.idmrg_energy[Ns_min:][ind_0modN]/chain_lengths[ind_0modN],
         marker='.', color='g', label='singlet')
ax0.plot(1.0/chain_lengths[ind_2modN], 
         engine.idmrg_energy[Ns_min:][ind_2modN]/chain_lengths[ind_2modN],
         marker='.', color='r', label='[1]')
ax0.plot(1.0/chain_lengths[ind_1modN], 
         engine.idmrg_energy[Ns_min:][ind_1modN]/chain_lengths[ind_1modN],
         marker='.', color='b', label='[1, 1]')
ax0.plot(0.0, e_bethe, marker='s', color='g', linestyle='none', label='Bethe ansatz')
ax0.plot(1.0/ed_lengths, ed_energies/ed_lengths, marker='o', markerfacecolor='none', color='k', linestyle='none', label='ED')
ax0.set_xlabel('1/N_s')
ax0.set_ylabel('Energy per site')
ax0.legend()

###############################################################################
