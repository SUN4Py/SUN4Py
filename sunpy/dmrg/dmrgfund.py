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
import time
import numpy as np
import scipy.linalg

from sunpy.dmrg.dmrgbase import DMRGSolver
from sunpy.dmrg.rme.rmereader import RMEReader
from sunpy.ed.edfund import EDSolverFund
from sunpy.ed.lattice import chainLattice
from sunpy.sun import sun
import sunpy.common.math



class DMRGSolverFund(DMRGSolver):
    """
    Class to perform Density Matrix Renormalization Group calculation on a 
    Heisenberg chain with local fundamental irrep of SU(N) at each site
    """
    
    def __init__(self, N, Ns, num_irreps, max_num_states, target, **kwargs):
        """
        Constructor
        
        Parameters
        ----------
        N : int
            SU(N)
        Ns : int
            chain length (at the end of iDMRG, each half-chain has length Ns/2)
        num_irreps : int
            number of irreps
        max_num_states : int
            max number of states to keep
        target : str
            target irrep
        Ns_min : int [optional][default: 4]
            initial half-chain length
        rme_filename : str [optional]
            path to filename of list of RME
        lanczos_max_iter : int [optional][default: 200]
            maximum number of Lanczos iterations
        lanczos_tol_residual : float [optional][default: 1.0e-13]
            tolerance on residual norm of eigenpairs
        lanczos_tol_ritz : float [optional][default: 1.0e-13]
            relative accuracy on eigenvalues
        
        Remarks
        -------
        - the 'target' is a string describing the target sector (sector for the
          entire chain). target='GS' means that the target sector will be:
                - the singlet sector when the total chain length is a multiple of N
                - the fundamenetal sector when the total chains length has a remainder of 1 modulo N
                - etc ...
          RMEs must have been computed accordingly
        - To target other sectors, proceed as follows:
            - define a str describing the new target rule    
            - update _target_irrep to define the Young diagrams associated to
              the newly defined target irrep
            - Perform similarly with dmrg.rme.rmefund to compute the relevant reduced
              matrix elements of the interaction
        """
        
        super().__init__(N, Ns, num_irreps, max_num_states, target, **kwargs)
        self._m = int(1)
        self._local_dimension = self._N
        
        # add a column of N boxes to each irrep (to easily identify a bottom corner in last row)
        self._irreps = self._irreps0 + np.full(shape=(self._num_irreps, self._N), fill_value=1, dtype=int)
        
        if 'rme_filename' in kwargs:
            self._rme_filename = kwargs['rme_filename']
        else:
            user_dir = os.getcwd()
            rme_dir = os.path.join(user_dir, 'sunpy_rmes')
            self._rme_filename = ''
            temp_f = f'RME_fund_SU{self._N}_{self._target}_numirreps{num_irreps}_'
            for tech in ['shortcut_cols', 'shortcut_rows', 'base']:
                file = temp_f + tech + '.pickle'
                file = os.path.join(rme_dir, file)
                if os.path.isfile(file):
                    self._rme_filename = file
                    break
            if self._rme_filename=='':
                print('WARNING : DMRGSolverFund : __init__ : RMEs not found. Computing RMEs now ...')
                from sunpy.dmrg.rme import RMEEngineFund
                tech = 'shortcut_cols'
                rmefund_engine = RMEEngineFund(self._N, 
                                               self._num_irreps, 
                                               target=self._target, 
                                               tech=tech, 
                                               restarting=True, 
                                               checkpointing=False)

                rmefund_engine.run()
                self._rme_filename = f'RME_fund_SU{self._N}_{self._target}_numirreps{num_irreps}_{tech}.pickle'
                self._rme_filename = os.path.join(rme_dir, self._rme_filename)
                if not os.path.isfile(self._rme_filename):
                    raise FileNotFoundError('DMRGSolverFund: __init__ : RMEs not found after computation.')
        
        self._rme_reader = RMEReader(self._N, self._num_irreps, self._rme_filename, m=int(1))
        
        self._do_check = False # set to True for checking versus ED on first iterations of iDMRG
        self._test_Ns_threshold = int(6) # tests will be performed up to this length (half-chain)
        
        print('DMRG Engine is now ready for iDMRG')
        
        return
    
    
    def idmrg(self):
        """
        Perform infinite-size DMRG, growing the half-chain from self._Ns_min 
        sites to self._Ns//2
        """
        print('Start iDMRG')
        
        self._idmrg_init()
        
        for n in range(self._Ns_min+1, self._Ns//2+1):
            print('--------------------------------------------------------------')
            print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
            print('iDMRG: n = ', n, ', Ns/2 = ', self._Ns//2)
            print('--------------------------------------------------------------')
                        
            TAB2 = self._valid_irreps(n)
            self._irreps_n[n] = TAB2
            
            # Perform states selection and DMRG truncation
            self._select_states_step1(n)
            Keep_val, ind_alpha_ascs = self._select_states_step2(n)
            
            # Measure total discarded weight
            self._get_discarded_weight(n, Keep_val)
            
            # Compute Hamiltonian on half-chain
            self._get_left_block_hamiltonians(n, ind_alpha_ascs)
            
            # Target irreps for the two blocks
            alphaGS = self._target_irrep(n)
            
            tensor_bool = self._get_tensor_bool(alphaGS, TAB2)
            
            ind_relevant_irreps, ind_irrelevant_irreps = self._analyze_tensor_bool(tensor_bool, self._num_states[n])
            
            bool_tensor_vec = self._get_bool_tensor_vec(tensor_bool, self._num_states[n])
            
            # build block-diagonal Hamiltonian on half-chain for relevant irreps
            Ai = self._get_block_left_H(n, ind_relevant_irreps)
            
            # build HLR
            HLR = self._get_HLR(n, tensor_bool, ind_relevant_irreps)
            
            # Perform Lanczos
            
            idmrg_energy, GS = self._diagonalize(n, HLR, Ai, bool_tensor_vec)

            #::::::::::::::::::
            # Sanity checks
            #::::::::::::::::::
            # idmrg_energy_v0, GS_v0 = self._lanczos_dmrg_full_form(n, HLR, Ai)
            #if (GS_v0[0]*GS[0]<0):
            #    GS_v0 *= -1
            
            #assert(abs(idmrg_energy - idmrg_energy_v0)<1.0e-12)
            #print('Error on 2 wrt base: ', np.sum(abs(GS_v0 - GS)))
            
            #lanczos_multiply = lambda v : self._multiply(HLR, Ai, bool_tensor_vec, v)
            #print('On base: np.sum(abs(H*GS - E*GS)) = ', np.sum(abs(lanczos_multiply(GS_v0) - idmrg_energy_v0*GS_v0)))
            #print('np.sum(abs(H*GS - E*GS)) = ', np.sum(abs(lanczos_multiply(GS) - idmrg_energy*GS)))
            #print('Difference on GS: ', np.sum(abs(GS - GS_v0)))
            #::::::::::::::::::
            # End sanity checks
            #::::::::::::::::::
            
            print('--------------------------')
            #print('EIGSH  : The energy at n=', n, ' (thus total chain length = ', 2*n, ') is: ', idmrg_energy_v0)
            print('LANCZOS: The energy at n=', n, ' (thus total chain length = ', 2*n, ') is: ', idmrg_energy)
            print('--------------------------')
            
            self.idmrg_energy[n] = idmrg_energy
            
            # Compute density matrices
            # for relevant irreps
            self._density_matrices_and_Hrotate_relevant_irreps(n, GS, TAB2, ind_relevant_irreps)            
            # for non-relevant irreps
            self._density_matrices_and_Hrotate_irrelevant_irreps(n, ind_irrelevant_irreps)
        
        return
    
    
    def _idmrg_init(self):
        """
        Initialization step in iDMRG, namely construction of the chain with 
        2*self._Ns_min sites
        """
        
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        print('iDMRG initialization ...')
        tstart = time.perf_counter()
        
        TAB0 = self._valid_irreps(self._Ns_min-1)
        i_TAB0 = TAB0.shape[0]
        TAB1 = self._valid_irreps(self._Ns_min)
        i_TAB1 = TAB1.shape[0]
        
        self._irreps_n[self._Ns_min-1] = TAB0
        self._irreps_n[self._Ns_min] = TAB1
        
        Genealogy_init = np.zeros(shape=(i_TAB1, i_TAB0), dtype=int)
        Genealogy_init_acc = np.zeros(shape=(i_TAB1, i_TAB0), dtype=int)
        
        # Genealogy_init[j, k] = number of SYTs for irrep TAB1[j] 
        #                          stemming from ascendant irrep TAB0[k]
        # Genealogy_init_acc[j, k] = total number of SYTs for irrep TAB1[j]
        #                              accumulated over all ascendants irreps TAB0[:k]
        
        self._H[self._Ns_min] = []
        
        num_states_init = np.zeros(shape=(i_TAB1,), dtype=int)
        
        lattice = chainLattice(Ns=self._Ns_min, isPBC=False)
        
        for q in range(0, i_TAB1):
            alpha = np.copy(TAB1[q])
            num_states_init[q] = sun.multiplicity(alpha)            
            # compute Hamiltonian for target irrep alpha
            ed_engine = EDSolverFund(self._N, 
                                     self._Ns_min, 
                                     alpha, 
                                     lattice, 
                                     basisOrder='iLLOS')
            self._H[self._Ns_min].append(ed_engine.sun_hamiltonian())
            
            bcs = sun.get_bottom_corner(alpha)
            Y = sun.get_SYT(alpha, order='iLLOS')
            cptsyt = int(0)
            for bc in bcs:
                alpha_asc = np.copy(alpha)
                alpha_asc[bc] -= 1
                index_asc = sunpy.common.math.find_row(TAB0, alpha_asc)
                assert(len(index_asc)==1)
                index_asc = index_asc[0]
                Genealogy_init[q, index_asc] = len(np.argwhere(Y[:, self._Ns_min-1]==bc))
                cptsyt += Genealogy_init[q, index_asc]
            
            for t in range(1, i_TAB0):
                Genealogy_init_acc[q, t] = np.sum(Genealogy_init[q, :t])
        
        self._num_states[self._Ns_min] = num_states_init
        self._Genealogy[self._Ns_min] = Genealogy_init
        self._Genealogy_acc[self._Ns_min] = Genealogy_init_acc
        
        # build irrep of target with 2*Ns_min sites
        alphaGS = self._target_irrep(self._Ns_min)
        
        # search which irreps of TAB1 combine into the target alphaGS
        tensor_bool_min = self._get_tensor_bool(alphaGS, TAB1)
        
        ind_relevant_irreps_init, ind_irrelevant_irreps_init = self._analyze_tensor_bool(tensor_bool_min, self._num_states[self._Ns_min])
        
        Ai_init = self._get_block_left_H(self._Ns_min, ind_relevant_irreps_init)
        
        # Compute HLR using the reduced matrix elements of the interaction, by used of SDCs
        HLR_init = self._get_HLR(self._Ns_min, tensor_bool_min, ind_relevant_irreps_init)
        
        # Diagonalize Hamiltonian on full chain
        energy_init, GS = self._lanczos_dmrg_full_form(self._Ns_min, HLR_init, Ai_init)
        
        print('DMRG: GS energy of chain with ', 2*self._Ns_min, ' sites: ', energy_init)
        self.idmrg_init_energy = energy_init
        self.idmrg_energy[self._Ns_min] = energy_init
        
        # density matrices for relevant irreps
        self._density_matrices_and_Hrotate_relevant_irreps(self._Ns_min, GS, TAB1, ind_relevant_irreps_init)
        
        # density matrices for irrelevant irreps
        self._density_matrices_and_Hrotate_irrelevant_irreps(self._Ns_min, ind_irrelevant_irreps_init)
        
        tend = time.perf_counter()
        print('Time iDMRG init = {}s'.format(tend-tstart))
        print('done.')
        
        return
    
    
    def _valid_irreps(self, Ns):
        """
        Find all relevant irreps with Ns boxes
        """        
        # find indices of irreps which have less or equal than Ns boxes, and 
        # whose number of boxes modulo N matches Ns (mod. N)
        ind = np.argwhere( (Ns>=np.sum(self._irreps0, axis=1)) & 
                           (Ns%self._N==np.sum(self._irreps0, axis=1)%self._N) ).flatten()
        
        irreps = np.copy(self._irreps0[ind])
        # add columns of N boxes to match ns
        for i in range(0, irreps.shape[0]):
            q = (Ns - np.sum(irreps[i])) // self._N
            irreps[i] += np.full(shape=(self._N,), fill_value=q, dtype=int)
        return irreps
        
    
    def _target_irrep(self, L):
        """
        Return the irrep of the target sector, for a chain with 2*L sites
        """
        if self._target=='GS':
            nc = 2*L // self._N
            alpha = np.full(shape=(self._N,), fill_value=nc, dtype=int)
            r = 2*L - self._N*nc
            ind = np.argwhere(np.sum(self._irreps_all, axis=1)==r).flatten()
            casimir = np.zeros(shape=ind.shape)
            for i in range(0, len(ind)):
                casimir[i] = sun.casimir_quadratic(self._irreps_all[ind[i]])
            indc = np.argmin(casimir)
            alpha += self._irreps_all[ind[indc]]
        else:
            raise ValueError('DMRGSolverFund : _target_irrep : Target undefined.')        
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Target irrep: ', alpha)
        return alpha
    
    
    def _get_ascendants_indices(self, alpha, TAB1):
        """
        Find the indices in TAB1 of all the ascendants of input irrep alpha
        
        Parameters
        ----------
        alpha : numpy array
            irrep
        TAB1 : numpy array
            collection of irreps with 1 box less than in alpha, stored in the rows
        
        Returns
        -------
        ind_alpha_ascs : numpy array
            indices
        bcs : numpy array
            positions (rows) of bottom corners in alpha
        """
        bcs = sun.get_bottom_corner(alpha)
        ind_alpha_ascs = np.zeros(self._N, dtype=int)
        cpt = int(0)
        for bc in bcs:
            # build ascendant irrep
            alpha_asc = np.copy(alpha)
            alpha_asc[bc] -= 1
            # search index in TAB1
            ind_asc = sunpy.common.math.find_row(TAB1, alpha_asc)
            if len(ind_asc)>0:
                ind_alpha_ascs[cpt] = ind_asc[0]
                cpt += 1
        ind_alpha_ascs = ind_alpha_ascs[:cpt]
        
        return ind_alpha_ascs, bcs
    
    
    def _select_states_step1(self, n):
        """
        DMRG truncation step 1: Determine how many states to keep in each 
        symmetry sector
        """
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Start states selection')
        tstart = time.perf_counter()
        
        TAB1 = self._irreps_n[n-1]
        TAB2 = self._irreps_n[n]
        i_TAB2 = TAB2.shape[0]
        
        rho_eigvals_all = np.array([], dtype=float)
        ind_rho_eigvals_all = np.array([], dtype=int)
        cpt_states = int(0)
        
        # TO DO : parallel loop on p, with each thread accumulating to its own 
        # rho_eigvals_all, ind_rho_eigvals_all, and a critical construct to 
        # build the common final rho_eigvals_all, ind_rho_eigvals_all
        for p in range(0, i_TAB2):
            # get the indices
            ind_alpha_ascs, _ = self._get_ascendants_indices(TAB2[p], TAB1)
            # accumulate the eigenvalues of the density matrix of each ascendant shape
            for s in range(0, len(ind_alpha_ascs)):
                ltemp = self._num_states[n-1][ind_alpha_ascs[s]]
                rho_eigvals_all = np.hstack((rho_eigvals_all, self._rho_eigvals[n-1][ind_alpha_ascs[s]]))
                ind_rho_eigvals_all = np.hstack((ind_rho_eigvals_all, np.full(ltemp, fill_value=p, dtype=int)))
                cpt_states += ltemp
        
        # DMRG truncation
        self._num_states[n] = np.zeros(i_TAB2, dtype=int)
        if cpt_states>self._max_num_states:
            print('Truncation from ', cpt_states, ' to ', self._max_num_states, 'states')
            self._truncated = True
        num_states = min(self._max_num_states, cpt_states)
        
        # find  indices which sort all eigenvalues of density matrices in 
        # descending order
        index_sort = np.argsort(rho_eigvals_all)
        index_sort = index_sort[::-1]
        
        # keep the first num_states indices
        index_sort = index_sort[:num_states]
        
        # apply the ordering
        ind_rho_eigvals_kept = ind_rho_eigvals_all[index_sort]
        
        # count the number of states kept for each irrep
        for p in range(0, i_TAB2):
            self._num_states[n][p] = len(np.argwhere(ind_rho_eigvals_kept==p).flatten())
        
        tend = time.perf_counter()
        print('Time = {}s'.format(tend-tstart))
        print('done.')
        
        self._print_distribution(TAB2, self._num_states[n])
        
        return
    
    
    def _select_states_step2(self, n):
        """
        DMRG truncation step 2: Truncate states in each symmetry sector, keeping
        the ones with the largest eigenvalues of the density matrices
        """
        TAB1 = self._irreps_n[n-1]
        TAB2 = self._irreps_n[n]
        i_TAB1 = TAB1.shape[0]
        i_TAB2 = TAB2.shape[0]
        
        Genealogy = np.zeros(shape=(i_TAB2, i_TAB1), dtype=int)
        Genealogy_acc = np.zeros(shape=(i_TAB2, i_TAB1), dtype=int)
        Keep_val = np.zeros(i_TAB2)
        ind_alpha_ascs = [None] * i_TAB2
        
        # TO DO : this is a parallel loop on p
        for p in range(0, i_TAB2):
            
            # get indices of ascendant shapes, and position of selected bottom corner
            ind_alpha_ascs[p], bcs = self._get_ascendants_indices(TAB2[p], TAB1)
            
            # accumulate the eigenvalues of the density matrix for sector p 
            # from each ascendant shape
            rho_eigvals = np.array([], dtype=float)
            ind_rho_eigvals = np.array([], dtype=int)
            cpt = int(0)
            for s in range(0, len(ind_alpha_ascs[p])):
                ltemp = self._num_states[n-1][ind_alpha_ascs[p][s]]
                rho_eigvals = np.hstack((rho_eigvals, self._rho_eigvals[n-1][ind_alpha_ascs[p][s]]))
                ind_rho_eigvals = np.hstack((ind_rho_eigvals, np.full(ltemp, fill_value=ind_alpha_ascs[p][s], dtype=int)))
                cpt += ltemp
            
            # find indices which sort in descending order accross all ascendant sectors
            index_sort = np.argsort(rho_eigvals)
            index_sort = index_sort[::-1]
            
            # perform DMRG truncation
            index_sort = index_sort[:self._num_states[n][p]]
            rho_eigvals = rho_eigvals[index_sort]
            ind_rho_eigvals = ind_rho_eigvals[index_sort]
            
            # kept weight in sector TAB2[p]
            Keep_val[p] = np.sum(rho_eigvals)
            
            # Build genealogy of irrep TAB2[p]
            cpt_st = int(0)
            for s in range(0, len(ind_alpha_ascs[p])):
                Genealogy[p, ind_alpha_ascs[p][s]] = len(np.argwhere(ind_rho_eigvals==ind_alpha_ascs[p][s]).flatten())
                cpt_st += Genealogy[p, ind_alpha_ascs[p][s]]
            
            for t in range(1, i_TAB1):
                Genealogy_acc[p, t] = np.sum(Genealogy[p, :t])
        
        self._Genealogy[n] = Genealogy
        self._Genealogy_acc[n] = Genealogy_acc
        
        return Keep_val, ind_alpha_ascs
                
    
    def _get_left_block_hamiltonians(self, n, ind_alpha_ascs):
        """
        Compute the Hamiltonian of the left (or right) block in each sector
        """
        TAB0 = self._irreps_n[n-2]
        TAB1 = self._irreps_n[n-1]
        TAB2 = self._irreps_n[n]
        i_TAB2 = TAB2.shape[0]
        
        Genealogy = self._Genealogy[n]
        Genealogy_acc = self._Genealogy_acc[n]
        
        self._H[n] = [None] * i_TAB2
        
        # TO DO : this is a parallel loop on p, with a critical construct when
        # printing to screen (checks wrt ED)
        for p in range(0, i_TAB2):
            
            ind_asc = ind_alpha_ascs[p]
            
            # Build the block-diagonal part of the Hamiltonian stemming from
            # each ascendant irrep of TAB2[p]
            H_asc = scipy.sparse.csr_matrix((self._num_states[n][p], self._num_states[n][p]))
            cpt = int(0)
            for s in range(0, len(ind_asc)):
                Gpis = Genealogy[p, ind_asc[s]]
                assert(self._H[n-1][ind_asc[s]].shape[0]>=Gpis)
                H_asc[cpt:cpt+Gpis, cpt:cpt+Gpis] = self._H[n-1][ind_asc[s]][:Gpis, :Gpis]
                cpt += Gpis
            assert(self._num_states[n][p]==cpt)
            
            # Build the part of the Hamiltonian which couples different ascendants
            # irreps sharing a common "grand-father" irrep
            
            #Hcoffdiag = np.zeros(shape=(self._num_states[n][p], self._num_states[n][p]))
            #Hcdiag = np.zeros(shape=(self._num_states[n][p], self._num_states[n][p]))
            Hcdiag_sparse = scipy.sparse.csr_matrix((self._num_states[n][p], self._num_states[n][p]))
            Hcoffdiag_sparse = scipy.sparse.csr_matrix((self._num_states[n][p], self._num_states[n][p]))
            
            for h in range(0, len(ind_asc)):
                
                A = self._rho_eigvecs[n-1][ind_asc[h]]
                
                # search for the indices of the ascendants of the h-th ascendant of TAB2[p]
                # these are the "grand-mother" irreps
                ind_asc_asc = np.argwhere(self._Genealogy[n-1][ind_asc[h], :]>0).flatten()
                
                gpiph = Genealogy[p, ind_asc[h]]
                Htemp = np.zeros(shape=(gpiph, gpiph))
                
                for k in range(0, len(ind_asc_asc)):
                    
                    # search for all the "sister" irreps of ind_asc[h], namely all irreps having
                    # the same "grand-mother" irrep
                    ind_sister = np.argwhere(self._Genealogy[n-1][ind_asc[h]+1:, ind_asc_asc[k]]>0).flatten()
                    ind_sister += ind_asc[h] + 1
                    
                    # index of the row of the bottom corner where there is the next to last box (--> n-2)
                    bc1 = np.argwhere(abs(TAB1[ind_asc[h]] - TAB0[ind_asc_asc[k]])>0).flatten()
                    assert(len(bc1)==1)
                    bc1 = bc1[0]
                    
                    # index of the row of the bottom corner where there is the last box (--> n-1)
                    bc2 = np.argwhere(abs(TAB1[ind_asc[h]] - TAB2[p])>0).flatten()
                    assert(len(bc2)==1)
                    bc2 = bc2[0]
                    
                    # position of next to last box (n-2)
                    yA1 = bc1
                    cA1 = TAB1[ind_asc[h], yA1] - 1
                    # position of last box (n-1)
                    yA2 = bc2
                    cA2 = TAB2[p, yA2] - 1
                    # axial distance from (n-2) to (n-1)
                    rhoA = 1.0/(cA1 - yA1 - cA2 + yA2)
                    
                    l_A = self._Genealogy[n-1][ind_asc[h], ind_asc_asc[k]]
                    ind_st_A = np.sum(self._Genealogy[n-1][ind_asc[h], :ind_asc_asc[k]])                    

                    # Diagonal part
                    DA_temp = A[ind_st_A:(ind_st_A+l_A), :gpiph]
                    Htemp += (-rhoA) * (DA_temp.transpose() @ DA_temp)
                    
                    # Off-diagonal part
                    for t in range(0, len(ind_sister)):
                        B = self._rho_eigvecs[n-1][ind_sister[t]]
                        l_B = self._Genealogy[n-1][ind_sister[t], ind_asc_asc[k]]
                        ind_st_B = np.sum(self._Genealogy[n-1][ind_sister[t], :ind_asc_asc[k]])
                        
                        l_A = min(l_A, l_B)
                        l_B = min(l_A, l_B)
                        
                        A_temp = A[ind_st_A:(ind_st_A+l_A), :gpiph]
                        B_temp = B[ind_st_B:(ind_st_B+l_B), :Genealogy[p, ind_sister[t]]]
                        
                        M = np.sqrt(1.0-rhoA**2) * A_temp.transpose() @ B_temp
                        # M: dimension Genealogy[p, ind_asc[h]] x Genealogy[p, ind_sister[t]]

                        indx = Genealogy_acc[p, ind_asc[h]] + np.arange(0, gpiph)
                        indy = Genealogy_acc[p, ind_sister[t]] + np.arange(0, Genealogy[p, ind_sister[t]])
                        '''
                        #-------------------
                        # full version
                        #-------------------
                        # fill the coupling part of the Hamiltonian
                        for ii, iindy in enumerate(indy):
                            Hcoffdiag[indx, iindy] = M[:, ii].flatten()
                        '''
                        #-------------------
                        # sparse version
                        #-------------------
                        indx_sp = np.repeat(indx, repeats=len(indy))
                        indy_sp = np.tile(indy, len(indx))
                        Hcoffdiag_sparse += scipy.sparse.csr_matrix((np.reshape(M, -1), (indx_sp, indy_sp)), 
                                                         (self._num_states[n][p], self._num_states[n][p]))
                        #assert(np.linalg.norm(Hcoffdiag_sparse.todense() -  Hcoffdiag)<1.0e-10)
                        
                        
                # end for k

                indx = Genealogy_acc[p, ind_asc[h]] + np.arange(0, gpiph)                
                '''
                #-------------------
                # full version
                #-------------------
                # fill the block-diagonal part of the Hamiltonian
                for ii, iindy in enumerate(indx):
                    Hcdiag[indx, iindy] = Htemp[:, ii].flatten()
                '''
                #-------------------
                # sparse version
                #-------------------
                indx_sp = np.repeat(indx, repeats=gpiph)
                indy_sp = np.tile(indx, gpiph)
                Hcdiag_sparse += scipy.sparse.csr_matrix((np.reshape(Htemp, -1), (indx_sp, indy_sp)), 
                                                         (self._num_states[n][p], self._num_states[n][p]))
                #assert(np.linalg.norm(Hcdiag_sparse.todense() -  Hcdiag)<1.0e-12)
                
            # end for h
            
            #self._H[n][p] = H_asc + Hcdiag + Hcoffdiag + Hcoffdiag.transpose()
            self._H[n][p] = H_asc + Hcdiag_sparse + Hcoffdiag_sparse + Hcoffdiag_sparse.transpose()
            
            # Check versus full ED to compute Hamiltonian in sector TAB2[p]
            if ((self._truncated==False) & (self._do_check==True) & (n<=self._test_Ns_threshold)):
                print('--------------------------')
                print('Perform check of Hamiltonian of left block in irrep: ', TAB2[p])
                test_dmrg_energy, _ = np.linalg.eigh(self._H[n][p].todense())
                #----------------------------
                test_lattice = chainLattice(Ns=np.sum(TAB2[p]), isPBC=False)
                test_edengine = EDSolverFund(N=self._N, 
                                             Ns=np.sum(TAB2[p]), 
                                             alpha=TAB2[p], 
                                             lattice=test_lattice)
                test_H = test_edengine.sun_hamiltonian().todense()
                test_ed_energy, _ = np.linalg.eigh(test_H)
                if (np.sum(abs(test_ed_energy-test_dmrg_energy))>1.0e-13):
                    print('Energy unmatch')
                    print('Energies from DMRG: ')
                    print(test_dmrg_energy)
                    print('Energies from ED: ')
                    print(test_ed_energy)
                    raise ValueError('Check with ED energy failed.')
                else:
                    print('GS energy = ', test_dmrg_energy[0])
                    print('Check ED OK.')
        
        return
    
    
    def _get_HLR(self, n, tensor_bool, ind_relevant_irreps):
        """
        Compute the matrix HLR, which couples the left block to the right block,
        using the Reduced Matrix Elements of the interaction previously obtained
        from the Subduction Coefficients
        """
        
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Computing HLR ...')
        tstart = time.perf_counter()
        
        num_states = self._num_states[n]
        TAB = self._irreps_n[n]
        TAB1 = self._irreps_n[n-1]
        
        num_relevant_irreps = len(ind_relevant_irreps)
        num_relevant_states = np.sum(num_states[ind_relevant_irreps])
        dim_superblock = num_relevant_states**2
        
        HLR = scipy.sparse.csr_matrix((dim_superblock, dim_superblock), dtype=float)
        
        vecHLRx = np.zeros(10*dim_superblock, dtype=int)
        vecHLRy = np.zeros(10*dim_superblock, dtype=int)
        vecHLRvalues = np.zeros(10*dim_superblock)
        index_cat = int(0)
        
        for q in range(0, num_relevant_irreps):
            print('--------------------------')
            print('irrep ', q, '/', num_relevant_irreps)
            
            nu1 = TAB[ind_relevant_irreps[q],:]
            
            ind_nu1 = sunpy.common.math.find_row(TAB[ind_relevant_irreps], nu1)
            assert(len(ind_nu1)==1)
            ind_nu1 = ind_nu1[0]
            assert(ind_nu1==q)
            
            ind_g_nu1 = sunpy.common.math.find_row(TAB, nu1)
            assert(len(ind_g_nu1)==1)
            ind_g_nu1 = ind_g_nu1[0]
            assert(ind_g_nu1==ind_relevant_irreps[q])
            assert(ind_relevant_irreps[ind_nu1]==ind_g_nu1)
            
            bcs1 = sun.get_bottom_corner(nu1)
            
            nu1_friends_indices = np.argwhere(tensor_bool[ind_relevant_irreps[q], :]==1).flatten()
            # nu1_friends_indices contains the indices of the friends irreps 
            # of irrep nu1, namely the indices of all irreps in TAB1 such 
            # that nu1 with such an irrep can lead to the target irrep
            
            for p in range(0, len(nu1_friends_indices)):
                
                nu2 = TAB[nu1_friends_indices[p]]
                
                ind_nu2 = sunpy.common.math.find_row(TAB[ind_relevant_irreps], nu2)
                assert(len(ind_nu2)==1)
                ind_nu2 = ind_nu2[0]                
                ind_g_nu2 = sunpy.common.math.find_row(TAB, nu2)
                assert(len(ind_g_nu2)==1)
                ind_g_nu2 = ind_g_nu2[0]
                assert(ind_g_nu2==nu1_friends_indices[p])
                
                bcs2 = sun.get_bottom_corner(nu2)
                length_bottom = len(bcs1)*len(bcs2)
                
                bottom_vecteur_cross = np.zeros(shape=(length_bottom, 2), dtype=int)
                bottom_vecteur_cross[:, 0] = np.repeat(bcs1, repeats=len(bcs2))
                bottom_vecteur_cross[:, 1] = np.tile(bcs2, len(bcs1))
                
                for k in range(0, length_bottom):
                    
                    bc1 = bottom_vecteur_cross[k, 0]
                    bc2 = bottom_vecteur_cross[k, 1]
                    
                    # build ascendant shapes of nu1 and nu2
                    alpha_asc1 = np.copy(nu1)
                    alpha_asc1[bc1] -= 1
                    alpha_asc2 = np.copy(nu2)
                    alpha_asc2[bc2] -= 1
                    
                    ind_alpha_asc1 = sunpy.common.math.find_row(TAB1, alpha_asc1)
                    ind_alpha_asc2 = sunpy.common.math.find_row(TAB1, alpha_asc2)
                    
                    if len(ind_alpha_asc1)==0:
                        l1 = int(0)
                        ind_st1 = int(0)
                    else:
                        ind_alpha_asc1 = ind_alpha_asc1[0]
                        l1 = self._Genealogy[n][ind_g_nu1, ind_alpha_asc1]
                        ind_st1 = self._Genealogy_acc[n][ind_g_nu1, ind_alpha_asc1]
                    
                    if len(ind_alpha_asc2)==0:
                        l2 = int(0)
                        ind_st2 = int(0)
                    else:
                        ind_alpha_asc2 = ind_alpha_asc2[0]
                        l2 = self._Genealogy[n][ind_g_nu2, ind_alpha_asc2]
                        ind_st2 = self._Genealogy_acc[n][ind_g_nu2, ind_alpha_asc2]
                    
                    # build all descendants of these ascendants shapes
                    alpha_desc1, row_pos1 = sun.get_descendants(alpha_asc1, self._N, m=1)
                    alpha_desc2, row_pos2 = sun.get_descendants(alpha_asc2, self._N, m=1)
                    
                    ndesc1 = len(row_pos1)
                    ndesc2 = len(row_pos2)
                    
                    for qq in range(0, ndesc1):
                        
                        nu3 = alpha_desc1[qq]
                        bc3 = row_pos1[qq]
                        
                        ind_nu3 = sunpy.common.math.find_row(TAB[ind_relevant_irreps], nu3)
                        
                        if len(ind_nu3)==0:
                            continue
                        
                        assert(len(ind_nu3)==1)
                        ind_nu3 = ind_nu3[0]
                        
                        ind_g_nu3 = sunpy.common.math.find_row(TAB, nu3)
                        assert(len(ind_g_nu3)==1)
                        ind_g_nu3 = ind_g_nu3[0]
                        
                        # build the ascendant irrep of nu3
                        nu3_asc = np.copy(nu3)
                        nu3_asc[bc3] -= 1
                        
                        # find index of nu3_asc in TAB1
                        ind_nu3_asc = sunpy.common.math.find_row(TAB1, nu3_asc)
                        
                        if len(ind_nu3_asc)==0:
                            l3 = int(0)
                            ind_st3 = int(0)
                        else:
                            ind_nu3_asc = ind_nu3_asc[0]
                            l3 = self._Genealogy[n][ind_g_nu3, ind_nu3_asc]
                            ind_st3 = self._Genealogy_acc[n][ind_g_nu3, ind_nu3_asc]
                        
                        assert(q==ind_nu1)
                        lenL = min(l1, l3)
                        iG = ind_st1 + np.sum(num_states[ind_relevant_irreps[:ind_nu1]])
                        jG = ind_st3 + np.sum(num_states[ind_relevant_irreps[:ind_nu3]])
                        
                        for pp in range(0, ndesc2):
                            
                            nu4 = alpha_desc2[pp]
                            bc4 = row_pos2[pp]                            
                            
                            ind_nu4 = sunpy.common.math.find_row(TAB[ind_relevant_irreps], nu4)
                            
                            if len(ind_nu4)==0:
                                continue
                            
                            assert(len(ind_nu4)==1)
                            ind_nu4 = ind_nu4[0]
                            
                            ind_g_nu4 = sunpy.common.math.find_row(TAB, nu4)
                            assert(len(ind_g_nu4)==1)
                            ind_g_nu4 = ind_g_nu4[0]
                            
                            # build the ascendant irrep of nu4
                            nu4_asc = np.copy(nu4)
                            nu4_asc[bc4] -= 1
                            
                            # find index of nu4_asc in TAB1
                            ind_nu4_asc = sunpy.common.math.find_row(TAB1, nu4_asc)
                            
                            if len(ind_nu4_asc)==0:
                                l4 = int(0)
                                ind_st4 = int(0)
                            else:
                                ind_nu4_asc = ind_nu4_asc[0]
                                l4 = self._Genealogy[n][ind_g_nu4, ind_nu4_asc]
                                ind_st4 = self._Genealogy_acc[n][ind_g_nu4, ind_nu4_asc]
                            
                            lenR = min(l2, l4)
                            iD = ind_st2 + np.sum(num_states[ind_relevant_irreps[:ind_nu2]])
                            jD = ind_st4 + np.sum(num_states[ind_relevant_irreps[:ind_nu4]])
                            
                            if (tensor_bool[ind_relevant_irreps[ind_nu3], ind_relevant_irreps[ind_nu4]]==1):
                                
                                coeff = self._rme_reader.read(nu1, bc1, 
                                                              nu2, bc2, 
                                                              nu3, bc3, 
                                                              nu4, bc4)
                                
                                iAB1 = iG*num_relevant_states + iD
                                jAB1 = jG*num_relevant_states + jD
                                
                                vecxAB_temp = np.arange(iAB1, iAB1+lenR)
                                vecyAB_temp = np.arange(jAB1, jAB1+lenR)
                                
                                vecxAB = np.tile(vecxAB_temp, lenL) + np.repeat(np.arange(0, lenL)*num_relevant_states, repeats=lenR)
                                vecyAB = np.tile(vecyAB_temp, lenL) + np.repeat(np.arange(0, lenL)*num_relevant_states, repeats=lenR)
                                
                                vecHLRx[index_cat:index_cat+lenL*lenR] = vecxAB
                                vecHLRy[index_cat:index_cat+lenL*lenR] = vecyAB
                                vecHLRvalues[index_cat:index_cat+lenL*lenR] = coeff                                
                                index_cat += lenL*lenR
        
        vecHLRx = vecHLRx[:index_cat]
        vecHLRy = vecHLRy[:index_cat]
        vecHLRvalues = vecHLRvalues[:index_cat]
        
        HLR = scipy.sparse.csr_matrix( (vecHLRvalues, (vecHLRx, vecHLRy)), 
                                      shape=(dim_superblock, dim_superblock) )
        
        tend = time.perf_counter()
        print('Time = {}s'.format(tend-tstart))
        print('done.')
        
        return HLR
