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
import scipy.sparse
from abc import ABC, abstractmethod

from sunpy.sun import sun
from sunpy.lanczos import lanczos



class DMRGSolver(ABC):
    """
    Abstract class to represent a Density Matrix Renormalization Group solver 
    for the SU(N) Heisenberg model
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
        """
        self._N = int(N)
        assert(self._N>=2)
        assert(int(Ns)%2==0)
        self._Ns = int(Ns)
        self._num_irreps = num_irreps
        self._max_num_states = max_num_states
        self._truncated = False
        self._target = kwargs.get('target', 'GS')
        self._Ns_min = int(kwargs.get('Ns_min', 4))
        
        self._lanczos_max_iter = int(kwargs.get('lanczos_max_iter', 200))
        self._lanczos_tol_ritz = kwargs.get('lanczos_tol_ritz', 1.0e-13)
        self._lanczos_tol_residual = kwargs.get('lanczos_tol_residual', 1.0e-13)
        
        irreps_filename = f'SU{self._N}_irreps_300.npy'
        if (self._num_irreps>int(300)):
            irreps_filename = f'SU{self._N}_irreps_{num_irreps}.npy'
        
        user_dir = os.getcwd()
        irreps_dir = os.path.join(user_dir, 'sunpy_irreps')
        irreps_filename = os.path.join(irreps_dir, irreps_filename)
        
        if not os.path.exists(irreps_filename):
            if not os.path.isdir(irreps_dir):
                os.makedirs(irreps_dir)
            print(f'Generating list of {max(int(300), num_irreps)} first irreps of SU({self._N})')
            self._irreps_all = sun.get_irreps_by_casimir(self._N, max(int(300), num_irreps))
            print('Done. Dumping to:', irreps_filename)
            np.save(irreps_filename, self._irreps_all)
        else:
            print(f'Loading list of {max(int(300), num_irreps)} first irreps of SU({self._N})')
            self._irreps_all = np.load(irreps_filename).astype(int)
        
        self._irreps0 = self._irreps_all[:self._num_irreps] # does not have column(s) with N boxes
        
        self._num_states = {}
        self._Genealogy = {}
        self._Genealogy_acc = {}
        self._irreps_n = {}
        self._H = {}
        self._rho_eigvals = {}
        self._rho_eigvecs = {}
        
        self.idmrg_discarded_weight = np.zeros(self._Ns//2+1)
        self.idmrg_kept_weight = np.zeros(self._Ns//2+1)
        self.idmrg_entropy = np.zeros(self._Ns//2+1)
        self.idmrg_energy = np.zeros(self._Ns//2+1)
        
        self.threshold_dimension_fullform = int(16000)
        self.threshold_dimension_numpy_eigh = int(200)
        
        return
    
    @abstractmethod
    def idmrg(self):
        """
        Perform infinite-size DMRG, growing the half-chain from self._Ns_min 
        sites to self._Ns//2
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _idmrg_init(self):
        """
        Initialization step in iDMRG, namely construction of the chain with 
        2*self._Ns_min sites
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _valid_irreps(self, Ns):
        """
        Find all relevant irreps with Ns boxes
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _target_irrep(self, L):
        """
        Return the irrep of the target sector, for a chain with 2*L sites
        """
        raise NotImplementedError()
    
    @abstractmethod
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
        raise NotImplementedError()
    
    @abstractmethod
    def _select_states_step1(self, n):
        """
        DMRG truncation step 1: Determine how many states to keep in each 
        symmetry sector
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _select_states_step2(self, n):
        """
        DMRG truncation step 2: Truncate states in each symmetry sector, keeping
        the ones with the largest eigenvalues of the density matrices
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _get_left_block_hamiltonians(self, n, ind_alpha_ascs):
        """
        Compute the Hamiltonian of the left (or right) block in each sector
        """
        raise NotImplementedError()
    
    @abstractmethod
    def _get_HLR(self, n, tensor_bool, ind_relevant_irreps):
        """
        Compute the matrix HLR, which couples the left block to the right block,
        using the Reduced Matrix Elements of the interaction previously obtained
        from the Subduction Coefficients
        """
        raise NotImplementedError()
    
    def _get_discarded_weight(self, n, Keep_val):
        """
        Measure the total discarded weight, taking into account the irreps' 
        dimensions
        """
        TAB2 = self._irreps_n[n]
        i_TAB2 = TAB2.shape[0]
        
        dim_irrep = np.zeros(i_TAB2)
        for p in range(0, i_TAB2):
            dim_irrep[p] = sun.dim_irrep_sun(TAB2[p], self._N)
        
        self.idmrg_kept_weight[n] = np.sum(np.multiply(Keep_val, dim_irrep))/self._local_dimension
        self.idmrg_discarded_weight[n] = 1. - self.idmrg_kept_weight[n]
        
        print('--------------------------')
        print('Discarded weight = ', self.idmrg_discarded_weight[n])
        print('--------------------------')
        
        return
    
    def _diagonalize(self, n, HLR, Ai, bool_tensor_vec):
        """
        Diagonalize the Hamiltonian on the entire chain
        """
        dimension = HLR.shape[0] # full Hilbert space dimension
        
        if (dimension<=self.threshold_dimension_fullform):
            # use a full-form representation of the Hamiltonian
            # diagonalize with numpy.linalg.eigh or scipy.sparse.linalg.eigsh
            energy, GS = self._lanczos_dmrg_full_form(n, HLR, Ai)
        else:
            # use a custom, light-weight, implementation of the Lanczos algorithm
            
            # random starting vector
            rng = np.random.default_rng(seed=(42+n))
            v_init = rng.uniform(low=-1.0, high=1.0, size=dimension)
            v_init = v_init/np.linalg.norm(v_init)
            
            # for n a multiple of N, we increase the maximum number of iterations
            max_iter = self._lanczos_max_iter + (n%self._N==0) * self._lanczos_max_iter//2
            
            lanczos_multiply = lambda v : self._multiply(HLR, Ai, bool_tensor_vec, v)
            
            energy, GS = lanczos.lanczos(lanczos_multiply, 
                                         v_init, 
                                         max_iter=max_iter, 
                                         tol_residual=self._lanczos_tol_residual, 
                                         tol_ritz=self._lanczos_tol_ritz)
        
        return energy, GS
    
    def _lanczos_dmrg_full_form(self, n, HLR, Ai):
        """
        Perform Lanczos using a full-form representation of the matrix, and 
        calling numpy.linalg.eigh or scipy.sparse.eigsh.
        This is the privileged diagonalization routine for small size 
        Hamiltonian.
        """
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Start Lanczos full form ...')
        tstart = time.perf_counter()
        
        dim_superbloc = HLR.shape[0]
        num_valid_states = int(np.sqrt(dim_superbloc))
        
        Bi = scipy.sparse.eye(m=num_valid_states, n=num_valid_states, k=0)
        
        HLi = scipy.sparse.kron(Ai, Bi, format='csr')
        HRi = scipy.sparse.kron(Bi, Ai, format='csr')
        
        H = HLR + HLi + HRi
        
        print('Hamiltonian dimensions: ', dim_superbloc, 'x', dim_superbloc)
        
        if (dim_superbloc==1):
            energy = H[0, 0]
            GS = np.array([1.0])
        elif (dim_superbloc<self.threshold_dimension_numpy_eigh):
            H_eigvals, GS = np.linalg.eigh(H.todense())
            energy = H_eigvals[0]
            GS = np.asarray(GS)
            GS = GS[:, 0].flatten()
        else:
            H_eigvals, GS = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
            energy = H_eigvals[0]
            GS = np.asarray(GS)
            GS = GS[:, 0].flatten()
        
        tend = time.perf_counter()
        print('Elapsed time full-form Hamiltonian diagonalization = {}s'.format((tend - tstart)))
        print('done.')
        
        return energy, GS
    
    def _multiply(self, HLR, Ai, bool_tensor_vec, V):
        """
        Compute W = Hamiltonian @ V without building a full representation of
        the Hamiltonian
        """
        tstart = time.perf_counter()
        
        dim_superblock = V.shape[0]
        num_states = Ai.shape[0]
        assert(num_states**2==dim_superblock)
        assert(HLR.shape[0]==dim_superblock)
        assert(len(bool_tensor_vec)==dim_superblock)
        
        # V is an array of dimension dim_superblock and has indices (i,j), 
        
        W = np.reshape(V, (num_states, num_states)) # (i, j)
        W = Ai @ W # (k, i) x (i, j) --> (k, j)
        W = np.reshape(W, (dim_superblock,)) # (k, j), 
        W = np.multiply(bool_tensor_vec, W)
        
        WL = np.multiply(bool_tensor_vec, V)
        WL = np.reshape(WL, (num_states, num_states))
        WL = WL @ Ai.transpose()
        WL = np.reshape(WL, (dim_superblock,))
        
        W += WL
        del WL
        
        W += HLR @ V
        
        tend = time.perf_counter()
        telapsed = tend-tstart
        if (telapsed>=10):
            print('multiply: Time = {}s'.format(telapsed))
        
        return W
    
    def _print_distribution(self, irreps, num_states):
        """
        Print the distribution of states
        """
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Distribution of states:')
        for j, irrep in enumerate(irreps):
            print(irrep, ': ', num_states[j])
        print('--------------------------')
        print('Total: ', np.sum(num_states))
        return
    
    def _get_tensor_bool(self, alphaTarget, irreps):
        """
        Compute the matrix which summarizes which pairs of irreps can lead to 
        the target irrep
        
        Parameters
        ----------
        alphaTarget : numpy array
            target irrep
        irreps : numpy array
            collection of irreps (stored in the rows)
        
        Returns
        -------
        out : numpy array (dimension irreps.shape[0] x irreps.shape[0])
            out[i, j] == 1 if irreps[i]xirreps[j] --> alphaTarget
                         0 otherwise
        """
        num_irreps = irreps.shape[0]
        out = np.zeros(shape=(num_irreps, num_irreps), dtype=int)
        for i in range(0, num_irreps):
            out[i, i] = (sun.multiplicity_irrep_mixed(alphaTarget, np.array([irreps[i], irreps[i]]), self._N)>0)
            for j in range(i+1, num_irreps):
                out[i, j] = (sun.multiplicity_irrep_mixed(alphaTarget, np.array([irreps[i], irreps[j]]), self._N)>0)
                out[j, i] = out[i, j]
        return out
    
    def _analyze_tensor_bool(self, tensor_bool, num_states):
        """
        Find indices of relevant and irrelevant irreps
        """
        ind_relevant_irreps = np.argwhere(np.sum(tensor_bool, axis=1)).flatten()
        ind_irrelevant_irreps = np.setdiff1d(np.arange(0, tensor_bool.shape[0]), ind_relevant_irreps)
        return ind_relevant_irreps, ind_irrelevant_irreps
    
    def _get_bool_tensor_vec(self, tensor_bool, num_states):
        """
        
        """
        ind_relevant_irreps, _ = self._analyze_tensor_bool(tensor_bool, num_states)
        num_relevant_irreps = len(ind_relevant_irreps)
        num_relevant_states = np.sum(num_states[ind_relevant_irreps])
        dim_superbloc = num_relevant_states**2
        
        bool_tensor_vec = np.zeros(shape=(dim_superbloc,), dtype=int)
        
        for t in range(0, num_relevant_irreps):
            for s in range(0, num_relevant_irreps):
                if (tensor_bool[ind_relevant_irreps[t], ind_relevant_irreps[s]]==1):
                    for q in range(0, num_states[ind_relevant_irreps[t]]):
                        sum_t = np.sum(num_states[ind_relevant_irreps[:t]])
                        sum_s = np.sum(num_states[ind_relevant_irreps[:s]])
                        ist = num_relevant_states * ( sum_t + q ) + sum_s
                        ied = num_relevant_states * ( sum_t + q ) + sum_s + num_states[ind_relevant_irreps[s]]
                        bool_tensor_vec[ist:ied] = 1
        
        return bool_tensor_vec
    
    def _get_block_left_H(self, n, ind_relevant_irreps):
        """
        Build a block-diagonal matrix with Hamiltonians of the relevant irrep
        on a half-chain in each block
        """
        num_relevant_irreps = len(ind_relevant_irreps)
        Hblock = self._H[n][ind_relevant_irreps[0]]
        for i in range(1, num_relevant_irreps):
            Hblock = scipy.sparse.block_diag((Hblock, self._H[n][ind_relevant_irreps[i]]), format='csr' )
        return Hblock
    
    def _density_matrices_and_Hrotate_relevant_irreps(self, n, GS, TAB, ind_relevant_irreps):
        """
        Compute density matrices of relevant irreps, digaonalize them and rotate 
        the associated block Hamiltonians according to the eigenvectors of the 
        density matrices
        
        Parameters
        ----------
        n : int
            step
        GS : numpy array
            |GS>
        TAB : numpy array
            collection of irreps (stored in the rows)
        ind_relevant_irreps : numpy array
            indices of the relevant irreps
        """
        print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
        print('Compute density matrices and diagonalize them ...')
        tstart = time.perf_counter()
        
        i_TAB = TAB.shape[0]
        self._rho_eigvals[n] = [None] * i_TAB
        self._rho_eigvecs[n] = [None] * i_TAB
        
        num_relevant_irreps = len(ind_relevant_irreps)
        num_states = self._num_states[n]
        
        entropy = 0.0
        
        # TO DO : This loop should be parallel, with a critical construct (or 
        # atomics) when incrementing entropy
        for q in range(0, num_relevant_irreps):
            
            # compute density matrix
            rho = self._get_density_matrix(q, GS, num_states, ind_relevant_irreps)
            
            # diagonalize density matrix
            rho_eigvals, rho_eigvecs = np.linalg.eigh(rho)
            rho_eigvecs = np.asarray(rho_eigvecs)
            
            # sort eigenvalues and eigenvectors according to decreasing eigenvalues
            rho_eigvals = rho_eigvals[::-1]
            rho_eigvecs = rho_eigvecs[:, ::-1]
            
            # renormalize according to irrep dimension
            dimq = sun.dim_irrep_sun(TAB[ind_relevant_irreps[q]], self._N)
            rho_eigvals = rho_eigvals / dimq
            
            # measure entanglement entropy
            ind_nv = np.argwhere(rho_eigvals<=0.0).flatten()
            if not all(abs(rho_eigvals[ind_nv])<1.0e-14):
                raise ValueError('ERROR : DMRG : density matrix has large negative eigenvalues')
            ind_pv = np.setdiff1d(np.arange(len(rho_eigvals)), ind_nv).flatten()
            entropy -= dimq * np.sum( np.multiply(rho_eigvals[ind_pv], np.log(rho_eigvals[ind_pv])) )
            
            # rotate Hamiltonian
            self._H[n][ind_relevant_irreps[q]] = rho_eigvecs.transpose() @ self._H[n][ind_relevant_irreps[q]] @ rho_eigvecs
            
            self._rho_eigvals[n][ind_relevant_irreps[q]] = rho_eigvals
            self._rho_eigvecs[n][ind_relevant_irreps[q]] = rho_eigvecs
        
        self.idmrg_entropy[n] = entropy
        
        tend = time.perf_counter()
        print('Time = {}s'.format(tend-tstart))
        print('done.')
        
        return
    
    def _get_density_matrix(self, q, GS, num_states, ind_relevant_irreps):
        """
        Compute density matrices associated to a relevant irrep
        
        Parameters
        ----------
        q : int
            index of relevant irrep for which the density matrix is computed
        GS : numpy array
            ground-state vector
        num_states : numpy array
            number of states kept for each irrep
        ind_relevant_irreps : numpy array
            indices of relevant irreps
        """
        
        dim_rho = num_states[ind_relevant_irreps[q]]
        rho = np.zeros(shape=(dim_rho, dim_rho), dtype=float)
        
        num_relevant_states = np.sum(num_states[ind_relevant_irreps])
        index_offset = num_relevant_states * np.sum(num_states[ind_relevant_irreps[:q]])
        
        for i in range(0, dim_rho):                
            
            index_vec_i = index_offset + np.arange(num_relevant_states*i, num_relevant_states*(i+1))
            rho[i, i] = np.dot(GS[index_vec_i], GS[index_vec_i])
            
            for j in range(i+1, dim_rho):
                index_vec_j = index_offset + np.arange(num_relevant_states*j, num_relevant_states*(j+1))
                rho[i, j] = np.dot(GS[index_vec_i], GS[index_vec_j])
                rho[j, i] = rho[i, j]
        
        return rho
    
    def _density_matrices_and_Hrotate_irrelevant_irreps(self, n, ind_irrelevant_irreps):
        """
        Rotate block Hamiltonians associated to irrelevant irreps (which have  a
        vanishing density matrix) according to the eigenvectors of these 
        Hamiltonians
        
        Parameters
        ----------
        n : int
            step
        ind_relevant_irreps : numpy array
            indices of irrelevant irreps
        """
        print('Deal with irrelevant irreps ...')
        tstart = time.perf_counter()
        
        if len(ind_irrelevant_irreps)==0:
            print('There is no irrelevant irrep.')
        
        num_states = self._num_states[n]
        
        # TO DO : this is a parallel loop
        for q in range(0, len(ind_irrelevant_irreps)):
            
            dim_rho = num_states[ind_irrelevant_irreps[q]]
            Hq = self._H[n][ind_irrelevant_irreps[q]]
            assert(Hq.shape[0]==dim_rho)
            
            # diagonalize associated Hamiltonian
            if Hq.shape[0]==1:
                Hq_eigvecs = np.array([[1.0]], dtype=float)
            else:
                _, Hq_eigvecs = np.linalg.eigh(Hq.todense())
            
            # eigenvalues of density matrix are all 0
            self._rho_eigvals[n][ind_irrelevant_irreps[q]] = np.zeros(shape=(dim_rho,), dtype=float)
            
            # for eigenvectors, we take the eigenvectors of the Hamiltonian Hq
            self._rho_eigvecs[n][ind_irrelevant_irreps[q]] = np.asarray(Hq_eigvecs)
            
            # rotate Hamiltonian
            self._H[n][ind_irrelevant_irreps[q]] = Hq_eigvecs.transpose() @ Hq @ Hq_eigvecs
        
        tend = time.perf_counter()
        print('Time = {}s'.format(tend-tstart))
        print('done.')
        
        return
