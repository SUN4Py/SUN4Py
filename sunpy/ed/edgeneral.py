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

import sys
import time
import numpy as np
# from numpy.typing import NDArray # numpy>=1.20
import scipy.sparse
import itertools
import math
import copy

from sunpy.ed.edbase import EDSolver
from sunpy.ed.lattice import Lattice
from sunpy.sun import sun



class LocalStates:
    
    def __init__(self, ec, site, y, cy, beta):
        """
        Generate the local states for an equivalence class at a given site
        
        Parameters
        ----------
        ec : int
            index of equivalence class
        site : int
            site
        y : numpy array
            SYT of equivalence class
        cy : numpy array
            associated column positions
        beta : numpy array
            all local irreps
        
        Remark
        ------
        States are obtained by computing the kernel of a projector obtained 
        from orthogonal units
        """
        
        self.site = np.array([site], dtype=int)
        self.ec = ec
        self.y = np.copy(y)
        self.cy = np.copy(cy)
        self.beta = np.copy(beta)
        beta_loc = np.copy(beta[site])
        self.particles = np.sum(np.sum(beta[0:site])) + np.arange(0, np.sum(beta_loc))
        
        self.__get_local_states()
        
        return
    
    
    
    def __get_local_states(self) -> None:
        """
        Compute the states at a local site, obtained as the kernel of the projector
        onto the local irrep
        
        Returns
        -------
        coeffs : numpy array
            expansion coefficients
            states are stored in the columns
        YProj : numpy array
            SYTs
        CYProj : numpy array
            associated column positions
        alphaM, alphaB, alphaP  : numpy arrays
            alphaB+alphaP=alphaM ; alphaM minimal irrep
        offset : int
            row offset
        """
        
        beta_loc = self.beta[self.site[0]]
        
        Proj, self.Ydev, self.CYdev, self.alphaM, self.alphaB, self.alphaP, self.offset = get_projector(
                                                                beta_loc, 
                                                                self.y, 
                                                                self.cy, 
                                                                self.particles)
        NY = Proj.shape[0]
        
        if NY==1:
            if abs(Proj[0,0]-1)<1.0e-13:
                V = np.array([[1.0]], dtype=float)
                self.n = int(1)
            else:
                V = np.zeros(shape=(0, 0), dtype=float)
                self.n = int(0)
        else:
            #V = scipy.linalg.null_space( (Proj - scipy.sparse.eye(n)).toarray(), overwrite_a=False )
            V = sg_null_space( (Proj - scipy.sparse.eye(NY)).toarray() )            
            self.n = V.shape[1] # dimension of null space
            # vectors spanning the null space of Proj are stored in the columns of V
            if self.n>0:
                for i in range(self.n):
                    if V[0,i]<0:
                        V[:,i] *= (-1.0)
            else:
                V = np.zeros(shape=(NY, 0), dtype=float)
        
        self.coeffs = V
        
        return



class GeneralBasis:
    
    def __init__(self, alpha, beta, N):
        """
        Generate basis of states as tensor product of local states on each site
        for each equivalence class
        
        Parameters
        ----------
        alpha : numpy array
            global irrep
        beta : numpy array
            collection of local irreps on each site
        N : int
            SU(N)
        """
        
        print('==========================')
        print('Generating basis')
        print('==========================')
        
        self.alpha = alpha
        self.beta = beta
        self.N = int(N)
        self.Ns = beta.shape[0]
        
        self.Y, self.CY = sun.get_SYT_general(self.alpha, self.beta)
        self.NY = self.Y.shape[0]
        
        self.local_states = []
        self.dimB = np.zeros(shape=(self.Ns, self.NY), dtype=int)
        
        si = int(0)
        
        for site in range(self.Ns):
            for i in range(self.NY):
                self.local_states.append(LocalStates(i, site, self.Y[i], self.CY[i], beta))
                self.dimB[site, i] = self.local_states[si].n
                si += 1
        
        self.statesPerClass = np.prod(self.dimB, axis=0)
        self.NH = np.sum( self.statesPerClass ) # total Hilbert space dimension
        
        if not (self.NH==sun.multiplicity_irrep_mixed(self.alpha, self.beta, self.N)):
            sys.exit('Problem: Number of states does not match multiplicity.')
        
        self.ind = np.argwhere( self.statesPerClass>0 ).flatten()
        if len(self.ind)!=self.NY:
            sys.exit('There is ', self.NY-len(self.ind), ' equivalence classes leading to 0 state.')
        
        return
    
        
    def __get_index_sec(self, ec, site):
        """
        Get the linear index of an equivalence class at a given site
        """
        si = site*self.NY + ec        
        return si    
    
    
    def get_states_of_class(self, ec, sites):
        """
        Compute all states corresponding to a given equivalence class, as a 
        development on multiple sites
        
        Parameters
        ----------
        ec : int
            index of equivalence class
        sites : numpy array
            sites on which to compute the states
        
        Returns
        -------
        out : LocalStates
            states
        
        Remark
        ------
        This corresponds to performing a kronecker product of local states to 
        generate a collection of states living on several sites
        """
        
        site1 = np.min(sites)
        site2 = np.max(sites)
        
        si = self.__get_index_sec(ec, site2)
        out = copy.deepcopy(self.local_states[si])

        mtot = np.sum(self.beta[site2])
        
        for site in range(site2-1, site1-1, -1):
            
            si = self.__get_index_sec(ec, site)
            local_states_to_add = copy.deepcopy(self.local_states[si])
            m_to_add = np.sum(self.beta[site])
            
            diffoffset = out.offset - local_states_to_add.offset
            
            # diffoffset>0 ==> site+1 particles start at a row BELOW site particles
            # diffoffset<0 ==> site+1 particles start at a row ABOVE site particles
            # diffoffset==0 ==> site+1 particles start at the same row as site particles
            
            if (diffoffset==0):
                off1 = int(0)
                off2 = int(0)
            elif (diffoffset>0):
                off1 = int(0)
                off2 = diffoffset
            else:
                off1 = -diffoffset
                off2 = int(0)
            
            # number of SYTs in total development so far
            q = out.Ydev.shape[0]
            
            # number of SYTs in the developments at <site>
            qadd = local_states_to_add.Ydev.shape[0]
            
            # "kronecker cross" SYTs (merging)
            A = np.repeat(local_states_to_add.Ydev[:,-m_to_add:] + off1, repeats=q, axis=0)
            B = np.tile(out.Ydev[:,-mtot:] + off2, (qadd, 1))
            out.Ydev = np.concatenate((A, B), axis=1)
            
            # compute coefficients
            out.coeffs = np.kron(local_states_to_add.coeffs, out.coeffs)
            
            # update attributes
            out.offset = min(out.offset, local_states_to_add.offset)
            out.particles = np.sort( np.concatenate( (out.particles, local_states_to_add.particles) ) )
            out.site = np.sort( np.concatenate( (out.site, np.array([site]) ) ) )
            out.n *= local_states_to_add.n
            
            mtot += m_to_add
        
        # update shapes
        out.alphaM, out.alphaB, out.alphaP, out.offset = sun.get_subshape(
                                                            out.y, 
                                                            out.cy, 
                                                            out.particles)
        
        return out
    
    
    
    def find_indices(self, ec, site1, site2) -> np.ndarray:
        """
        Find the indices in the global basis of states of a given equivalence class
        for a set of sites
        
        Returns
        -------
        index : numpy array
            array of dimension n x U
            n = number of states
            U = number of duplicates
            
            the i-th state of the equivalence class must be duplicated at 
            positions index[i,:]
        """        
        
        # distribution of states among the different sites in the class
        statesSpEq = self.dimB[:, ec]
        
        # number of states generated by the input sites
        NbStates = np.prod(statesSpEq[site1:site2+1])
        
        # number of states BEFORE input equivalence class
        offsetClass = np.sum(self.statesPerClass[:ec])
        
        # number of states in the input equivalence class
        statesEqcl = self.statesPerClass[ec]
        
        # number of times duplication will occur
        nDublicates = statesEqcl // NbStates
        
        if site1>0:
            mL = np.prod(statesSpEq[:site1])
        else:
            mL = int(1)
        
        if site2==self.Ns:
            mR = int(1)
        else:
            mR = np.prod(statesSpEq[site2+1:])
        
        assert mL*mR == nDublicates
        
        m1 = statesSpEq[site1]
        m2 = statesSpEq[site2]
        assert m1*m2 == NbStates
        
        '''
        if site1+1==site2:
            
            # Duplicate because of offset Right
            index1 = np.arange(offsetClass, offsetClass+mR)
            l1 = len(index1)
            
            # Duplicate because of site2
            index2 = np.zeros(shape=(m2, l1), dtype=int)
            index2[0, :] = index1
            
            for i2 in range(1, m2):
                index2[i2, :] = index2[i2-1, :] + mR
    
            # Duplicate because of site1
            index3 = np.zeros(shape=(m2*m1, l1), dtype=int)
            index3[:m2, :] = index2
    
            for i1 in range(1, m1):
                index3[i1*m2:(i1+1)*m2, :] = index3[(i1-1)*m2:i1*m2, :] + m2*mR
            
            
            # Duplicate because of offset Left
            index4 = np.zeros(shape=(m2*m1, l1*mL), dtype=int)
            
            assert mR == index3.shape[1]
            
            index4[:,:mR] = index3
    
            for iL in range(1, mL):
                index4[:,iL*mR:(iL+1)*mR] = index4[:, (iL-1)*mR:iL*mR] + m1*m2*mR
            
            index = index4
            
        else:
        '''    
        mS = NbStates
        
        # duplicate because of offset Right
        index1 = np.arange(offsetClass, offsetClass+mR)
        l1 = len(index1)
        
        # duplicate because of site1--site2
        index2 = np.zeros(shape=(mS, l1), dtype=int)
        
        index2[0, :] = index1
        for i2 in range(1, mS):
            index2[i2, :] = index2[i2-1, :] + mR
        
        # Duplicate because of offset Left
        index3 = np.zeros(shape=(mS, l1*mL), dtype=int)
        
        index3[:, :mR] = index2
        for iL in range(1, mL):
            index3[:, iL*mR:(iL+1)*mR] = index3[:,(iL-1)*mR:iL*mR] + mS*mR
        
        index = index3
        
        return index
    
    
    
    def __check_conditions_equivalence_class(self, ec1, ec2, particles1, particles2) -> bool:
        """
        Check if 2 equivalence classes are compatible
        
        Returns
        -------
        iseq : Bool
            True if ec1 and ec2 are compatible
        """
        
        y1 = self.Y[ec1]
        cy1 = self.CY[ec1]
        y2 = self.Y[ec2]
        cy2 = self.CY[ec2]
        
        p1 = np.min(particles1)
        p2 = np.max(particles2)
        n = self.Y.shape[1]
        
        iseq = True
        
        # all particles before p1 must be identically placed
        i = int(1)
        while (i<p1):
            if not ( (y1[i]==y2[i]) & (cy1[i]==cy2[i]) ):
                return False
            i += 1
        
        # all particles after p2 must be identically placed
        i = p2 + 1
        while (i<n):                
            if not ( (y1[i]==y2[i]) & (cy1[i]==cy2[i]) ):
                return False
            i += 1   
        '''
        # vectorize the above - not really faster
        iseq2 = True
        ylowdiff = np.sum(abs(y1[:p1] - y2[:p1]))
        if not ylowdiff==0:
            iseq2 = False
            return iseq
        yhidiff = np.sum(abs(y1[p2+1:] - y2[p2+1:]))
        if not yhidiff==0:
            iseq2 = False
            return iseq
        if iseq2==False:
            sys.exit('Problem: we should not have entered here')
        '''
        # check that there is at most 1 interchange
        v1 = np.vstack((y1[particles1], cy1[particles1])).T
        v2 = np.vstack((y2[particles1], cy2[particles1])).T
        
        cpt = int(0)
        for j in range(0, len(particles1)):
            ind = np.argwhere( np.sum(abs( v2 - v1[j, :] ), axis=1)==0 ).flatten()
            if len(ind)==0:
                cpt += 1
        
        if cpt>1:
            iseq = False
        
        return iseq
    
    
    
    def get_sister_equivalence_class(self, ec1, particles1, particles2) -> np.ndarray:
        """
        Determine the relevant equivalence classes coupling with an input class
        
        Returns
        -------
        ind_ec2 : numpy array
            indices of compatible equivalence classes
        """
        
        ind_ec2 = np.array([ec1], dtype=int)
        
        # one should parallelize this loop, taking care of the creation (append) of ind_ec2
        for ec2 in range(ec1+1, self.NY):
            iseq = self.__check_conditions_equivalence_class(ec1, ec2, particles1, particles2)
            if iseq:
                ind_ec2 = np.hstack((ind_ec2, ec2))
        
        return ind_ec2



class OrthogonalUnits:
    
    def __init__(self, alpha, only00='True'):
        self.alpha = alpha
        
        # build all SYTs associated to the irrep alpha
        Y = sun.get_SYT(self.alpha)
        self.falpha = Y.shape[0]
        CY = sun.get_column(Y)
        
        # construct matrices of adjacent transpositions
        P = sun.get_adjacent_transposition_matrices(alpha, Y, CY)
        self.P = P
        
        # get all permutations of S_n
        n = np.sum(alpha)
        ps = itertools.permutations(np.arange(0, n, dtype=int))
        self.nn = math.factorial(n)
        
        # compute the matrices of the inverse permutations, obtained by reading
        # the sequence of adjacent transpositions in reverse order
        ps = itertools.permutations(np.arange(0, n, dtype=int))
        self.permutations = []
        self.sigmaMat = [scipy.sparse.eye(self.falpha)] * self.nn
        self.sigmaMatinverse = [scipy.sparse.eye(self.falpha).tocsr()] * self.nn
        self.adjaTranspo = [None] * self.nn
        for i, sigma in enumerate(ps):
            self.permutations.append(sigma)
            at = sun.permutation_to_adjacent_transpositions(np.array(sigma, dtype=int))
            self.adjaTranspo[i] = at
            for j in range(0, len(at)):
                self.sigmaMat[i] = self.sigmaMat[i] @ P[at[j]]
                self.sigmaMatinverse[i] = P[at[j]] @ self.sigmaMatinverse[i]
        
        if only00==True:
            self.coeffs = np.zeros(shape=(1, 1, self.nn))
            for i in range(self.nn):
                self.coeffs[0][0][i] = self.sigmaMatinverse[i][0,0]
        else:
            self.coeffs = np.zeros(shape=(self.falpha, self.falpha, self.nn))
            for r in range(self.falpha):
                for s in range(self.falpha):
                    for i in range(self.nn):
                        self.coeffs[r][s][i] = self.sigmaMatinverse[i][s,r]
        self.coeffs *= self.falpha/self.nn



def get_projector(beta_loc, y, cy, particles):
    """
    Compute projector onto a local irrep from an orthogonal unit
    
    Parameters
    ----------
    beta_loc : numpy array
        local irrep for projector
    y : numpy array
        SYT of equivalence class
    cy : numpy array
        associated column positions
    particles : numpy array
        positions in the SYT corresponding to the local irrep
    
    Returns
    -------
    Proj : scipy.sparse.csr.csr_matrix
        Projector onto local irrep, expressed in a basis of SYTs
    YMc : numpy array
        basis of SYTs used to express the projector
    CYMc : numpy array
        associated column positions
    alphaM, alphaB, alphaP : numpy arrays
        alphaB+alphaP = alphaM ; alphaM minimal irrep
    offset : int
        row offset of particles
    """
    
    particles = np.sort(particles)
    npart = len(particles)
    
    # get the (0, 0) orthogonal unit
    ortho = OrthogonalUnits(beta_loc, only00='True')
    
    # identify locations of boxes associated to the particles
    alphaM, alphaB, alphaP, offset = sun.get_subshape(y, cy, particles)
    
    # get associated SYTs
    YM = sun.get_subSYT(alphaM, alphaB, order='LLOS')
    YMc = sun.fill_subSYT(YM, alphaM, fill_type='smallest')
    CYMc = sun.get_column(YMc)
    NYMc = YMc.shape[0]
    
    # get matrices of adjacent transpositions of these boxes
    particles_shifted = particles - particles[0] + np.sum(alphaB)
    P = [None] * (npart -  1)
    for i in range(0, npart-1):
        P[i] = sun.get_adjacent_transposition_matrix(
                    alphaM, 
                    YMc, 
                    CYMc, 
                    k=particles_shifted[i])
        
    # build matrix of projector
    Proj = scipy.sparse.csr_matrix((NYMc, NYMc))
    for i in range(ortho.nn):
        Permuti = sun.get_matrix_permutation(P, ortho.adjaTranspo[i])
        Proj += ortho.coeffs[0,0][i] * Permuti
    
    return Proj, YMc, CYMc, alphaM, alphaB, alphaP, offset



class EDSolverGeneral(EDSolver):
    """
    Class to represent an exact diagonalization solver for SU(N) Heisenberg 
    models with any irreducible representation at each site
    """
    
    def __init__(self, N: int, Ns: int, alpha, beta, lattice: Lattice):
        """
        Constructor
        
        Parameters
        ----------
        N : int
            SU(N)
        Ns : int
            number of sites
        alpha : numpy array
            irrep
        beta : numpy array
            local irreps (stored in the rows)
        lattice : Lattice
            lattice of bonds
        
        Remarks
        -------
        - The number of sites is given by the number of rows in beta
        - The number of boxes in alpha must match with the sum of the number of 
          boxes in each local irrep
        """
        
        super().__init__(N, Ns, alpha, lattice)
        self._beta = np.copy(beta)
        
        if not self._n==np.sum(beta):
            sys.exit('ERROR : EDEngineGeneral : __init__ : missmatch bewteen alpha and beta.')
        
        if not self._Ns==beta.shape[0]:
            sys.exit('ERROR : EDEngineGeneral : __init__ : missmatch bewteen beta and Ns.')
        
        if not beta.shape[1]<=self._N:
            sys.exit('ERROR : EDEngineGeneral : __init__ : missmatch bewteen beta and N.')
        
        return
    
    def get_basis(self):
        """
        Compute basis states
        """
        tstart = time.perf_counter()
        self._Basis = GeneralBasis(self._alpha, self._beta, self._N)
        self._basis_computed = True
        self._NY = self._Basis.NH
        tend = time.perf_counter()
        print('Elapsed time Basis construction = {}s'.format((tend - tstart)))
        return
    
    def sun_hamiltonian(self):
        """
        Compute the Hamiltonian
        """
        
        if (self._basis_computed==False):
            self.get_basis()
        
        tstart = time.perf_counter()
        
        print('==========================')
        print('Generating Hamiltonian')
        print('==========================')
        
        self._H = scipy.sparse.csr_matrix((self._Basis.NH, self._Basis.NH))
        
        for i, link in enumerate(self._lattice.links):
            
            site1 = link[0]
            site2 = link[1]

            print('==========================')
            print('link = ', site1, ' -- ', site2)
            print('==========================')
            
            m1 = np.sum(self._beta[site1])
            m2 = np.sum(self._beta[site2])
            
            particles1 = np.sum(self._beta[0:site1]) + np.arange(m1)
            particles2 = np.sum(self._beta[0:site2]) + np.arange(m2)
            
            # all particles from site1 to site2
            allparticles = np.sum(self._beta[0:site1]) + np.arange(np.sum(self._beta[site1:site2+1]))
            
            Hbond = scipy.sparse.csr_matrix((self._Basis.NH, self._Basis.NH))
            
            for ec1 in range(self._Basis.NY):
                # generates the states living on all sites [site1, ... , site2]
                states_class1 = self._Basis.get_states_of_class(ec=ec1, sites=link)
                states_class1.Ydev = sun.fill_subSYT(states_class1.Ydev, states_class1.alphaM, fill_type='largest')
                
                # generate all SYTs associated to [site1, ..., site2]
                Yall_ordered = sun.get_subSYT(states_class1.alphaM, states_class1.alphaB, order='iLLOS')
                Yall_ordered = sun.fill_subSYT(Yall_ordered, states_class1.alphaM, fill_type='largest')
                CYall_ordered = sun.get_column(Yall_ordered)
                
                # re-index components of states onto the full local basis
                coeffs1 = sun.reorder_development(Yall_ordered, 
                                                  states_class1.Ydev, 
                                                  states_class1.coeffs, 
                                                  'cols')
                
                # generate all matrices of adjacent transpositions from <Yall_ordered>
                P1 = []
                for k in range(np.sum(states_class1.alphaB), np.sum(states_class1.alphaM)-1):
                    P1.append( sun.get_adjacent_transposition_matrix(
                                    states_class1.alphaM, 
                                    Yall_ordered, 
                                    CYall_ordered, 
                                    k) )
                
                # compute all transpositions between site1 and site2
                perms = np.stack( (np.repeat(particles1, repeats=len(particles2)), 
                                   np.tile(particles2, len(particles1))),
                                   axis=1 )
                
                # Compute shift between global numbering (particles1, particles2)
                # and state-local numbering
                shift = -np.min(allparticles) + np.sum(states_class1.alphaB)
                assert np.max(allparticles)+shift == np.sum(states_class1.alphaM)-1
                perms += shift
                
                # Generate the matrix Hint representing the sum of all transpositions between
                # particles of site1 and those of site2
                #Hint = np.zeros(shape=P1[0].shape, dtype=float)
                Hint = scipy.sparse.csr_matrix(P1[0].shape)
                
                for perm in perms:
                    atr = sun.transposition_to_adjacent_transpositions(perm)   
                    Hp = sun.get_matrix_permutation(P1, atr-np.sum(states_class1.alphaB))
                    #Hint += Hp.todense()
                    Hint += Hp
                
                Hcoeffs1 = Hint @ coeffs1
                
                # Determine all relevant equivalence classes for the <bra|
                ind_ec2 = self._Basis.get_sister_equivalence_class(ec1, particles1, particles2)
                
                for ec2 in ind_ec2:
                    
                    if ec2==ec1:
                        states_class2 = states_class1
                    else:
                        states_class2 = self._Basis.get_states_of_class(ec=ec2, sites=link)
                        states_class2.Ydev = sun.fill_subSYT(states_class2.Ydev, states_class2.alphaM, fill_type='largest')                    
                    
                    # re-index components of states onto the full local basis              
                    if ec2==ec1:
                        coeffs2 = coeffs1
                    else:
                        coeffs2 = sun.reorder_development(
                                Yall_ordered, 
                                states_class2.Ydev, 
                                states_class2.coeffs, 
                                'cols')
                    
                    # build local interaction Hamiltonian
                    Hloc = coeffs2.T @ Hcoeffs1
                    
                    # duplicate and embedd elements in interaction Hamiltonian in full basis
                    ind_st_1 = self._Basis.find_indices(ec1, site1, site2)
                    ind_st_2 = self._Basis.find_indices(ec2, site1, site2)
                    
                    nD = ind_st_1.shape[1]
                    n1 = coeffs1.shape[1]
                    n2 = coeffs2.shape[1]
                    assert nD==ind_st_2.shape[1]
                    

                    rows = np.tile(ind_st_2, (1, n1))
                    rows = np.reshape(rows, (nD*n1*n2, ))
                    cols = np.tile(ind_st_1, (n2, 1))
                    cols = np.reshape(cols, (nD*n1*n2, ))
                    vals = np.repeat(Hloc, repeats=nD)
                    HTemp = scipy.sparse.csr_matrix(
                             (vals, (rows, cols)),
                             shape=(self._Basis.NH, self._Basis.NH))
                    if not ec1==ec2:
                        HTemp += scipy.sparse.csr_matrix(
                                    (vals, (cols, rows)),
                                    shape=(self._Basis.NH, self._Basis.NH))
                    
                    Hbond += HTemp
            
            self._add_H_link(i, Hbond)
        
        self._H_computed = True
        
        tend = time.perf_counter()
        print('Elapsed time Hamiltonian construction = {}s'.format((tend - tstart)))
        
        return self._H

def sg_null_space(A, rcond=None):
    """
    Basic implementation for
        scipy.linalg.null_space( A, rcond=None )
    """
    u, s, vh = scipy.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

