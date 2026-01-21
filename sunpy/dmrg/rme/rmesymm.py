# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np
import sys
import os
import time

from sunpy.dmrg.rme.rmebase import RMEEngine
from sunpy.dmrg.rme.utils import states_rme
from sunpy.dmrg.rme.rmereader import RMEReader
import sunpy.common.math
from sunpy.sun import sun
from sunpy.sdc.dmrg import sdcdmrgbase # basic method (slow)
from sunpy.sdc.dmrg import sdcdmrg # shortcut method ("fast")



class RMEEngineSymm(RMEEngine):
    """
    RME Engine for symmetric irrep at each site
    """
    
    def __init__(self, N, m, num_irreps, target, tech, **kwargs):
        """
        Constructor of RMEEngineSymm
        
        Parameters
        ----------
        N : int
            SU(N)
        m : int
            number of particles in symmetric irrep
        num_irreps : int
            number of irreps
        target : str
            string describing the target irrep
        tech : str
            string describing the technique to use to compute RMEs ('base', 'shortcut_cols', 'shortcut_rows')
        
        [optional]
        checkpointing : bool  [default]False
            if True, perform checkpointing
        chkpt_method : str [default]'log2', 'custom'
            checkpointing strategy
        chkpt_ni : numpy array
            array of num_irreps to checkpoint
        restarting : bool [default]True
            if True, attempt to restart from a saved file
        restarting_folder : str
            directory to search for restarting file
        restarting_filename : str
            path to filename from which to restart
        
        Possible targets are:
            N=3 - m=2:
                'GS_idmrg'
                'GS_idmrg_edgeAdjoint'
                'ES_idmrg'
                'ES_idmrg_edgeAdjoint'
                'ES300_dmrg'
            N=3 - m=3:
                'GS_idmrg_edgeAdjoint'
                'ES300_idmrg_edgeAdjoint'
                'ES330_idmrg_edgeAdjoint'                
        """
        
        self._m = m
        
        if 'filename_prefix' in kwargs:
            self._filename_prefix = kwargs['filename_prefix']
        else:
            self._filename_prefix = 'RME_symm_m' + str(self._m)
        super().__init__(N, num_irreps, self._filename_prefix, target, tech, **kwargs)
        
        return
    
    
    def __str__(self):
        return f"RMEEngineSymm, N={self._N}, m={self._m}, num_irreps={self._num_irreps}, target={self._target}, tech={self._tech}"
    
    
    def _init_irreps(self, num_irreps):
        """
        
        """
        # add m columns with N boxes to each irrep
        return self._irreps_all[:num_irreps] + np.full(shape=(num_irreps, self._N), fill_value=self._m, dtype=int)
    
    
    def _target_irrep(self, n):
        """
        Obtain the effective target irrep with n boxes
        
        Parameters
        ----------
        n : int
            number of boxes in the target irrep
        
        Returns
        -------
        alphaTarget : numpy array
            target irrep
        """
        alphaTarget = np.zeros(shape=(self._N, ), dtype=int)
        
        if (self._N==2):
            
            sys.exit('RMEEngineSymm.target_irrep: target_irrep not yet implemented for N=2.')
            # this is very simple to implement though ...
        
        elif (self._N==3):
            
            if (self._m==2):
                
                if self._target=='GS_idmrg':
                    # infinite-size DMRG, without edge irreps
                    # Ground State
                    # 
                    # The GS is situated in:
                    #   - [0, 0, 0] for Ns=0 (mod. 3)
                    #   - [2, 2, 0] for Ns=1 (mod. 3)
                    #   - [2, 0, 0] for Ns=2 (mod. 3)
                    # 
                    # where Ns is the total chain length (Ns = 2*L)
                    # 
                    # See Gozel et al., Phys. Rev. B 104, L180411 (2021)
                    r = n % self._N
                    if r==1:
                        alphaTarget = np.array([2, 2, 0], dtype=int)
                    elif r==2:
                        alphaTarget = np.array([2, 0, 0], dtype=int)
                    
                elif self._target=='GS_idmrg_edgeAdjoint':
                    # infinite-size DMRG, with adjoint edge irreps
                    # Ground State
                    # 
                    # The GS is situated in:
                    #   - [0, 0, 0] for Ns=0 (mod. 3)
                    #   - [1, 0, 0] for Ns=1 (mod. 3)
                    #   - [1, 1, 0] for Ns=2 (mod. 3)
                    # 
                    # where Ns is the total chain length (Ns = 2*L)
                    # 
                    # See Gozel et al., Phys. Rev. B 104, L180411 (2021)
                    r = n % self._N
                    if r==1:
                        alphaTarget = np.array([1, 0, 0], dtype=int)
                    elif r==2:
                        alphaTarget = np.array([1, 1, 0], dtype=int)
                    
                elif self._target=='ES_idmrg':
                    # infinite-size DMRG, without edge irreps
                    # Excited State
                    # 
                    # The ES is situated in:
                    #   - [1, 0, 0] for Ns=0 (mod. 3)
                    #   - [1, 1, 0] for Ns=1 (mod. 3)
                    #   - [3, 0, 0] for Ns=2 (mod. 3)
                    # 
                    # where Ns is the total chain length (Ns = 2*L)
                    # 
                    # See Gozel et al., Phys. Rev. B 104, L180411 (2021)
                    r = n % self._N
                    if r==0:
                        alphaTarget = np.array([1, 0, 0], dtype=int)
                    elif r==1:
                        alphaTarget = np.array([1, 1, 0], dtype=int)
                    else:
                        alphaTarget = np.array([3, 0, 0], dtype=int)
                    
                elif self._target=='ES_idmrg_edgeAdjoint':
                    # infinite-size DMRG, with adjoint edge irreps
                    # Excited State
                    # 
                    # The ES is situated in:
                    #   - [3, 0, 0] for Ns=0 (mod. 3)
                    #   - [2, 2, 0] for Ns=1 (mod. 3)
                    #   - [2, 0, 0] for Ns=2 (mod. 3)
                    # 
                    # where Ns is the total chain length (Ns = 2*L)
                    # 
                    # See Gozel et al., Phys. Rev. B 104, L180411 (2021)
                    r = n % self._N
                    if r==0:
                        alphaTarget = np.array([3, 0, 0], dtype=int)
                    elif r==1:
                        alphaTarget = np.array([2, 2, 0], dtype=int)
                    else:
                        alphaTarget = np.array([2, 0, 0], dtype=int)
                
                elif self._target=='ES300_dmrg':
                    # finite-size DMRG, without adjoint edge irreps
                    # Excited state
                    # 
                    # One can reach this sector provided Ns=2 (mod. 3)
                    # 
                    # See Gozel et al., Phys. Rev. B 104, L180411 (2021)
                    alphaTarget = np.array([3, 0, 0], dtype=int)
                    
                else:
                    print(self._target)
                    sys.exit('RMEEngineSymm.target_irrep: undefined target sector.')
                    
            elif self._m==3:
                
                if self._target=='GS_idmrg_edgeAdjoint':
                    # For SU(3), m=3, with adjoint edge irreps
                    # the GS is always the singlet sector
                    # See Gozel et al., Phys. Rev. Lett. 125, 057202 (2020)
                    alphaTarget = np.array([0, 0, 0], dtype=int)
                    
                elif self._target=='ES300_idmrg_edgeAdjoint':
                    # For SU(3), m=3, with adjoint edge irreps
                    # one of the excited states is in [3, 0, 0] sector
                    # See Gozel et al., Phys. Rev. Lett. 125, 057202 (2020)
                    alphaTarget = np.array([3, 0, 0], dtype=int)
                    
                elif self._target=='ES330_idmrg_edgeAdjoint':
                    # For SU(3), m=3, with adjoint edge irreps
                    # another excited state is in [3, 3, 0] sector
                    # See Gozel et al., Phys. Rev. Lett. 125, 057202 (2020)
                    alphaTarget = np.array([3, 3, 0], dtype=int)
                
                elif self._target=='ES210_idmrg_edgeAdjoint':
                    # For SU(3), m=3, with adjoint edge irreps
                    # the lowest excited state is likely the [2, 1, 0] sector
                    # See Gozel et al., Phys. Rev. Lett. 125, 057202 (2020)
                    alphaTarget = np.array([2, 1, 0], dtype=int)
                    sys.exit('RMEEngineSymm.target_irrep: for SU(3) m=3, the target adjoint has mult>1.')
                    # we do not currently know how to deail with this target
                    # adjoint irrep in DMRG
                    
                else:
                    print(self._target)
                    sys.exit('RMEEngineSymm.target_irrep: undefined target sector.')
                
            else:
                sys.exit('RMEEngineSymm.target_irrep: for N=3, m>3 not implemented.')
            
        else:
            sys.exit('RMEEngineSymm.target_irrep: target irreps not implemented for N>3.')
        
        ncols = (n - np.sum(alphaTarget))//self._N
        alphaTarget += np.full(shape=(self._N,), fill_value=ncols, dtype=int)
        
        return alphaTarget
    
    
    def _evaluate_compatibility(self, alpha1, alpha2):
        """
        Test if alpha1, alpha2 are valid irreps for the target sector
        """
        
        condition = True
        
        n1 = np.sum(alpha1)
        n2 = np.sum(alpha2)
        n = n1 + n2
        
        if self._N==3:
            
            if self._m==2:
                
                if self._target in ['GS_idmrg', 'GS_idmrg_edgeAdjoint', 'ES_idmrg', 'ES_idmrg_edgeAdjoint']:
                    condition = ((n1 - n2) % self._N == 0)
                    
                elif self._target in ['EX300_dmrg']:
                    condition = (n % self._N == 0)
            
            elif self._m==3:
                
                cond1 = (n1 % self._N == 0)
                cond2 = (n2 % self._N == 0)
                condition = cond1 * cond2
        
        return condition
    
    
    def _get_filename(self, num_irreps):
        """
        
        """
        filename = self._filename_prefix + '_SU' + str(self._N) + '_' + self._target + '_numirreps' + str(num_irreps) +  '_' + self._tech +  '.pickle'
        filename = os.path.join(os.getcwd(), 'sunpy', 'rme_coefficients', filename)
        return filename
    
    
    def _get_sdc(self, alpha, alpha1, l1, alpha2, l2):
        """
        Compute SDCs for |alpha; alpha1, l1; alpha2, l2>
        """
        assert(len(l1)==self._m)
        assert(len(l2)==self._m)
        
        if not self._tech in ['base', 'shortcut_cols', 'shortcut_rows']:
            sys.exit('tech undefined.')
        
        if self._tech=='base':
            
            ydev, cydev, coeff = sdcdmrgbase.get_SDC(self._N, alpha, 
                                                     alpha1, l1, 
                                                     alpha2, l2, 
                                                     ref2firstLLOS=False, 
                                                     ref1firstLLOS=True,
                                                     symmetry='symmetric')
            
        elif self._tech=='shortcut_cols':
            
            ydev, cydev, coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=False, 
                                                 ref1firstLLOS=True,
                                                 symmetry='symmetric')
            '''
            ydev0, cydev0, coeff0 = sdcdmrgbase.get_SDC(self._N, alpha, 
                                                     alpha1, l1, 
                                                     alpha2, l2, 
                                                     ref2firstLLOS=False, 
                                                     ref1firstLLOS=True,
                                                     symmetry='symmetric')
            assert(np.linalg.norm(ydev-ydev0)==0)
            assert(np.linalg.norm(cydev-cydev0)==0)
            assert(np.linalg.norm(coeff-coeff0)<1.0e-12)
            '''
        elif self._tech=='shortcut_rows':
            
            ydev, cydev, coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=True, 
                                                 ref1firstLLOS=True,
                                                 symmetry='symmetric')
            '''
            ydev0, cydev0, coeff0 = sdcdmrgbase.get_SDC(self._N, alpha, 
                                                     alpha1, l1, 
                                                     alpha2, l2, 
                                                     ref2firstLLOS=True, 
                                                     ref1firstLLOS=True,
                                                     symmetry='symmetric')
            assert(np.linalg.norm(ydev-ydev0)==0)
            assert(np.linalg.norm(cydev-cydev0)==0)
            assert(np.linalg.norm(coeff-coeff0)<1.0e-12)
            '''
        
        assert(len(coeff.shape)==1)
        
        return ydev, cydev, coeff
    
    
    def _atomic_run(self):
        """
        Compute the reduced matrix elements
        """
        
        print('Start computing RME num_irreps = ', self._num_irreps)
        ts = time.time()
        
        if self._restarting==True:
            rme_reader_old = RMEReader(self._N, 
                                       self._num_irreps_old, 
                                       self._restarting_filename, 
                                       m=self._m, 
                                       symmetry='symmetric')
        
        states = states_rme(self._N, self._num_irreps, self._irreps, m=self._m, symmetry='symmetric') # all ket states |alpha1, l1; alpha2, l2>
        num_states = states.shape[0]
        
        self._liste_rme = []
        self._indices_liste_rme = {}
        index_liste_rme = int(0)
        bool_indliste = np.full(shape=(num_states,), fill_value=False, dtype=bool)
        
        for p in range(0, num_states):
            
            doCalcs = []
            index_bra_smaller_than_p = []
            rme_to_compute = []
            
            alpha1 = self._irreps[states[p, 0]]
            lvec1 = states[p, 1:self._m+1]
            alpha2 = self._irreps[states[p, self._m+1]]
            lvec2 = states[p, self._m+2:]
            # |p> = |alpha1, l1; alpha2, l2>
            
            # build target irrep
            alphaGS = self._target_irrep(np.sum(alpha1)+np.sum(alpha2))
            
            condition_1 = self._evaluate_compatibility(alpha1, alpha2)
            
            if condition_1:
                mult_1 = sun.multiplicity_irrep_mixed(alphaGS, np.array([alpha1, alpha2]), self._N)
            else:
                mult_1 = int(0)
            
            if (mult_1>1):
                
                sys.exit('Problem: mult_1>1')
                
            elif ((mult_1==1) & condition_1):
                
                # construct ascendant shapes
                alpha1_asc = np.copy(alpha1)
                for l1 in lvec1:
                    alpha1_asc[l1] -= 1
                alpha2_asc = np.copy(alpha2)
                for l2 in lvec2:
                    alpha2_asc[l2] -= 1
                
                # get all descendants of the ascendants shapes
                alpha1_desc, pos1_desc = sun.get_descendants(alpha1_asc, self._N, m=self._m, symmetry='symmetric')
                alpha2_desc, pos2_desc = sun.get_descendants(alpha2_asc, self._N, m=self._m, symmetry='symmetric')
                
                nbd_1 = len(pos1_desc) # number of descendants of alpha1_asc
                nbd_2 = len(pos2_desc) # number of descendants of alpha2_asc
                
                # search for <bra| states leading to RMEs to compute/extract
                
                for d1 in range(0, nbd_1):
                    for d2 in range(0, nbd_2):
                        
                        alpha3 = alpha1_desc[d1]
                        lvec3 = pos1_desc[d1]
                        
                        alpha4 = alpha2_desc[d2]
                        lvec4 = pos2_desc[d2]
                        
                        mult_2 = sun.multiplicity_irrep_mixed(alphaGS, 
                                                              np.array([alpha3, alpha4]), 
                                                              self._N)
                        
                        if (mult_2>1):
                            
                            sys.exit('Problem: mult_2>1')
                            
                        elif (mult_2==1):
                            
                            # ensure there is a single column with N boxes
                            alpha3p = alpha3 + (self._m-alpha3[-1])*np.full(shape=(self._N,), fill_value=1, dtype=int)
                            alpha4p = alpha4 + (self._m-alpha4[-1])*np.full(shape=(self._N,), fill_value=1, dtype=int)
                            
                            ind3 = sunpy.common.math.find_row(self._irreps, alpha3p)
                            ind4 = sunpy.common.math.find_row(self._irreps, alpha4p)
                            
                            if ((len(ind3)>0) & (len(ind4)>0)):
                                
                                doCalc = True
                                
                                if self._restarting:
                                    # search for indices in old list
                                    ind1_old = sunpy.common.math.find_row(self._irreps_old, alpha1)
                                    ind2_old = sunpy.common.math.find_row(self._irreps_old, alpha2)
                                    ind3_old = sunpy.common.math.find_row(self._irreps_old, alpha3p)
                                    ind4_old = sunpy.common.math.find_row(self._irreps_old, alpha4p)
                                
                                    if len(ind1_old)*len(ind2_old)*len(ind3_old)*len(ind4_old)==1:
                                        doCalc = False
                                
                                doCalcs.append(doCalc)
                                
                                bra = np.hstack((ind3[0], lvec3, ind4[0], lvec4))
                                index_bra = sunpy.common.math.find_row(states, bra)[0]
                                # <index_bra| = <alpha3, l3; alpha4, l4|
                                index_bra_smaller_than_p.append(index_bra<p)
                                
                                # register a new element to compute/extract
                                td = {}
                                if doCalc==True:
                                    td['alpha3'] = alpha3
                                    td['alpha4'] = alpha4
                                else:
                                    td['alpha3'] = alpha3p
                                    td['alpha4'] = alpha4p
                                td['index_bra'] = index_bra
                                td['lvec3'] = lvec3
                                td['lvec4'] = lvec4
                                td['doCalc'] = doCalc
                                rme_to_compute.append(td)
                
                # perform calculation of RMEs associated with |p> = |alpha1, lvec1; alpha2, lvec2>
                # for all identified <bra| states
                
                if len(rme_to_compute)>0:
                
                    Matp_ind = np.zeros(shape=(len(rme_to_compute),), dtype=int)
                    Matp_coeff = np.zeros(shape=(len(rme_to_compute),), dtype=float)
                    
                    bool_compute = [a and (not b) for a, b in zip(doCalcs, index_bra_smaller_than_p)]
                    
                    if any(bool_compute):
                        ket_y, ket_cy, ket_coeff = self._get_sdc(alphaGS, 
                                                                 alpha1, lvec1, 
                                                                 alpha2, lvec2)
                        Pket_y, Pket_cy, Pket_coeff = sun.develop_consecutive_number(alphaGS, 
                                                                                     np.copy(ket_y), 
                                                                                     np.copy(ket_cy), 
                                                                                     np.copy(ket_coeff), 
                                                                                     np.sum(alpha1)-1)
                    else:
                        # we entirely read from memory, either from an old list, or 
                        #  from already computed elements of the current list
                        pass
                    
                    for i in range(0, len(rme_to_compute)):
                        td = rme_to_compute[i]
                        if td['doCalc']==True:
                            if (td['index_bra']<p):                                
                                # we use the fact that <index_bra|P|p>==<p|P|index_bra>
                                # to read from the current list
                                ics = self._liste_rme[self._indices_liste_rme[td['index_bra']]]
                                indp = np.argwhere(ics[0]==p).flatten()
                                assert(len(indp)==1)
                                coeff = ics[1][indp[0]]
                            else:
                                # we really need to perform the calculation
                                if (td['index_bra']==p):
                                    # the <bra| state is identical to the |ket> state
                                    # avoid re-computing the SDCs for the <bra|
                                    bra_y, bra_cy, bra_coeff = ket_y, ket_cy, ket_coeff
                                else:
                                    # td['index_bra']>p --> need to compute SDCs for the <bra|
                                    bra_y, bra_cy, bra_coeff = self._get_sdc(alphaGS, 
                                                                             td['alpha3'], td['lvec3'], 
                                                                             td['alpha4'], td['lvec4'])
                                # compute overlap <bra|Pket>
                                _, bra_ind, Pket_ind = sunpy.common.math.intersect_row(bra_cy, Pket_cy)
                                coeff = np.sum( np.multiply(bra_coeff[bra_ind], Pket_coeff[Pket_ind]) )
                        else:
                            # read coefficient from old list
                            coeff = rme_reader_old.read(alpha1, lvec1, 
                                                        alpha2, lvec2, 
                                                        td['alpha3'], td['lvec3'], 
                                                        td['alpha4'], td['lvec4'], )
                        Matp_ind[i] = td['index_bra']
                        Matp_coeff[i] = coeff
                    
                    self._liste_rme.append([Matp_ind, Matp_coeff])
                    bool_indliste[p] = True
                    self._indices_liste_rme[p] = index_liste_rme
                    index_liste_rme += 1
        
        self._indliste = np.argwhere(bool_indliste==True).flatten()
        
        te = time.time()
        print('Done. Time = ', (te-ts)/60, 'min')
        
        self.save(self._filename)
        
        return

