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

# deprecated methods for checking
#from sunpy.sdc.dmrg.deprecated import sdcdmrgbasedeprecated
#from sunpy.sdc.dmrg.deprecated import sdcdmrgfunddeprecated



class RMEEngineFund(RMEEngine):
    """
    RME Engine for fundamental irrep at each site
    """
    
    def __init__(self, N, num_irreps, target='GS', **kwargs):
        """
        Constructor of RMEEngineFund
        
        Parameters
        ----------
        N : int
            SU(N)
        num_irreps : int
            number of irreps
        target : str
            string describing the target irrep
        
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
        
        Examples
        --------
        Example 1:
        rme_engine1 = RMEEngineFund(int(3), int(12), 
                                    restarting=False, 
                                    checkpointing=True, 
                                    chkpt_method='custom', 
                                    chkpt_ni=np.array([4, 6, 8]), dtype=int)
        
        --> will checkpoint at num_irreps = 4, 6, 8, and finally compute 
            num_irreps=12
        
        Example 2: 
        rme_engine2 = RMEEngineFund(int(3), int(24), 
                                    restarting=True, 
                                    checkpointing=True, 
                                    chkpt_method='log2')
        --> will search for an already computed list of RMEs. If the list is 
            insufficient (namely, <24 irreps), it will proceed to the calculation
            for num_irreps=24, checkpointing on a log2 scale between the found 
            list and 24. If the found list has more than 24 irreps, it will be 
            used to create the "reduced" list with 24 irreps.
        """
        
        assert(target in ['GS'])
        
        if 'filename_prefix' in kwargs:
            self._filename_prefix = kwargs['filename_prefix']
        else:
            self._filename_prefix = 'pythonRME_fund'
        
        super().__init__(N, num_irreps, filename_prefix=self._filename_prefix, target=target, **kwargs)
    
    
    def __str__(self):
        return f"RMEEngineFund, N={self._N}, num_irreps={self._num_irreps}, target={self._target}"
    
    
    def _init_irreps(self, num_irreps):
        """
        
        """
        return self._irreps_all[:num_irreps] + np.full(shape=(num_irreps, self._N), fill_value=1, dtype=int)
    
    
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
        if self._target=='GS':
            num_col_GS = n // self._N
            alphaTarget = num_col_GS*np.full(shape=(self._N,), fill_value=1, dtype=int)
            nr = n - num_col_GS * self._N
            ind = np.argwhere(np.sum(self._irreps_all, axis=1)==nr).flatten()
            alphaTarget += self._irreps_all[ind[0]]
        else:
            sys.exit('RMEEngineFund.target_irrep: Target irrep undefined.')
        
        return alphaTarget
    
    
    def _evaluate_compatibility(self, alpha1, alpha2):
        """
        Test if alpha1, alpha2 are valid irreps for the target sector
        """
        
        condition = True
        
        n1 = np.sum(alpha1)
        n2 = np.sum(alpha2)
        
        if self._target=='GS':
            condition = ( ((n1-n2)%self._N==0) | ((n1+n2)%self._N==0) )
        
        return condition
    
    
    def _get_filename(self, num_irreps):
        """
        
        """
        filename = self._filename_prefix + '_SU' + str(self._N) + '_' + self._target + '_numirreps' + str(num_irreps) + '.pickle'
        filename = os.path.join(os.getcwd(), 'sunpy', 'rme_coefficients', filename)
        return filename
    
    
    def _get_sdc(self, alpha, alpha1, l1, alpha2, l2, tech):
        """
        Compute SDCs for |alpha; alpha1, l1; alpha2, l2>
        """
        
        if not tech in ['base', 'shortcut_cols', 'shortcut_rows']:
            sys.exit('tech undefined.')
        
        if tech=='base':
            
            ydev, cydev, coeff = sdcdmrgbase.get_SDC(self._N, alpha, 
                                                     alpha1, l1, 
                                                     alpha2, l2, 
                                                     ref2firstLLOS=False, 
                                                     ref1firstLLOS=True)
            '''
            # old, deprecated basic method (without shortcut in Chen's method)
            ydev0, cydev0, coeff0 = sdcdmrgbasedeprecated.get_SDC(self._N, alpha, 
                                                     alpha1, l1, 
                                                     alpha2, l2, 
                                                     ref2firstLLOS=False, 
                                                     ref1firstLLOS=True)
            
            assert(np.linalg.norm(ydev-ydev0)==0)
            assert(np.linalg.norm(cydev-cydev0)==0)
            assert(np.linalg.norm(coeff-coeff0)<1.0e-12)
            '''
        elif tech=='shortcut_cols':
            
            ydev, cydev, coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=False, 
                                                 ref1firstLLOS=True)
            '''
            # old, deprecated method with shortcut
            ydev0, cydev0, coeff0 = sdcdmrgfunddeprecated.get_SDC(self._N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=False, 
                                                 ref1firstLLOS=True)
            assert(np.linalg.norm(ydev-ydev0)==0)
            assert(np.linalg.norm(cydev-cydev0)==0)
            assert(np.linalg.norm(coeff-coeff0)<1.0e-12)
            '''
        elif tech=='shortcut_rows':
            
            ydev, cydev, coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=True, 
                                                 ref1firstLLOS=True)
            '''
            # old, deprecated method with shortcut
            ydev0, cydev0, coeff0 = sdcdmrgfunddeprecated.get_SDC(self._N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=True, 
                                                 ref1firstLLOS=True)
            assert(np.linalg.norm(ydev-ydev0)==0)
            assert(np.linalg.norm(cydev-cydev0)==0)
            assert(np.linalg.norm(coeff-coeff0)<1.0e-12)
            '''
        
        assert(len(coeff.shape)==1)
        
        return ydev, cydev, coeff
    
    
    def _atomic_run(self, tech='base'):
        """
        Compute the reduced matrix elements
        """
        
        print('Start computing RME num_irreps = ', self._num_irreps)
        ts = time.time()
        
        if self._restarting==True:
            rme_reader_old = RMEReader(self._N, 
                                       self._num_irreps_old, 
                                       self._restarting_filename,
                                       m=1)
        
        states = states_rme(self._N, self._num_irreps, self._irreps) # all ket states |alpha1, l1; alpha2, l2>
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
            l1 = states[p, 1]
            alpha2 = self._irreps[states[p, 2]]
            l2 = states[p, 3]
            # |p> = |alpha1, l1; alpha2, l2>
            
            # build target irrep
            alphaGS = self._target_irrep(np.sum(alpha1)+np.sum(alpha2))
            
            mult_1 = sun.multiplicity_irrep_mixed(alphaGS, np.array([alpha1, alpha2]), self._N)
            
            # condition_1 = ( ((np.sum(alpha1)-np.sum(alpha2))%self.N==0) | ((np.sum(alpha1) + np.sum(alpha2))%self.N==0) )
            condition_1 = self._evaluate_compatibility(alpha1, alpha2)
            
            if (mult_1>1):
                sys.exit('Problem: mult_1>1')
            elif ((mult_1==1) & condition_1):
                
                # construct ascendant shapes
                alpha1_asc = np.copy(alpha1)
                alpha1_asc[l1] -= 1
                alpha2_asc = np.copy(alpha2)
                alpha2_asc[l2] -= 1
                
                # get all descendants of the ascendants shapes
                alpha1_desc, pos1_desc = sun.get_descendants(alpha1_asc, self._N)
                alpha2_desc, pos2_desc = sun.get_descendants(alpha2_asc, self._N)
                
                nbd_1 = len(pos1_desc) # number of descendants of alpha1_asc
                nbd_2 = len(pos2_desc) # number of descendants of alpha2_asc
                
                # search for <bra| states leading to RMEs to compute/extract
                
                for d1 in range(0, nbd_1):
                    for d2 in range(0, nbd_2):
                        
                        alpha3 = alpha1_desc[d1]
                        l3 = pos1_desc[d1]
                        
                        alpha4 = alpha2_desc[d2]
                        l4 = pos2_desc[d2]
                        
                        mult_2 = sun.multiplicity_irrep_mixed(alphaGS, 
                                                              np.array([alpha3, alpha4]), 
                                                              self._N)
                        
                        #condition_2 = ( ((np.sum(alpha3)-np.sum(alpha4))%self.N==0) | ((np.sum(alpha3) + np.sum(alpha4))%self.N==0) )
                        condition_2 = self._evaluate_compatibility(alpha3, alpha4)
                        
                        if (mult_2>1):
                            sys.exit('Problem: mult_2>1')
                        elif ((mult_2==1) & condition_2):
                            
                            doCalc = True
                            
                            # ensure there is a single column with N boxes
                            alpha3p = alpha3 + (1-alpha3[-1])*np.full(shape=(self._N,), fill_value=1, dtype=int)
                            alpha4p = alpha4 + (1-alpha4[-1])*np.full(shape=(self._N,), fill_value=1, dtype=int)
                            
                            if self._restarting:
                                # search for indices in old list
                                ind1_old = sunpy.common.math.find_row(self._irreps_old, alpha1)
                                ind2_old = sunpy.common.math.find_row(self._irreps_old, alpha2)
                                ind3_old = sunpy.common.math.find_row(self._irreps_old, alpha3p)
                                ind4_old = sunpy.common.math.find_row(self._irreps_old, alpha4p)
                                
                                if len(ind1_old)*len(ind2_old)*len(ind3_old)*len(ind4_old)==1:
                                    doCalc = False
                            
                            ind3 = sunpy.common.math.find_row(self._irreps, alpha3p)
                            ind4 = sunpy.common.math.find_row(self._irreps, alpha4p)
                            
                            if ((len(ind3)>0) & (len(ind4)>0)):
                                
                                doCalcs.append(doCalc)
                                
                                bra = np.array([ind3[0], l3, ind4[0], l4], dtype=int)
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
                                td['l3'] = l3
                                td['l4'] = l4
                                td['doCalc'] = doCalc
                                rme_to_compute.append(td)
                
                # perform calculation of RMEs associated with |p> = |alpha1, l1; alpha2, l2>
                # for all identified <bra| states
                
                if len(rme_to_compute)>0:
                
                    Matp_ind = np.zeros(shape=(len(rme_to_compute),), dtype=int)
                    Matp_coeff = np.zeros(shape=(len(rme_to_compute),), dtype=float)
                    
                    bool_compute = [a and (not b) for a, b in zip(doCalcs, index_bra_smaller_than_p)]
                    
                    if any(bool_compute):
                        ket_y, ket_cy, ket_coeff = self._get_sdc(alphaGS, 
                                                                 alpha1, l1, 
                                                                 alpha2, l2, 
                                                                 tech)
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
                                                                             td['alpha3'], td['l3'], 
                                                                             td['alpha4'], td['l4'], 
                                                                             tech)
                                # compute overlap <bra|Pket>
                                _, bra_ind, Pket_ind = sunpy.common.math.intersect_row(bra_cy, Pket_cy)
                                coeff = np.sum( np.multiply(bra_coeff[bra_ind], Pket_coeff[Pket_ind]) )
                        else:
                            # read coefficient from old list
                            coeff = rme_reader_old.read(alpha1, l1, 
                                                        alpha2, l2, 
                                                        td['alpha3'], td['l3'], 
                                                        td['alpha4'], td['l4'], )
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
    
    
    def _atomic_run_old(self, tech='base'):
        """
        Compute the reduced matrix elements
        """
        print('DEPRECATED _atomic_run_old This is an old, less efficient, method.')
        
        print('Start computing RME num_irreps = ', self.num_irreps)
        ts = time.time()
        
        if self.restarting==True:
            rme_reader_old = RMEReader(self.N, self.num_irreps_old, self.restarting_filename)
        
        states = states_rme(self.N, self.num_irreps, self.irreps)
        num_states2 = states.shape[0]
        
        # compute reduced matrix elements
    
        liste_rme = []
        indices_liste_rme = {}
        index_liste_rme = int(0)
        bool_indliste = np.full(shape=(num_states2,), fill_value=False, dtype=bool)
        
        for p in range(0, num_states2):
            
            cpt = int(0)        
            Matp_ind = np.zeros(shape=(20,), dtype=int)
            Matp_coeff = np.zeros(shape=(20,), dtype=float)
            
            alpha1 = self.irreps[states[p, 0]]
            l1 = states[p, 1]
            alpha2 = self.irreps[states[p, 2]]
            l2 = states[p, 3]
            
            # build target irrep
            alphaGS = self.target_irrep(np.sum(alpha1)+np.sum(alpha2))
            
            mult_1 = sun.multiplicity_irrep_mixed(alphaGS, np.array([alpha1, alpha2]), self.N)
            
            condition_1 = ( ((np.sum(alpha1)-np.sum(alpha2))%self.N==0) | ((np.sum(alpha1) + np.sum(alpha2))%self.N==0) )
            
            if ((mult_1==1) & condition_1):
                
                # construct ascendant shapes
                alpha1_asc = np.copy(alpha1)
                alpha1_asc[l1] -= 1
                alpha2_asc = np.copy(alpha2)
                alpha2_asc[l2] -= 1
                
                # get all descendants of the ascendants shapes
                alpha1_desc, pos1_desc = sun.get_descendants(alpha1_asc, self.N)
                alpha2_desc, pos2_desc = sun.get_descendants(alpha2_asc, self.N)
                
                nbf_1 = len(pos1_desc)
                nbf_2 = len(pos2_desc)
                
                for qq in range(0, nbf_1):
                    for pp in range(0, nbf_2):
                        
                        alpha3 = alpha1_desc[qq]
                        l3 = pos1_desc[qq]
                        
                        alpha4 = alpha2_desc[pp]
                        l4 = pos2_desc[pp]
                        
                        # ensure there is a column with N boxes
                        alpha3p = alpha3 + (1-alpha3[-1])*np.full(shape=(self.N,), fill_value=1, dtype=int)
                        alpha4p = alpha4 + (1-alpha4[-1])*np.full(shape=(self.N,), fill_value=1, dtype=int)
                        
                        mult_2 = sun.multiplicity_irrep_mixed(alphaGS, np.array([alpha3, alpha4]), self.N)
                        
                        condition_2 = ( ((np.sum(alpha3)-np.sum(alpha4))%self.N==0) | ((np.sum(alpha3) + np.sum(alpha4))%self.N==0) )
                        
                        if ((mult_2==1) & condition_2):
                            
                            doCalc = True
                            
                            if self.restarting:
                                # search for indices in old list
                                ind1_old = sunpy.common.math.find_row(self.irreps_old, alpha1)
                                ind2_old = sunpy.common.math.find_row(self.irreps_old, alpha2)
                                ind3_old = sunpy.common.math.find_row(self.irreps_old, alpha3p)
                                ind4_old = sunpy.common.math.find_row(self.irreps_old, alpha4p)
                                
                                if len(ind1_old)*len(ind2_old)*len(ind3_old)*len(ind4_old)==1:
                                    doCalc = False
                            
                            ind3 = sunpy.common.math.find_row(self.irreps, alpha3p)
                            ind4 = sunpy.common.math.find_row(self.irreps, alpha4p)
                            
                            if ((len(ind3)>0) & (len(ind4)>0)):
                                
                                bra = np.array([ind3[0], l3, ind4[0], l4], dtype=int)
                                index_bra = sunpy.common.math.find_row(states, bra)[0]
                                
                                if doCalc==True:
                                    coeff = self.__rme(alphaGS, 
                                                       alpha1, l1, 
                                                       alpha2, l2, 
                                                       alpha3, l3, 
                                                       alpha4, l4, 
                                                       tech)
                                else:
                                    coeff = rme_reader_old.read(alpha1, l1, 
                                                                alpha2, l2, 
                                                                alpha3p, l3, 
                                                                alpha4p, l4)
                                
                                Matp_ind[cpt] = index_bra
                                Matp_coeff[cpt] = coeff
                                cpt += 1
                        
                        elif (mult_2>1):
                            sys.exit('Problem: mult_2>1')
            elif (mult_1>1):
                sys.exit('Problem: mult_1>1')
            
            if cpt>0:
                Matp_ind = Matp_ind[:cpt]
                Matp_coeff = Matp_coeff[:cpt]
                liste_rme.append([Matp_ind, Matp_coeff])
                bool_indliste[p] = True
                indices_liste_rme[p] = index_liste_rme
                index_liste_rme += 1
        
        indliste = np.argwhere(bool_indliste==True).flatten()
        
        self.indliste = indliste
        self.indices_liste_rme = indices_liste_rme
        self.liste_rme = liste_rme
        
        te = time.time()
        print('Done. Time = ', (te-ts)/60, 'min')
        
        self.save(self.filename)
        
        return
    
    
    def __rme(self, alpha, alpha1, l1, alpha2, l2, alpha3, l3, alpha4, l4, tech='base'):
        """
        Reduced matrix element
            <alpha3, l3; alpha4, l4 | P_{alpha} | alpha1, l1; alpha2, l2>
        in SU(N)
        
        Parameters
        ----------
        self : RMEEngine
            current instance
        alpha : numpy array
            global target irrep
        alpha1 : numpy array
            ket irrep, left block
        l1 : int
            position of last particle in ket irrep, left block
        alpha2 : numpy array
            ket irrep, right block
        l2 : int
            position of last particle in ket irrep, right block
        alpha3 : numpy array
            ket irrep, left block
        l3 : int
            position of last particle in bra irrep, left block
        alpha4 : numpy array
            ket irrep, right block
        l4 : int
            position of last particle in bra irrep, right block
        tech : str
            technique to use to compute the SDCs
            'base', 'shortcut_cols', 'shortcut_rows'
        
        Returns
        -------
        out : float
            reduced matrix element
        
        Details
        -------
        tech='base' : basic method for the SDCS, using the group co-chain (see Ref. [1])
        tech='shortcut_cols' : shortcut in Chen's method, developing the irrep as a 
                               product of its columns (see Ref. [2])
        tech='shortcut_rows' : shortcut in Chen's method, developing the irrep as a
                               product of its rows
        
        References
        ----------
        [1]         Group Representation Theory for Physicists
                    Jin-Quan Chen, Hialun Ping and Fan Wang
                    World Scientific, 2nd edition, (2002)
        [2]         DMRG simulations of SU(N) Heisenberg chains using standard ...
                    Pierre Nataf, Frederic Mila
                    Phys. Rev. B 97, 134420 (2018)
        """
        
        if not tech in ['base', 'shortcut_cols', 'shortcut_rows']:
            sys.exit('tech undefined.')
        
        if tech=='base':
            
            ket_y, ket_cy, ket_coeff = sdcdmrgbase.get_SDC(self._N, alpha, 
                                                    alpha1, l1, 
                                                    alpha2, l2, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
            
            bra_y, bra_cy, bra_coeff = sdcdmrgbase.get_SDC(self._N, alpha, 
                                                    alpha3, l3, 
                                                    alpha4, l4, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
        elif tech=='shortcut_cols':
            
            ket_y, ket_cy, ket_coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                    alpha1, l1, 
                                                    alpha2, l2, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
            
            bra_y, bra_cy, bra_coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                    alpha3, l3, 
                                                    alpha4, l4, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
            
        elif tech=='shortcut_rows':
            
            ket_y, ket_cy, ket_coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                    alpha1, l1, 
                                                    alpha2, l2, 
                                                    ref2firstLLOS=True, 
                                                    ref1firstLLOS=True)
            
            bra_y, bra_cy, bra_coeff = sdcdmrg.get_SDC(self._N, alpha, 
                                                    alpha3, l3, 
                                                    alpha4, l4, 
                                                    ref2firstLLOS=True, 
                                                    ref1firstLLOS=True)
        
        ket_coeff = ket_coeff.flatten()
        bra_coeff = bra_coeff.flatten()
        
        n1 = np.sum(alpha1)
        ket_y, ket_cy, ket_coeff = sun.develop_consecutive_number(alpha, ket_y, ket_cy, ket_coeff, n1-1)
        
        _, bra_ind, ket_ind = sunpy.common.math.intersect_row(bra_cy, ket_cy)
        
        out = np.sum( np.multiply(bra_coeff[bra_ind], ket_coeff[ket_ind]) )
        
        return out
