# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np
import sys
import pickle
import os
import re
import time

import sunpy.common.math
from sunpy.sun import sun
from sunpy.sdc.dmrg import sdcdmrgbase
from sunpy.sdc.dmrg import sdcdmrg



def get_states(N, num_irreps, irreps):
    """
    Create the ensemble of all states (irrep, bottom corner) for a given ensemble
    of input irreps
    
    Parameters
    ----------
    N : int
        SU(N)
    num_irreps : int
        number of irreps
    irreps : numpy array
        array of num_irreps first irreps of SU(N)
    """
    assert(irreps.shape[0]==num_irreps)
    
    states = np.zeros(shape=(N*num_irreps, 2), dtype=int)
    cpt = int(0)
    for p in range(0, num_irreps):
        bc_vec = sun.get_bottom_corner(irreps[p])
        for q in range(0, len(bc_vec)):
            states[cpt+q, 0] = p
            states[cpt+q, 1] = bc_vec[q]
        cpt += len(bc_vec)
    
    num_states = cpt
    states = states[0:num_states]
    
    return states


def states_rme(N, num_irreps, irreps):
    """
    Create the ensemble of all states (irrep, bottom corner) for a given ensemble
    of input irreps
    
    Parameters
    ----------
    N : int
        SU(N)
    num_irreps : int
        number of irreps
    irreps : numpy array
        array of num_irreps first irreps of SU(N)
    """
    assert(irreps.shape[0]==num_irreps)
    
    states1 = get_states(N, num_irreps, irreps)
    num_states = states1.shape[0]
    num_states2 = num_states * num_states
    
    states = np.zeros(shape=(num_states2, 4), dtype=int)
    
    states[:, 0:2] = np.repeat(states1, repeats=num_states, axis=0)
    states[:, 2:] = np.matlib.repmat(states1, num_states, 1)
    
    return states
    

class RMEEngine:
    """
    Class to deal with the computation of reduced matrix elements in DMRG
    """
    
    def __init__(self, N, num_irreps, target='GS', **kwargs):
        """
        Constructor of RMEEngine
        
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
        rme_engine1 = rmefund.RMEEngine(int(3), int(12), 
                               restarting=False, 
                               checkpointing=True, 
                               chkpt_method='custom', 
                               chkpt_ni=np.array([4, 6, 8]), dtype=int)
        
        --> will checkpoint at num_irreps = 4, 6, 8, and finally compute 
            num_irreps=12
        
        Example 2: 
        rme_engine2 = rmefund.RMEEngine(int(3), int(24), 
                               restarting=True, 
                               checkpointing=True, 
                               chkpt_method='log2')
        --> will search for an already computed list of RMEs. If the list is 
            insufficient (namely, <24 irreps), it will proceed to the calculation
            for num_irreps=24, checkpointing on a log2 scale between the found 
            list and 24
        """
        
        self.N = N
        if self.N==3:
            self.irreps_all = np.load('sunpy/irreps/SU3_irreps_300.npy')
        else:
            sys.exit('List of irreps not yet computed for N>3.')
        
        self.num_irreps = num_irreps
        self.irreps = self.__init_irreps(self.num_irreps)
        
        self.target = target
        
        if 'restarting' in kwargs:
            self.restarting = kwargs['restarting']
        else:
            self.restarting = True
        
        if self.restarting==True:
            if 'restarting_filename' in kwargs:
                path_head, path_tail = os.path.split(kwargs['restarting_filename'])
                if path_head=='':
                    self.restarting_filename = os.path.join(os.getcwd(), 
                                                            'sunpy', 
                                                            'rme_coefficients', 
                                                            kwargs['restarting_filename'])
                else:
                    self.restarting_filename = kwargs['restarting_filename']
                if not os.path.isfile(self.restarting_filename):
                    print('Restarting file : ', self.restarting_filename, ' was not found.')
                    sys.exit('Exit')
                pattern = re.compile(r'^pythonRME_fund_SU(\d+)_' + self.target + '_numirreps(\d+)\.pickle$')
                m = pattern.match(self.restarting_filename)
                assert( int(m.group(1)) == self.N )
                self.num_irreps_old = int(m.group(2))
            else:
                if 'restarting_folder' in kwargs:
                    self.restarting_folder = kwargs['restarting_folder']
                else:
                    self.restarting_folder = os.path.join(os.getcwd(), 'sunpy', 'rme_coefficients')
                    # self.restarting_folder = os.path.dirname(os.path.abspath(__file__))
                # search for restart file
                files_in_dir = [f for f in os.listdir(self.restarting_folder) if os.path.isfile(os.path.join(self.restarting_folder, f))]
                # search for pattern
                pattern = re.compile(r'^pythonRME_fund_SU(\d+)_' + self.target + '_numirreps(\d+)\.pickle$')
                self.num_irreps_old = int(0)
                for file in files_in_dir:
                    m = pattern.match(file)
                    if m:
                        fN = int(m.group(1))
                        fni = int(m.group(2))
                        if fN==self.N:
                            if fni>self.num_irreps_old:
                                self.num_irreps_old = fni
                                self.restarting_filename = os.path.join(self.restarting_folder, file)
                if self.num_irreps_old==0:
                    # we did not find a restarting file
                    self.restarting = False
        else:
            self.num_irreps_old = int(0)
        
        if self.restarting==True:
            print('Found restarting file with ', self.num_irreps_old, ' irreps')
            print(self.restarting_filename)
            self.irreps_old = self.__init_irreps(self.num_irreps_old)
        
        if 'checkpointing' in kwargs:
            self.checkpointing = kwargs['checkpointing']
        else:
            self.checkpointing = False
        
        if self.num_irreps<self.num_irreps_old:
            # the list of RME for self.num_irreps will be extracted by reading
            # in a list with more irreps ===> no need to checkpoint
            self.checkpointing = False
        
        if self.checkpointing==True:
            if 'chkpt_method' in kwargs:
                self.chkpt_method = kwargs['chkpt_method']
            else:
                self.chkpt_method = 'log2'
            
            if self.chkpt_method=='custom':
                if 'chkpt_ni' in kwargs:
                    assert(isinstance(kwargs['chkpt_ni'], np.ndarray))
                    self.chkpt_ni = kwargs['chkpt_ni']
                    if not self.chkpt_ni[-1]==num_irreps:
                        self.chkpt_ni = np.hstack((self.chkpt_ni, self.num_irreps))
                else:
                    sys.exit('Missing input argument chkpt_ni for custom checkpointing.')
            else:
                t = np.floor(np.log2(self.num_irreps-self.num_irreps_old)) + 1
                xx = np.arange(1, t+1, 1)
                self.chkpt_ni = self.num_irreps_old + np.ceil( (self.num_irreps-self.num_irreps_old) * (1 - 1/2**xx ) ).astype(int)
                assert(self.chkpt_ni[-1]==self.num_irreps)
                assert(self.chkpt_ni[0]>self.num_irreps_old)
        
        self.filename = os.path.join(os.getcwd(), 
                                     'sunpy', 
                                     'rme_coefficients', 
                                     self.__get_filename(self.num_irreps))
        
        return
    
    
    def run(self, tech='base'):
        """
        Compute the reduced matrix elements
        """
        if self.num_irreps==self.num_irreps_old:
            return
        if self.checkpointing==False:
            self.__atomic_run(tech)
        else:
            self.ultimate_num_irreps = self.num_irreps # not really necessary
            for i, ni in enumerate(self.chkpt_ni):
                print(':::::::::::::::::::')
                print('Checkpoint run ', i+1, '/', len(self.chkpt_ni), ' : num_irreps = ', ni)
                self.num_irreps = ni
                self.filename = self.__get_filename(ni)
                self.irreps = self.__init_irreps(ni)
                # peform calculation
                self.__atomic_run(tech)
                # use current checkpoint as restarting in next iteration
                self.restarting = True
                self.restarting_filename = self.filename
                self.num_irreps_old = ni
                self.irreps_old = self.__init_irreps(ni)
        
        return
    
    
    def __init_irreps(self, num_irreps):
        """
        
        """
        return self.irreps_all[:num_irreps] + np.full(shape=(num_irreps, self.N), fill_value=1, dtype=int)
    
    
    def __get_filename(self, num_irreps):
        """
        
        """
        filename = 'pythonRME_fund_SU' + str(self.N) + '_' + self.target + '_numirreps' + str(num_irreps) + '.pickle'
        filename = os.path.join(os.getcwd(), 'sunpy', 'rme_coefficients', filename)
        return filename
    
    
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
            
            ket_y, ket_cy, ket_coeff = sdcdmrgbase.get_SDC(self.N, alpha, 
                                                    alpha1, l1, 
                                                    alpha2, l2, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
            
            bra_y, bra_cy, bra_coeff = sdcdmrgbase.get_SDC(self.N, alpha, 
                                                    alpha3, l3, 
                                                    alpha4, l4, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
        elif tech=='shortcut_cols':
            
            ket_y, ket_cy, ket_coeff = sdcdmrg.get_SDC(self.N, alpha, 
                                                    alpha1, l1, 
                                                    alpha2, l2, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
            
            bra_y, bra_cy, bra_coeff = sdcdmrg.get_SDC(self.N, alpha, 
                                                    alpha3, l3, 
                                                    alpha4, l4, 
                                                    ref2firstLLOS=False, 
                                                    ref1firstLLOS=True)
            
        elif tech=='shortcut_rows':
            
            ket_y, ket_cy, ket_coeff = sdcdmrg.get_SDC(self.N, alpha, 
                                                    alpha1, l1, 
                                                    alpha2, l2, 
                                                    ref2firstLLOS=True, 
                                                    ref1firstLLOS=True)
            
            bra_y, bra_cy, bra_coeff = sdcdmrg.get_SDC(self.N, alpha, 
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
    
    
    def __get_sdc(self, alpha, alpha1, l1, alpha2, l2, tech):
        """
        Compute SDCs for |alpha; alpha1, l1; alpha2, l2>
        """
        
        if not tech in ['base', 'shortcut_cols', 'shortcut_rows']:
            sys.exit('tech undefined.')
        
        if tech=='base':
            ydev, cydev, coeff = sdcdmrgbase.get_SDC(self.N, alpha, 
                                                     alpha1, l1, 
                                                     alpha2, l2, 
                                                     ref2firstLLOS=False, 
                                                     ref1firstLLOS=True)
        elif tech=='shortcut_cols':            
            ydev, cydev, coeff = sdcdmrg.get_SDC(self.N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=False, 
                                                 ref1firstLLOS=True)            
        elif tech=='shortcut_rows':            
            ydev, cydev, coeff = sdcdmrg.get_SDC(self.N, alpha, 
                                                 alpha1, l1, 
                                                 alpha2, l2, 
                                                 ref2firstLLOS=True, 
                                                 ref1firstLLOS=True)
        coeff = coeff.flatten()
        return ydev, cydev, coeff
        
    
    def __target_irrep(self, n):
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
        if self.target=='GS':
            num_col_GS = n // self.N
            alphaTarget = num_col_GS*np.full(shape=(self.N,), fill_value=1, dtype=int)
            nr = n - num_col_GS * self.N
            ind = np.argwhere(np.sum(self.irreps_all, axis=1)==nr).flatten()
            alphaTarget += self.irreps_all[ind[0]]
        else:
            sys.exit('Target irrep undefined.')
        
        return alphaTarget
    
    
    def __save(self, filename):
        """
        Save reduced matrix elements to file
        """
        data = {}
        data['indliste'] = self.indliste
        data['indices_liste_rme'] = self.indices_liste_rme
        data['liste_rme'] = self.liste_rme
        with open(filename, 'wb') as file:
            print('Saving RME to file: ', filename)
            pickle.dump(data, file)
        return    
    
    
    def __get_index_irrep(self, nu) -> int:
        index = sunpy.common.math.find_row(self.irreps, nu)
        assert(len(index)==1)
        return index[0]
    
    
    def __get_index_state(self, state) -> int:
        index = sunpy.common.math.find_row(self.states, state)
        assert(len(index)==1)
        return index[0]
    
    
    def __atomic_run(self, tech='base'):
        """
        Compute the reduced matrix elements
        """
        
        print('Start computing RME num_irreps = ', self.num_irreps)
        ts = time.time()
        
        if self.restarting==True:
            rme_reader_old = RMEReader(self.N, self.num_irreps_old, self.restarting_filename)
        
        states = states_rme(self.N, self.num_irreps, self.irreps) # all ket states |alpha1, l1; alpha2, l2>
        num_states = states.shape[0]
    
        self.liste_rme = []
        self.indices_liste_rme = {}
        index_liste_rme = int(0)
        bool_indliste = np.full(shape=(num_states,), fill_value=False, dtype=bool)
        
        for p in range(0, num_states):
            
            doCalcs = []
            index_bra_smaller_than_p = []
            rme_to_compute = []
            
            alpha1 = self.irreps[states[p, 0]]
            l1 = states[p, 1]
            alpha2 = self.irreps[states[p, 2]]
            l2 = states[p, 3]
            # |p> = |alpha1, l1; alpha2, l2>
            
            # build target irrep
            alphaGS = self.__target_irrep(np.sum(alpha1)+np.sum(alpha2))
            
            mult_1 = sun.multiplicity_irrep_mixed(alphaGS, np.array([alpha1, alpha2]), self.N)
            
            condition_1 = ( ((np.sum(alpha1)-np.sum(alpha2))%self.N==0) | ((np.sum(alpha1) + np.sum(alpha2))%self.N==0) )
            
            if (mult_1>1):
                sys.exit('Problem: mult_1>1')
            elif ((mult_1==1) & condition_1):
                
                # construct ascendant shapes
                alpha1_asc = np.copy(alpha1)
                alpha1_asc[l1] -= 1
                alpha2_asc = np.copy(alpha2)
                alpha2_asc[l2] -= 1
                
                # get all descendants of the ascendants shapes
                alpha1_desc, pos1_desc = sun.get_descendants(alpha1_asc, self.N)
                alpha2_desc, pos2_desc = sun.get_descendants(alpha2_asc, self.N)
                
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
                                                              self.N)
                        
                        condition_2 = ( ((np.sum(alpha3)-np.sum(alpha4))%self.N==0) | ((np.sum(alpha3) + np.sum(alpha4))%self.N==0) )
                        
                        if (mult_2>1):
                            sys.exit('Problem: mult_2>1')
                        elif ((mult_2==1) & condition_2):
                            
                            doCalc = True
                            
                            # ensure there is a single column with N boxes
                            alpha3p = alpha3 + (1-alpha3[-1])*np.full(shape=(self.N,), fill_value=1, dtype=int)
                            alpha4p = alpha4 + (1-alpha4[-1])*np.full(shape=(self.N,), fill_value=1, dtype=int)
                            
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
                        ket_y, ket_cy, ket_coeff = self.__get_sdc(alphaGS, 
                                                                  alpha1, 
                                                                  l1, 
                                                                  alpha2, 
                                                                  l2, 
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
                                ics = self.liste_rme[self.indices_liste_rme[index_bra]]
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
                                    bra_y, bra_cy, bra_coeff = self.__get_sdc(alphaGS, 
                                                                              td['alpha3'], 
                                                                              td['l3'], 
                                                                              td['alpha4'], 
                                                                              td['l4'], 
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
                    
                    self.liste_rme.append([Matp_ind, Matp_coeff])
                    bool_indliste[p] = True
                    self.indices_liste_rme[p] = index_liste_rme
                    index_liste_rme += 1
        
        self.indliste = np.argwhere(bool_indliste==True).flatten()
        
        te = time.time()
        print('Done. Time = ', (te-ts)/60, 'min')
        
        self.__save(self.filename)
        
        return
    
    
    def __atomic_run_old(self, tech='base'):
        """
        Compute the reduced matrix elements
        """
        print('DEPRECATED __atomic_run_old This is an old, less efficient, method.')
        
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
            alphaGS = self.__target_irrep(np.sum(alpha1)+np.sum(alpha2))
            
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
        
        self.__save(self.filename)
        
        return


class RMEReader:
    """
    Class for reading reduced matrix elements from file
    """
    
    def __init__(self, N, num_irreps, filename):
        """
        Load file containing the reduced matrix elements
        
        Parameters
        ----------
        N : int
            SU(N)
        num_irreps : int
            number of irreps
        filename : str
            path to filename
        """
        self.N = N
        self.num_irreps = num_irreps
        self.filename = filename
        
        if N==3:
            self.irreps = np.load('sunpy/irreps/SU3_irreps_300.npy')
        else:
            sys.exit('List of irreps not yet computed for N>3.')
        self.irreps = self.irreps[:num_irreps] + np.full(shape=(num_irreps, N), fill_value=1, dtype=int)
        
        with open(filename, 'rb') as file:
            print('Reading RME from: ', filename)
            data = pickle.load(file)
        self.rme = data['liste_rme']
        self.indliste = data['indliste']
        self.indices_liste_rme = data['indices_liste_rme']
        self.states = states_rme(self.N, self.num_irreps, self.irreps)
        self.num_states = self.states.shape[0]
    
    
    def __get_index_irrep(self, nu) -> int:
        index = sunpy.common.math.find_row(self.irreps, nu)
        assert(len(index)==1)
        return index[0]
    
    
    def __get_index_state(self, state) -> int:
        index = sunpy.common.math.find_row(self.states, state)
        assert(len(index)==1)
        return index[0]
    
    
    def read(self, nu1, l1, nu2, l2, nu3, l3, nu4, l4):
        """
        Read reduced matrix element
        """
        
        indnu1 = self.__get_index_irrep(nu1)
        indnu2 = self.__get_index_irrep(nu2)
        ket_state = np.array([indnu1, l1, indnu2, l2])
        ket_state_ind = self.__get_index_state(ket_state)
        
        if ket_state_ind in self.indices_liste_rme:
            ket_rme_ind = self.indices_liste_rme[ket_state_ind]
            # one can also obtain the index through a binary search in self.indliste
            #ket_rme_ind2 = np.searchsorted(self.indliste, ket_state_ind)
            #assert(ket_rme_ind==ket_rme_ind2)
        else:
            return None
        
        indnu3 = self.__get_index_irrep(nu3)
        indnu4 = self.__get_index_irrep(nu4)
        bra_state = np.array([indnu3, l3, indnu4, l4])
        bra_state_ind = self.__get_index_state(bra_state)
        
        if not bra_state_ind in self.indices_liste_rme:
            return None        
        
        bra_inds = self.rme[ket_rme_ind][0]
        
        ind = np.argwhere(bra_inds==bra_state_ind).flatten()
        assert(len(ind)==1)
        ind = ind[0]
        
        # print('Reading RME: <', bra_state_ind, '|P|', ket_state_ind, '>')
        
        return self.rme[ket_rme_ind][1][ind]
