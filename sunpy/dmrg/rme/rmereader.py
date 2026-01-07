# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np
import sys
import pickle

import sunpy.common.math
from sunpy.dmrg.rme.utils import states_rme


class RMEReader:
    """
    Class for reading reduced matrix elements from file
    """
    
    def __init__(self, N, num_irreps, filename, m, **kwargs):
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
        m : int
            number of particles
        
        
        """
        self._N = N
        self._m = m
        self._num_irreps = num_irreps
        self._filename = filename
        
        if N==3:
            self._irreps = np.load('sunpy/irreps/SU3_irreps_300.npy')
            self._irreps = self._irreps.astype(int)
        else:
            sys.exit('List of irreps not yet computed for N>3.')
        self._irreps = self._irreps[:self._num_irreps] + np.full(shape=(self._num_irreps, self._N), fill_value=self._m, dtype=int)
        
        with open(self._filename, 'rb') as file:
            print('Reading RME from: ', self._filename)
            data = pickle.load(file)
        self._rme = data['liste_rme']
        self._indliste = data['indliste']
        self._indices_liste_rme = data['indices_liste_rme']
        if self._m==1:
            self._states = states_rme(self._N, self._num_irreps, self._irreps)
        else:
            if 'symmetry' in kwargs:
                if kwargs['symmetry'] in ['symm', 'symmetric', 'antisymm', 'antisymmetric']:
                    self._states = states_rme(self._N, self._num_irreps, self._irreps, m=self._m, symmetry=kwargs['symmetry'])
                else:
                    sys.exit('RMEReader: symmetry undefined.')
            else:
                sys.exit('RMEReader: missing symmetry specification for m>1.')
        
        self._num_states = self._states.shape[0]
    
    
    def __get_index_irrep(self, nu) -> int:
        index = sunpy.common.math.find_row(self._irreps, nu)
        assert(len(index)==1)
        return index[0]
    
    
    def __get_index_state(self, state) -> int:
        index = sunpy.common.math.find_row(self._states, state)
        assert(len(index)==1)
        return index[0]
    
    
    def read(self, nu1, l1, nu2, l2, nu3, l3, nu4, l4):
        """
        Read reduced matrix element
        """
        
        indnu1 = self.__get_index_irrep(nu1)
        indnu2 = self.__get_index_irrep(nu2)
        ket_state = np.hstack((indnu1, l1, indnu2, l2))
        ket_state_ind = self.__get_index_state(ket_state)
        
        if ket_state_ind in self._indices_liste_rme:
            ket_rme_ind = self._indices_liste_rme[ket_state_ind]
            # one can also obtain the index through a binary search in self.indliste
            #ket_rme_ind2 = np.searchsorted(self.indliste, ket_state_ind)
            #assert(ket_rme_ind==ket_rme_ind2)
        else:
            return None
        
        indnu3 = self.__get_index_irrep(nu3)
        indnu4 = self.__get_index_irrep(nu4)
        bra_state = np.hstack((indnu3, l3, indnu4, l4))
        bra_state_ind = self.__get_index_state(bra_state)
        
        if not bra_state_ind in self._indices_liste_rme:
            return None        
        
        bra_inds = self._rme[ket_rme_ind][0]
        
        ind = np.argwhere(bra_inds==bra_state_ind).flatten()
        assert(len(ind)==1)
        ind = ind[0]
        
        # print('Reading RME: <', bra_state_ind, '|P|', ket_state_ind, '>')
        
        return self._rme[ket_rme_ind][1][ind]
