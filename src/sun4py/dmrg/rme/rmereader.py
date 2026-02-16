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

import numpy as np
import pickle

import sun4py.common.math
from sun4py.dmrg.rme.utils import states_rme


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
        symmetry : str [optional]['symmetric' or 'antisymmetric']
            symmetry type, when m>1
        
        """
        self._N = int(N)
        self._m = int(m)
        self._num_irreps = int(num_irreps)
        self._filename = filename
        
        with open(self._filename, 'rb') as file:
            print('Reading RME from: ', self._filename)
            data = pickle.load(file)
        
        self._rme = data['liste_rme']
        self._indliste = data['indliste']
        self._indices_liste_rme = data['indices_liste_rme']
        self._irreps = data['irreps']

        assert(self._irreps.shape[0]==self._num_irreps)
        assert(self._irreps[0][-1]==self._m) # singlet irrep must be represented with self._m columns
        
        if self._m==1:
            self._states = states_rme(self._N, self._num_irreps, self._irreps)
        else:
            if 'symmetry' in kwargs:
                if kwargs['symmetry'] in ['symm', 'symmetric', 'antisymm', 'antisymmetric']:
                    self._states = states_rme(self._N, self._num_irreps, self._irreps, m=self._m, symmetry=kwargs['symmetry'])
                else:
                    raise ValueError('RMEReader: __init__ : symmetry undefined.')
            else:
                raise ValueError('RMEReader: __init__ : missing symmetry specification for m>1.')
        
        self._num_states = self._states.shape[0]
    
    
    def __get_index_irrep(self, nu):
        index = sun4py.common.math.find_row(self._irreps, nu)
        assert(len(index)==1)
        return index[0]
    
    
    def __get_index_state(self, state):
        index = sun4py.common.math.find_row(self._states, state)
        assert(len(index)==1)
        return index[0]
    
    
    def __adapt_columns(self, nu):
        """
        Return an irrep equivalent to nu, but with exactly self._m column(s) with self._N boxes
        """
        # nu[-1] is the number of columns with N boxes in nu
        nup = np.copy(nu)
        nup += np.full(shape=(self._N,), fill_value=(self._m-nu[-1]), dtype=int)
        return nup
    
    def read(self, nu1, l1, nu2, l2, nu3, l3, nu4, l4):
        """
        Read reduced matrix element
        """
        
        nu1p = self.__adapt_columns(nu1)
        nu2p = self.__adapt_columns(nu2)
        nu3p = self.__adapt_columns(nu3)
        nu4p = self.__adapt_columns(nu4)
        
        indnu1 = self.__get_index_irrep(nu1p)
        indnu2 = self.__get_index_irrep(nu2p)
        ket_state = np.hstack((indnu1, l1, indnu2, l2))
        ket_state_ind = self.__get_index_state(ket_state)
        
        if ket_state_ind in self._indices_liste_rme:
            ket_rme_ind = self._indices_liste_rme[ket_state_ind]
            # one can also obtain the index through a binary search in self.indliste
            #ket_rme_ind2 = np.searchsorted(self.indliste, ket_state_ind)
            #assert(ket_rme_ind==ket_rme_ind2)
        else:
            return None
        
        indnu3 = self.__get_index_irrep(nu3p)
        indnu4 = self.__get_index_irrep(nu4p)
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
