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
import os
import re
from abc import ABC, abstractmethod



class RMEEngine(ABC):
    """
    Class to deal with the computation of reduced matrix elements in DMRG
    """
    
    def __init__(self, N, num_irreps, filename_prefix, target, tech, **kwargs):
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
        tech : str
            string describing the technique to use to compute RMEs ('base', 'shortcut_cols', 'shortcut_rows')
        
        [optional]
        rme_folder : str [default]'path/to/user/script/sun4py_rmes/'
            path to directory containing files with RMEs
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
        """
        
        self._N = int(N)
        
        if (num_irreps<=int(300)):
            irreps_filename = f'SU{self._N}_irreps_300.npy'
        else:
            irreps_filename = f'SU{self._N}_irreps_{num_irreps}.npy'
        
        user_dir = os.getcwd()
        irreps_dir = os.path.join(user_dir, 'sun4py_irreps')
        irreps_filename = os.path.join(irreps_dir, irreps_filename)
        
        if not os.path.exists(irreps_filename):            
            if not os.path.isdir(irreps_dir):
                os.makedirs(irreps_dir)
            # generate list of 300 (or num_irreps if 300<num_irreps) first irreps of SU(N)
            from sun4py.sun import sun
            print(f'Generating list of {max(int(300), num_irreps)} first irreps of SU({self._N})')
            self._irreps_all = sun.get_irreps_by_casimir(N, max(int(300), num_irreps))
            print('Done. Dumping to:', irreps_filename)
            np.save(irreps_filename, self._irreps_all)
        else:
            print(f'Loading list of {max(int(300), num_irreps)} first irreps of SU({self._N})')
            self._irreps_all = np.load(irreps_filename).astype(int)
        
        self._num_irreps = int(num_irreps)
        self._irreps = self._init_irreps(self._num_irreps)
        
        self._target = target
        
        if not tech in ['base', 'shortcut_cols', 'shortcut_rows']:
            raise ValueError(f'ERROR : RMEEngine.__init__ : Tech {tech} undefined')
        self._tech = tech
        
        self._rme_folder = kwargs.get('rme_folder', os.path.join(user_dir, 'sun4py_rmes'))
        if not os.path.isdir(self._rme_folder):
            os.makedirs(self._rme_folder)
        
        self._restarting = kwargs.get('restarting', True)
        
        if self._restarting==True:
            if 'restarting_filename' in kwargs:
                path_head, path_tail = os.path.split(kwargs['restarting_filename'])
                if path_head=='':
                    self._restarting_filename = os.path.join(self._rme_folder, 
                                                             kwargs['restarting_filename'])
                else:
                    self._restarting_filename = kwargs['restarting_filename']
                if not os.path.isfile(self._restarting_filename):
                    raise FileNotFoundError(f'Restarting file : {self._restarting_filename} was not found. Provide absolute path.')
                pattern = re.compile(rf'^{filename_prefix}_SU(\d+)_{self._target}_numirreps(\d+)_{self._tech}\.pickle$')
                pm = pattern.match(os.path.basename(self._restarting_filename))
                assert( int(pm.group(1)) == self._N )
                self._num_irreps_old = int(pm.group(2))
            else:
                self._restarting_folder = kwargs.get('restarting_folder', self._rme_folder)
                if os.path.isdir(self._restarting_folder):
                    # search for restart file
                    files_in_dir = [f for f in os.listdir(self._restarting_folder) if os.path.isfile(os.path.join(self._restarting_folder, f))]
                    pattern = re.compile(rf'^{filename_prefix}_SU(\d+)_{self._target}_numirreps(\d+)_{self._tech}\.pickle$')
                    self._num_irreps_old = int(0)
                    for file in files_in_dir:
                        pm = pattern.match(file)
                        if pm:
                            fN = int(pm.group(1))
                            fni = int(pm.group(2))
                            if fN==self._N:
                                if fni>self._num_irreps_old:
                                    self._num_irreps_old = fni
                                    self._restarting_filename = os.path.join(self._restarting_folder, file)
                else:
                    raise ValueError('Provided restarding folder does not exist.')
                if self._num_irreps_old==0:
                    self._restarting = False
        else:
            self._num_irreps_old = int(0)
        
        if self._restarting==True:
            print(f'Found restarting file with {self._num_irreps_old} irreps')
            print(self._restarting_filename)
            self._irreps_old = self._init_irreps(self._num_irreps_old)
        
        self._checkpointing = kwargs.get('checkpointing', False)
        
        if self._num_irreps<self._num_irreps_old:
            # the list of RME for self.num_irreps will be extracted by reading
            # in a list with more irreps ===> no need to checkpoint
            self._checkpointing = False
        
        if self._checkpointing==True:
            self._chkpt_method = kwargs.get('chkpt_method', 'log2')
            if self._chkpt_method=='custom':
                if 'chkpt_ni' in kwargs:
                    assert(isinstance(kwargs['chkpt_ni'], np.ndarray))
                    self._chkpt_ni = kwargs['chkpt_ni']
                    if not self._chkpt_ni[-1]==self._num_irreps:
                        self._chkpt_ni = np.hstack((self._chkpt_ni, self._num_irreps))
                else:
                    raise ValueError('Missing input argument chkpt_ni for custom checkpointing.')
            else:
                t = np.floor(np.log2(self._num_irreps-self._num_irreps_old)) + 1
                xx = np.arange(1, t+1, 1)
                self._chkpt_ni = self._num_irreps_old + np.ceil( (self._num_irreps-self._num_irreps_old) * (1 - 1/2**xx ) ).astype(int)
                assert(self._chkpt_ni[-1]==self._num_irreps)
                assert(self._chkpt_ni[0]>self._num_irreps_old)
        
        self._filename = os.path.join(self._rme_folder, 
                                      self._get_filename(self._num_irreps))
        
        return
    
    
    @property
    def filename(self):
        return self._filename
    
    
    def run(self):
        """
        Compute the reduced matrix elements
        """
        
        if self._num_irreps==self._num_irreps_old:
            return
        if self._checkpointing==False:
            self._atomic_run()
        else:
            self._ultimate_num_irreps = self._num_irreps # not really necessary
            for i, ni in enumerate(self._chkpt_ni):
                print(':::::::::::::::::::')
                print('Checkpoint run ', i+1, '/', len(self._chkpt_ni), ' : num_irreps = ', ni)
                self._num_irreps = ni
                self._filename = self._get_filename(ni)
                self._irreps = self._init_irreps(ni)
                # peform calculation
                self._atomic_run()
                # use current checkpoint as restarting in next iteration
                self._restarting = True
                self._restarting_filename = self._filename
                self._num_irreps_old = ni
                self._irreps_old = self._init_irreps(ni)
        
        return
    
    
    @abstractmethod
    def _init_irreps(self, num_irreps):
        """
        
        """
        raise NotImplementedError()
    
    
    @abstractmethod
    def _atomic_run(self):
        """
        Compute the reduced matrix elements
        """
        raise NotImplementedError()
    
    
    def _get_filename(self, num_irreps):
        """
        
        """
        filename = f'{self._filename_prefix}_SU{self._N}_{self._target}_numirreps{num_irreps}_{self._tech}.pickle'
        return filename
    
    
    @abstractmethod
    def _get_sdc(self, alpha, alpha1, l1, alpha2, l2):
        """
        Compute SDCs for |alpha; alpha1, l1; alpha2, l2>
        """
        raise NotImplementedError()
    
    
    @abstractmethod
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
        raise NotImplementedError()
    
    
    @abstractmethod
    def _evaluate_compatibility(self, alpha1, alpha2):
        """
        Test if alpha1, alpha2 are valid irreps for the target sector
        """
        raise NotImplementedError()
    
    
    def save(self, filename):
        """
        Save reduced matrix elements to file
        """
        data = {}
        data['indliste'] = self._indliste
        data['indices_liste_rme'] = self._indices_liste_rme
        data['liste_rme'] = self._liste_rme
        data['tech'] = self._tech
        data['irreps'] = self._irreps
        with open(filename, 'wb') as file:
            print('Saving RME to file: ', filename)
            pickle.dump(data, file)
        return
