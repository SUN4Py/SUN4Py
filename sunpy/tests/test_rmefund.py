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

import pytest
import numpy as np
import pickle
import os

from sunpy.dmrg.rme import rmefund

prec = 1.0e-13



@pytest.mark.parametrize("N, num_irreps, tech, target", [
    #------------
    # SU(3)
    (int(3), int(12), 'base', 'GS'),
    (int(3), int(12), 'shortcut_cols', 'GS'),
    (int(3), int(16), 'shortcut_cols', 'GS'),
    #------------
    # SU(4)
    (int(4), int(6), 'shortcut_cols', 'GS'),
    (int(4), int(10), 'shortcut_cols', 'GS'),
    #------------
    # SU(5)
    #(int(5), int(12), 'shortcut_cols', 'GS')
])
def test_rme_fund(N, num_irreps, tech, target):
    # arrange
    temp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    testdata_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     'testdata', 
                                     f'testdata_rmefund_SU{N}_numirreps{num_irreps}_{target}_{tech}.pickle')
    with open(testdata_filename, 'rb') as file:
        testdata = pickle.load(file)
    
    # act
    rme_engine = rmefund.RMEEngineFund(N, 
                                       num_irreps, 
                                       target=target, 
                                       tech=tech, 
                                       rme_folder=temp_folder,
                                       restarting=False, 
                                       checkpointing=False)
    rme_engine.run()
    filename = rme_engine.filename
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    # assert
    assert(data['tech']==testdata['tech'])
    assert(np.sum(abs(data['irreps'] - testdata['irreps']))==0)
    assert(np.sum(abs(data['indliste'] - testdata['indliste']))==0)
    assert(len(data['liste_rme'])==len(testdata['liste_rme']))
    for i in range(0, len(data['liste_rme'])):
        assert(np.sum(abs(data['liste_rme'][i][0] - testdata['liste_rme'][i][0]))==0)
        assert(np.sum(abs(data['liste_rme'][i][1] - testdata['liste_rme'][i][1]))<prec)
    