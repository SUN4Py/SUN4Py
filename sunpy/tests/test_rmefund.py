# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
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
    # arange
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
    