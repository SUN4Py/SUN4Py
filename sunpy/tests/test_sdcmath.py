# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

#import pytest
import numpy as np

from sunpy.sdc.math import sdcmath

prec = 1.0e-13



def test_SDC():
    """
    See Eq. (G.8), (G.14), (G.19) of:
        Asymptotic freedom, Haldane gap and edge states of SU(N) spin chains
        Samuel Gozel
        EPFL, 2020
    """
    
    # arange
    N = int(3)
    nu = np.array([4, 4, 3], dtype=int)
    nu1 = np.array([3, 3, 1], dtype=int)
    nu2 = np.array([2, 1, 1], dtype=int)
    
    expected_coeff0 = np.array([0.0, np.sqrt(15)/8, 5./8., -np.sqrt(5)/8, 
                                -5./(8*np.sqrt(3)), 1./np.sqrt(6)])
    expected_coeff1 = np.array([np.sqrt(5/2)/3, 5*np.sqrt(5)/24, -5./(8*np.sqrt(3)),
                                -np.sqrt(15)/8, 3.0/8, 0.0])
    expected_coeff2 = np.array([np.sqrt(5)/3, -np.sqrt(5/2)/3, 1./np.sqrt(6),
                                0.0, 0.0, 0.0])
    # act
    Y, CY, coeff = sdcmath.get_SDC(N, nu, nu1, nu2, ref2firstLLOS=True, ref1firstLLOS=True)
    
    # assert
    assert(np.linalg.norm(coeff[0, :, 0] - expected_coeff0)<prec)
    assert(np.linalg.norm(coeff[1, :, 0] - expected_coeff1)<prec)
    assert(np.linalg.norm(coeff[2, :, 0] - expected_coeff2)<prec)
    