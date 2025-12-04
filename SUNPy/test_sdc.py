# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import pytest
import numpy as np

import subduction

prec = 1.0e-13

@pytest.mark.parametrize("y, ytarget, expected", [
    (np.array([0, 1, 0, 2], dtype=int), np.array([0, 0, 1, 2], dtype=int), (np.array([1], dtype=int), np.array([1./2.], dtype=int))),
    (np.array([0, 0, 1, 2], dtype=int), np.array([0, 1, 2, 0], dtype=int), (np.array([2, 1], dtype=int), np.array([-1./3., -1./2.], dtype=int)))
])
def test_SYT_to_target_1(alpha, y, ytarget, expected):
    # arange
    # ...
    # act
    sigma, rho = subduction.SYT_to_target(y, ytarget)
    # assert
    assert(np.linalg.norm( sigma - expected[0] )==0)
    assert(np.linalg.norm( rho - expected[1] )<prec)


@pytest.mark.parametrize("y, ytarget", [
    (np.array([0, 1, 0, 2], dtype=int), np.array([0, 0, 1, 2], dtype=int)), 
    (np.array([0, 0, 1, 2], dtype=int), np.array([0, 1, 2, 0], dtype=int))
])
def test_SYT_to_target_2(alpha, y, ytarget, expected):
    # arange
    # ...
    # act
    sigma, rho = subduction.SYT_to_target(y, ytarget)
    sigmap, rhop = subduction.SYT_to_target(ytarget, y)
    # assert
    assert(np.linalg.norm(sigma - np.flip(sigmap))==0)
    assert(np.linalg.norm(rho - np.flip(-rho))<prec)
