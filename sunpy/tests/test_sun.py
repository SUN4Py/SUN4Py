#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import pytest
import numpy as np

from sunpy.sun import sun



prec = 1.0e-13



@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), np.array([[0]], dtype=int)),
    (np.array([2], dtype=int), np.array([[0, 0]], dtype=int)),
    (np.array([1,1], dtype=int), np.array([[0, 1]], dtype=int)),
    (np.array([2,1], dtype=int), np.array([[0, 0, 1], [0, 1, 0]], dtype=int))
])
def test_get_SYT(alpha, expected):
    assert(np.linalg.norm( sun.get_SYT(alpha) - expected )==0)



@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), int(1)),
    (np.array([2], dtype=int), int(1)),
    (np.array([1, 1], dtype=int), int(1)),
    (np.array([2, 1], dtype=int), int(2)),
    # See Nataf & Mila, PRL 113, 127204 (2014) for the 4 following dimensions
    (np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=int), int(1430)),
    (np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=int), int(16796)),
    (np.array([4, 4, 4, 4, 4], dtype=int), int(1662804)),
    (np.array([5, 5, 5, 5, 5], dtype=int), int(701149020))
])
def test_multiplicity(alpha, expected):
    assert(sun.multiplicity(alpha)==expected)



@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), 0),
    (np.array([2], dtype=int), 1),
    (np.array([1,1], dtype=int), -1),
    (np.array([3], dtype=int), 3),
    (np.array([2,1], dtype=int), 0),
    (np.array([1,1,1], dtype=int), -3),
    (np.array([4], dtype=int), 6),
    (np.array([3,1], dtype=int), 2),
    (np.array([2,2], dtype=int), 0),
    (np.array([2,1,1], dtype=int), -2),
    (np.array([1,1,1,1], dtype=int), -6),
    (np.array([5], dtype=int), 10),
    (np.array([4,1], dtype=int), 5),
    (np.array([3,2], dtype=int), 2),
    (np.array([3,1,1], dtype=int), 0),
    (np.array([2,2,1], dtype=int), -2),
    (np.array([2,1,1,1], dtype=int), -5),
    (np.array([1,1,1,1,1], dtype=int), -10),
])
def test_casimir(alpha, expected):
    assert abs( sun.casimir_quadratic(alpha) - expected )<prec



@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), True),
    (np.array([2], dtype=int), True),
    (np.array([1,1], dtype=int), True),
    (np.array([3], dtype=int), True),
    (np.array([2,1], dtype=int), True),
    (np.array([1,1,1], dtype=int), True),
    (np.array([3,3,3,3], dtype=int), True)
])
def test_binary_search_SYT(alpha, expected):
    actual = True
    # extracts all SYTs
    Y = sun.get_SYT(alpha=alpha, order='iLLOS')
    # test for each SYT
    for i in range(0, Y.shape[0]):
        ii = sun.binary_search_SYT(Y[i,:], Y)
        if not ii==i:
            actual = False
            break
    assert actual==expected



@pytest.mark.parametrize("alpha, m, order, expected", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - LLOS
    (np.array([1], dtype=int), int(1), 'LLOS', True),
    (np.array([2], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1], dtype=int), int(1), 'LLOS', True),
    (np.array([3], dtype=int), int(1), 'LLOS', True),
    (np.array([2,1], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'LLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - iLLOS
    (np.array([1], dtype=int), int(1), 'iLLOS', True),
    (np.array([2], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3], dtype=int), int(1), 'iLLOS', True),
    (np.array([2,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'iLLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - LLOS
    (np.array([2,2,2], dtype=int), int(2), 'LLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'LLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - iLLOS
    (np.array([2,2,2], dtype=int), int(2), 'iLLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'iLLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - LLOS
    (np.array([3,3,3], dtype=int), int(3), 'LLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'LLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - iLLOS
    (np.array([3,3,3], dtype=int), int(3), 'iLLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'iLLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'iLLOS', True),
])
def test_map_syt_to_index(alpha, m, order, expected):
    actual = True
    # extracts all SYTs
    if m==1:
        Y = sun.get_SYT(alpha=alpha, order=order)
    else:
        Y = sun.get_SYT_symm(alpha=alpha, m=m, order=order)
    # check for each SYT
    for i in range(0, Y.shape[0]):
        if m==1:
            ii = sun.SYT_to_index(Y[i,:], alpha, order)
        else:
            ii = sun.SYT_to_index_symm(Y[i,:], alpha, m, order)
    
        if not ii==i:
            actual = False
            break
    assert actual==expected



@pytest.mark.parametrize("alpha, m, order, expected", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - LLOS
    (np.array([1], dtype=int), int(1), 'LLOS', True),
    (np.array([2], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1], dtype=int), int(1), 'LLOS', True),
    (np.array([3], dtype=int), int(1), 'LLOS', True),
    (np.array([2,1], dtype=int), int(1), 'LLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'LLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - iLLOS
    (np.array([1], dtype=int), int(1), 'iLLOS', True),
    (np.array([2], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3], dtype=int), int(1), 'iLLOS', True),
    (np.array([2,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'iLLOS', True),
    # (np.array([3,3,3,3], dtype=int), int(1), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - LLOS
    (np.array([2,2,2], dtype=int), int(2), 'LLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'LLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - iLLOS
    (np.array([2,2,2], dtype=int), int(2), 'iLLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'iLLOS', True),
    # (np.array([6,6,2], dtype=int), int(2), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - LLOS
    (np.array([3,3,3], dtype=int), int(3), 'LLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'LLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - iLLOS
    (np.array([3,3,3], dtype=int), int(3), 'iLLOS', True),
    # (np.array([5,5,5], dtype=int), int(3), 'iLLOS', True),
    # (np.array([6,5,4], dtype=int), int(3), 'iLLOS', True),
])
def test_map_index_to_syt(alpha, m, order, expected):
    actual = True
    # extracts all SYTs
    if m==1:
        Y = sun.get_SYT(alpha, order=order)
    else:
        Y = sun.get_SYT_symm(alpha, m, order=order)
    
    # check for each SYT
    for i in range(0, Y.shape[0]):
        if m==1:
            yi = sun.index_to_SYT(i, alpha, order)
        else:
            yi = sun.index_to_SYT_symm(i, alpha, m, order)
        
        if np.linalg.norm(Y[i,:]-yi)>0:
            actual = False
            break
    
    assert actual==expected



@pytest.mark.parametrize("y, ytarget, expected", [
    (np.array([0, 0, 1, 2], dtype=int), np.array([0, 1, 0, 2], dtype=int), (np.array([1], dtype=int), np.array([1./2.]))),
    (np.array([0, 1, 2, 0], dtype=int), np.array([0, 0, 1, 2], dtype=int), (np.array([2, 1], dtype=int), np.array([-1./3., -1./2.])))
])
def test_SYT_to_target_1(y, ytarget, expected):
    # arange
    # ...
    # act
    sigma, rho = sun.SYT_to_target(y, ytarget)
    # assert
    assert(np.linalg.norm( sigma - expected[0] )==0)
    assert(np.linalg.norm( rho - expected[1] )<prec)



@pytest.mark.parametrize("y, ytarget", [
    (np.array([0, 1, 0, 2], dtype=int), np.array([0, 0, 1, 2], dtype=int)), 
    (np.array([0, 0, 1, 2], dtype=int), np.array([0, 1, 2, 0], dtype=int))
])
def test_SYT_to_target_2(y, ytarget):
    # arange
    # ...
    # act
    sigma, rho = sun.SYT_to_target(y, ytarget)
    sigmap, rhop = sun.SYT_to_target(ytarget, y)
    # assert
    assert(np.linalg.norm(sigma - np.flip(sigmap))==0)
    assert(np.linalg.norm(rho - np.flip(-rhop))<prec)
