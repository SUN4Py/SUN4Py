#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:31:51 2023

@author: sgozel
"""
# Copyright 2023 Samuel GOZEL, GNU GPLv3

import pytest
import numpy as np

import sun



prec = 1.0e-13



@pytest.mark.parametrize("alpha, expected", [
    (np.array([1], dtype=int), np.array([[0]], dtype=int)),
    (np.array([2], dtype=int), np.array([[0, 0]], dtype=int)),
    (np.array([1,1], dtype=int), np.array([[0, 1]], dtype=int)),
    (np.array([2,1], dtype=int), np.array([[0, 0, 1], [0, 1, 0]], dtype=int))
])
def test_get_SYT(alpha, expected):
    assert np.linalg.norm( sun.get_SYT(alpha) - expected )==0



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



@pytest.mark.parametrize("N, alpha, nlookupboxes", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(3)
    #:::::::::::::::::::::::::::::::::::::::::::
    (int(3), np.array([3,2,2], dtype=int), int(2)),
    (int(3), np.array([3,2,2], dtype=int), int(3)),
    (int(3), np.array([3,2,2], dtype=int), int(4)),
    (int(3), np.array([3,2,2], dtype=int), int(5)),
    (int(3), np.array([3,2,2], dtype=int), int(6)),
    #------------------------
    # (int(3), np.array([3,3,3], dtype=int), int(2)),
    # (int(3), np.array([3,3,3], dtype=int), int(3)),
    (int(3), np.array([3,3,3], dtype=int), int(4)),
    (int(3), np.array([3,3,3], dtype=int), int(5)),
    (int(3), np.array([3,3,3], dtype=int), int(6)),
    # (int(3), np.array([3,3,3], dtype=int), int(7)),
    # (int(3), np.array([3,3,3], dtype=int), int(8)),
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(4)
    #:::::::::::::::::::::::::::::::::::::::::::
    # (int(4), np.array([4,4,4,4], dtype=int), int(2)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(3)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(4)),
    (int(4), np.array([4,4,4,4], dtype=int), int(5)),
    (int(4), np.array([4,4,4,4], dtype=int), int(6)),
    (int(4), np.array([4,4,4,4], dtype=int), int(7)),
    (int(4), np.array([4,4,4,4], dtype=int), int(8)),
    (int(4), np.array([4,4,4,4], dtype=int), int(9)),
    (int(4), np.array([4,4,4,4], dtype=int), int(10)),
    (int(4), np.array([4,4,4,4], dtype=int), int(11)),
    (int(4), np.array([4,4,4,4], dtype=int), int(12)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(13)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(14)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(15)),
])
def test_partial_lookup_get_SYT(N, alpha, nlookupboxes):
    expected = True
    
    # Generate lookup tool and initialize it
    lkptool = sun.PartialLookupTool(N, alpha, nlookupboxes)
    lkptool.init_lookup()
    
    # Generate all SYTs in iLLOS
    Y = sun.get_SYT(alpha, order='iLLOS')
    
    # Test
    actual = True
    for i in range(0, Y.shape[0]):
        yi = Y[i]
        y = lkptool.get_SYT(i)
        if not np.sum(abs(y-yi))==0:
            actual = False
            break
    
    assert actual==expected



@pytest.mark.parametrize("N, alpha, nlookupboxes", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(3)
    #:::::::::::::::::::::::::::::::::::::::::::
    (int(3), np.array([3,2,2], dtype=int), int(2)),
    (int(3), np.array([3,2,2], dtype=int), int(3)),
    (int(3), np.array([3,2,2], dtype=int), int(4)),
    (int(3), np.array([3,2,2], dtype=int), int(5)),
    (int(3), np.array([3,2,2], dtype=int), int(6)),
    #------------------------
    # (int(3), np.array([3,3,3], dtype=int), int(2)),
    # (int(3), np.array([3,3,3], dtype=int), int(3)),
    (int(3), np.array([3,3,3], dtype=int), int(4)),
    (int(3), np.array([3,3,3], dtype=int), int(5)),
    (int(3), np.array([3,3,3], dtype=int), int(6)),
    (int(3), np.array([3,3,3], dtype=int), int(7)),
    (int(3), np.array([3,3,3], dtype=int), int(8)),
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(4)
    #:::::::::::::::::::::::::::::::::::::::::::
    # (int(4), np.array([4,4,4,4], dtype=int), int(6)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(7)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(8)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(9)),
    (int(4), np.array([4,4,4,4], dtype=int), int(10)),
    (int(4), np.array([4,4,4,4], dtype=int), int(11)),
    (int(4), np.array([4,4,4,4], dtype=int), int(12)),
    (int(4), np.array([4,4,4,4], dtype=int), int(13)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(14)),
    # (int(4), np.array([4,4,4,4], dtype=int), int(15)),
])
def test_partial_lookup_get_index(N, alpha, nlookupboxes):
    expected = True
    
    # Generate lookup tool and initialize it
    lkptool = sun.PartialLookupTool(N, alpha, nlookupboxes)
    lkptool.init_lookup()
    
    # Generate all SYTs in iLLOS
    Y = sun.get_SYT(alpha, order='iLLOS')
    
    # Test
    actual = True
    for i in range(0, Y.shape[0]):
        ii = lkptool.get_index(Y[i])
        if not ii==i:
            actual = False
            break
    
    assert actual==expected


