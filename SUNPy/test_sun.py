#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:31:51 2023

@author: sgozel
"""
# Copyright 2023 Samuel GOZEL, GNU GPLv3

import pytest

import sys
import numpy as np
import scipy.sparse

import sun
import Lattice



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
    (np.array([3,3,3,3], dtype=int), int(1), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - iLLOS
    (np.array([1], dtype=int), int(1), 'iLLOS', True),
    (np.array([2], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3], dtype=int), int(1), 'iLLOS', True),
    (np.array([2,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3,3,3,3], dtype=int), int(1), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - LLOS
    (np.array([2,2,2], dtype=int), int(2), 'LLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'LLOS', True),
    (np.array([6,6,2], dtype=int), int(2), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - iLLOS
    (np.array([2,2,2], dtype=int), int(2), 'iLLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'iLLOS', True),
    (np.array([6,6,2], dtype=int), int(2), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - LLOS
    (np.array([3,3,3], dtype=int), int(3), 'LLOS', True),
    (np.array([5,5,5], dtype=int), int(3), 'LLOS', True),
    (np.array([6,5,4], dtype=int), int(3), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - iLLOS
    (np.array([3,3,3], dtype=int), int(3), 'iLLOS', True),
    (np.array([5,5,5], dtype=int), int(3), 'iLLOS', True),
    (np.array([6,5,4], dtype=int), int(3), 'iLLOS', True),
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
    (np.array([3,3,3,3], dtype=int), int(1), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=1 - iLLOS
    (np.array([1], dtype=int), int(1), 'iLLOS', True),
    (np.array([2], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3], dtype=int), int(1), 'iLLOS', True),
    (np.array([2,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([1,1,1], dtype=int), int(1), 'iLLOS', True),
    (np.array([3,3,3,3], dtype=int), int(1), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - LLOS
    (np.array([2,2,2], dtype=int), int(2), 'LLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'LLOS', True),
    (np.array([6,6,2], dtype=int), int(2), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=2 - iLLOS
    (np.array([2,2,2], dtype=int), int(2), 'iLLOS', True),
    (np.array([4,4,4], dtype=int), int(2), 'iLLOS', True),
    (np.array([6,6,2], dtype=int), int(2), 'iLLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - LLOS
    (np.array([3,3,3], dtype=int), int(3), 'LLOS', True),
    (np.array([5,5,5], dtype=int), int(3), 'LLOS', True),
    (np.array([6,5,4], dtype=int), int(3), 'LLOS', True),
    #:::::::::::::::::::::::::::::::::::::::::::
    # m=3 - iLLOS
    (np.array([3,3,3], dtype=int), int(3), 'iLLOS', True),
    (np.array([5,5,5], dtype=int), int(3), 'iLLOS', True),
    (np.array([6,5,4], dtype=int), int(3), 'iLLOS', True),
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
    (int(3), np.array([3,3,3], dtype=int), int(2)),
    (int(3), np.array([3,3,3], dtype=int), int(3)),
    (int(3), np.array([3,3,3], dtype=int), int(4)),
    (int(3), np.array([3,3,3], dtype=int), int(5)),
    (int(3), np.array([3,3,3], dtype=int), int(6)),
    (int(3), np.array([3,3,3], dtype=int), int(7)),
    (int(3), np.array([3,3,3], dtype=int), int(8)),
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(4)
    #:::::::::::::::::::::::::::::::::::::::::::
    (int(4), np.array([4,4,4,4], dtype=int), int(2)),
    (int(4), np.array([4,4,4,4], dtype=int), int(3)),
    (int(4), np.array([4,4,4,4], dtype=int), int(4)),
    (int(4), np.array([4,4,4,4], dtype=int), int(5)),
    (int(4), np.array([4,4,4,4], dtype=int), int(6)),
    (int(4), np.array([4,4,4,4], dtype=int), int(7)),
    (int(4), np.array([4,4,4,4], dtype=int), int(8)),
    (int(4), np.array([4,4,4,4], dtype=int), int(9)),
    (int(4), np.array([4,4,4,4], dtype=int), int(10)),
    (int(4), np.array([4,4,4,4], dtype=int), int(11)),
    (int(4), np.array([4,4,4,4], dtype=int), int(12)),
    (int(4), np.array([4,4,4,4], dtype=int), int(13)),
    (int(4), np.array([4,4,4,4], dtype=int), int(14)),
    (int(4), np.array([4,4,4,4], dtype=int), int(15)),
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
    (int(3), np.array([3,3,3], dtype=int), int(2)),
    (int(3), np.array([3,3,3], dtype=int), int(3)),
    (int(3), np.array([3,3,3], dtype=int), int(4)),
    (int(3), np.array([3,3,3], dtype=int), int(5)),
    (int(3), np.array([3,3,3], dtype=int), int(6)),
    (int(3), np.array([3,3,3], dtype=int), int(7)),
    (int(3), np.array([3,3,3], dtype=int), int(8)),
    #:::::::::::::::::::::::::::::::::::::::::::
    # SU(4)
    #:::::::::::::::::::::::::::::::::::::::::::
    (int(4), np.array([4,4,4,4], dtype=int), int(6)),
    (int(4), np.array([4,4,4,4], dtype=int), int(7)),
    (int(4), np.array([4,4,4,4], dtype=int), int(8)),
    (int(4), np.array([4,4,4,4], dtype=int), int(9)),
    (int(4), np.array([4,4,4,4], dtype=int), int(10)),
    (int(4), np.array([4,4,4,4], dtype=int), int(11)),
    (int(4), np.array([4,4,4,4], dtype=int), int(12)),
    (int(4), np.array([4,4,4,4], dtype=int), int(13)),
    (int(4), np.array([4,4,4,4], dtype=int), int(14)),
    (int(4), np.array([4,4,4,4], dtype=int), int(15)),
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



@pytest.mark.parametrize("alpha, m, symmetry, N, isPBC, expected", [
    #:::::::::::::::::::::::::::::::::::::::::::
    # FUNDAMENTAL
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=2, m=1
    (np.array([2,0], dtype=int), int(1), 'fundamental', int(2), False, 1.0),
    (np.array([3,0], dtype=int), int(1), 'fundamental', int(2), False, 2.0),
    (np.array([4,0], dtype=int), int(1), 'fundamental', int(2), False, 3.0),
    (np.array([2,1], dtype=int), int(1), 'fundamental', int(2), False, -1.0),
    (np.array([2,2], dtype=int), int(1), 'fundamental', int(2), False, -1.732050807568877),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - OBC - Singlet
    (np.array([3,3,3], dtype=int), int(1), 'fundamental', int(3), False, -6.162794759771583),
    (np.array([4,4,4], dtype=int), int(1), 'fundamental', int(3), False, -8.263188004910717),
    (np.array([5,5,5], dtype=int), int(1), 'fundamental', int(3), False, -10.366907518918115),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - PBC - Singlet
    (np.array([3,3,3], dtype=int), int(1), 'fundamental', int(3), True, -6.579435759086468),
    (np.array([4,4,4], dtype=int), int(1), 'fundamental', int(3), True, -8.624526804645157),
    (np.array([5,5,5], dtype=int), int(1), 'fundamental', int(3), True, -10.696221973842384),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - OBC - Adjoint
    (np.array([4,3,2], dtype=int), int(1), 'fundamental', int(3), False, -5.713816495016518),
    (np.array([5,4,3], dtype=int), int(1), 'fundamental', int(3), False, -7.908329500628006),
    # (np.array([6,5,4], dtype=int), int(1), 'fundamental', int(3), False, -10.072858996269495), # remove this test to save time
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=1 - PBC - Adjoint
    (np.array([4,3,2], dtype=int), int(1), 'fundamental', int(3), True, -5.647199911888293),
    (np.array([5,4,3], dtype=int), int(1), 'fundamental', int(3), True, -7.924221842740132),
    # (np.array([6,5,4], dtype=int), int(1), 'fundamental', int(3), True, -10.134975270128132), # remove this test to save time
    #:::::::::::::::::::::::::::::::::::::::::::
    # SYMMETRIC
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=2, m=2
    (np.array([2,2], dtype=int), int(2), 'symmetric', int(2), False, -2.0),
    (np.array([3,1], dtype=int), int(2), 'symmetric', int(2), False, 0.0),
    (np.array([4,0], dtype=int), int(2), 'symmetric', int(2), False, 4.0),
    (np.array([3,3], dtype=int), int(2), 'symmetric', int(2), False, 0.0),
    (np.array([4,2], dtype=int), int(2), 'symmetric', int(2), False, -2.0),
    (np.array([5,1], dtype=int), int(2), 'symmetric', int(2), False, 2.0),
    (np.array([6,0], dtype=int), int(2), 'symmetric', int(2), False, 8.0),
    (np.array([4,4], dtype=int), int(2), 'symmetric', int(2), False, -3.291502622129181),
    (np.array([5,3], dtype=int), int(2), 'symmetric', int(2), False, -2.273163042697025),
    (np.array([6,2], dtype=int), int(2), 'symmetric', int(2), False, 0.417424305044159),
    (np.array([7,1], dtype=int), int(2), 'symmetric', int(2), False, 5.171572875253810),
    (np.array([8,0], dtype=int), int(2), 'symmetric', int(2), False, 12.0),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=2
    (np.array([2,2,2], dtype=int), int(2), 'symmetric', int(3), False, -4.0),
    (np.array([3,2,1], dtype=int), int(2), 'symmetric', int(3), False, -3.0),
    (np.array([4,1,1], dtype=int), int(2), 'symmetric', int(3), False, 0.0),
    (np.array([4,2,0], dtype=int), int(2), 'symmetric', int(3), False, -2.0),
    (np.array([5,1,0], dtype=int), int(2), 'symmetric', int(3), False, 2.0),
    (np.array([6,0,0], dtype=int), int(2), 'symmetric', int(3), False, 8.0),
    (np.array([3,3,2], dtype=int), int(2), 'symmetric', int(3), False, -3.414213562373094),
    (np.array([4,3,1], dtype=int), int(2), 'symmetric', int(3), False, -4.130648586880582),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=2, m=3
    (np.array([3,3], dtype=int), int(3), 'symmetric', int(2), False, -3.0),
    (np.array([4,2], dtype=int), int(3), 'symmetric', int(2), False, -1.0),
    (np.array([5,1], dtype=int), int(3), 'symmetric', int(2), False, 3.0),
    (np.array([6,0], dtype=int), int(3), 'symmetric', int(2), False, 9.0),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=3
    (np.array([3,3,3], dtype=int), int(3), 'symmetric', int(3), False, -6.0),
    (np.array([4,3,2], dtype=int), int(3), 'symmetric', int(3), False, -5.0),
    (np.array([5,3,1], dtype=int), int(3), 'symmetric', int(3), False, -4.0),
    (np.array([4,4,4], dtype=int), int(3), 'symmetric', int(3), False, -3.0),
    (np.array([5,4,3], dtype=int), int(3), 'symmetric', int(3), False, -5.758895915294917),
    (np.array([5,5,5], dtype=int), int(3), 'symmetric', int(3), False, -6.567134109811708),
    (np.array([6,5,4], dtype=int), int(3), 'symmetric', int(3), False, -8.683291472392161),
    #:::::::::::::::::::::::::::::::::::::::::::
    # ANTISYMMETRIC
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=3, m=2
    (np.array([2,1,1], dtype=int), int(2), 'antisymmetric', int(3), False, 0.0),
    (np.array([2,2,0], dtype=int), int(2), 'antisymmetric', int(3), False, 2.0),
    (np.array([2,2,2], dtype=int), int(2), 'antisymmetric', int(3), False, 0.0),
    (np.array([3,2,1], dtype=int), int(2), 'antisymmetric', int(3), False, 1.0),
    (np.array([3,3,0], dtype=int), int(2), 'antisymmetric', int(3), False, 4.0),
    (np.array([3,3,2], dtype=int), int(2), 'antisymmetric', int(3), False, 0.585786437626905),
    (np.array([4,3,1], dtype=int), int(2), 'antisymmetric', int(3), False, 2.585786437626907),
    (np.array([4,4,0], dtype=int), int(2), 'antisymmetric', int(3), False, 6.0),
    (np.array([4,4,2], dtype=int), int(2), 'antisymmetric', int(3), False, 1.763932022500210),
    (np.array([5,4,1], dtype=int), int(2), 'antisymmetric', int(3), False, 4.381966011250105),
    #:::::::::::::::::::::::::::::::::::::::::::
    # N=4, m=2
    (np.array([3,3,3,3], dtype=int), int(2), 'antisymmetric', int(4), False, -10.598765818237256),
    (np.array([4,3,3,2], dtype=int), int(2), 'antisymmetric', int(4), False, -7.895930963124478),
])
def test_energy(alpha, m, symmetry, N, isPBC, expected):
    if m==1:
        Ns = np.sum(alpha)
        lattice = Lattice.Lattice(Ns=Ns, typeLattice='chain', isPBC=isPBC)
        Engine = sun.SUNFundamental(Ns, N, alpha, lattice)
    else:
        Ns = np.sum(alpha)//m
        lattice = Lattice.Lattice(Ns=Ns, typeLattice='chain', isPBC=isPBC)
        if symmetry=='symmetric':
            Engine = sun.SUNSymmetric(Ns, N, m, alpha, lattice)
        elif symmetry=='antisymmetric':
            Engine = sun.SUNAntiSymmetric(Ns, N, m, alpha, lattice)
        else:
            sys.exit('symmetry undefined.')
    
    H = Engine.sun_hamiltonian()
    
    if H.shape[0]<100:
        H = H.todense()
        E, _ = np.linalg.eigh(H)
        E = E[0]
    else:    
        E, _ = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
        E = E[0]
    assert abs(E-expected)<prec
