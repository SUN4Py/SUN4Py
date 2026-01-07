# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np

from sunpy.sdc.dmrg import sdcdmrgbase
from sunpy.sdc.dmrg import sdcdmrg
from sunpy.sun import sun

prec = 1.0e-13


def compute_rme(N, alphaTarget, alpha1, l1, alpha2, l2, alpha3, l3, alpha4, l4, ref2firstLLOS, tech, symmetry):
    
    if tech=='base':
        ket_ydev, ket_cydev, ket_coeff = sdcdmrgbase.get_SDC(
                                            N, alphaTarget, 
                                            alpha1, l1, 
                                            alpha2, l2, 
                                            ref2firstLLOS=ref2firstLLOS, 
                                            ref1firstLLOS=True,
                                            symmetry=symmetry)
        bra_ydev, bra_cydev, bra_coeff = sdcdmrgbase.get_SDC(
                                            N, alphaTarget, 
                                            alpha3, l3, 
                                            alpha4, l4, 
                                            ref2firstLLOS=ref2firstLLOS, 
                                            ref1firstLLOS=True,
                                            symmetry=symmetry)
    elif tech=='shortcut':
        ket_ydev, ket_cydev, ket_coeff = sdcdmrg.get_SDC(
                                            N, alphaTarget, 
                                            alpha1, l1, 
                                            alpha2, l2, 
                                            ref2firstLLOS=ref2firstLLOS, 
                                            ref1firstLLOS=True,
                                            symmetry=symmetry)
        bra_ydev, bra_cydev, bra_coeff = sdcdmrg.get_SDC(
                                            N, alphaTarget, 
                                            alpha3, l3, 
                                            alpha4, l4, 
                                            ref2firstLLOS=ref2firstLLOS, 
                                            ref1firstLLOS=True,
                                            symmetry=symmetry)
    
    Pket_ydev, Pket_cydev, Pket_coeff = sun.develop_consecutive_number(
                                        alphaTarget, 
                                        np.copy(ket_ydev), 
                                        np.copy(ket_cydev), 
                                        np.copy(ket_coeff), 
                                        np.sum(alpha1)-1)
    # compute overlap <bra|Pket>
    rme = sun.overlap(bra_cydev, bra_coeff, Pket_cydev, Pket_coeff)
    factor = 1.0
    if symmetry in ['symmetric', 'antisymmetric']:
        factor = len(l1)**2
    rme *= factor
    
    return rme


def test_rme_0_baseA():
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([2, 1, 1], dtype=int)
    l1 =  int(2)
    alpha2 = np.array([2, 1, 1], dtype=int)
    l2 = int(2)
    # bra
    alpha3 = np.array([2, 2, 0], dtype=int)
    l3 = int(1)
    alpha4 = np.array([3, 1, 0], dtype=int)
    l4 = int(0)
    # target
    alphaTarget = np.array([3, 3, 2], dtype=int)
    
    expected = 0.4192627457812104
    
    #-----------------
    # act
    #-----------------
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      False, 'base', '')
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_0_cols():
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([2, 1, 1], dtype=int)
    l1 =  int(2)
    alpha2 = np.array([2, 1, 1], dtype=int)
    l2 = int(2)
    # bra
    alpha3 = np.array([2, 2, 0], dtype=int)
    l3 = int(1)
    alpha4 = np.array([3, 1, 0], dtype=int)
    l4 = int(0)
    # target
    alphaTarget = np.array([3, 3, 2], dtype=int)
    
    expected = 0.4192627457812104
    
    #-----------------
    # act
    #-----------------
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      False, 'shortcut', '')
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_0_baseB():
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([2, 1, 1], dtype=int)
    l1 =  int(2)
    alpha2 = np.array([2, 1, 1], dtype=int)
    l2 = int(2)
    # bra
    alpha3 = np.array([2, 2, 0], dtype=int)
    l3 = int(1)
    alpha4 = np.array([3, 1, 0], dtype=int)
    l4 = int(0)
    # target
    alphaTarget = np.array([3, 3, 2], dtype=int)
    
    expected = -0.4192627457812104
    
    #-----------------
    # act
    #-----------------
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      True, 'base', '')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_0_rows():
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([2, 1, 1], dtype=int)
    l1 =  int(2)
    alpha2 = np.array([2, 1, 1], dtype=int)
    l2 = int(2)
    # bra
    alpha3 = np.array([2, 2, 0], dtype=int)
    l3 = int(1)
    alpha4 = np.array([3, 1, 0], dtype=int)
    l4 = int(0)
    # target
    alphaTarget = np.array([3, 3, 2], dtype=int)
    
    expected = -0.4192627457812104
    
    #-----------------
    # act
    #-----------------
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      True, 'shortcut', '')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_1_baseB():
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([3, 2, 2], dtype=int)
    l1 = np.array([0, 2], dtype=int)
    alpha2 = np.array([3, 2, 2], dtype=int)
    l2 = np.array([0, 2], dtype=int)
    # bra
    alpha3 = np.array([4, 2, 1], dtype=int)
    l3 = np.array([0, 0], dtype=int)
    alpha4 = np.array([4, 2, 1], dtype=int)
    l4 = np.array([0, 0], dtype=int)
    # target
    alphaTarget = np.array([5, 5, 4], dtype=int)
    
    expected = 0.9682458365518546
    
    #-----------------
    # act
    #-----------------
    # Compute using basic method
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      True, 'base', 'symmetric')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_1_rows():
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([3, 2, 2], dtype=int)
    l1 = np.array([0, 2], dtype=int)
    alpha2 = np.array([3, 2, 2], dtype=int)
    l2 = np.array([0, 2], dtype=int)
    # bra
    alpha3 = np.array([4, 2, 1], dtype=int)
    l3 = np.array([0, 0], dtype=int)
    alpha4 = np.array([4, 2, 1], dtype=int)
    l4 = np.array([0, 0], dtype=int)
    # target
    alphaTarget = np.array([5, 5, 4], dtype=int)
    
    expected = 0.9682458365518546
    
    #-----------------
    # act
    #-----------------
    # Compute using shortcut rows method
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      True, 'shortcut', 'symmetric')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_2_baseA():
    """
    See Supplemental Material of Phys. Rev. Lett. 125, 057202 (2020), Gozel et al.
    Eq.(S35)
    """
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([4, 2, 0], dtype=int)
    l1 = np.array([0, 0, 1], dtype=int)
    alpha2 = np.array([4, 2, 0], dtype=int)
    l2 = np.array([0, 0, 1], dtype=int)
    # bra
    alpha3 = np.array([3, 2, 1], dtype=int)
    l3 = np.array([0, 1, 2], dtype=int)
    alpha4 = np.array([3, 2, 1], dtype=int)
    l4 = np.array([0, 1, 2], dtype=int)    
    # target
    alphaTarget = np.array([4, 4, 4], dtype=int)
    
    expected = 2*np.sqrt(6)/5
    
    #-----------------
    # act
    #-----------------
    # Compute using basic method
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      False, 'base', 'symmetric')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_2_baseB():
    """
    See Supplemental Material of Phys. Rev. Lett. 125, 057202 (2020), Gozel et al.
    Eq.(S35)
    """
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([4, 2, 0], dtype=int)
    l1 = np.array([0, 0, 1], dtype=int)
    alpha2 = np.array([4, 2, 0], dtype=int)
    l2 = np.array([0, 0, 1], dtype=int)
    # bra
    alpha3 = np.array([3, 2, 1], dtype=int)
    l3 = np.array([0, 1, 2], dtype=int)
    alpha4 = np.array([3, 2, 1], dtype=int)
    l4 = np.array([0, 1, 2], dtype=int)    
    # target
    alphaTarget = np.array([4, 4, 4], dtype=int)
    
    expected = 2*np.sqrt(6)/5
    
    #-----------------
    # act
    #-----------------
    # Compute using basic method
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      True, 'base', 'symmetric')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_2_cols():
    """
    See Supplemental Material of Phys. Rev. Lett. 125, 057202 (2020), Gozel et al.
    Eq.(S35)
    """
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([4, 2, 0], dtype=int)
    l1 = np.array([0, 0, 1], dtype=int)
    alpha2 = np.array([4, 2, 0], dtype=int)
    l2 = np.array([0, 0, 1], dtype=int)
    # bra
    alpha3 = np.array([3, 2, 1], dtype=int)
    l3 = np.array([0, 1, 2], dtype=int)
    alpha4 = np.array([3, 2, 1], dtype=int)
    l4 = np.array([0, 1, 2], dtype=int)    
    # target
    alphaTarget = np.array([4, 4, 4], dtype=int)
    
    expected = 2*np.sqrt(6)/5
    
    #-----------------
    # act
    #-----------------
    # Compute using shortcut cols method
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      False, 'shortcut', 'symmetric')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)


def test_rme_2_rows():
    """
    See Supplemental Material of Phys. Rev. Lett. 125, 057202 (2020), Gozel et al.
    Eq.(S35)
    """
    
    #-----------------
    # arange
    #-----------------
    N = int(3)
    # ket
    alpha1 = np.array([4, 2, 0], dtype=int)
    l1 = np.array([0, 0, 1], dtype=int)
    alpha2 = np.array([4, 2, 0], dtype=int)
    l2 = np.array([0, 0, 1], dtype=int)
    # bra
    alpha3 = np.array([3, 2, 1], dtype=int)
    l3 = np.array([0, 1, 2], dtype=int)
    alpha4 = np.array([3, 2, 1], dtype=int)
    l4 = np.array([0, 1, 2], dtype=int)    
    # target
    alphaTarget = np.array([4, 4, 4], dtype=int)
    
    expected = 2*np.sqrt(6)/5
    
    #-----------------
    # act
    #-----------------
    # Compute using shortcut rows method
    rme = compute_rme(N, alphaTarget, 
                      alpha1, l1, 
                      alpha2, l2, 
                      alpha3, l3, 
                      alpha4, l4, 
                      True, 'shortcut', 'symmetric')
    
    #-----------------
    # assert
    #-----------------
    assert(abs(rme-expected)<prec)

