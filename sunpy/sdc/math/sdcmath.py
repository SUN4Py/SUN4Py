# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np
import scipy
import scipy.sparse

from sunpy.sun import sun
from sunpy.sdc.common import sdcutils



def get_SDC(N, nu, nu1, nu2, Y2=None, ref2firstLLOS=True, ref1firstLLOS=True):
    """
    Compute the permutational Subduction Coefficients (SDCs)
        < [nu], m | [nu1], m1; [nu2], m2>  for all m.
    If m2 is/are not provided through its/their SYT(s) Y2, the SDCs are computed 
    for all m2's.
    
    See for instance Eq. (4-165b) of Ref. [1].
    
    Parameters
    ----------
    N : int
        SU(N)
    nu : numpy array
        global irrep with n particles
    nu1 : numpy array
        irrep with n1 particles
    nu2 : numpy array
        irrep with n2 particles, n=n1+n2
    Y2 : numpy array or None
        SYT or None - defines which SDCs to compute
    ref1firstLLOS : bool
        reference SYT used for y1
    ref2firstLLOS : bool
        if True, the first SYT in the ascending order of the LLOS is used as 
        reference for fixing the overall phase convention.  If False, the last 
        SYT is used (namely the first in the descending order of LLOS)
    
    Returns
    -------
    Y : numpy array
        SYTs for the irrep nu, with n1 first particles according to input ref1firstLLOS
    CY : numpy array
        associated column positions
    SDCs : numpy array
        SDCs is of dimension (NY2, NY, Ntau) where
            NY2  : number of SYTs in Y2 if Y2 is provided. Otherwise, total number
                   of SYTs for nu2
            NY   : number of SYTs for irrep nu with n1 first particles fixed 
                   according to nu1
            Ntau : multiplicity of nu in the tensor product of nu1 with nu2
    
    Remarks
    -------
    - Computes the permutational subduction coefficients
      < [nu], m | [nu1], m1; [nu2], m2> as well as the states |[nu], m> (SYTs) 
      of the expansion. See Eq. (4-165b) of Ref. [1].
      
    - The states |[nu]m> are given as SYTs (here = reversed Yamanouchi symbols).
    
    - The SDCs are independant of the state m1, thus there is no input y1. However,
      expansion basis depends on m1. Thus the possibility to select the first (default)
      SYT (ref1firstLLOS=True) or the last one (ref1firstLLOS=False) in the 
      ascending order of the LLOS
    
    - ref1firstLLOS has no influence on the SDCs. It does have an influence on
      the SYTs in the output Y.
    
    Example
    -------
    Example 1:
    N = int(3)
    nu = np.array([4, 1, 0], dtype=int)
    nu1 = np.array([1, 0, 0], dtype=int)
    nu2 = np.array([3, 1, 0], dtype=int)
    Y, CY, SDCs = get_SDC(N, nu, nu1, nu2)
    
    This reproduces the 2nd, 3rd and 4th lines of Table 4.18 2a page 187 of
    Ref. [1].
    
    Example 2:
    N = int(3)
    nu = np.array([4, 4, 3], dtype=int)
    nu1 = np.array([3, 3, 1], dtype=int)
    nu2 = np.array([2, 1, 1], dtype=int)
    Y, CY, SDCs = get_SDC(N, nu, nu1, nu2)
    
    This reproduces equations (G.8), (G.14) and (G.19) pages 163-164 of Ref. [3].
    
    References
    ----------
    [1]         Group Representation Theory for Physicists
                Jin-Quan Chen, Hialun Ping and Fan Wang
                World Scientific, 2nd edition, (2002)
    [2]         Transformation coefficients of permutation groups
                Jin-Quand Chen, David F. Collinson and Mei-Juan Gao
                J. Math. Phys. 24, 1695 (1983)
    [3]         Asymptotic freedom, Haldane gap and edge states of SU(N) spin chains
                Samuel Gozel
                Thesis EPFL #8417 (2020)
    """
    
    assert len(nu)<=N
    assert len(nu1)<=N
    assert len(nu2)<=N
    
    # ensure irreps have exactly N rows
    if len(nu)<N:
        nu = np.hstack((nu, np.zeros(N-len(nu), dtype=int)))
    if len(nu)<N:
        nu1 = np.hstack((nu1, np.zeros(N-len(nu1), dtype=int)))
    if len(nu2)<N:
        nu2 = np.hstack((nu2, np.zeros(N-len(nu2), dtype=int)))
    
    nu1nu2nu = sun.multiplicity_irrep_mixed(nu, np.array([nu1, nu2], dtype=int), N)
    assert(nu1nu2nu>0)
    
    n = np.sum(nu)
    n1 = np.sum(nu1)
    n2 = np.sum(nu2)
    assert n==n1+n2
    
    # construct SYT for nu1 (no impact on SDCs)
    if ref1firstLLOS==True:
        y1 = sun.index_to_SYT(0, alpha=nu1, order='LLOS')
    else:
        y1 = sun.index_to_SYT(0, alpha=nu1, order='iLLOS')
    
    # construct reference SYT for nu2
    if ref2firstLLOS==True:
        # the reference is the first SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='LLOS')
    else:
        # the reference is the last SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='iLLOS')        
    
    # Construct all SYTs associated to the shape nu-nu1, in increasing order of LLOS
    Ys = sun.get_subSYT(nu, nu1, order='LLOS')
    NY = Ys.shape[0]
    
    # Fill all obtained SYTs with numbers from 0 to n1-1
    Y = np.zeros(shape=(NY, n), dtype=int)
    Y[:, :n1] = np.matlib.repmat(y1, NY, 1)
    Y[:, n1:] = Ys
    CY = sun.get_column(Y)
    
    # Compute the matrices of the adjacent transpositions for S_{n_2}
    MatAdjaTranspo = []
    for k in range(n1, n-1):
        MatAdjaTranspo.append( sun.get_adjacent_transposition_matrix(nu, Y, CY, k) )
    
    # Compute Casimir (eigenvalue of CSCO-II) for the canonical chain
    # associated to S_{n_2}
    casimir = sdcutils.casimir_canonical_chain(nu2, y2_ref)
    
    # construct projection operator
    M = sdcutils.build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, casimir, ordering='math')
    
    # compute kernel of projection operator
    if NY==1:
        if (abs(M[0, 0])>1.0e-13):
            sys.exit('Problem: M is 1x1 matrix, but kernel is empty.')
        DD = np.array([0.0])
        COEFF_REF = np.array([[1.0]])
    elif NY<200:
        DD, COEFF_REF = np.linalg.eigh(M.todense())
        COEFF_REF = np.asarray(COEFF_REF)
    else:
        DD, COEFF_REF = scipy.sparse.linalg.eigsh(M, k=min(4, NY-1), which='SA')
    
    # keep only eigenvectors with eigenvalue==0
    ind = np.argwhere(abs(DD)<1.0e-12).flatten()
    assert len(ind)==nu1nu2nu
    COEFF_REF = COEFF_REF[:, ind]
    
    # Fix the overall phase
    COEFF_REF, IND_REF = sdcutils.set_overall_phase(COEFF_REF)
    
    # Format SYTs of nu2 for which SDCs are to be computed
    
    if Y2 is not None:
       if len(Y2.shape)==1:
           Y2 = np.reshape(Y2, (1, len(Y2)))
    else:
        if ref2firstLLOS==True:
            Y2 = sun.get_SYT(nu2, order='LLOS')
        else:
            Y2 = sun.get_SYT(nu2, order='iLLOS')

    NY2 = Y2.shape[0]

    SDCs = np.zeros(shape=(NY2, NY, nu1nu2nu), dtype=float)
    # indices for SDCs are (m2, m, tau)

    # Compute SDCs
    
    for i2 in range(0, NY2):
    
        y2 = Y2[i2]
        
        deal_with_phase = False
        
        if (np.sum(abs(y2-y2_ref))!=0):
            deal_with_phase = True
        
        if deal_with_phase==False:
            
            SDCs[i2, :, :] = COEFF_REF
            
        else:
            
            # find the sequence of operations which transfrom y2_ref into y2
            sigma, rho = sun.SYT_to_target(y2_ref, y2)
            # y2 = sigma[-1] ... sigma[0] y2_ref
            
            # shift sigma by n1 to go from local to global numbering
            sigma = n1 + sigma            
            
            # apply the sequence of transpositions
            
            for tau in range(0, nu1nu2nu):
                ydev1, cydev1, coeff1 = sun.apply_transpositions_v2(
                                                nu, 
                                                sigma, 
                                                rho, 
                                                np.copy(Y[IND_REF[tau]]), 
                                                np.copy(CY[IND_REF[tau]]), 
                                                np.copy(COEFF_REF[IND_REF[tau], tau].flatten()))
                
                # now the SYTs in ydev1 are not necessarily stored in ascending 
                # order of LLOS
                # sort SYTs according to Y (--> LLOS)
                SDCs[i2, :, tau] = sun.sort_development(Y, ydev1, coeff1)
    
    return Y, CY, SDCs

