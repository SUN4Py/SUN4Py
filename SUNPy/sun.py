#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:10:43 2023

@author: sgozel
"""
# Copyright 2023 Samuel GOZEL, GNU GPLv3

import sys
import math
import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg
import itertools
import copy

import partitions



def get_column(Y):
    # extract column positions for all SYTs in Y.
    # Y: array of SYTs, each row is a SYT
    # 
    
    if len(Y.shape)==1:
        NY = 1
        n = Y.shape[0]
        Y = np.reshape(Y, (NY, n))
    else:    
        NY = Y.shape[0]
        n = Y.shape[1]
    
    CY = np.zeros((NY, n), dtype=int)
    for k in range(1, n):
        CY[:,k] = np.sum( ( np.ones((k+1, 1), dtype=int)*Y[:,k] == np.transpose(Y[:,0:k+1]) ), axis=0 ) - 1
    
    return CY



def transpose_shape(alpha):
    # Transpose the irrep alpha (each row becomes a column).
    # 
    
    alphaT = np.zeros(alpha[0], dtype=int)
    for i in range(0, len(alpha)):
        for j in range(0, alpha[i]):
            alphaT[j] += 1
    return alphaT



def casimir_quadratic(alpha):
    # Compute the permutational quadratic Casimir of an irrep.
    # 
    # Refs:
    # - Eq. (31) of PRB 97, 134420 (2018)
    # - Eq. (4-25) [for a different formulation] of Group Representation Theory
    #   for Physicists, Jian-Quan Chen, Jialun Ping and Fan Wang 
    # - K. Pilch and A. N. Schellekens, J. of Mathematical Physics 25, 3455 (1984)
    # 
    
    k = len(np.argwhere(alpha>0).flatten())
    s = transpose_shape(alpha[0:k])
    
    B = np.sum(alpha**2)
    A = np.sum(s**2)
    
    c = 0.5 * (B - A)
    
    # other formula
    lvec = np.arange(1, k+1)
    cprime = 0.5 * ( np.sum(alpha[0:k]*(alpha[0:k] - 2*lvec + 1)) )
    
    if not cprime==c:
        sys.exit('Problem computing Casimir')
    
    return c



def multiplicity(alpha):
    # Compute the total number of SYTs associated to the shape alpha.
    # 
    
    n = np.sum(alpha)
    nl = len(np.argwhere(alpha>0))
    m = alpha[0]
    alphafull = np.zeros((nl, m), dtype=int)
    for i in range(0, nl):
        for j in range(0, alpha[i]):
            alphafull[i,j] = 1
    
    arnum = np.arange(1, n+1) # sequence for n!
    
    denom = int(1)
    for i in range(0, nl):
        for j in range(0, alpha[i]):
            sumi = np.sum(alphafull[i:nl,j]) + np.sum(alphafull[i,j+1:m])
            if not sumi==0:
                ind = np.argwhere(arnum==sumi).flatten()
                if len(ind)==1:
                    arnum[ind[0]] = 1
                else:
                    denom *= sumi
    
    # we further reduce the numerator and the denominator by finding common divisors
    for i in range(0, len(arnum)):
        k = arnum[i]
        if denom%k==0:
            denom = denom//k
            arnum[i] = 1
    
    # we further reduce by finding common divisors among prime numbers
    divisors = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=int) # 11 first prime numbers
    for i in range(0, len(arnum)):
        for t in range(0, 4): # 4 repetitions of the loop below
            for d in divisors:
                if ((denom%d==0) and (arnum[i]%d==0)):
                    denom = denom//d
                    arnum[i] = arnum[i]//d
    
    num = np.prod(arnum)
    
    falpha = num//denom
    
    if (num/denom-falpha)>1.0e-14:
        sys.exit('Problem when computing falpha. Probable overflow.')
    
    return falpha



def get_SYT(alpha, order='LLOS'):
    # Get all SYTs for the irrep alpha.
    # 
    # By default, in the resulting array, the SYTs are sorted in the ascending
    # order of the Last Letter Order Sequence, namely:
    #   Y[0,:] < Y[1,:] < Y[2,:] < ... < Y[-1,:]
    # 
    # This can be changed by setting order='iLLOS' in the input arguments.
    # 
    # Remarks: 
    # - the output is an array where SYTs are in the rows
    # - Y[i,0] = 0, namely the top left box is row=0, coluumn=0 (Python convention)
    # 
    
    n = np.sum(alpha)
    falpha = multiplicity(alpha)
    
    Y = np.zeros((falpha, n), dtype=int)
    nl = len(np.argwhere(alpha>0).flatten())
    alpha = alpha[0:nl]
    alphaT = transpose_shape(alpha)
    
    # construct first SYT
    y1 = np.zeros((n,), dtype=int)
    cptel = 0
    for j in range(0, len(alphaT)):
        for row in range(0, alphaT[j]):
            y1[cptel] = row
            cptel += 1
    
    Y[0,:] = y1
    
    # construct all remaining SYT's sequentially
    s = int(1)
    while s<falpha:
        
        y = np.copy(Y[s-1,:])
        
        lbd = np.zeros((nl+1,), dtype=int)
        lbd[0] = 1
        
        j = int(1)
        while (j<n) and (y[j]>=y[j-1]):
            lbd[y[j]] += 1
            j += 1
        lbd[y[j]] += 1
        
        t = lbd[y[j]+1]
        i = nl
        while not lbd[i-1]==t:
            i -= 1
        y[j] = i-1
        lbd[i-1] -= 1
        
        t = j
        l = int(1)
        while l<=t:
            r = int(1)
            while lbd[r-1]>0:
                y[l-1] = r-1
                lbd[r-1] = lbd[r-1] - 1
                l += 1
                r += 1
        
        Y[s, :] = y
        s += 1
    
    if order=='LLOS':
        Y = np.flipud(Y)
    
    if not Y.shape[0]==multiplicity(alpha):
        sys.exit('Problem: number of SYTs is not correct')
    
    return Y



def get_subSYT(alpha, alphaB, order='iLLOS'):
    """
    Build all SYTs associated to the subshape alpha-alpha0
    
    where alphaB is the base top-left irrep
    
    Inputs
    ------
    alpha : numpy array
        irrep
    alphaB : numpy array
        base top-left irrep
    order : str
        [default] 'iLLOS' or 'LLOS'
    
    Returns
    -------
    Y : numpy array
        SYTs associated to alpha-alphaB
    
    Example
    -------
    alpha = [3, 1]
    alphaB = [2, 0]
    The two SYTs are:
      |x|x|3|     |x|x|2|
      |2|      &  |3|
     y=[1, 0]    y=[0, 1]
    output: Y=[[1, 0], [0, 1]] (order='iLLOS')
    """
    
    n = np.sum(alpha)
    nB = np.sum(alphaB)
    
    nl = len(np.argwhere(alpha>0).flatten())
    nlB = len(np.argwhere(alphaB>0).flatten())
    
    if (nl<nlB):
        sys.exit('Problem: irreps alpha and alpha0 do not match')
    
    if not len(alpha)==len(alphaB):
        sys.exit('Problem: alpha and alphaB do not have the same number of elements.')
    
    if len(np.argwhere(alpha<alphaB).flatten())>0:
        sys.exit('Problem: alphaB is not contained in alpha.')
    
    alphap = np.copy(alpha)
    alphaBp = np.copy(alphaB)
    alphap = np.hstack([alpha, 0])
    alphaBp = np.hstack([alphaB, 0])
    
    #if len(np.argwhere(alphap<alphaBp).flatten())>0:
    #    sys.exit('Problem: alphaB is not contained in alpha.')
    
    nP = n - nB
    Y = np.zeros(shape=(1, nP), dtype=int)
    
    NY = int(1)
    
    for j in range(nP-1, -1, -1):
        
        cpt = int(0)
        Ynew = np.zeros(shape=(0, nP), dtype=int)
        
        for q in range(0, NY):
            # generate the remaining shape
            alphap_q = np.copy(alphap)
            for t in range(nP-1, j, -1):
                alphap_q[Y[q][t]] -= 1
            
            alphap_r = alphap_q - alphaBp
            
            ind = np.argwhere(alphap_r>0).flatten()
            
            for t in range(0, len(ind)):
                if alphap_q[ind[t]]>alphap_q[ind[t]+1]:
                    # row ind[t] has a bottom corner
                    ytmp = np.copy(Y[q])
                    ytmp[j] = ind[t]
                    Ynew = np.vstack([Ynew, ytmp])
                    cpt += 1
        NY = cpt
        Y = np.copy(Ynew)
    
    if order=='LLOS':
        Y = np.flipud(Y)
    
    return Y



def fill_subSYT(Y, alpha, **kwargs):
    """
    Fill all remaining boxes of a collection of sub-SYTs in order to be a 
    collection of valid SYTs for a given irrep
    
    Inputs
    ------
    Y : numpy array
        collection of SYTs
    alpha : numpy array
        irrep
    fill_type : str
        [default] 'largest' or 'smallest'
        order of the base filling in LLOS
    
    Returns
    -------
    Y : numpy array
        collection of SYTs
    
    Example
    -------
    Y = np.array([[0, 1, 2], [1, 0, 2]], dtype=int)
    alpha = np.array([3, 2, 1], dtype=int)
    
    The input sub-SYTs are:
    x x 3         x x 4
    x 4     and   x 3
    5             5
    
    For fill_type='largest', we have:
                0 2                 0 2 3     0 2 4
     filling =  1      --> output = 1 4   and 1 3
                                    5         5
    For fill_type='smallest', we have:
                0 1                 0 1 3     0 1 4
     filling =  2      --> output = 2 4   and 2 3
                                    5         5
    """
    
    if not 'fill_type' in kwargs:
        kwargs['fill_type'] = 'largest'
    
    if not kwargs['fill_type'] in ['largest', 'smallest']:
        sys.exit('Problem: undefined fill_type.')
    
    n = np.sum(alpha)
    NY = Y.shape[0]
    n1 = Y.shape[1]
    n0 = n - n1 # number of particles to place
    
    alpha1 = np.zeros(shape=alpha.shape, dtype=int)
    
    for i in range(0, n1):
        alpha1[Y[0][i]] += 1
    
    alpha0 = alpha - alpha1
    
    
    
    y1 = np.zeros((n0,), dtype=int)
    cpt = int(0)
    
    if kwargs['fill_type']=='largest':
        alpha0T = transpose_shape(alpha0)
        for ic in range(0, len(alpha0T)):
            for il in range(0, alpha0T[ic]):
                y1[cpt] = il
                cpt += 1
    else:
        for il in range(0, len(alpha0)):
            for ic in range(0, alpha0[il]):    
                y1[cpt] = il
                cpt += 1
    
    Y = np.hstack([np.tile(y1, [NY, 1]), Y])
    
    return Y



def index_to_SYT(i, alpha, order):
    # Build the SYT corresponding to index <i> in the <order> for the irrep alpha.
    # 
    # Inputs:
    #   i       index, ranging from 0 to multiplicity(alpha)-1
    #   alpha   irrep
    #   order   'LLOS' or 'iLLOS', to specify in which order the index is taken
    # 
    
    if order=='LLOS':
        y = index_to_SYT_LLOS(i, alpha, order)
    elif order=='iLLOS':
        y = index_to_SYT_iLLOS(i, alpha, order)
    else:
        sys.exit('Problem: undefined order.')
    
    return y



def index_to_SYT_LLOS(i, alpha, order='LLOS'):
    # Build the SYT corresponding to index <i> in <order> for the irrep <alpha>.
    # 
    # Inputs:
    #   i       index, ranging from 0 to multiplicity(alpha)-1
    #   alpha   irrep
    #   order   [optional] default: order='LLOS', to specify in which order the index is taken
    # 
    # recall: increasing order of LLOS:
    #   1st  SYT: numbers filled row-wise
    #   last SYT: numbers filled column-wise
    # 
    # Remarks:
    #   - The default order is the increasing order of the LLOS, unless the user 
    #     puts order='iLLOS' (for inverse Last Letter Order Sequence)
    #   - If the user puts order='iLLOS', the algorithm needs to compute the 
    #     multiplicity of the irrep <alpha>. It is thus better to use 
    #     index_to_SYT_iLLOS(i, alpha) to save execution time.
    # 
    
    alphap = np.copy(alpha)
    n = np.sum(alphap)
    
    if order=='iLLOS':
        # need to reverse the order
        falpha = multiplicity(alpha)
        i = falpha - 1 - i
    
    nl = len(np.argwhere(alphap>0).flatten())
    alphap = alphap[0:nl]
    
    y = np.full(shape=(n,), fill_value=-1, dtype=int)
    y[0] = 0 # first particle is always in the first row
    
    borne_inf = int(0)
    
    k = n-1 # number to place
    
    while k>0: # we will place number k
        bc = get_bottom_corner(alphap)
        nbc = len(bc)
        b = nbc
        while ( (b>0) and (i>=borne_inf) ):
            b -= 1
            alphapp = np.copy(alphap)
            alphapp[bc[b]] -= 1
            tmp = multiplicity(alphapp)
            borne_inf += tmp
        y[k] = bc[b]
        alphap[bc[b]] -= 1
        borne_inf -= tmp
        k -= 1
    
    return y



def index_to_SYT_iLLOS(i, alpha, order='iLLOS'):
    # Build the SYT corresponding to index <i> in <order> for the irrep <alpha>.
    # 
    # Inputs:
    #   i       index, ranging from 0 to multiplicity(alpha)-1
    #   alpha   irrep
    #   order   [optional] default: order='iLLOS', to specify in which order the index is taken
    # 
    # recall: iLLOS = inverse Last Letter Order Sequence:
    #   1st  SYT: numbers filled column-wise
    #   last SYT: numbers filled row-wise
    # 
    # Remarks:
    #   - The default order is the decreasing order of the LLOS (so-called iLLOS), 
    #     unless the user puts order='LLOS'.
    #   - If the user puts order='LLOS', the algorithm needs to compute the 
    #     multiplicity of the irrep <alpha>. It is thus better to use 
    #     index_to_SYT_LLOS(i, alpha) to save execution time.
    # 
    
    alphap = np.copy(alpha)
    n = np.sum(alphap)
    
    if order=='LLOS':
        # need to reverse the order
        falpha = multiplicity(alpha)
        i = falpha - 1 - i
    
    nl = len(np.argwhere(alphap>0).flatten())
    alphap = alphap[0:nl]
    
    y = np.full(shape=(n,), fill_value=-1, dtype=int)
    y[0] = 0 # first particle is always in the first row
    
    borne_inf = int(0)
    
    k = n-1 # number to place
    
    while k>0: # we will place number k
        bc = get_bottom_corner(alphap)
        nbc = len(bc)
        b = int(-1)
        while ((b<nbc-1) and (i>=borne_inf)):
            b += 1
            alphapp = np.copy(alphap)
            alphapp[bc[b]] -= 1
            tmp = multiplicity(alphapp)
            borne_inf += tmp
        y[k] = bc[b]
        alphap[bc[b]] -= 1
        borne_inf -= tmp
        k -= 1
    
    return y



def index_to_SYT_symm(i, alpha, m, order='LLOS'):
    # Build the SYT corresponding to index <i> for the irrep alpha, in the case
    # of local symmetric irrep with <m> boxes (local constraints).
    # 
    # By default, the index <i> is considered in the ascending order of the 
    # Last Letter Order Sequence, namely:
    #   i=0 --> lowest SYT in LLOS
    #   i=multiplicity_symm(alpha, m)-1 ---> highest SYT in LLOS
    # 
    # This can be changed by setting order='iLLOS', in which case the 
    # decreasing order of the LLOS is used.
    # 
    
    alpha0 = np.copy(alpha)
    n = np.sum(alpha0)
    
    if order=='LLOS':
        falpha = multiplicity_symm(alpha, m)
        i = falpha - 1 - i
    
    nl = len(np.argwhere(alpha0>0).flatten())
    alpha0 = alpha0[0:nl]
    alpha0 = np.hstack([alpha0, 0])
    
    y = np.full(shape=(n,), fill_value=-1, dtype=int)
    
    Ns = n//m
    
    borne_inf = int(0)
    
    s = int(Ns-1)
    
    if m==2:
        
        # the 2 first particles are necessarily in the first row
        y[0] = 0
        y[1] = 0
        
        while s>0:
            # we will place numbers s*m, s*m+1, ..., (s+1)*m-1
            
            # the two particles of site s are:
            p0 = s*m
            p1 = s*m + 1
            
            bc = get_bottom_corner(alpha0)
            cbc = alpha0[bc] - 1 # columns of bottom corners
            nbc = len(bc)
            b = nbc
            while ((b>0) and (i>=borne_inf)):
                # place particle p1 in bottom corner bc[b]
                b -= 1
                alpha1 = np.copy(alpha0)
                alpha1[bc[b]] -= 1
                w0 = bc[b]
                while ((w0>=0) and (i>= borne_inf)):
                    alpha2 = np.copy(alpha1)
                    # trying to place p0 in row w0
                    if ((alpha2[w0]>alpha2[w0+1]) and (not (alpha2[w0]-1==cbc[b]))):
                        # we are in a legal configuration for placing p0 at row w0
                        alpha2[w0] -= 1
                        tmp = multiplicity_symm(alpha2, m)
                        borne_inf += tmp
                    # end if
                    w0 -= 1
                # end while w2
            # end while b
            y[p1] = bc[b]
            y[p0] = w0 + 1
            borne_inf -= tmp
            alpha0[bc[b]] -= 1
            alpha0[w0+1] -= 1
            s -= 1
        # end while s
        
    elif m==3:
        
        # the 2 first particles are necessarily in the first row
        y[0] = 0
        y[1] = 0
        
        while s>0:
            # we will place numbers s*m, s*m+1, ..., (s+1)*m-1
            
            # the three particles of site s are:
            p0 = s*m
            p1 = s*m + 1
            p2 = s*m + 2
            
            bc = get_bottom_corner(alpha0)
            cbc = alpha0[bc] - 1 # columns of bottom corners
            nbc = len(bc)
            b = nbc
            
            while ( (b>0) and (i>=borne_inf) ):
                # place particle p3 of site <s> in bottom corner bc[b]
                b -= 1
                alpha1 = np.copy(alpha0)
                alpha1[bc[b]] -= 1
                w1 = bc[b]
                while ((w1>=0) and (i>=borne_inf)):
                    alpha2 = np.copy(alpha1)
                    # trying to place p1 in row w1
                    if ((alpha2[w1]>alpha2[w1+1]) and (not (alpha2[w1]-1==cbc[b]))):
                        # we are in a legal configuration for placing p1 at row w1
                        alpha2[w1] -= 1
                        w0 = w1
                        while ((w0>=0) and (i>=borne_inf)):
                            alpha3 = np.copy(alpha2)
                            # trying to place p0 in row w0
                            if ((alpha3[w0]>alpha3[w0+1]) and (not (alpha3[w0]-1==cbc[b])) and (not (alpha3[w0]-1==alpha2[w1]))):
                                # we are in a legal configuration for placing p0 at row w0
                                alpha3[w0] -= 1
                                tmp = multiplicity_symm(alpha3, m)
                                borne_inf += tmp
                            # end if
                            w0 -= 1
                        # end while w0
                    # end if
                    w1 -= 1
                # end while w1
            # end while b
            
            y[p2] = bc[b]
            y[p1] = w1 + 1
            y[p0] = w0 + 1
            
            borne_inf -= tmp
            alpha0[bc[b]] -= 1
            alpha0[w1+1] -= 1
            alpha0[w0+1] -= 1
            s -= 1
        # end while s
        
        # now place the 3rd particle of site 0 in the lowest remaining bottom corner
        y[2] = np.argwhere(alpha0>0).flatten()[-1]
        
    else:
        sys.exit('Problem: m>3 not yet implemented')
    
    return y



def SYT_to_index(y, alpha, order):
    # Map a SYT to its index in <order> in the list of all SYTs.
    # 
    # Inputs:
    #   y       SYT
    #   alpha   irrep
    #   order   'LLOS' or 'iLLOS', to specify in which order the index is taken
    # 
    
    if order=='LLOS':
        i = SYT_to_index_LLOS(y, alpha, order)
    elif order=='iLLOS':
        i = SYT_to_index_iLLOS(y, alpha, order)
    else:
        sys.exit('Problem: order undefined.')
    
    return i



def SYT_to_index_LLOS(y, alpha, order='LLOS'):
    # Map a SYT to its index in the list of all SYTs.
    # 
    # The default order is inceasing order of Last Letter Order Sequence, 
    # unless the user specifies order='iLLOS' for the reverse order
    # 
    
    alphap = np.copy(alpha)
    n = np.sum(alphap)
    
    nl = len(np.argwhere(alphap>0).flatten())
    alphap = alphap[0:nl]

    k = n-1
    
    i = int(0)
    
    while k>0:
        bc = get_bottom_corner(alphap)
        nbc = len(bc)
        b = nbc-1
        while bc[b]>y[k]:
            alphapp = np.copy(alphap)    
            alphapp[bc[b]] -= 1
            i += multiplicity(alphapp)
            b -= 1
        alphap[bc[b]] -= 1
        k -= 1
    
    if order=='iLLOS':
        # reverse order
        falpha = multiplicity(alpha)
        i = falpha - 1 - i
    
    return i



def SYT_to_index_iLLOS(y, alpha, order='iLLOS'):
    # Map a SYT to its index in the list of all SYTs.
    # 
    # The default order is decreasing order of Last Letter Order Sequence, 
    # unless the user specifies order='LLOS' for the reverse order
    # 
    
    alphap = np.copy(alpha)
    n = np.sum(alphap)
    
    nl = len(np.argwhere(alphap>0).flatten())
    alphap = alphap[0:nl]

    k = n-1
    
    i = int(0)
    
    while k>0:
        bc = get_bottom_corner(alphap)
        b = int(0)
        while (bc[b]<y[k]):
            alphapp = np.copy(alphap)    
            alphapp[bc[b]] -= 1
            i += multiplicity(alphapp)
            b += 1
        alphap[bc[b]] -= 1
        k -= 1
    
    if order=='LLOS':
        # reverse order
        falpha = multiplicity(alpha)
        i = falpha - 1 - i
    
    return i



def SYT_to_index_symm(y, alpha, m, order='LLOS'):
    # Map a SYT to its index in the list of all SYTs.
    # 
    # The default order is inceasing order of Last Letter Order Sequence, 
    # unless the user specifies order='iLLOS' for the reverse order
    # 
    
    alpha0 = np.copy(alpha)
    n = np.sum(alpha0)
    
    nl = len(np.argwhere(alpha0>0).flatten())
    alpha0 = alpha0[0:nl]
    alpha0 = np.hstack([alpha0, 0])
    
    Ns = n//m # number of sites
    
    i = int(0)
    
    s = int(Ns-1)
    
    if m==2:
        
        while s>0:
            
            # particles of site s are:
            p0 = s*m
            p1 = s*m + 1
            
            bc = get_bottom_corner(alpha0)
            nbc = len(bc)
            cbc = alpha0[bc] - 1 # columns of bottom corners
            b = nbc - 1
            
            while ((b>=0) and (bc[b]>=y[p1])):
                
                alpha1 = np.copy(alpha0)
                alpha1[bc[b]] -= 1
                
                if bc[b]==y[p1]:
                    limit = y[p0]
                else:
                    limit = int(-1)
                
                w0 = bc[b]
                
                while w0>limit:
                    alpha2 = np.copy(alpha1)
                    if ((alpha2[w0]>alpha2[w0+1]) and (not (alpha2[w0]-1==cbc[b]))):
                        # it is a legal configuration: p1 at bc[b] and p0 at w0
                        alpha2[w0] -= 1
                        i += multiplicity_symm(alpha2, m)
                    # end if
                    w0 -= 1
                # end while w0
                b -= 1
            # end while b
            
            alpha0[bc[b+1]] -= 1
            alpha0[w0] -= 1
            s -= 1            
    
    elif m==3:
        
        while s>0:
            
            # particles of site s are:
            p0 = s*m
            p1 = s*m + 1
            p2 = s*m + 2
            
            bc = get_bottom_corner(alpha0)
            nbc = len(bc)
            cbc = alpha0[bc] - 1 # columns of bottom corners
            b = nbc - 1
            
            while ((b>=0) and (bc[b]>=y[p2])):
                
                alpha1 = np.copy(alpha0)
                alpha1[bc[b]] -= 1
                
                if bc[b]==y[p2]:
                    limit2 = y[p1]
                else:
                    limit2 = int(-1)
                
                w1 = bc[b]
                
                while w1>=limit2:
                    alpha2 = np.copy(alpha1)
                    if ((alpha2[w1]>alpha2[w1+1]) and (not (alpha2[w1]-1==cbc[b]))):
                        
                        alpha2[w1] -= 1
                        
                        if ((bc[b]==y[p2]) and (w1==y[p1])):
                            limit3 = y[p0]
                        else:
                            limit3 = int(-1)
                        
                        w0 = w1
                        
                        while w0>limit3:
                            alpha3 = np.copy(alpha2)
                            if ((alpha3[w0]>alpha3[w0+1]) and (not (alpha3[w0]-1==cbc[b])) and (not (alpha3[w0]-1==alpha2[w1]))):
                                # it is a legal configuration: p2 at bc[b], p1 at w1 and p0 at w0
                                alpha3[w0] -= 1
                                i += multiplicity_symm(alpha3, m)
                            # end if
                            w0 -= 1
                        # end while w0
                    # end if
                    w1 -= 1
                # end while w1
                b -= 1
            # end while b
            alpha0[bc[b+1]] -= 1
            alpha0[w1+1] -= 1
            alpha0[w0] -= 1
            s -= 1
        # en while s
    
    else:
        sys.exit('Problem: case m>3 not yet implemented.')
    
    if order=='LLOS':
        falpha = multiplicity_symm(alpha, m)
        i = falpha - 1 - i
    
    return i



def binary_search_SYT(y, Y):
    # Search the index of SYT <y> in the list of all SYTs <Y> through a binary
    # search.
    # 
    
    NY = Y.shape[0]
    n = Y.shape[1]
    
    l = int(0)
    u = NY - 1
    m = int(0)
    
    while (l<=u):
        m = (l+u)//2
        t = n - 1
        while (t>0):
            if (Y[m][t]<y[t]):
                l = m + 1
                break
            elif (Y[m][t]>y[t]):
                u = m - 1
                break
            else:
                # box t is correctly placed
                t -= 1
        if (t==0):
            l = u+1
    
    return m



def multiplicity_symm(alpha, m):
    # Compute the total number of SYTs for the irrep alpha, satisfying the 
    # constraints of local symmetry with m particles per site in the symmetric
    # irrep.
    # 
    
    if not np.sum(alpha)%m==0:
        sys.exit('Problem: number of boxes in alpha must be a multiple of m')
    
    n = np.sum(alpha)//int(m)
    
    nmax = int(2*10**5)
    nk = int(m)
    N = len(np.argwhere(alpha>0))
    alpha = alpha[0:N]
    
    local_dimension = int(1)
    for w in range(0, nk):
        local_dimension = local_dimension * (N+w)
    local_dimension = local_dimension//math.factorial(nk)
    
    Forme = np.zeros((nmax, N+1), dtype=int)
    Forme[0,0] = nk
    Forme[0,N] = 1
    
    s = 0
    PI = np.pi**np.arange(0,N)
    PI = np.reshape(PI, [N, 1])
    
    p = np.zeros((n,), dtype=int)
    p[0] = 1
    
    for tt in range(2, n+1):
        
        index_sum = np.sum(p[0:tt-1])
        
        for pp in range(s+1, s+p[tt-2]+1): # Matlab: for pp=s+1:s+p(tt-1)
            W_k = np.argwhere(Forme[pp-1,0:N]).flatten()+1
            N_W = len(W_k)
            
            nombre_max_de_filles_sur_toutes_les_generations = 2 + (N_W+1)**nk
            
            Forme_temp_pp = np.zeros((nombre_max_de_filles_sur_toutes_les_generations, N+1), dtype=int)
            
            Forme_temp_pp[0,:] = Forme[pp-1,:]
            lignes_added_pp = np.zeros((nombre_max_de_filles_sur_toutes_les_generations, nk+1), dtype=int)
            colonnes_added_pp = 100 * np.full((nombre_max_de_filles_sur_toutes_les_generations,nk+1), fill_value=1, dtype=int)
            
            index_courant_fille = 2
            start_mere = 0
            end_mere = 0
            
            for k in range(1, nk+1):
                start_mere = end_mere + 1
                end_mere = index_courant_fille - 1
                
                for index_courant_mere in range(start_mere, end_mere+1):
                
                    tmp = Forme_temp_pp[index_courant_mere-1, (max(lignes_added_pp[index_courant_mere-1,k-1]-1, 1)-1):N-1]
                    W_k = np.argwhere(tmp).flatten() + 1
                    N_W = len(W_k)
                    
                    W_k = max(lignes_added_pp[index_courant_mere-1, k-1]-1, 1) + W_k - 1
                    
                    ligne_bottom = int(1)
                    colonne_bottom = int(1) + np.sum(Forme_temp_pp[index_courant_mere-1, 0:N])
                    
                    if lignes_added_pp[index_courant_mere-1, k-1]<=1:
                        
                        Forme_temp_pp[index_courant_fille-1,:] = np.copy(Forme_temp_pp[index_courant_mere-1,:])
                        Forme_temp_pp[index_courant_fille-1,0] = Forme_temp_pp[index_courant_fille-1,0] + 1
                        lignes_added_pp[index_courant_fille-1,k] = ligne_bottom
                        colonnes_added_pp[index_courant_fille-1,k] = colonne_bottom
                        colonnes_added_pp[index_courant_fille-1,0:k] = np.copy(colonnes_added_pp[index_courant_mere-1,0:k])
                        index_courant_fille = index_courant_fille + 1
                    
                    for hh in range(1, N_W+1):
                        
                        # lignehh = hh
                        ligne_bottom = W_k[hh-1] + 1
                        colonne_bottom = 1 + np.sum(Forme_temp_pp[index_courant_mere-1, ligne_bottom-1:N])
                        
                        if lignes_added_pp[index_courant_mere-1,k-1]==ligne_bottom:
                            
                            prod_bool_col = int(1)
                            
                            for z in range(1, k+1):
                                prod_bool_col = prod_bool_col * ( colonne_bottom - colonnes_added_pp[index_courant_mere-1,z-1] )
                            
                            if prod_bool_col:
                                
                                Forme_temp_pp[index_courant_fille-1,:] = np.copy(Forme_temp_pp[index_courant_mere-1, :])
                                
                                Forme_temp_pp[index_courant_fille-1,W_k[hh-1]-1] = Forme_temp_pp[index_courant_fille-1,W_k[hh-1]-1] - 1
                                Forme_temp_pp[index_courant_fille-1,W_k[hh-1]] = Forme_temp_pp[index_courant_fille-1,W_k[hh-1]] + 1

                                lignes_added_pp[index_courant_fille-1,k] = ligne_bottom
                                colonnes_added_pp[index_courant_fille-1,k] = colonne_bottom
                                colonnes_added_pp[index_courant_fille-1,0:k] = np.copy(colonnes_added_pp[index_courant_mere-1,0:k])
                                index_courant_fille = index_courant_fille + 1
                            
                        elif lignes_added_pp[index_courant_mere-1,k-1]<ligne_bottom:
                            
                            if colonne_bottom<min(colonnes_added_pp[index_courant_mere-1,0:k]):
                                
                                Forme_temp_pp[index_courant_fille-1,:] = np.copy(Forme_temp_pp[index_courant_mere-1,:])
                                Forme_temp_pp[index_courant_fille-1,W_k[hh-1]-1] = Forme_temp_pp[index_courant_fille-1,W_k[hh-1]-1] - 1
                                Forme_temp_pp[index_courant_fille-1,W_k[hh-1]] = Forme_temp_pp[index_courant_fille-1,W_k[hh-1]] + 1
                                
                                lignes_added_pp[index_courant_fille-1,k] = ligne_bottom
                                colonnes_added_pp[index_courant_fille-1,k] = colonne_bottom
                                colonnes_added_pp[index_courant_fille-1,0:k] = np.copy(colonnes_added_pp[index_courant_mere-1,0:k])
                                index_courant_fille = index_courant_fille + 1
                            
                    # end for hh
                # end for index_courant_mere
            # end for k
            
            Forme[index_sum:index_sum+index_courant_fille-1-end_mere,:] = np.copy(Forme_temp_pp[end_mere:index_courant_fille-1,:])
            index_sum = index_sum + index_courant_fille - 1 - end_mere
            
        # end for pp
        
        s = s + p[tt-2]
        
        # implement the equivalent of Matlab's sortrows
        A = np.asarray(Forme[s:index_sum,:])
        if A.shape[0]>1:
            tmp = []
            for i in range(A.shape[1]-1,-1,-1):
                tmp.append(tuple(A[:,i]))
            ix = np.lexsort(tmp)
            B = np.copy(A)
            B = B[ix,:]
            Forme[s:index_sum,:] = B
        
        FormePrec = np.zeros((index_sum, N), dtype=int)
        FormePrec[s+1:index_sum,0:N] = np.copy(Forme[s:index_sum-1,0:N])
        
        Fdiff = np.argwhere( ((Forme[s:index_sum,0:N] - FormePrec[s:index_sum,0:N]) @ PI).flatten() ).flatten() + 1
        
        N_temp_ordre = np.copy(Forme[s:index_sum,N])
        
        N_Fdiff = Fdiff.shape[0]
        p[tt-1] = N_Fdiff
        if N_Fdiff>0:
            Forme[s:s+N_Fdiff,:] = Forme[s+Fdiff-1,:]
        else:
            sys.exit('toto')
        
        Fdiffsup = np.zeros((N_Fdiff,), dtype=int)
        Fdiffsup[0:N_Fdiff-1] = np.copy(Fdiff[1:N_Fdiff]) - 1
        Fdiffsup[N_Fdiff-1] = index_sum - s
        
        for aa in range(1, N_Fdiff+1):
            Forme[s+aa-1,N] = np.sum(N_temp_ordre[Fdiff[aa-1]-1:Fdiffsup[aa-1]])
        
        if (index_sum>s+N_Fdiff):
            Forme[s+N_Fdiff:index_sum,:] = np.zeros(shape=(index_sum-(s+N_Fdiff),N+1), dtype=int)
        
    # end for tt
    
    Forme_par_lignes = np.copy(Forme[0:np.sum(p[0:n]), 0:N]) @ np.transpose(np.triu(np.full(shape=(N,N), fill_value=1)))
    
    Mattest = np.transpose(np.abs(Forme_par_lignes - np.full(shape=(np.sum(p[0:n]), 1), fill_value=1, dtype=int) @ np.reshape(alpha[0:N], [1, N])))
    lM = Mattest.shape[0]
    
    if lM>1:
        vectest = np.sum(abs(Mattest), axis=0)
        vecindex = np.argwhere(vectest==0).flatten() + 1
    elif lM==1:
        vecindex = np.argwhere(Mattest==0).flatten() + 1
    
    if len(vecindex)>0:
        kostka = Forme[vecindex[0]-1, N]
    else:
        kostka = 0
    
    return kostka



def get_bottom_corner(alpha):
    # Extract the bottom corners of the shape alpha.
    # 
    
    alphap = np.hstack([alpha, 0])
    delta = alpha - alphap[1:]
    bc = np.argwhere(delta!=0).flatten()
    
    return bc



def get_transpositions(links):
    # Compute the string of transpositions for a list of permutations.
    # 
    # Input:
    #   links   list of numpy arrays. Each numpy array must be of dimension 2
    # 
    # Example:
    #   links = [np.array([0, 3], dtype=int)]
    #   links = [np.array([0, 3], dtype=int), np.array([1, 6], dtype=int)]
    # 
    # Outputs:
    #   listtranspositions  list of transpositions for each link (permutation)
    #   nbtranspositions    number of transpositions for each link
    # 
    # Example 1:
    #   links = [np.array([0, 3], dtype=int)]
    #   --->
    #   listtranspositions = [np.array([0, 1, 2, 1, 0], dtype=int)]
    #   nbtranspositions = np.array([5], dtype=int)
    # 
    # Example 2:
    #   links = [np.array([0, 3], dtype=int), np.array([1, 6], dtype=int)]
    #   --->
    #   listtranspositions = [np.array([0, 1, 2, 1, 0], dtype=int), np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=int)]
    #   nbtranspositions = np.array([5, 9], dtype=int)
    # 
    
    nlinks = len(links)
    nbtranspositions = np.zeros((nlinks,), dtype=int)
    # nbtranspositions[i] = number of transpositions needed to rewrite the
    # permutation of the i-th link as a product of transpositions
    
    '''
    for i in range(0, nlinks):
        nbtranspositions[i] = 2*abs(links[i][1]-links[i][0]) - 1
    '''
    
    listtranspositions = [None] * nlinks
    
    '''
    for i in range(0, nlinks):
        mini = np.min(links[i])
        maxi = np.max(links[i])
        
        listtranspositions[i] = np.zeros((nbtranspositions[i], ), dtype=int)
        listtranspositions[i][0:((nbtranspositions[i]-1)/2+1).astype(int)] = np.arange(mini, maxi)
        listtranspositions[i][((nbtranspositions[i]-1)/2+1).astype(int):] = np.arange(maxi-2, mini-1, -1)
    '''
    for i in range(0, nlinks):
        listtranspositions[i] = transposition_to_adjacent_transpositions(links[i])
        nbtranspositions[i] = len(listtranspositions[i])
    
    return listtranspositions, nbtranspositions


def get_axial_distance(y, cy, i, j):
    # Compute axial distance from i to j in SYT y (and columns cy).
    # 
    
    ad = cy[i] - y[i] - cy[j] + y[j]
    
    return ad


def get_new_shape(alpha, y):
    # Get the remaining shape once all boxes corrresponding to the particles 
    # already placed in y are removed from alpha
    # 
    
    n = len(y)
    idst = np.argwhere(y>=0).flatten()
    alphap = np.copy(alpha)
    if len(idst)>0:
        idst = idst[0]
        for p in range(n-1,idst-1,-1):
            alphap[y[p]] -= 1
    
    return alphap



def dim_irrep_sun(alpha, N):
    """
    Dimension of irrep of SU(N)
    
    Parameters
    ----------
    alpha : numpy array
        irrep
    N : int
        SU(N)
    
    Returns
    -------
    dimension : int
        dimension of irrep
    
    Description
    -----------
    Compute the dimension of the irrep of SU(N) using the Hook-length formula
    """
    
    nl = len(np.argwhere(alpha>0).flatten())
    if nl>N:
        sys.exit('Problem: alpha has more rows than N.')
    alphap = np.zeros((N,), dtype=int)
    alphap[0:nl] = alpha[0:nl]
    alpha = alphap
    alpha = alpha - np.full(shape=(N,), fill_value=alpha[N-1], dtype=int) # remove columns with N boxes
    n = np.sum(alpha)
    
    # Numerator
    numvec = np.zeros(shape=n, dtype=int)
    cpt = int(0)
    for i in range(0, nl):
        for j in range(0, alpha[i]):
            numvec[cpt] = N-i+j
            cpt += 1
    numvec = np.sort(numvec)
    
    # Denominator: product of Hook lengths
    m = alpha[0]
    alphafull = np.zeros((nl, m), dtype=int)
    for i in range(0, nl):
        for j in range(0, alpha[i]):
            alphafull[i,j] = 1
    
    denomvec = np.zeros(shape=n, dtype=int)
    cpt = int(0)
    for i in range(0, nl):
        for j in range(0, alpha[i]):
            sumi = np.sum(alphafull[i:nl,j]) + np.sum(alphafull[i,j+1:m])
            denomvec[cpt] = sumi
            cpt += 1
    denomvec = np.sort(denomvec)
    
    # check for equality between factors in numerator and denominator
    for i in range(n):
        di = denomvec[i]
        ind = np.argwhere(numvec==di).flatten()
        if len(ind)>0:
            numvec[ind[0]] = 1
            denomvec[i] = 1
    
    # keep only non-1's
    numvec = numvec[np.argwhere(numvec!=1)].flatten()
    denomvec = denomvec[np.argwhere(denomvec!=1)].flatten()
    
    if len(denomvec)==0:
        denomvec = np.array([1], dtype=int)
    else:
        # we further reduce by finding common divisors among prime numbers
        divisors = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=int) # 11 first prime numbers
        for d in divisors:
            for t in range(0, 4): # 4 repetitions of the loops below
                for i in range(0, len(numvec)):
                    if (numvec[i]%d==0):
                        for j in range(0, len(denomvec)):
                            if (denomvec[j]%d==0):
                                numvec[i] = numvec[i]//d
                                denomvec[j] = denomvec[j]//d
    
    num = np.prod(numvec)
    denom = np.prod(denomvec)
    dimension = num//denom
    
    return dimension



def get_SSYT(alpha, N):
    # Get the collection of semi-standard Young tableaux for the irrep alpha of
    # SU(N).
    # 
    # Output:
    #   vec     numpy array. The SSYTs are in rows in vec
    # 
    
    nl = len(np.argwhere(alpha>0).flatten())
    if nl>N:
        sys.exit('Problem: alpha has more rows than N.')
    alphap = np.zeros((N,), dtype=int)
    alphap[0:nl] = alpha[0:nl]
    alpha = alphap
    
    dimension = dim_irrep_sun(alpha, N)
    
    vec = np.zeros(shape=(dimension, N*(N+1)//2), dtype=int)
    long_ligne = np.zeros((N,), dtype=int)
    long_ligne[0] = N
    long_ligne_offset = np.zeros((N,), dtype=int)
    acc = int(0)
    
    for p in range(2, N+1):
        long_ligne[p-1] = N - p + 1
        acc += long_ligne[p-2]
        long_ligne_offset[p-1] = acc
    
    # create max pattern
    vec[0,0:N] = np.copy(alpha)
    for p in range(1, N):
        vec[0, p+long_ligne_offset[1:N-p+1]-1] = np.full(shape=(1, N-p), fill_value=vec[0, p-1], dtype=int)
    
    for s in range(1, dimension):
        vectemp = np.copy(vec[s-1, :])
        
        # find pivoting index
        indice = N*(N+1)//2
        j_indice = np.argwhere( (long_ligne_offset - indice*np.full(shape=(N,), fill_value=1, dtype=int))<0).flatten()[-1] +1
        i_indice = indice - long_ligne_offset[j_indice-1]
        indice_friend = i_indice +1 + long_ligne_offset[j_indice-2]
        
        while (vectemp[indice-1]==vectemp[indice_friend-1]):
            indice -= 1
            j_indice = np.argwhere( (long_ligne_offset - indice*np.full(shape=(N,), fill_value=1, dtype=int))<0).flatten()[-1] +1
            i_indice = indice - long_ligne_offset[j_indice-1]
            indice_friend = i_indice + 1 + long_ligne_offset[j_indice-2]
        
        # apply (C9) from Alex' article
        vectemp[indice-1] -= 1
        for k in range(indice+1, N*(N+1)//2+1):
            j_k = np.argwhere( (long_ligne_offset-k*np.full(shape=(N,), fill_value=1, dtype=int))<0 )[-1] +1
            i_k = k - long_ligne_offset[j_k-1]
            vectemp[k-1] = vectemp[i_k+long_ligne_offset[j_k-2]-1]
        
        vec[s,:] = vectemp
    
    return vec



def tensor_product_irrep(alpha1, alpha2, N):
    # Compute the tensor product of two irreps of SU(N).
    # 
    # Remark:
    # - The total number of boxes is preserved.
    # 
    
    nl1 = len(np.argwhere(alpha1>0).flatten())
    nl2 = len(np.argwhere(alpha2>0).flatten())
    
    if ( (nl1>N) or (nl2>N) ):
        sys.exit('Problem in the shapes of the irreps.')
    
    alpha1p = np.zeros((N,), dtype=int)
    alpha1p[0:nl1] = alpha1[0:nl1]
    alpha1 = np.copy(alpha1p)
    alpha2p = np.zeros((N,), dtype=int)
    alpha2p[0:nl2] = alpha2[0:nl2]
    alpha2 = np.copy(alpha2p)
    
    vec = get_SSYT(alpha1, N)
    if len(vec.shape)==1:
        lvec = 1
    else:
        lvec = vec.shape[0] # number of SSYT
    
    dim1 = dim_irrep_sun(alpha1, N)
    
    if not lvec==dim1:
        sys.exit('Problem: dimension of irrep alpha1 does not match number of Gelfand patters.')
    
    long_ligne = np.zeros((N,), dtype=int)
    long_ligne[0] = N
    long_ligne_offset = np.zeros((N,), dtype=int)
    acc = int(0)
    for p in range(2, N+1):
        long_ligne[p-1] = N - p + 1
        acc += long_ligne[p-2]
        long_ligne_offset[p-1] = acc
    
    bool_beca = np.zeros(shape=(N*(N+1)//2,), dtype=int)
    bool_beca[-1] = 1
    
    index_ligne_diagonal = np.zeros(shape=(N*(N+1)//2,), dtype=int)
    
    ist = int(0)
    
    for i in range(1, N+1):
        ist += 1
        ied = ist + N - i
        index_ligne_diagonal[ist-1:ied] = i + long_ligne_offset[0:N-i+1]
        bool_beca[ied-1] = 1
        ist = ied
    
    k = int(1)
    cpt = int(0)
    alpha = np.zeros((1000, N), dtype=int)
    
    while (k<(dim1+1)):
        tvec = np.copy(alpha2)
        gel = np.copy(vec[k-1, :])
        becal = np.zeros((N*(N+1)//2,), dtype=int)
        
        for j in range(1, N*(N+1)//2+1):
            if bool_beca[j-1]>0:
                becal[j-1] = gel[index_ligne_diagonal[j-1]-1]
            else:
                becal[j-1] = gel[index_ligne_diagonal[j-1]-1] - gel[index_ligne_diagonal[j]-1]
        # end for j
        
        jj = int(0)
        
        while (jj<N*(N+1)//2):
            
            jj += 1
            
            toto = long_ligne_offset - index_ligne_diagonal[jj-1]*np.full(shape=(N,), fill_value=1, dtype=int)
            l = N - (np.argwhere(toto<0).flatten()[-1]+1) + 1
            
            tvec[l-1] += becal[jj-1]
            
            if (l>1):
                if (tvec[l-1]>tvec[l-2]):
                    break # go out of loop on jj
        # end while
        
        if (jj==N*(N+1)//2):
            cpt += 1
            alpha[cpt-1,:] = np.copy(tvec)
        
        k += 1
    # end while k
    
    alpha = alpha[0:cpt,:]
    
    return alpha



def reduce_shape(alpha):
    # Count the number of occurences of all shapes contained in the input
    # 
    # Input: 
    #   alpha   collection of irreps. irreps are stored in the rows
    # 
    
    if len(alpha.shape)==1:
        # there is only one shape
        return alpha, np.array([1], dtype=int)
    
    n = alpha.shape[0] # number of shapes
    # nl = alpha.shape[1] # unused
    
    multi1 = np.zeros((n,), dtype=int)
    multi1[0] = 1
    alpha1 = np.zeros(alpha.shape, dtype=int)
    alpha1[0,:] = np.copy(alpha[0,:])
    cpt = 1
    
    for i in range(2, n+1):    
        ind = np.argwhere( np.sum(abs(alpha1[0:cpt,:] - alpha[i-1,:]), axis=1) == 0 ).flatten()
        if len(ind)==0:
            cpt += 1
            alpha1[cpt-1,:] = np.copy(alpha[i-1,:])
            multi1[cpt-1] = 1
        else:
            multi1[ind[0]] += 1
    
    alpha1 = alpha1[0:cpt,:]
    multi1 = multi1[0:cpt]
    
    return alpha1, multi1



def merge_shapes(alpha1, multi1, alpha2, multi2):
    # merge two combinations of shapes with their multiplicities.
    # 
    
    n1 = len(multi1)
    n2 = len(multi2)
    
    if ( (n1==1) and (len(alpha1.shape)==1) ):
        alpha1 = np.reshape(alpha1, (1, alpha1.shape[0]))
    if ( (n2==1) and (len(alpha2.shape)==1) ):
        alpha2 = np.reshape(alpha2, (1, alpha2.shape[0]))
    
    nl = alpha1.shape[1]
    
    alpha = np.zeros((n1+n2, nl), dtype=int)
    alpha[0:n1,:] = np.copy(alpha1)
    multi = np.zeros((n1+n2,), dtype=int)
    multi[0:n1] = np.copy(multi1)
    cpt = n1
    
    for j2 in range(0, n2):
        ind = np.argwhere(np.sum(abs(alpha - alpha2[j2,:]), axis=1)==0).flatten()
        if len(ind)==0:
            alpha[cpt,:] = np.copy(alpha2[j2,:])
            multi[cpt] = multi2[j2]
            cpt += 1
        else:
            multi[ind[0]] += multi2[j2]
    
    alpha = alpha[0:cpt,:]
    multi = multi[0:cpt]
    
    return alpha, multi



def multiplicity_irrep_mixed(alpha, beta, N):
    # Compute the number of times that alpha occurs in the tensor product of 
    # all irreps in beta.
    # 
    # Remark:
    # - beta must be a numpy array of irreps, each irrep stored in the rows.
    # 
    
    n = np.sum(alpha)
    
    if not n==np.sum(beta):
        sys.exit('Problem: number of boxes in beta does not match alpha.')
    
    nl = len(np.argwhere(alpha>0).flatten())
    if nl>N:
        sys.exit('Problem: Shape alpha not legal for SU(N=' + N + ')')
    
    alphap = np.zeros((N,), dtype=int)
    alphap[0:nl] = alpha[0:nl]
    alpha = np.copy(alphap)
    
    if len(beta.shape)==1:
        # there is only 1 irrep in beta
        Ns = 1
        beta = np.reshape(beta, (Ns, -1))
    else:
        # irreps are stored in the rowss
        Ns = beta.shape[0]
    
    if Ns==1:
        if alpha==beta:
            fmixedalpha = int(1)
        else:
            fmixedalpha = int(0)
    else:
        alpha1 = tensor_product_irrep(beta[0,:], beta[1,:], N)
        alpha1, multi1 = reduce_shape(alpha1)
    
        for j in range(2, Ns):
            alpha2 = tensor_product_irrep(alpha1[0,:], beta[j,:], N)
            alpha2, multi2 = reduce_shape(alpha2)
            multi2 *= multi1[0]
            
            for i in range(1, len(multi1)):
                alpha2i = tensor_product_irrep(alpha1[i,:], beta[j,:], N)
                alpha2i, multi2i = reduce_shape(alpha2i)
                multi2i *= multi1[i]
                alpha2, multi2 = merge_shapes(alpha2, multi2, alpha2i, multi2i)
            
            alpha1 = np.copy(alpha2)
            multi1 = np.copy(multi2)
        
        ind = np.argwhere(np.sum(abs(alpha1 - alpha), axis=1)==0).flatten()
        
        if len(ind)==0:
            fmixedalpha = int(0)
        else:
            fmixedalpha = multi1[ind[0]]
    
    return fmixedalpha



def sortrows(A):
    # Equivalent to Matlab's sortrows.
    # 
    # Input:
    #   A       2d numpy array
    # 
    # Outputs:
    #   B       sorted array of rows
    #   index   numpy array of indices such that B = A[index, :] 
    # 
    # Remark:
    #   See Matlab documentation: 
    #   https://ch.mathworks.com/help/matlab/ref/double.sortrows.html
    # 
    
    if A.shape[0]>1:
        tmp = []
        for i in range(A.shape[1]-1,-1,-1):
            tmp.append(tuple(A[:,i]))
        index = np.lexsort(tmp)
        B = np.copy(A)
        B = B[index,:]
    else:
        B = A
        index = np.array([0], dtype=int)
    
    return B, index



def get_list_irreps(N, num_irreps, n):
    # Generate the list of the <num_irreps> first irreps of SU(N), sorted by 
    # increasing number of the quadratic Casimir, and by taking the fundamental 
    # irrep to the power <n>.
    # 
    # Inputs:
    #   N           SU(N)
    #   num_irreps  number of irreps to generate
    #   n           generated irreps live in the tensor product (fundamental)^{\otimes n}
    # 
    # Outputs:
    #   alpha       numpy array of irreps. Each row is an irrep.
    #   casimir     numpy array, each element is the quadratic Casimir of the irreps
    # 
    # Remark:
    # CAUTION: That is a rather dirty implementation, were some arrays change 
    #          size in an uncontrolled way ! To be cleaned !
    # 
    
    
    if N==2:
        
        alpha = np.zeros((num_irreps, 2), dtype=int)
        alpha[:, 0] = np.arange(0, num_irreps)
        
        casimir = alpha[:,0]/2 * (alpha[:,0]/2 + 1)
        
    else:
        
        p = np.zeros((n+1,), dtype=int)
        p[0] = 1
        for i in range(1, n+1):
            j = int(1)
            k = int(1)
            s = int(0)
            while j>0:
                j = i - (3*k*k+k)//2
                if j>=0:
                    s -= ((-1)**k) * p[j]
                j = i - (3*k*k-k)//2
                if j>=0:
                    s -= ((-1)**k) * p[j]
                k += 1
            # end while
            p[i] = s
        # end for i
        p[0] = 0
        
        
        Forme = np.zeros((np.sum(p[1:]), n), dtype=int)
        Forme[0, 0] = 1
        Forme_tt_N = np.zeros((2, n), dtype=int)
        Forme_tt_N[1,0] = 1
        
        s = int(0)
        PI = np.pi**np.arange(0, n)
        index_N = int(2)
        
        for tt in range(2, n+1):
            print('tt = ', tt, '/', n)
            
            index_sum = np.sum(p[0:tt])
            
            for pp in range(s+1, s+p[tt-1]+1):
                
                if index_sum<Forme.shape[0]:
                    Forme[index_sum, :] = np.copy(Forme[pp-1, :]) + np.copy(Forme[0,:])
                elif index_sum==Forme.shape[0]:
                    # add a row to Forme
                    Forme = np.vstack([Forme, np.copy(Forme[pp-1, :]) + np.copy(Forme[0,:])])
                else:
                    sys.exit('Problem A')
                
                index_sum += 1
                
                W_k = np.argwhere(Forme[pp-1, :]>0).flatten() + 1
                N_W = len(W_k)
                
                for hh in range(1, N_W+1):
                    if index_sum+hh-1<Forme.shape[0]:
                        Forme[index_sum+hh-1, :] = np.copy(Forme[pp-1, :])
                        Forme[index_sum+hh-1, W_k[hh-1]-1] -= 1
                        Forme[index_sum+hh-1, W_k[hh-1]] += 1
                    elif index_sum+hh-1==Forme.shape[0]:
                        # add a row to Forme
                        tmp = np.copy(Forme[pp-1, :])
                        tmp[W_k[hh-1]-1] -= 1
                        tmp[W_k[hh-1]] += 1
                        Forme = np.vstack([Forme, tmp])
                    else:
                        sys.exit('Problem B')
                
                index_sum += N_W
            # end for pp
            
            s += p[tt-1]
            
            temp, _ = sortrows(Forme[s:index_sum, :])
            Forme[s:index_sum, :] = temp
            
            FormePrec = np.zeros((index_sum-s, n), dtype=int)
            FormePrec[1:, :] = np.copy(Forme[s:index_sum-1, :])
            
            Fdiff = np.argwhere( (Forme[s:index_sum, :] - FormePrec) @ PI ).flatten() + 1
            N_Fdiff = len(Fdiff)
            
            Forme[s:s+N_Fdiff, :] = np.copy(Forme[s+Fdiff-1, :])
            
            # That is dirty: remove rows to Forme --> change its shape
            Forme = np.delete(Forme, obj=np.arange(s+N_Fdiff, index_sum), axis=0)
            
            Forme_tt = np.copy(Forme[np.sum(p[0:tt]):np.sum(p[0:tt+1]), :])
            
            for q in range(1, p[tt]+1):
                if np.sum(Forme_tt[q-1, N-1:])==0:
                    index_N += 1
                    # add a row to Forme_tt_N
                    Forme_tt_N = np.vstack([Forme_tt_N, np.copy(Forme_tt[q-1, :])])
            # end for q
        # end for tt
        
        Number_N = index_N
        
        '''
        if Number_N<1.5*num_irreps:
            sys.exit('Problem: considering (fund irrep)^{\\otimes q} ' + 'for q=1, ..., {n}, we did not extract more than 3/2*num_irreps={3/2*num_irreps} different irreps of SU(N).')
        '''
        print('Generated ', Number_N, ' irreps.')
        
        alpha = Forme_tt_N @ np.tril(np.full(shape=(n, n), fill_value=1, dtype=int))
        alpha = alpha[:, 0:N]
        
        casimir = np.zeros((Number_N, ))
        
        for qq in range(0, Number_N):
            sumqq = np.sum(alpha[qq,:])
            sumqq2 = np.sum(alpha[qq,:]**2)
            casimir[qq] = sumqq * (N - sumqq/N) + sumqq2 - np.sum( Forme_tt_N[qq,:] * (np.arange(1, n+1)**2) )
        
        ind = np.argsort(casimir)
        alpha = alpha[ind, :]
        casimir = casimir[ind]
        # alpha = alpha[0:num_irreps, :]
        # casimir = casimir[0: num_irreps]
    # end if
    
    return alpha, casimir



def get_SYT_symm(alpha, m, order='LLOS'):
    # Compute all SYTs for the irrep alpha with m particles per site in the 
    # symmetric irrep with m boxes.
    # 
    # By default, in the resulting array, the SYTs are sorted in the ascending
    # order of the Last Letter Order Sequence, namely:
    #   Y[0,:] < Y[1,:] < Y[2,:] < ... < Y[-1,:]
    # 
    # This can be changed by setting order='iLLOS' in the input arguments.
    # 

    n = np.sum(alpha)
    Ns = n//m
    
    nly = len(np.argwhere(alpha>0).flatten())
    alpha = np.hstack([alpha[0:nly], 0])
    
    NY = multiplicity_symm(alpha, m)
    
    Y = np.full(shape=(10*NY, n), fill_value=-1, dtype=int)
    nbsyt = int(1)
    
    if m==2:
        
        for j in range(Ns-1, -1, -1):
            
            cpt = int(0)
            Ynew = np.full(shape=(0, n), fill_value=-1, dtype=int)
            
            for q in range(0, nbsyt):
                
                alphap = get_new_shape(alpha, Y[q, :])
                
                bc = get_bottom_corner(alphap)
                nbc = len(bc)
                cbc = alphap[bc]-1 # columns of bottom corners
                
                for b in range(nbc-1, -1, -1):
                    
                    alphapp = np.copy(alphap)
                    alphapp[bc[b]] -= 1
                    
                    for w2 in range(bc[b], -1, -1):
                        
                        if ((alphapp[w2]>alphapp[w2+1]) and (not (alphapp[w2]-1==cbc[b]))):
                            ytmp = np.copy(Y[q,:])
                            ytmp[m*j+1] = bc[b]
                            ytmp[m*j] = w2
                            Ynew = np.vstack([Ynew, ytmp])
                            cpt += 1
                        # end if
                    # end for w2
                # end for b
            # end for q
            
            Y[0:cpt, :] = np.copy(Ynew)
            nbsyt = cpt
            
        # end for j
        
    elif m==3:
        
        for j in range(Ns-1, -1, -1):
            
            cpt = int(0)
            Ynew = np.full(shape=(0, n), fill_value=-1, dtype=int)
            
            for q in range(0, nbsyt):
                
                alphap = get_new_shape(alpha, Y[q, :])
                
                bc = get_bottom_corner(alphap)
                nbc = len(bc)
                cbc = alphap[bc]-1 # columns of bottom corners
                
                for b in range(nbc-1, -1, -1):
                    
                    alphapp = np.copy(alphap)
                    alphapp[bc[b]] -= 1
                    
                    for w2 in range(bc[b], -1, -1):
                        
                        if ( (alphapp[w2]>alphapp[w2+1]) and (not alphapp[w2]-1==cbc[b]) ):
                            alphappp = np.copy(alphapp)
                            alphappp[w2] -= 1
                            for w3 in range(w2, -1, -1):
                                if ( (alphappp[w3]>alphappp[w3+1]) and (not (alphappp[w3]-1-cbc[b])*(alphappp[w3]-alphapp[w2])==0)):
                                    ytmp = np.copy(Y[q, :])
                                    ytmp[m*j+2] = bc[b]
                                    ytmp[m*j+1] = w2
                                    ytmp[m*j] = w3
                                    Ynew = np.vstack([Ynew, ytmp])
                                    cpt += 1
                                # end if
                            # end for w3
                        # end if
                    # end for w2
                # end for b
            # end for q
            
            Y[0:cpt, :] = np.copy(Ynew)
            nbsyt = cpt
            
        # end for j
        
    else:
        sys.exit('m>3 not implemented')
    
    if not nbsyt==NY:
        sys.exit('Problem: we have not generated the correct number of SYTs.')
    
    Y = Y[0:nbsyt,:]
    
    if order=='LLOS':
        Y = np.flipud(Y)
    
    return Y



def get_SYT_antisymm(alpha, m, order='LLOS'):
    # Compute all SYTs for the irrep alpha with m particles per site in the 
    # antisymmetric irrep with m boxes.
    # 
    # By default, in the resulting array, the SYTs are sorted in the ascending
    # order of the Last Letter Order Sequence, namely:
    #   Y[0,:] < Y[1,:] < Y[2,:] < ... < Y[-1,:]
    # 
    # This can be changed by setting order='iLLOS' in the input arguments.
    # 
    
    n = np.sum(alpha)
    Ns = n//m
    
    nly = len(np.argwhere(alpha>0).flatten())
    alpha = np.hstack([alpha[0:nly], 0])
    
    # NY = multiplicitiy_antisymm(alpha, m)
    NY = int(1e5)
    
    Y = np.full(shape=(10*NY, n), fill_value=-1, dtype=int)
    nbsyt = int(1)
    
    if m==2:
        
        for j in range(Ns-1, -1, -1):
            
            cpt = int(0)
            Ynew = np.full(shape=(0, n), fill_value=-1, dtype=int)
            
            for q in range(0, nbsyt):
                
                alphap = get_new_shape(alpha, Y[q, :])
                
                bc = get_bottom_corner(alphap)
                nbc = len(bc)
                
                for b in range(nbc-1, -1, -1):
                    
                    alphapp = np.copy(alphap)
                    alphapp[bc[b]] -= 1
                    
                    for w2 in range(bc[b]-1, -1, -1):
                        if (alphapp[w2]>alphapp[w2+1]):
                            ytmp = np.copy(Y[q,:])
                            ytmp[m*j+1] = bc[b]
                            ytmp[m*j] = w2
                            Ynew = np.vstack([Ynew, ytmp])
                            cpt += 1
                        # end if
                    # end for w2
                # end for b
            # end for q
            
            Y[0:cpt, :] = np.copy(Ynew)
            nbsyt = cpt
        
        # end for j
        
    elif m==3:
        
        for j in range(Ns-1, -1, -1):
            
            cpt = int(0)
            Ynew = np.full(shape=(0, n), fill_value=-1, dtype=int)
            
            for q in range(0, nbsyt):
                
                alphap = get_new_shape(alpha, Y[q, :])
                
                bc = get_bottom_corner(alphap)
                nbc = len(bc)
                
                for b in range(nbc-1, -1, -1):
                    
                    alphapp = np.copy(alphap)
                    alphapp[bc[b]] -= 1
                    
                    for w2 in range(bc[b]-1, -1, -1):
                        
                        if (alphapp[w2]>alphapp[w2+1]):
                            
                            alphappp = np.copy(alphapp)
                            alphappp[w2] -= 1
                            
                            for w3 in range(w2-1, -1, -1):
                                
                                if (alphappp[w3]>alphappp[w3+1]):
                                    ytmp = np.copy(Y[q, :])
                                    ytmp[m*j+2] = bc[b]
                                    ytmp[m*j+1] = w2
                                    ytmp[m*j] = w3
                                    Ynew = np.vstack([Ynew, ytmp])
                                    cpt += 1
                                # end if
                            # end for w3
                        # end if
                    # end for w2
                # end for b
            # end for q
            
            Y[0:cpt, :] = np.copy(Ynew)
            nbsyt = cpt
            
        # end for j
        
    else:
        sys.exit('m>3 not yet implemented')
    
    Y = Y[0:nbsyt,:]
    
    if order=='LLOS':
        Y = np.flipud(Y)
    
    return Y



def get_SYT_general(alpha, beta, order='LLOS'):
    """
    SYTs of equivalence classes for general local constraints
    
    Parameters
    ----------
    alpha : numpy array
        global irrep
    beta : numpy array of numpy arrays
        local irreps
    order : str - [default] LLOS or iLLOS
        order in which to arrange the SYTs
    
    Returns
    -------
    Y : numpy array
        SYTs of equivalence classes
    CY: numpy array
        column positions of SYTs of equivalence classes
    
    Description
    -----------
    Generate all SYTs of equivalence classes associated to the global irrep 
    alpha, satisfying the local constraints imposed by irreps in beta
    """
    
    Ns = beta.shape[0]
    n = np.sum(alpha)
    
    nly = len(np.argwhere(alpha>0).flatten())
    alpha = np.hstack([alpha[0:nly], 0])
    
    Y = np.full(shape=(1, n), fill_value=-1, dtype=int)
    CY = np.full(shape=(1, n), fill_value=-1, dtype=int)
    nbsyt = int(1)
    
    for j in range(Ns-1, -1, -1):
        
        cpt = int(0)
        Ynew = np.full(shape=(0, n), fill_value=-1, dtype=int)
        CYnew = np.full(shape=(0, n), fill_value=-1, dtype=int)
        
        for q in range(0, nbsyt):
            
            alphap = get_new_shape(alpha, Y[q,:])
            
            bc = get_bottom_corner(alphap)
            nbc = len(bc)
            cbc = alphap[bc]-1 # columns of bottom corners
            
            for b in range(nbc-1, -1, -1):
                
                w1 = bc[b] # row position of bottom corner
                cw1 = cbc[b] # column position of bottom corner
                
                if np.sum(beta[j])==1:
                    cpt += 1
                    pos = n - np.sum(np.sum(beta[j+1:,:])) - 1
                    Ynew[cpt, :] = Y[q, :]
                    Ynew[cpt, pos] = w1
                    CYnew[cpt, :] = CY[q, :]
                    CYnew[cpt, pos] = cw1
                else:
                    
                    alphapp = np.copy(alphap)
                    alphapp[w1] -= 1
                    
                    Ynew, CYnew, cpt = place_recursive(
                                            int(1), 
                                            beta[j], 
                                            alphapp, 
                                            np.array([w1], dtype=int), 
                                            np.array([cw1], dtype=int), 
                                            Y[q,:], 
                                            CY[q,:], 
                                            Ynew, 
                                            CYnew, 
                                            cpt)
        
        Y = np.copy(Ynew)
        CY = np.copy(CYnew)
        nbsyt = cpt
    
    if order=='LLOS':
        Y = np.flipud(Y)
        CY = np.flipud(CY)
    
    return Y, CY



def place_recursive(k, beta_loc, alphap, wvec, cwvec, y, cy, Ynew, CYnew, cpt):
    """
    Recursive placement satisfying local constraints
    
    Parameters
    ----------
    k : int
        step
    beta_loc : numpy array
        local irrep
    alphap : numpy array
        remaining global irrep
    wvec : numpy array
        rows of already placed particles
    cwvec : numpy array
        columns of already placed particles
    y : numpy array
        subSYT under consideration
    cy : numpy array
        columns positions associated to y
    Ynew : numpy array
        all SYTs generated so far
    CYnew : numpy array
        column positions associated with Ynew
    cpt : counter
    
    Returns
    -------
    Ynew : numpy array
        SYTs of equivalence classes generated so  far
    CYnew : numpy array
        column positions associated with Ynew
    cpt : int
        counter
    
    Description
    -----------
    Try to place the k-th particle of the local irrep beta_loc in the remaining
    global irrep alphap in the (yet incomplete) SYTs in Ynew which satisfy the
    pattern from (incomplete) SYT y
    """
    
    m_loc = np.sum(beta_loc)
    
    nr = len(np.argwhere(beta_loc>0)) # number of rows in the shape
    nc = beta_loc[0] # number of columns in the shape
        
    for w in range(wvec[0], -1, -1):
        
        if alphap[w]>alphap[w+1]:
            # bottom corner in row w
            cw = alphap[w] - 1
            isOk = True
            nw = len(np.argwhere(wvec==w)) # nb of part already placed in row w
            if (nw+1>nc):
                isOk = False
            else:
                cnw = len(np.argwhere(cwvec==cw)) # nb of part already placed in column cw
                if (cnw+1>nr):
                    isOk = False
            
            if isOk:
                wvecp = np.concatenate(([w], wvec))
                cwvecp = np.concatenate(([cw], cwvec))
                
                if k==m_loc-1:
                    Ynew = np.vstack([Ynew, y])
                    CYnew = np.vstack([CYnew, cy])
                    pos = np.max(np.argwhere(y<0).flatten())                    
                    for ll in range(m_loc-1, -1, -1):
                        Ynew[cpt, pos] = wvecp[ll]
                        CYnew[cpt, pos] = cwvecp[ll]
                        pos -= 1
                    cpt += 1
                
                else:
                    alphapp = np.copy(alphap)
                    alphapp[w] -= 1
                    Ynew, CYnew, cpt = place_recursive(k+1, 
                                                beta_loc, 
                                                alphapp, 
                                                wvecp, 
                                                cwvecp, 
                                                y, 
                                                cy, 
                                                Ynew, 
                                                CYnew, 
                                                cpt)
    
    return Ynew, CYnew, cpt



def permutation_to_cycles(sigma):
    """
    Transform a permutation of S_n into a product of disjoint cycles
    """
    
    n = len(sigma)
    
    cycles = []
    used = []
    
    f = lambda v : min(set(np.arange(0, n)) - set(v))
    
    while not len(used)==n:
        s = f(used)
        if sigma[s]==s:
            used.append(s)
        else:
            cycle = np.array([s, sigma[s]], dtype=int)
            used.append(s)
            used.append(sigma[s])
            while not sigma[cycle[-1]]==cycle[0]:
                cycle = np.append(cycle, sigma[cycle[-1]])
                used.append(cycle[-1])
            cycles.append(cycle)
    return cycles



def cycle_to_transpositions(cycle):
    """
    Transform a cycle nto a product of 2-cycles (transpositions)
    
    Parameters
    ----------
    cycle : numpy array
        cycle of S_n
    
    Returns
    -------
    t : list of numpy arrays
        product of 2-cycles
    """
    
    t = [None] * (len(cycle)-1)
    
    for i in range(0, len(cycle)-1):
        a = cycle[i]
        b = cycle[i+1]
        t[i] = np.array([min(a, b), max(a, b)])
    
    return t



def transposition_to_adjacent_transpositions(t):
    """
    Transform an arbitrary 2-cycle (transposition) into a product of adjacent 
    2-cycles (transpositions)
    
    Parameters
    ----------
    t : numpy array
        2-cycle of S_n
    
    Returns
    -------
    z : numpy array
        product of adjacent 2-cycles (adjacent transpositions)
    
    Example
    -------
    t = (2, 5)
    z = (2, 3)(3, 4)(4, 5)(3, 4)(2, 3) --> z=np.array([2, 3, 4, 3, 2])
    """
    
    a = np.min(t)
    b = np.max(t)
    
    n = 2*abs(b - a) - 1
    
    z = np.zeros(shape=(n, ), dtype=int)
    z[0:((n-1)/2+1).astype(int)] = np.arange(a, b)
    z[((n-1)/2+1).astype(int):] = np.arange(b-2, a-1, -1)
    
    return z



def permutation_to_adjacent_transpositions_v1(sigma):
    """
    Transform a permutation into a product of adjacent 2-cycles (transpositions)
    
    Remark
    ------
    First method
    """
    
    # Method A:
    cycles = permutation_to_cycles(sigma)
    
    transpositions = []
    for cycle in cycles:
        transpositions += cycle_to_transpositions(cycle)
    
    at = np.zeros(shape=(0, ), dtype=int)
    
    for t in transpositions:
        z = transposition_to_adjacent_transpositions(t)
        at = np.append(at, z)
    
    # remove potential even number multiplying adjacent 2-cycles, such as (2 3)(2 3)=id
    atr = reduce_adjacent_transpositions(at)
    
    # Method B:
    
    
    return atr



def reduce_adjacent_transpositions(at):
    """
    Remove even numbers of following identical adjacent transposition in a 
    product of adjacent transpositions
    
    Description
    -----------
    If a product of adjacent transpositions contains identical elements multiplying
    each other, replace them by the identity
    
    Example
    -------
    sigma = {3, 2, 0, 1} (permutation) = (0, 3, 1, 2) = cycle = 
                                       = (0, 3)(1, 3)(1, 2)
                                       = (0,1)(1,2)(2,3)(1,2)(0,1)(1,2)(2,3)(1,2)(1,2)
                                                                              |____|
                                                                                =id
                                       = (0,1)(1,2)(2,3)(1,2)(0,1)(1,2)(2,3)
    at = [0 1 2 1 0 1 2 1 1] ---> atr = [0 1 2 1 0 1 2]
    """
    
    atr = np.zeros(shape=(0,), dtype=int)
    if len(at)<=1:
        atr = at
    else:
        i = int(0)
        while i<len(at):
            cpt = int(1)
            j = i+1
            while (j<len(at)):
                if at[j]==at[i]:
                    j += 1
                    cpt += 1
                else:
                    break
            if cpt%2==1:
                atr = np.append(atr, at[i])
            i = j
    
    return atr



def permutation_to_transpositions(sigma):
    """
    Transform a permutation of S_n into a product of transpositions (2-cycles)
    
    Parameters
    ----------
    sigma : numpy array
        permutation of S_n, as a permutation of [0, 1, ..., n-1]
    
    Returns
    -------
    tau : numpy array
        transpositions
    
    Description
    -----------
    Decompose a permutation of S_n (such as (3, 2, 0, 5, 4, 1) \in S_6) as a 
    product of 2-cycles (transpositions).
    
    Example
    -------
    sigma = (3, 2, 5, 0, 1, 4, 6, 8, 7) \in S_9
    ---> sigma = (0, 3)(1, 2)(2, 5)(4, 5)(7, 8)
    """
    
    n = len(sigma)
    cpt = 0
    
    tau = np.zeros(shape=(n, 2), dtype=int)
    
    for i in range(0, n):
        if not sigma[i]==i:
            tau[cpt] = [i, sigma[i]]
            j = np.argwhere(sigma==i).flatten()[0]
            sigma[j] = sigma[i]
            sigma[i] = i
            cpt += 1
    
    tau = tau[0:cpt,:]
    
    return tau



def permutation_to_adjacent_transpositions_v2(sigma):
    """
    Transform a permutation into a product of adjacent 2-cycles (transpositions)
    
    Remark
    ------
    Second method
    """
    
    transpo = permutation_to_transpositions(sigma)
    
    at = np.zeros(shape=(0,), dtype=int)
    
    for t in transpo:
        z = transposition_to_adjacent_transpositions(t)
        at = np.append(at, z)
    
    atr = reduce_adjacent_transpositions(at)
    
    return atr



def permutation_to_adjacent_transpositions(sigma):
    """
    Transform a permutation into a product of adjacent 2-cycles (transpositions)
    
    Remark
    ------
    Provides the best decomposition between the first and second method
    """
    at1 = permutation_to_adjacent_transpositions_v1(sigma)
    at2 = permutation_to_adjacent_transpositions_v2(sigma)
    
    if len(at1)<len(at2):
        atr = at1
    else:
        atr = at2
    
    return atr



class OrthogonalUnits:
    
    def __init__(self, alpha, only00='True'):
        self.alpha = alpha
        
        # build all SYTs associated to the irrep alpha
        Y = get_SYT(self.alpha)
        self.falpha = Y.shape[0]
        CY = get_column(Y)
        
        # construct matrices of adjacent transpositions
        P = get_adjacent_transposition_matrices(alpha, Y, CY)
        self.P = P
        
        # get all permutations of S_n
        n = np.sum(alpha)
        ps = itertools.permutations(np.arange(0, n, dtype=int))
        self.nn = math.factorial(n)
        
        # compute the matrices of the inverse permutations, obtained by reading
        # the sequence of adjacent transpositions in reverse order
        ps = itertools.permutations(np.arange(0, n, dtype=int))
        self.permutations = []
        self.sigmaMat = [scipy.sparse.eye(self.falpha)] * self.nn
        self.sigmaMatinverse = [scipy.sparse.eye(self.falpha).tocsr()] * self.nn
        self.adjaTranspo = [None] * self.nn
        for i, sigma in enumerate(ps):
            self.permutations.append(sigma)
            at = permutation_to_adjacent_transpositions(np.array(sigma, dtype=int))
            self.adjaTranspo[i] = at
            for j in range(0, len(at)):
                self.sigmaMat[i] = self.sigmaMat[i] @ P[at[j]]
                self.sigmaMatinverse[i] = P[at[j]] @ self.sigmaMatinverse[i]
        
        if only00==True:
            self.coeffs = np.zeros(shape=(1, 1, self.nn))
            for i in range(self.nn):
                self.coeffs[0][0][i] = self.sigmaMatinverse[i][0,0]
        else:
            self.coeffs = np.zeros(shape=(self.falpha, self.falpha, self.nn))
            for r in range(self.falpha):
                for s in range(self.falpha):
                    for i in range(self.nn):
                        self.coeffs[r][s][i] = self.sigmaMatinverse[i][s,r]
        self.coeffs *= self.falpha/self.nn



def get_projector(beta_loc, y, cy, particles):
    """
    
    """
    
    particles = np.sort(particles)
    npart = len(particles)
    
    # get the (0, 0) orthogonal unit
    ortho = OrthogonalUnits(beta_loc, only00='True')
    
    # identify locations of boxes associated to the particles
    alphaM, alphaB, alphaP, offset = get_subshape(y, cy, particles)
    
    # get associated SYTs
    YM = get_subSYT(alphaM, alphaB, order='LLOS')
    YMc = fill_subSYT(YM, alphaM, fill_type='smallest')
    CYMc = get_column(YMc)
    NYMc = YMc.shape[0]
    
    # get matrices of adjacent transpositions of these boxes
    particles_shifted = particles - particles[0] + np.sum(alphaB)
    P = [None] * (npart -  1)
    for i in range(0, npart-1):
        P[i] = get_adjacent_transposition_matrix(alphaM, YMc, CYMc, k=particles_shifted[i])
        
    # build matrix of projector
    Proj = scipy.sparse.csr_matrix((NYMc, NYMc))
    for i in range(ortho.nn):
        Permuti = get_matrix_permutation(P, ortho.adjaTranspo[i])
        Proj += ortho.coeffs[0,0][i] * Permuti
    
    return Proj, YMc, CYMc, alphaM, alphaB, alphaP, offset



def sg_null_space(A, rcond=None):
    u, s, vh = scipy.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q



class LocalStates:
    
    def __init__(self, ec, site, y, cy, beta):
        """
        ec: equivalence class
        site: site
        y : SYT of equivalence class
        cy : column positions
        beta : all local irreps
        """
        
        self.site = np.array([site], dtype=int)
        self.ec = ec
        self.y = np.copy(y)
        self.cy = np.copy(cy)
        beta_loc = np.copy(beta[site])
        self.particles = np.sum(np.sum(beta[0:site])) + np.arange(0, np.sum(beta_loc))
        
        self.coeffs, self.Ydev, self.CYdev, self.alphaM, self.alphaB, self.alphaP, self.offset =  get_local_states(y, cy, beta, site)
        # recall alphaP + alphaB = alphaM
        
        # number of states  !!! CAUTION !!!
        self.n = self.coeffs.shape[1]
        
        return
    



def get_local_states(y, cy, beta, site):
    """
    
    !!! IMPORTANT: states are stored in column in the output <coeffs> !!!
    
    """
    
    beta_loc = beta[site]
    
    particles = np.sum(np.sum(beta[0:site])) + np.arange(0, np.sum(beta_loc))
    
    Proj, YProj, CYProj, alphaM, alphaB, alphaP, offset = get_projector(beta_loc, y, cy, particles)
    n = Proj.shape[0]
    
    if n==1:
        if abs(Proj[0,0]-1)<1.0e-13:
            V = np.array([1.0], dtype=float)
        else:
            V = np.array(shape=(0,), dtype=float)
    else:
        
        #ns = scipy.linalg.null_space( (Proj - scipy.sparse.eye(n)).toarray(), overwrite_a=False )
        ns = sg_null_space( (Proj - scipy.sparse.eye(n)).toarray() )
        
        d = ns.shape[1] # dimension of null space
        
        #---------------------
        # CAUTION : vectors spanning the null space are stored in the COLUMNS !!!
        #---------------------
        
        if d>0:
            V = ns
            for i in range(d):
                if V[0,i]<0:
                    V[:,i] *= (-1.0)
        else:
            V = np.array(shape=(0,), dtype=float)
    
    coeffs = V
    
    return coeffs, YProj, CYProj, alphaM, alphaB, alphaP, offset



class GeneralBasis:
    
    def __init__(self, alpha, beta, N):
        """
        Generate basis of states as tensor product of local states on each site
        for each equivalence class
        
        Parameters
        ----------
        alpha : numpy array
            global irrep
        beta : numpy array
            collection of local irreps on each site
        N : int
            SU(N)
        """
        
        print('==========================')
        print('Generating basis')
        print('==========================')
        
        self.alpha = alpha
        self.beta = beta
        self.N = N
        self.Ns = beta.shape[0]
        
        self.Y, self.CY = get_SYT_general(self.alpha, self.beta)
        self.NY = self.Y.shape[0]
        
        # TO DO: To be improved in the future. Compute orthogonal units for the 
        # different irreps appearing in beta
        #beta_loc = beta[0] # here assume same irrep on each site # UNUSED
        #self.orthoUnit = OrthogonalUnits(beta_loc, only00='True') # UNUSED
        
        #self.local_states = [[None] * self.Ns] * self.NY
        self.local_states = []
        self.dimB = np.zeros(shape=(self.Ns, self.NY), dtype=int)
        
        si = int(0)
        
        for site in range(self.Ns):
            for i in range(self.NY):
                self.local_states.append(LocalStates(i, site, self.Y[i], self.CY[i], beta))
                if self.local_states[si].coeffs.shape[0]>0:
                    self.dimB[site, i] = self.local_states[si].coeffs.shape[1]
                si += 1
        
        self.statesPerClass = np.prod(self.dimB, axis=0)
        self.NH = np.sum( self.statesPerClass ) # total Hilbert space dimension
        
        if not (self.NH==multiplicity_irrep_mixed(self.alpha, self.beta, self.N)):
            sys.exit('Problem: Number of states does not match multiplicity.')
        
        self.ind = np.argwhere( self.statesPerClass>0 ).flatten()
        if len(self.ind)!=self.NY:
            sys.exit('There is ', self.NY-len(self.ind), ' equivalence classes leading to 0 state.')
        
        return
    
    
    
    def __get_index_sec(self, ec, site):
        
        si = site*self.NY + ec
        
        return si
    
    
    
    def get_states_of_class(self, ec, sites):
        """
        Compute all states corresponding to a given equivalence class, as a 
        development on multiple sites
        
        Parameters
        ----------
        ec : int
            index of equivalence class
        sites : numpy array
            sites on which to compute the states
        
        Returns
        -------
        out : LocalStates
            states
        
        Remark
        ------
        This corresponds to performing a kronecker product of local states to 
        generate a collection of states living on several sites
        """
        
        site1 = np.min(sites)
        site2 = np.max(sites)
        
        si = self.__get_index_sec(ec, site2)
        out = copy.deepcopy(self.local_states[si])

        mtot = np.sum(self.beta[site2])
        
        for site in range(site2-1, site1-1, -1):
            
            si = self.__get_index_sec(ec, site)
            local_states_to_add = copy.deepcopy(self.local_states[si])
            m_to_add = np.sum(self.beta[site])
            
            diffoffset = out.offset - local_states_to_add.offset
            
            # diffoffset > 0 ==> <site>+1 particles start at a row BELOW <site> particles
            # diffoffset < 0 ==> <site>+1 particles start at a row ABOVE <site> particles
            # diffoffset = 0 ==> <site>+1 particles start at the same row as <site> particles
            
            if (diffoffset==0):
                off1 = 0
                off2 = 0
            elif (diffoffset>0):
                off1 = 0
                off2 = diffoffset
            else:
                off1 = -diffoffset
                off2 = 0
            
            # number of SYTs in total development so far
            q = out.Ydev.shape[0]
            
            # number of SYTs in the developments at <site>
            qadd = local_states_to_add.Ydev.shape[0]
            
            # "kronecker cross" SYTs (merging)
            A = np.repeat(local_states_to_add.Ydev[:,-m_to_add:] + off1, repeats=q, axis=0)
            B = np.matlib.repmat(out.Ydev[:,-mtot:] + off2, qadd, 1)
            out.Ydev = np.concatenate((A, B), axis=1)
            
            # compute coefficients
            out.coeffs = np.kron(local_states_to_add.coeffs, out.coeffs)
            
            # update attributes
            out.offset = min(out.offset, local_states_to_add.offset)
            out.particles = np.sort( np.concatenate( (out.particles, local_states_to_add.particles) ) )
            out.site = np.sort( np.concatenate( (out.site, np.array([site]) ) ) )
            out.n *= local_states_to_add.n
            
            mtot += m_to_add
        
        # update shapes
        out.alphaM, out.alphaB, out.alphaP, out.offset = get_subshape(out.y, out.cy, out.particles)
        
        return out
    
    
    
    def find_index_state(self, ec, site1, site2):
        """
        Find the indices in the global basis of states of a given equivalence class
        for a set of sites
        
        Returns
        -------
        index : numpy array
            array of dimension n x U
            n = number of states
            U = number of duplicates
            
            the i-th state of the equivalence class must be duplicated at 
            positions index[i,:]
        """
        
        # Recall that
        #   self.Basis.dimB : self.Basis.Ns x self.Basis.NY : number of states at (site, eqclass)
        # 
        #   self.Basis.statesPerClass : self.Basis.NY : number of states for each equivalence class
        # 
        #   self.Basis.NH : int : total Hilbert space dimension
        
        
        # distribution of states among the different sites in the class
        statesSpEq = self.dimB[:, ec]
        
        # number of states generated by the input sites
        NbStates = np.prod(statesSpEq[site1:site2+1])
        
        # number of states BEFORE input equivalence class
        offsetClass = np.sum(self.statesPerClass[:ec])
        
        # number of states in the input equivalence class
        statesEqcl = self.statesPerClass[ec]
        
        # number of times duplication will occur
        nDublicates = statesEqcl // NbStates
        
        if site1>0:
            mL = np.prod(statesSpEq[:site1])
        else:
            mL = int(1)
        
        if site2==self.Ns:
            mR = int(1)
        else:
            mR = np.prod(statesSpEq[site2+1:])
        
        assert mL*mR == nDublicates
        
        if site1+1==site2:
            
            m1 = statesSpEq[site1]
            m2 = statesSpEq[site2]
            
            # Duplicate because of offset Right
            index1 = np.arange(offsetClass, offsetClass+mR)
            l1 = len(index1)
            
            # Duplicate because of site2
            index2 = np.zeros(shape=(m2, l1), dtype=int)
            index2[0, :] = index1
            
            for i2 in range(1, m2):
                index2[i2, :] = index2[i2-1, :] + mR
    
            # Duplicate because of site1
            index3 = np.zeros(shape=(m2*m1, l1), dtype=int)
            index3[:m2, :] = index2
    
            for i1 in range(1, m1):
                index3[i1*m2:(i1+1)*m2, :] = index3[(i1-1)*m2:i1*m2, :] + m2*mR
            
            
            # Duplicate because of offset Left
            index4 = np.zeros(shape=(m2*m1, l1*mL), dtype=int)
            
            assert mR == index3.shape[1]
            
            index4[:,:mR] = index3
    
            for iL in range(1, mL):
                index4[:,iL*mR:(iL+1)*mR] = index4[:, (iL-1)*mR:iL*mR] + m1*m2*mR
            
            index = index4
            
        else:
            
            mS = np.prod(statesSpEq[site1:site2+1])
            assert mS==NbStates
            # thus mS should not be recomputed
            
            # duplicate because of offset Right
            index1 = np.arange(offsetClass, offsetClass+mR)
            l1 = len(index1)
            
            # duplicate because of site1--site2
            index2 = np.zeros(shape=(mS, l1), dtype=int)
            
            index2[0, :] = index1
            for i2 in range(1, mS):
                index2[i2, :] = index2[i2-1, :] + mR
            
            # Duplicate because of offset Left
            index3 = np.zeros(shape=(mS, l1*mL), dtype=int)
            
            index3[:, :mR] = index2
            for iL in range(1, mL):
                index3[:, iL*mR:(iL+1)*mR] = index3[:,(iL-1)*mR:iL*mR] + mS*mR
            
            index = index3
        
        return index



class SUNGeneral:
    
    def __init__(self, alpha, beta, N, lattice):
        
        self.N = N
        self.Ns = beta.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.lattice = lattice
        
        self.Basis = GeneralBasis(alpha, beta, N)
        
        return
    
    
    
    def __check_conditions_equivalence_class(self, ec1, ec2, particles1, particles2):
        """
        Check if 2 equivalence classes are compatible
        
        Returns
        -------
        iseq : Bool
            True if ec1 and ec2 are compatible
        """
        
        y1 = self.Basis.Y[ec1]
        cy1 = self.Basis.CY[ec1]
        y2 = self.Basis.Y[ec2]
        cy2 = self.Basis.CY[ec2]
        
        p1 = np.min(particles1)
        p2 = np.max(particles2)
        n = self.Basis.Y.shape[1]
        
        iseq = True
        
        # all particles before p1 must be identically placed
        i = int(1)
        while (i<p1):
            if not ( (y1[i]==y2[i]) & (cy1[i]==cy2[i]) ):
                return False
            i += 1
        
        # all particles after p2 must be identically placed
        i = p2 + 1
        while (i<n):                
            if not ( (y1[i]==y2[i]) & (cy1[i]==cy2[i]) ):
                return False
            i += 1   
        '''
        # vectorize the above - not really faster
        iseq2 = True
        ylowdiff = np.sum(abs(y1[:p1] - y2[:p1]))
        if not ylowdiff==0:
            iseq2 = False
            return iseq
        yhidiff = np.sum(abs(y1[p2+1:] - y2[p2+1:]))
        if not yhidiff==0:
            iseq2 = False
            return iseq
        if iseq2==False:
            sys.exit('Problem: we should not have entered here')
        '''
        # check that there is at most 1 interchange
        v1 = np.vstack((y1[particles1], cy1[particles1])).T
        v2 = np.vstack((y2[particles1], cy2[particles1])).T
        
        cpt = int(0)
        for j in range(0, len(particles1)):
            ind = np.argwhere( np.sum(abs( v2 - v1[j, :] ), axis=1)==0 ).flatten()
            if len(ind)==0:
                cpt += 1
        
        if cpt>1:
            iseq = False
        
        return iseq
    
    
    
    def __get_sister_equivalence_class(self, ec1, particles1, particles2):
        """
        Determine the relevant equivalence classes for the <bra|
        
        Returns
        -------
        ind_ec2 : numpy array
            indices of compatible equivalence classes
        """
        
        ind_ec2 = np.array([ec1], dtype=int)
        
        # one should parallelize this loop, taking care of the creation (append) of ind_ec2
        for ec2 in range(ec1+1, self.Basis.NY):
            iseq = self.__check_conditions_equivalence_class(ec1, ec2, particles1, particles2)
            if iseq:
                ind_ec2 = np.hstack((ind_ec2, ec2))
        
        return ind_ec2
    
    
    
    def sun_hamiltonian(self):
        
        H = scipy.sparse.csr_matrix((self.Basis.NH, self.Basis.NH))
        
        for link in self.lattice.links:
            
            site1 = link[0]
            site2 = link[1]

            print('==========================')
            print('link = ', site1, ' -- ', site2)
            print('==========================')
            
            m1 = np.sum(self.beta[site1])
            m2 = np.sum(self.beta[site2])
            
            particles1 = np.sum(self.beta[0:site1]) + np.arange(m1)
            particles2 = np.sum(self.beta[0:site2]) + np.arange(m2)
            
            # all particles from site1 to site2
            allparticles = np.sum(self.beta[0:site1]) + np.arange(np.sum(self.beta[site1:site2+1]))
            
            Hbond = scipy.sparse.csr_matrix((self.Basis.NH, self.Basis.NH))
            
            for ec1 in range(self.Basis.NY):
                # generates the states living on all sites [site1, ... , site2]
                states_class1 = self.Basis.get_states_of_class(ec=ec1, sites=link)
                states_class1.Ydev = fill_subSYT(states_class1.Ydev, states_class1.alphaM, fill_type='largest')
                
                # generate all SYTs associated to [site1, ..., site2], in LLOS order ---> Yall_ec1_ordered
                Yall_ec1_ordered = get_subSYT(states_class1.alphaM, states_class1.alphaB, order='iLLOS')
                Yall_ec1_ordered = fill_subSYT(Yall_ec1_ordered, states_class1.alphaM, fill_type='largest')
                CYall_ec1_ordered = get_column(Yall_ec1_ordered)
                
                # create a new matrix which will contain all coefficients expanded in the new ordered basis
                # this matrix is of dimension: #SYTs x #states
                coeffs1 = np.zeros(shape=(Yall_ec1_ordered.shape[0], states_class1.n), dtype=float)
                
                # re-index components of states living on all sites [site1, ..., site2]
                # onto the full local basis
                index_map1 = np.zeros(states_class1.Ydev.shape[0], dtype=int)
                for i in range(0, states_class1.Ydev.shape[0]):
                    ind = np.argwhere( np.sum(abs(Yall_ec1_ordered - states_class1.Ydev[i]), axis=1) < 1e-13).flatten()
                    assert len(ind)==1
                    index_map1[i] = ind[0]       
                coeffs1[index_map1, :] = states_class1.coeffs
                
                # generate all matrices of adjacent transpositions from <Yall_ec1_ordered>
                P1 = []
                for k in range(np.sum(states_class1.alphaB), np.sum(states_class1.alphaM)-1):
                    P1.append( get_adjacent_transposition_matrix(states_class1.alphaM, 
                                                                 Yall_ec1_ordered, 
                                                                 CYall_ec1_ordered, 
                                                                 k) )
                
                # compute all transpositions between site1 and site2
                perms = np.stack( (np.repeat(particles1, repeats=len(particles2)), 
                                   np.matlib.repmat(particles2, 1, len(particles1)).flatten()), axis=1 )
                
                # Compute shift between global numbering (particles1, particles2)
                # and state-local numbering
                shift = -np.min(allparticles) + np.sum(states_class1.alphaB)
                assert np.max(allparticles)+shift == np.sum(states_class1.alphaM)-1
                perms += shift
                
                # Generate the matrix Hint representing the sum of all transpositions between
                # particles of site1 and those of site2
                #Hint = np.zeros(shape=P1[0].shape, dtype=float)
                Hint = scipy.sparse.csr_matrix(P1[0].shape)
                
                for perm in perms:
                    # decompose permutation into product of adjacent transpositions
                    atr = transposition_to_adjacent_transpositions(perm)
                    # compute matrix of permutation
                    Hp = get_matrix_permutation(P1, atr-np.sum(states_class1.alphaB))
                    #Hint += Hp.todense()
                    Hint += Hp
                
                # Apply operator H0 on all states belonging to [site1, ..., site2]
                Hcoeffs1 = Hint @ coeffs1
                
                # Determine all relevant equivalence classes for the <bra|
                ind_ec2 = self.__get_sister_equivalence_class(ec1, particles1, particles2)
                
                # Iterate over <bra| equivalence class
                for ec2 in ind_ec2:
                    
                    if ec2==ec1:
                        states_class2 = states_class1
                    else:
                        # generates the states living on all sites [site1, ... , site2] for the 2nd equiv class
                        states_class2 = self.Basis.get_states_of_class(ec=ec2, sites=link)
                        states_class2.Ydev = fill_subSYT(states_class2.Ydev, states_class2.alphaM, fill_type='largest')
                    
                    '''
                    # generate all SYTs associated to [site1, ..., site2], in LLOS order ---> Yall_ec2_ordered
                    Yall_ec2_ordered = get_subSYT(states_class2.alphaM, states_class2.alphaB, order='iLLOS')
                    Yall_ec2_ordered = fill_subSYT(Yall_ec2_ordered, states_class2.alphaM, fill_type='largest')
                    
                    assert np.sum(abs(Yall_ec1_ordered-Yall_ec2_ordered))==0
                    # as a consequence, we could save some time by not computing 
                    # Yall_ec2_ordered
                    '''
                    Yall_ec2_ordered = Yall_ec1_ordered
                    
                    
                    # create a new matrix which will contain all coefficients again
                    # this matrix is of dimension: #SYTs x #states
                    
                    if ec2==ec1:
                        coeffs2 = coeffs1
                    else:
                        coeffs2 = np.zeros(shape=(Yall_ec2_ordered.shape[0], states_class2.n), dtype=float)
                        # re-index components of states living on all sites [site1, ..., site2]
                        # onto the full local basis              
                        index_map2 = np.zeros(states_class2.Ydev.shape[0], dtype=int)
                        for i in range(0, states_class2.Ydev.shape[0]):
                            ind = np.argwhere( np.sum(abs(Yall_ec2_ordered - states_class2.Ydev[i]), axis=1) < 1e-13).flatten()
                            assert len(ind)==1
                            index_map2[i] = ind[0]
                        coeffs2[index_map2, :] = states_class2.coeffs
                    
                    # build local interaction Hamiltonian
                    Hloc = coeffs2.T @ Hcoeffs1
                    # Hloc is of dimension states_class2.n x states_class1.n
                    
                    # duplicate and embedd elements in interaction Hamiltonian in full basis
                    ind_st_1 = self.Basis.find_index_state(ec1, site1, site2)
                    ind_st_2 = self.Basis.find_index_state(ec2, site1, site2)
                    
                    # number of duplicates
                    nD = ind_st_1.shape[1]
                    
                    n1 = coeffs1.shape[1]
                    n2 = coeffs2.shape[1]
                    
                    assert nD==ind_st_2.shape[1]                    
                    
                    #---------------------------------------
                    # duplicate elements
                    
                    '''
                    #---------------------------------------                
                    # METHOD 1 - for loops
                    #---------------------------------------                
                    HTemp = scipy.sparse.csr_matrix((self.Basis.NH, self.Basis.NH))
                    
                    for row in range(0, n2):
                        for col in range(0, n1):
                            newrows = ind_st_2[row]
                            newcols = ind_st_1[col]
                            
                            assert len(newrows)==nD
                            assert len(newcols)==nD
                            
                            for i in range(0, nD):
                                rowi = newrows[i]
                                coli = newcols[i]
                                HTemp[rowi, coli] = Hloc[row, col]
                                if not ec1==ec2:
                                    HTemp[coli, rowi] = Hloc[row, col]
                    #---------------------------------------                
                    '''
                    
                    #---------------------------------------                
                    # METHOD 2 - vectorized
                    #---------------------------------------                
                    rows = np.matlib.repmat(ind_st_2, 1, n1)
                    rows = np.reshape(rows, (nD*n1*n2, ))
                    cols = np.matlib.repmat(ind_st_1, n2, 1)
                    cols = np.reshape(cols, (nD*n1*n2, ))
                    vals = np.repeat(Hloc, repeats=nD)
                    HTemp = scipy.sparse.csr_matrix(
                             (vals, (rows, cols)),
                             shape=(self.Basis.NH, self.Basis.NH))
                    if not ec1==ec2:
                        HTemp += HTemp.T
                    
                    #---------------------------------------
                    '''
                    #---------------------------------------
                    # Check
                    #---------------------------------------
                    assert np.linalg.norm(HTemp.todense() - HTemp2.todense())<1.0e-13
                    #---------------------------------------
                    '''
                    Hbond += HTemp
            
            H += Hbond
            
        return H



def get_matrix_permutation(P, at):
    """
    Compute the matrix of a permutation
    
    Parameters
    ----------
    P : list
        list of matrices of adjacent transpositions
    at : numpy array
        sequence of adjacent transpositions
    
    Returns
    -------
    M : scipy.sparse.csr_matrix
        matrix of the permutation
    
    Remark
    ------
    The permutation should have already been written as a product of adjacent 
    transpositions
    """
    
    M = scipy.sparse.eye(P[0].shape[0])
    for k in at:
        M = M @ P[k]
    
    return M



def get_subshape(y, cy, particles):
    """
    Get the subshape associated to given particles in a SYT
    
    Parameters
    ----------
    y : numpy array
        SYT
    cy : numpy array
        column positions
    particles : numpy array
        boxes to consider for extracting subshape
    
    Returns
    -------
    alphaM : numpy array
        minimal legal irrep
    alphaP : numpy array
        number of boxes associated to particles in each row
    alphaB : numpy array
        base top-left corner irrep
    offset : int
        offset compared to initial irrep associated to input SYT
    
    Description
    -----------
    alphaP + alphaB = alphaM
    
    Example
    -------
    y = [0, 0, 0, 1, 0, 0, 2, 1, 1, 2]
    cy = [0, 1, 2, 0, 3, 4, 0, 1, 2, 1]
    irrep = [5, 3, 1] (not required in the input arguments)
    particles = [4, 5, 6]
    The tableau looks like:
    | 0 | 1 | 2 | 4 | 5 |        | x | x | x | 4 | 5 |      | x | x | x | 4 | 5 |
    | 3 | 7 | 8 |          --->  | x | x | x |         ---> | x |
    | 6 | 9 |                    | 6 | x |                  | 6 |
    
    offset = 0 [there is at least 1 particle in the first row]
    alphaP = [2, 0, 1] = number of particles in each row
    alphaM = [5, 1, 1] = minimal legal irrep which contains the particles at exterior positions
    alphaB = [3, 1, 0] = base top-left corner irrep such that alphaB + alphaP = alphaM
    """
    
    yp = y[particles]
    cyp = cy[particles]
    
    offset = np.min(yp)
    offsetcol = np.min(cyp)
    
    # shift such that top row and top column are numbered 0
    yp = yp - offset
    cyp = cyp - offsetcol
    
    nr = np.max(yp) - np.min(yp) + 1
    
    alphaP = np.zeros(shape=(nr,), dtype=int)
    for i in range(nr):
        alphaP[i] = len( np.argwhere(yp==i).flatten() )
    
    alphaM = np.zeros(shape=(nr,), dtype=int)
    alphaB = np.zeros(shape=(nr,), dtype=int)
    
    alphaB[nr-1] = 1
    alphaM[nr-1] = alphaB[nr-1] + alphaP[nr-1]
    
    for i in range(nr-1,-1,-1):
        if alphaP[i]==0:
            alphaM[i] = alphaM[i+1]
            alphaB[i] = alphaM[i+1]
        else:
            ind = np.argwhere(yp==i).flatten()
            nbi = np.max(cyp[ind]) + 1
            alphaM[i] = nbi
            alphaB[i] = alphaM[i] - alphaP[i]
    
    return alphaM, alphaB, alphaP, offset



def get_orthogonal_units(alpha):
    """
    Generate the first orthogonal unit for the irrep alpha
    
    Parameters
    ----------
    alpha : numpy array
        irrep
    
    Returns
    -------
    o : numpy array
        orthogonal unit
    
    Description
    -----------
    
    """
    print('Calling DEPRECATED get_orthogonal_units')
    
    # build all SYTs associated to the irrep alpha
    Y = get_SYT(alpha)
    falpha = Y.shape[0]
    CY = get_column(Y)
    
    # construct matrices of adjacent transpositions
    P = get_adjacent_transposition_matrices(alpha, Y, CY)
    
    # get all permutations of S_n
    n = np.sum(alpha)
    ps = itertools.permutations(np.arange(0, n, dtype=int))
    nn = math.factorial(n)
    
    import time
    '''
    #----------------------
    # not the optimal method
    #----------------------
    # compute the matrices of the permutations, and then their inverse
    start = time.perf_counter()
    sigmaMat = [scipy.sparse.eye(falpha)] * nn
    sigmaMatinverse0 = [None] * nn
    for i, sigma in enumerate(ps):
        at = permutation_to_adjacent_transpositions(np.array(sigma, dtype=int))
        for j in range(0, len(at)):
            sigmaMat[i] = sigmaMat[i] @ P[at[j]]
        sigmaMatinverse0[i] = scipy.sparse.linalg.inv(sigmaMat[i])
    end = time.perf_counter()
    elapsed1 = end-start
    print("Elapsed method 1 = {}s".format(elapsed1))
    #----------------------
    '''
    
    #----------------------
    # better
    #----------------------
    # compute the matrices of the inverse permutations, obtained by reading
    # the sequence of adjacent transpositions in reverse order
    ps = itertools.permutations(np.arange(0, n, dtype=int))
    start = time.perf_counter()
    sigmaMatinverse = [scipy.sparse.eye(falpha)] * nn
    for i, sigma in enumerate(ps):
        at = permutation_to_adjacent_transpositions(np.array(sigma, dtype=int))
        for j in range(0, len(at)):
            sigmaMatinverse[i] = P[at[j]] @ sigmaMatinverse[i]
    end = time.perf_counter()
    elapsed2 = end-start
    print("Elapsed method 2 = {}s".format((elapsed2)))
    #print("factor = {}s".format(elapsed1/elapsed2))
    
    '''
    #----------------------
    # verify equivalence of 2 methods
    #----------------------
    for i in range(0, nn):
        res = np.linalg.norm(sigmaMatinverse0[i].toarray()-sigmaMatinverse[i].toarray())
        if res>1.0e-14:
            print('Problem: matrix of inverse is not inverse of matrix. i=', i, ' res = ', res)
    #----------------------
    '''
    
    # compute orthogonal units
    orthogonalUnits = [[None] * nn] * nn
    
    for r in range(falpha):
        for s in range(falpha):
            orthogonalUnits[r][s] = np.zeros(nn, dtype=float)
            for i in range(nn):
                orthogonalUnits[r][s][sigma] = sigmaMatinverse[sigma][s,r]
            orthogonalUnits[r][s] *= falpha/nn
    
    return orthogonalUnits



def fullsimplify_development(ydev, cydev, coeff):
    # Sum coefficients of equal SYTs and remove SYTs having vanishing 
    # coefficients.
    # 
    
    if len(ydev.shape)==1:
        # only 1 SYT in the development
        ydev1 = np.reshape(ydev, (1, ydev.shape[0]))
        cydev1 = np.reshape(cydev, (1, cydev.shape[0]))
        coeff1 = np.copy(coeff)
    else:
        Ny = ydev.shape[0] # number of SYTs in the development
        ydev1, ia, ic = np.unique(ydev, axis=0, return_index=True, return_inverse=True)
        cydev1 = np.copy(cydev[ia,:])
        M = scipy.sparse.csr_matrix( (np.full(fill_value=1, shape=(Ny,), dtype=int), (ic, np.arange(0, Ny))), shape=(len(ia), Ny) )
        
        coeff1 = M @ coeff
    
    ind = np.argwhere(abs(coeff1)>1.0e-12).flatten()
    ydev1 = ydev1[ind,:]
    cydev1 = cydev1[ind,:]
    coeff1 = coeff1[ind]
    
    return ydev1, cydev1, coeff1



def develop_consecutive_number(alpha, ydev, cydev, coeff, k):
    # Apply the transposition \tau_{k,k+1} on the development.
    # 
    
    n = np.sum(alpha)
    
    if len(ydev.shape)==1:
        # only 1 SYT in the development
        Ny = 1
        n = ydev.shape[0]
        ydev = np.reshape(ydev, (1, n))
        cydev = np.reshape(cydev, (1, n))
    else:
        Ny = ydev.shape[0]
        n = ydev.shape[1]
    
    if not n==np.sum(alpha):
        sys.exit('Problem')
    
    if k>=n-1:
        sys.exit('Problem: k too large. k must be striclty less than n-1.')
    
    ydev1 = np.full(shape=(2*Ny, n), fill_value=-1, dtype=int)
    cydev1 = np.full(shape=(2*Ny, n), fill_value=-1, dtype=int)
    coeff1 = np.zeros((2*Ny,), dtype=float)
    
    ydev1[0:Ny,:] = np.copy(ydev)
    cydev1[0:Ny,:] = np.copy(cydev)
    coeff1[0:Ny] = np.copy(coeff)
    
    count = Ny

    for i in range(1, Ny+1):
        
        y = np.copy(ydev[i-1,:])
        cy = np.copy(cydev[i-1,:])
        
        if not y[k]==y[k+1]:
            # not in the same row
            
            c1 = cy[k]
            c2 = cy[k+1]
            
            if c1==c2:
                coeff1[i-1] *= (-1)
            else:
                
                count += 1
                
                ax = get_axial_distance(y, cy, k+1, k) # axial distance from k+1 to k
                rho = 1.0/ax
                yfriend = np.copy(y)
                cyfriend = np.copy(cy)
                yfriend[k] = y[k+1]
                yfriend[k+1] = y[k]
                cyfriend[k] = cy[k+1]
                cyfriend[k+1] = cy[k]
                
                ydev1[count-1,:] = np.copy(yfriend)
                cydev1[count-1,:] = np.copy(cyfriend)
                
                coeff1[count-1] = coeff1[i-1] * np.sqrt(1.0-rho*rho)
                coeff1[i-1] *= rho
    # end for i
    ydev1 = ydev1[0:count,:]
    cydev1 = cydev1[0:count,:]
    coeff1 = coeff1[0:count]
    
    ydev2, cydev2, coeff2 = fullsimplify_development(ydev1, cydev1, coeff1)
    
    return ydev2, cydev2, coeff2



def develop_transposition(alpha, ydev, cydev, coeff, i, j):
    # Apply the permutation P_{i,j} on the development. 
    # 
    
    if not i==j:
        listtranspositions, nbtranspositions = get_transpositions([np.array([min(i, j), max(i, j)], dtype=int)])
        for l in range(nbtranspositions[0]-1, -1, -1):
            k = listtranspositions[0][l]
            ydev, cydev, coeff = develop_consecutive_number(alpha, ydev, cydev, coeff, k)
    
    return ydev, cydev, coeff



def sum_develop(ydev1, cydev1, coeff1, ydev2, cydev2, coeff2):
    # Compute the sum of two developments.
    # 
    
    ydev = np.vstack([ydev1, ydev2])
    cydev = np.vstack([cydev1, cydev2])
    coeff = np.hstack([coeff1, coeff2]).flatten()
    
    ydev, cydev, coeff = fullsimplify_development(ydev, cydev, coeff)
    
    return ydev, cydev, coeff



def developp_symmetric(alpha, y, cy, m, n1, n2):
    # Apply the permutation of sites (n1, n2) on the SYT y, in the case of m
    # particles per site in the symmetric irrep.
    # 
    
    j1 = min(n1, n2)
    j2 = max(n1, n2)
    
    n = np.sum(alpha)
    
    if not n==len(y):
        sys.exit('Problem: y does not match n')
    
    if n%m:
        sys.exit('Problem: n not a multiple of m')
    
    ydevfinal = np.zeros(shape=(1, n), dtype=int)
    cydevfinal = np.zeros(shape=(1, n), dtype=int)
    coefffinal = np.array([0.0], dtype=float)
    
    if m==2:
        
        cpt = int(1)
        
        ydev = np.zeros(shape=(2**(j2-j1+1), n), dtype=int)
        cydev = np.zeros(shape=(2**(j2-j1+1), n), dtype=int)
        coeff = np.zeros(shape=(2**(j2-j1+1),), dtype=float)
        
        ydev[0,:] = np.copy(y)
        cydev[0,:] = np.copy(cy)
        coeff[0] = 1.0
        
        for j in range(j1, j2+1):
            
            p1 = 2*j
            p2 = 2*j+1
            
            if not y[p1]==y[p2]:
                # not in the same row
                
                axialdist = get_axial_distance(y, cy, p1, p2)
                
                for t in range(1, cpt+1):
                    rp1 = ydev[t-1, p1]
                    rp2 = ydev[t-1, p2]
                    cp1 = cydev[t-1, p1]
                    cp2 = cydev[t-1, p2]
                    # create new SYT
                    ydev[cpt+t-1,:] = np.copy(ydev[t-1,:])
                    cydev[cpt+t-1,:] = np.copy(cydev[t-1,:])
                    # perform exchange of particles
                    ydev[cpt+t-1, p2] = rp1
                    ydev[cpt+t-1, p1] = rp2
                    cydev[cpt+t-1, p2] = cp1
                    cydev[cpt+t-1, p1] = cp2
                    # update coefficients
                    coeff[cpt+t-1] = coeff[t-1] * np.sqrt((1.0 + 1.0/axialdist)/2.0)
                    coeff[t-1] *= np.sqrt((1.0 - 1.0/axialdist)/2.0)
                # end for t
                
                cpt *= 2
            # end if not y[p1]==y[p2]
        # end for j
        
        ydev = ydev[0:cpt,:]
        cydev = cydev[0:cpt,:]
        coeff = coeff[0:cpt]
        
        # STEP 2:
        ydev, cydev, coefficients = develop_transposition(alpha, ydev, cydev, coeff, 2*j1+1, 2*j2)
        Nydev = ydev.shape[0]

        # STEP 3:
        for qq in range(1, Nydev+1):
            for kk in range(j1, j2+1):
                
                # particles at site kk
                p1 = 2*kk
                p2 = 2*kk+1
                
                if not cydev[qq-1, p2]==cydev[qq-1,p1]:
                    distanceproj = cydev[qq-1,p1] - cydev[qq-1,p2] + ydev[qq-1,p2] - ydev[qq-1,p1]
                    # This is the axial distance. To check.
                    ax = get_axial_distance(ydev[qq-1,:], cydev[qq-1,:], p1, p2)
                    
                    if not ax==distanceproj:
                        sys.exit('Problem: need to swap 2*kk and 2*kk+1 in ax')
                    
                    coefficients[qq-1] *= np.sqrt((1.0-(1.0/distanceproj))/2.0)
                    
                    if ydev[qq-1, p2]<ydev[qq-1, p1]:
                        ytemp = np.copy(ydev[qq-1, :])
                        ctemp = np.copy(cydev[qq-1, :])
                        ydev[qq-1, p2] = ytemp[p1]
                        ydev[qq-1, p1] = ytemp[p2]
                        cydev[qq-1, p2] = ctemp[p1]
                        cydev[qq-1, p1] = ctemp[p2]
                else:
                    coefficients[qq-1] = 0
            # end for kk
        # end for qq
        
        newcoeff = 4.0*np.copy(coefficients)
        
        ydev, cydev, coeff = fullsimplify_development(ydev, cydev, newcoeff)
        
        ydevfinal, cydevfinal, coefffinal = sum_develop(ydevfinal, cydevfinal, coefffinal, ydev, cydev, coeff)

        
    elif m==3:
        
        cpt = int(1)
        
        ydev = np.zeros(shape=(10*3**(j2-j1+1), n), dtype=int)
        cydev = np.zeros(shape=(10*3**(j2-j1+1), n), dtype=int)
        coeff = np.zeros(shape=(10*3**(j2-j1+1),), dtype=float)
        
        ydev[0,:] = np.copy(y)
        cydev[0,:] = np.copy(cy)
        coeff[0] = 1.0
        
        for q in range(j1, j2+1):
            
            # the 3 particles at site q are:
            p1 = 3*q
            p2 = 3*q+1
            p3 = 3*q+2
            
            cpt_temp = cpt
            
            axdistx = get_axial_distance(y, cy, p1, p2)
            axdisty = get_axial_distance(y, cy, p1, p3)
            axdistz = get_axial_distance(y, cy, p2, p3)
            rhox = 1.0/axdistx
            rhoy = 1.0/axdisty
            rhoz = 1.0/axdistz
            
            if not y[p1]==y[p2]:
                
                for t in range(1, cpt_temp+1):
                    rp1 = ydev[t-1, p1]
                    rp2 = ydev[t-1, p2]
                    cp1 = cydev[t-1, p1]
                    cp2 = cydev[t-1, p2]
                    # create new SYT
                    ydev[cpt+t-1, :] = np.copy(ydev[t-1, :])
                    cydev[cpt+t-1, :] = np.copy(cydev[t-1, :])
                    # exchange p1 and p2
                    ydev[cpt+t-1, p2] = rp1
                    ydev[cpt+t-1, p1] = rp2
                    cydev[cpt+t-1, p2] = cp1
                    cydev[cpt+t-1, p1] = cp2
                    
                    coeff[cpt+t-1] = coeff[t-1] * np.sqrt(1+rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) * (1.0/np.sqrt(6.0))
                # end for t
                cpt += cpt_temp
                
                for t in range(1, cpt_temp+1):
                    rp1 = ydev[t-1, p1]
                    rp2 = ydev[t-1, p2]
                    rp3 = ydev[t-1, p3]
                    cp1 = cydev[t-1, p1]
                    cp2 = cydev[t-1, p2]
                    cp3 = cydev[t-1, p3]
                    #---------
                    ydev[cpt+t-1, :] = np.copy(ydev[t-1, :])
                    cydev[cpt+t-1, :] = np.copy(cydev[t-1, :])
                    #---------
                    ydev[cpt+t-1, p1] = rp2
                    ydev[cpt+t-1, p2] = rp3
                    ydev[cpt+t-1, p3] = rp1
                    cydev[cpt+t-1, p1] = cp2
                    cydev[cpt+t-1, p2] = cp3
                    cydev[cpt+t-1, p3] = cp1
                    
                    coeff[cpt+t-1] = coeff[t-1] * np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1-rhoz) * (1.0/np.sqrt(6.0))
                # end for t
                
                cpt += cpt_temp
            # end if not y[p1]==y[p2]
            
            ###################################################################
            
            if not y[p2]==y[p3]:
                
                for t in range(1, cpt_temp+1):
                    rp2 = ydev[t-1, p2]
                    rp3 = ydev[t-1, p3]
                    cp2 = cydev[t-1, p2]
                    cp3 = cydev[t-1, p3]
                    #---------
                    ydev[cpt+t-1, :] = np.copy(ydev[t-1, :])
                    cydev[cpt+t-1, :] = np.copy(cydev[t-1, :])
                    #---------
                    ydev[cpt+t-1, p2] = rp3
                    ydev[cpt+t-1, p3] = rp2
                    cydev[cpt+t-1, p2] = cp3
                    cydev[cpt+t-1, p3] = cp2
                    
                    coeff[cpt+t-1] = coeff[t-1] * np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1+rhoz) * (1.0/np.sqrt(6.0))
                # end for t
                
                cpt += cpt_temp
                
                for t in range(1, cpt_temp+1):
                    rp1 = ydev[t-1, p1]
                    rp2 = ydev[t-1, p2]
                    rp3 = ydev[t-1, p3]
                    cp1 = cydev[t-1, p1]
                    cp2 = cydev[t-1, p2]
                    cp3 = cydev[t-1, p3]
                    #---------
                    ydev[cpt+t-1, :] = np.copy(ydev[t-1, :])
                    cydev[cpt+t-1, :] = np.copy(cydev[t-1, :])
                    #---------
                    ydev[cpt+t-1, p1] = rp3
                    ydev[cpt+t-1, p2] = rp1
                    ydev[cpt+t-1, p3] = rp2
                    cydev[cpt+t-1, p1] = cp3
                    cydev[cpt+t-1, p2] = cp1
                    cydev[cpt+t-1, p3] = cp2
                    
                    coeff[cpt+t-1] = coeff[t-1] * np.sqrt(1-rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) * (1.0/np.sqrt(6.0))
                # end for t
                
                cpt += cpt_temp
                
                if not y[p1]==y[p2]:
                    
                    for t in range(1, cpt_temp+1):
                        rp1 = ydev[t-1, p1]
                        rp3 = ydev[t-1, p3]
                        cp1 = cydev[t-1, p1]
                        cp3 = cydev[t-1, p3]
                        #---------
                        ydev[cpt+t-1, :] = np.copy(ydev[t-1, :])
                        cydev[cpt+t-1, :] = np.copy(cydev[t-1, :])
                        #---------
                        ydev[cpt+t-1, p1] = rp3
                        ydev[cpt+t-1, p3] = rp1
                        cydev[cpt+t-1, p1] = cp3
                        cydev[cpt+t-1, p3] = cp1
                        
                        coeff[cpt+t-1] = coeff[t-1] * np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) * (1.0/np.sqrt(6.0))
                    # end for t
                    
                    cpt += cpt_temp
                # end if not y[p1]==y[p2]
            # end if not y[p2]==y[p3]
            
            ###################################################################
            
            for t in range(1, cpt_temp+1):
                coeff[t-1] *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) * (1.0/np.sqrt(6.0))
            
        # end for q
        
        ydev = ydev[0:cpt, :]
        cydev = cydev[0:cpt, :]
        coeff = coeff[0:cpt]
        
        ydev, cydev, coefficients = develop_transposition(alpha, ydev, cydev, coeff, 3*j1+2, 3*j2)
        
        Nydev = ydev.shape[0]
        
        for qq in range(1, Nydev+1):
            
            prod_if = int(1)
            
            for kk in range(j1, j2+1):
                p1 = 3*kk
                p2 = 3*kk+1
                p3 = 3*kk+2
                
                prod_if *= (cydev[qq-1, p2]-cydev[qq-1, p1]) * (cydev[qq-1, p3]-cydev[qq-1, p2]) * (cydev[qq-1, p3]-cydev[qq-1, p1])
            
            if not prod_if==0:
                
                for k in range(j1, j2+1):
                    p1 = 3*k
                    p2 = 3*k+1
                    p3 = 3*k+2
                    
                    axdistx = get_axial_distance(ydev[qq-1, :], cydev[qq-1, :], p1, p2)
                    axdisty = get_axial_distance(ydev[qq-1, :], cydev[qq-1, :], p1, p3)
                    axdistz = get_axial_distance(ydev[qq-1, :], cydev[qq-1, :], p2, p3)
                    rhox = 1.0/axdistx
                    rhoy = 1.0/axdisty
                    rhoz = 1.0/axdistz
                    
                    coefficients[qq-1] *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) * (1.0/np.sqrt(6.0))
                    
                    yd1 = ydev[qq-1, p1]
                    yd2 = ydev[qq-1, p2]
                    yd3 = ydev[qq-1, p3]
                    
                    a = ( yd1 + yd2 + abs(yd1 - yd2) )/2
                    b = ( yd2 + yd3 + abs(yd2 - yd3) )/2
                    c = ( yd1 + yd3 + abs(yd1 - yd3) )/2
                    d = ( yd1 + yd2 - abs(yd1 - yd2) )/2
                    e = ( yd3 + yd2 - abs(yd3 - yd2) )/2
                    f = ( a + b - abs(a-b) )/2
                    
                    ydev[qq-1, p1] = ( d + e - abs(d-e) )/2 # min(y[p1],y[p2],y[p3])
                    ydev[qq-1, p2] = ( f + c - abs(f-c) )/2 # intermediary value
                    ydev[qq-1, p3] = ( a + b + abs(a-b) )/2 # max(y[p1],y[p2],y[p3])
                    
                    cydev[qq-1, :] = get_column(ydev[qq-1, :]).flatten()
                    
                # end for k
                
            else:
                coefficients[qq-1] = 0.0
            # end if prod_if
        # end for qq
        
        newcoeff = 9.0 * np.copy(coefficients)
        
        ydev, cydev, coeff = fullsimplify_development(ydev, cydev, newcoeff)
        ydevfinal, cydevfinal, coefffinal = sum_develop(ydevfinal, cydevfinal, coefffinal, ydev, cydev, coeff)
        
    else:
        sys.exit('Problem: case m>3 not implemented.')
    
    
    return ydevfinal, cydevfinal, coefffinal



def developp_antisymmetric(alpha, y, cy, m, n1, n2):
    # Apply the permutation of sites (n1, n2) on the SYT y, in the case of m
    # particles per site in the antisymmetric irrep.
    # 
    
    j1 = min(n1, n2)
    j2 = max(n1, n2)
    
    n = np.sum(alpha)
    
    if not n==len(y):
        sys.exit('Problem: y does not match n')
    
    if n%m:
        sys.exit('Problem: n not a multiple of m')
    
    ydevfinal = np.zeros(shape=(1, n), dtype=int)
    cydevfinal = np.zeros(shape=(1, n), dtype=int)
    coefffinal = np.array([0.0], dtype=float)
    
    if m==2:
        
        cpt = int(1)
        
        ydev = np.zeros(shape=(2**(j2-j1+1), n), dtype=int)
        cydev = np.zeros(shape=(2**(j2-j1+1), n), dtype=int)
        coeff = np.zeros(shape=(2**(j2-j1+1),), dtype=float)
        
        ydev[0,:] = np.copy(y)
        cydev[0,:] = np.copy(cy)
        coeff[0] = 1.0
        
        for j in range(j1, j2+1):
            
            p1 = 2*j
            p2 = 2*j+1
            
            if not cy[p1]==cy[p2]:
                # not in the same column
                
                axialdist = get_axial_distance(y, cy, p1, p2)
                
                for t in range(1, cpt+1):
                    rp1 = ydev[t-1, p1]
                    rp2 = ydev[t-1, p2]
                    cp1 = cydev[t-1, p1]
                    cp2 = cydev[t-1, p2]
                    # create new SYT
                    ydev[cpt+t-1,:] = np.copy(ydev[t-1,:])
                    cydev[cpt+t-1,:] = np.copy(cydev[t-1,:])
                    # perform exchange of particles
                    ydev[cpt+t-1, p2] = rp1
                    ydev[cpt+t-1, p1] = rp2
                    cydev[cpt+t-1, p2] = cp1
                    cydev[cpt+t-1, p1] = cp2
                    # update coefficients
                    coeff[cpt+t-1] = - coeff[t-1] * np.sqrt((1.0 - 1.0/axialdist)/2.0)
                    coeff[t-1] *= np.sqrt((1.0 + 1.0/axialdist)/2.0)
                # end for t
                
                cpt *= 2
            # end if not cy[p1]==cy[p2]
        # end for j
        
        ydev = ydev[0:cpt,:]
        cydev = cydev[0:cpt,:]
        coeff = coeff[0:cpt]
        
        # STEP 2:
        ydev, cydev, coefficients = develop_transposition(alpha, ydev, cydev, coeff, 2*j1+1, 2*j2)
        Nydev = ydev.shape[0]
        
        # STEP 3:
        for qq in range(1, Nydev+1):
            for kk in range(j1, j2+1):
                
                # particles at site kk
                p1 = 2*kk
                p2 = 2*kk+1
                
                if not ydev[qq-1, p2]==ydev[qq-1,p1]:
                    distanceproj = cydev[qq-1,p1] - cydev[qq-1,p2] + ydev[qq-1,p2] - ydev[qq-1,p1]
                    # This is the axial distance. To check.
                    ax = get_axial_distance(ydev[qq-1,:], cydev[qq-1,:], p1, p2)
                    
                    if not ax==distanceproj:
                        sys.exit('Problem: need to swap 2*kk and 2*kk+1 in ax')
                    
                    coefficients[qq-1] *= np.sqrt((1.0+(1.0/distanceproj))/2.0)
                    
                    if ydev[qq-1, p2]<ydev[qq-1, p1]:
                        ytemp = np.copy(ydev[qq-1, :])
                        ctemp = np.copy(cydev[qq-1, :])
                        ydev[qq-1, p2] = ytemp[p1]
                        ydev[qq-1, p1] = ytemp[p2]
                        cydev[qq-1, p2] = ctemp[p1]
                        cydev[qq-1, p1] = ctemp[p2]
                        coefficients[qq-1] *= -1
                else:
                    coefficients[qq-1] = 0
            # end for kk
        # end for qq
        
        newcoeff = 4.0*np.copy(coefficients)

        ydev, cydev, coeff = fullsimplify_development(ydev, cydev, newcoeff)
        
        ydevfinal, cydevfinal, coefffinal = sum_develop(ydevfinal, cydevfinal, coefffinal, ydev, cydev, coeff)
        
    else:
        sys.exit('Problem: case m>2 not yet implemented.')
    
    
    return ydevfinal, cydevfinal, coefffinal



def print_to_latex(y, **kwargs):
    # Print a SYT to text in LaTeX format, using \ytableau
    # 
    # Optional arguments:
    #   zerobased [True]
    #   shift
    #   scale
    #   colors : dictionnary with keys-value pairs as i: 'col'
    # 
    
    if not 'shift' in kwargs:
        kwargs['shift'] = 0
    if not 'scale' in kwargs:
        kwargs['scale'] = 1
    if not 'zeroBased' in kwargs:
        kwargs['zeroBased'] = True
    
    if kwargs['zeroBased']==True:
        szb = int(0)
    else:
        szb = int(1)
    
    n = len(y)
    if not 'colors' in kwargs:
        c = [''] * n
        kwargs['colors'] = {i: c[i] for i in range(0, n)}
    else:
        allkeys = [i for i in range(0, n)]
        for i in allkeys:
            if i in kwargs['colors'].keys():
                if not ((kwargs['colors'][i][0:2]=='*(') and (kwargs['colors'][i][-1:]==')')):
                    kwargs['colors'][i] = '*('+kwargs['colors'][i]+')'
            else:
                kwargs['colors'][i] = ''
    
    nl = np.max(y) + 1 # number of rows in the shape
    
    print('\\shsc{' + '{:.1f}'.format(kwargs['shift']) + 'pt}{' + '{:.1f}'.format(kwargs['scale']) + '}{\\begin{ytableau}')
    
    for j in range(0, nl):
        # find all numbers situated in the j-th row
        indj = np.argwhere(y==j).flatten()
        
        # print all numbers of the j-th row
        line = ''
        for k in range(0, len(indj)-1):
            line += kwargs['colors'][indj[k]] + str(indj[k]+szb) + ' & '
        
        line += kwargs['colors'][indj[-1]] + str(indj[-1]+szb) + ' \\\\ '
        print(line)
    
    print('\\end{ytableau}}\n')
    
    return



def print_subSYT_to_latex(y, alpha, **kwargs):
    # Print the subSYT <y> in LaTeX format, using \ytableau.
    # 
    # Details:
    #  1) <alpha> is the global shape associated to <y>.
    #  2) <y> is of length n1, corresponding to the number of boxes in the
    #     remainder shape <alpha>-alpha0, where alpha0 is the subshape of alpha
    #
    # Optional arguments:
    #   shift
    #   scale
    #   lowcol      color for low part (alpha0)
    #   highcol     color for high part (<alpha>-alpha0)
    # 
    
    if not 'shift' in kwargs:
        kwargs['shift'] = 0
    if not 'scale' in kwargs:
        kwargs['scale'] = 1
    
    if not 'lowcol' in kwargs:
        kwargs['lowcol'] = '{}'
    else:
        kwargs['lowcol'] = '*(' + kwargs['lowcol'] + ')'
    
    if not 'highcol' in kwargs:
        kwargs['highcol'] = ''
    else:
        kwargs['highcol'] = '*(' + kwargs['highcol'] + ')'
    
    if not 'zeroBased' in kwargs:
        kwargs['zeroBased'] = True
    
    if kwargs['zeroBased']==True:
        szb = int(0)
    else:
        szb = int(1)
    
    n = np.sum(alpha)
    n1 = len(y)
    n0 = n - n1
    
    alpha1 = np.zeros(shape=alpha.shape, dtype=int)
    
    for i in range(0, n1):
        alpha1[y[i]] += 1
    
    y0 = np.full(shape=(n0,), fill_value=-1, dtype=int)
    yglobal = np.hstack([y0, y])
    
    nl = len(np.argwhere(alpha>0).flatten()) # number of rows in global shape alpha
    
    print('\\shsc{' + '{:.1f}'.format(kwargs['shift']) + 'pt}{' + '{:.1f}'.format(kwargs['scale']) + '}{\\begin{ytableau}')
    
    for j in range(0, nl):
        # find all numbers of subtableau situated in the j-th row
        indj = np.argwhere(yglobal==j).flatten()
        
        ns = len(indj) # number of filled boxes in the j-th row
        nb = alpha[j] - ns # number of blank boxes
        
        # print all numbers of the j-th row
        line = ''
        for k in range(0, nb-1):
            line += str(kwargs['lowcol'] + ' & ')
        if ns==0:
            line += str(kwargs['lowcol'] + ' \\\\')
        else:
            if nb>0:
                line += str(kwargs['lowcol'] + ' & ')
            for k in range(0, len(indj)-1):
                line += kwargs['highcol'] + str(indj[k]+szb) + ' & '
            line += kwargs['highcol'] + str(indj[-1]+szb) + ' \\\\ '
        
        print(line)
    
    print('\\end{ytableau}}\n')
    
    return



class PartialLookupTool:
    
    
    def __init__(self, N, alpha, nlookupboxes):
        
        if nlookupboxes==0:
            sys.exit('Problem: for nlookupboxes==0, no lookup. Use dedicated functions.')
        elif nlookupboxes==1:
            print('Warning: nlookupboxes=1 is equivalent to no lookup.')
        elif nlookupboxes>=np.sum(alpha):
            sys.exit('Problem: for partial lookup, nlookupboxes must be stricly smaller than the total number of boxes.')
        
        self.alpha = alpha
        self.N = N
        self.n = np.sum(alpha)
        self.nlookupboxes = nlookupboxes
        self.n0 = nlookupboxes
        self.n1 = self.n - self.n0
        
        return
    
    
    
    def init_lookup(self):
        # Generate all lookup tables
        # 
        
        # generate all subshapes of alpha with <self.n0> boxes
        pstarnm = partitions.pstarnm(self.n0, self.N) # number of partitions of <self.n0> in at most <self.N> parts
        
        alpha_all, _ = get_list_irreps(self.N, 2*pstarnm, self.n)
        
        # restrict to potential shapes
        ind = np.argwhere( (np.sum(alpha_all, axis=1)<=self.n0) & ((np.sum(alpha_all, axis=1)%self.N)==(self.n0%self.N)) ).flatten()
        alpha_all = alpha_all[ind]
        
        if not alpha_all.shape[0]==pstarnm:
            sys.exit('Problem: the number of generated shapes does not match the number of parititions.')
        
        # add columns with N boxes
        for i in range(0, alpha_all.shape[0]):
            r = (self.n0 - np.sum(alpha_all[i]))//self.N
            alpha_all[i] += np.full(shape=alpha_all[i].shape, fill_value=r, dtype=int)
        
        # remove shapes which are not subshapes
        ind = np.zeros(shape=(0,), dtype=int)
        for i in range(0, alpha_all.shape[0]):
            issubshape = True
            for l in range(0, self.N):
                if alpha_all[i][l]>self.alpha[l]:
                    issubshape = False
                    break
            if issubshape==True:
                ind = np.hstack([ind, i])
        
        self.alpha0 = np.copy(alpha_all[ind])
        
        ###########################################################################
        
        # reorder alpha0 so as to correspond to the iLLOS
        # irreps must be sorted according to the length of the rows
        
        self.alpha0, _ = sortrows(self.alpha0)
        self.Na = self.alpha0.shape[0] # total number of subshapes
        
        ###########################################################################
        
        # for each subshape of alpha, generate all SYTs
        
        self.Y0 = [None] * self.Na
        self.dims0 = np.zeros((self.Na,), dtype=int)
        
        for i in range(0, self.Na):
            self.Y0[i] = get_SYT(self.alpha0[i], order='iLLOS')
            self.dims0[i] = self.Y0[i].shape[0]
        
        ###########################################################################
        
        # Generate all remainder shapes
        
        self.alpha1 = self.alpha - self.alpha0
        
        ###########################################################################
        
        # for each remainder shape, generate all SYTs
        
        self.Y1 = [None] * self.Na
        self.dims1 = np.zeros((self.Na,), dtype=int)
        
        for i in range(0, self.Na):
            self.Y1[i] = get_subSYT(self.alpha, self.alpha0[i])
            self.dims1[i] = self.Y1[i].shape[0]
        
        ###########################################################################
        
        # Verify dimension
        
        self.dim = int(0)
        for i in range(0, self.Na):
            self.dim += (self.dims0[i] * self.dims1[i])
        
        if not self.dim==multiplicity(self.alpha):
            sys.exit('Problem: the sum of the product of the dimensions does not match the multiplicity.')
        
        ###########################################################################
        # GENERATE THE ORDERING TO MATCH THE ILLOS
        ###########################################################################
        
        # note that sorting the SYTs is not strictly necessary, as one can simply
        # redefine the basis, as in each "block", the SYTs are ordered in the iLLOS
        # by construction. However, the iLLOS is a convenient order.
        
        self.orderingIrrep = np.zeros(shape=(np.sum(self.dims1),), dtype=int)
        self.orderingState = np.zeros(shape=(np.sum(self.dims1),), dtype=int)
        
        self.lookup = [None] * self.Na
        for i in range(0, self.Na):
            self.lookup[i] = np.zeros(shape=(self.dims1[i],), dtype=int)
        
        self.lookup[0][0] = int(0)
        
        start_index = np.zeros(shape=(self.Na,), dtype=int)
        start_index[0] = int(1)
        
        cpt = int(1)
        while cpt<np.sum(self.dims1):
            
            Ytmp = np.zeros(shape=(0, self.n1), dtype=int)
            ar = []
            for i in range(0, self.Na):
                if start_index[i]<self.dims1[i]:
                    ar.append(i)
                    Ytmp = np.vstack([Ytmp, self.Y1[i][start_index[i]]])
            ar = np.array(ar, dtype=int)
            _, ind = sortrows(np.fliplr(Ytmp))
            
            k = ar[ind[0]]
            
            self.orderingIrrep[cpt] = k
            self.orderingState[cpt] = start_index[k]
            self.lookup[k][start_index[k]] = cpt
            
            start_index[ar[ind[0]]] += 1
            cpt += 1
        
        ###########################################################################
        # GENERATE A USEFUL LOOKUP TO GET SYT FROM INDEX
        ###########################################################################
        
        self.acc_dims = np.zeros(shape=(np.sum(self.dims1)+1,), dtype=int)
        
        for j in range(1, np.sum(self.dims1)+1):
            ind = self.orderingIrrep[j-1]
            self.acc_dims[j] = self.acc_dims[j-1] + self.dims0[ind]
        
        return
    
    
        
    def get_SYT(self, i):
        # Get the <i>-th SYT.
        # 
        # Input index <i> is in the iLLOS: i=0 ---> largest SYT.
        # 
        
        ind = np.argwhere(self.acc_dims<=i).flatten()[-1]
        
        k = self.orderingIrrep[ind] # index of irrep
        i0 = i - self.acc_dims[ind] # index of low part
        i1 = self.orderingState[ind] # index of high part
        
        y0 = np.copy(self.Y0[k][i0])
        y1 = np.copy(self.Y1[k][i1])
        
        y = np.hstack([y0, y1])
        
        return y
    
    
    
    def get_index(self, y):
        # get the index corresponding to SYT <y> in the ordered (wrt iLLOS) 
        # list of all SYTs.
        # 
        
        y0 = np.copy(y[0:self.n0]) # low part
        y1 = np.copy(y[self.n0:]) # high part
        
        # generate irrep for the low part
        a0 = np.zeros(shape=(self.N,), dtype=int)
        for i in range(0, self.n0):
            a0[y0[i]] += 1
        
        # search index of a0 in self.alpha0 [here linear search]
        k = np.argwhere( np.sum(abs(self.alpha0-a0), axis=1)==0 ).flatten()[0]
        
        # search index of state for low part [here linear search]
        #i0 = np.argwhere( np.sum(abs(self.Y0[k]-y0), axis=1)==0 ).flatten()[0]
        # [here binary search]
        i0 = binary_search_SYT(y0, self.Y0[k])
        
        # search index of state for high part [here linear search]
        # i1 = np.argwhere( np.sum(abs(self.Y1[k]-y1), axis=1)==0 ).flatten()[0]
        # [here binary search]
        i1 = binary_search_SYT(y1, self.Y1[k])
        
        # find index from high part
        # indhigh = np.argwhere( (self.orderingIrrep==k) & (self.orderingState==i1) ).flatten()[0]
        # use <lookup> technique: - slightly faster than argwhere, but requires self.lookup
        indhigh = self.lookup[k][i1]
        
        # compute index
        i = int(0)
        for j in range(0, indhigh):
            i += self.dims0[self.orderingIrrep[j]]
        i += i0
        
        return i



def get_adjacent_transposition_matrix(alpha, Y, CY, k):
    """
    Get matrix of adjacent transposition P_{k, k+1}
    
    Parameters
    ----------
    alpha : numpy array
        irrep
    Y : numpy array
        SYTs associated to irrep alpha
    k : int
        transposition to obtain
    
    Returns
    -------
    Pk : scipy.sparse.csr_matrix
        matrix of adjacent transposition P_{k, k+1}
    
    Description
    -----------
    Computes the adjacent transposition P_{k, k+1} for the irrep alpha
    """
    
    NY = Y.shape[0]
    
    Pk_diagval = np.zeros(shape=(NY,), dtype=float)
    Pk_offdiagrowindex = np.zeros(shape=(NY, ), dtype=int)
    Pk_offdiagcolindex = np.zeros(shape=(NY, ), dtype=int)
    Pk_offdiagval = np.zeros(shape=(NY, ), dtype=float)
    
    offdiagcounter = int(0)
    
    for i in range(0, NY):
        
        y = Y[i]
        cy = CY[i]
        
        if Pk_diagval[i]==0:
            
            if y[k]==y[k+1]:
            
                Pk_diagval[i] = 1
            
            else:
                
                c1 = cy[k]
                c2 = cy[k+1]
                
                if c1==c2:
                    Pk_diagval[i] = -1
                else:
                    rho = 1.0/get_axial_distance(y, cy, k, k+1)
                    yfriend = np.copy(y)
                    yfriend[k] = y[k+1]
                    yfriend[k+1] = y[k]
                    
                    # index = np.argwhere(np.sum(np.abs(self.Y - yfriend), axis=1) < 1e-13).flatten()[0]
                    index = i + np.argwhere(np.sum(np.abs(Y[i:,:] - yfriend), axis=1)<1.0e-13).flatten()[0]
                    
                    # diagonal elements
                    Pk_diagval[i] = -rho
                    Pk_diagval[index] = rho
                    
                    # off-diagonal elements
                    Pk_offdiagrowindex[offdiagcounter] = i
                    Pk_offdiagcolindex[offdiagcounter] = index
                    Pk_offdiagval[offdiagcounter] = np.sqrt(1.0-rho**2)
                    offdiagcounter += 1        
    
    
    Pkdiag = scipy.sparse.diags(Pk_diagval, offsets=0, shape=(NY, NY))
    Pkoffdiag = scipy.sparse.csr_matrix(
                    (Pk_offdiagval[0:offdiagcounter], 
                    (Pk_offdiagrowindex[0:offdiagcounter], 
                     Pk_offdiagcolindex[0:offdiagcounter])), 
                    shape=(NY, NY))    
    Pk = Pkdiag + Pkoffdiag + Pkoffdiag.transpose()
    
    return Pk



def get_adjacent_transposition_matrices(alpha, Y, CY):
    """
    Get all matrices of adjacent transpositions
    
    Parameters
    ----------
    alpha : numpy array
        irrep
    Y : numpy array
        SYTs associated to irrep alpha
    
    Returns
    -------
    P : list
        matrices of adjacent transpositions P_{k, k+1}
    
    Description
    -----------
    Computes the n-1 (where n is the number of boxes in alpha) adjacent transpositions
    P_{k, k+1} for the irrep alpha.
    """
    
    n = np.sum(alpha)
    P = []
    
    for k in range(n-1):
        P.append( get_adjacent_transposition_matrix(alpha, Y, CY, k) )
    
    '''
    NY = Y.shape[0]
    
    Pk_diagval = np.zeros(shape=(n-1, NY), dtype=float)
    Pk_offdiagrowindex = np.zeros(shape=(n-1, NY), dtype=int)
    Pk_offdiagcolindex = np.zeros(shape=(n-1, NY), dtype=int)
    Pk_offdiagval = np.zeros(shape=(n-1, NY), dtype=float)
    
    offdiagcounter = np.zeros(shape=(n-1, ), dtype=int)
    
    for k in range(0, n-1):
        
        count = 0
        
        for i in range(0, NY):
            
            y = Y[i]
            cy = CY[i]
            
            if Pk_diagval[k, i]==0:
                
                if y[k]==y[k+1]:
                
                    Pk_diagval[k, i] = 1
                
                else:
                    
                    c1 = cy[k]
                    c2 = cy[k+1]
                    
                    if c1==c2:
                        Pk_diagval[k, i] = -1
                    else:
                        rho = 1.0/get_axial_distance(y, cy, k, k+1)
                        yfriend = np.copy(y)
                        yfriend[k] = y[k+1]
                        yfriend[k+1] = y[k]
                        
                        # index = np.argwhere(np.sum(np.abs(self.Y - yfriend), axis=1) < 1e-13).flatten()[0]
                        index = i + np.argwhere(np.sum(np.abs(Y[i:,:] - yfriend), axis=1)<1.0e-13).flatten()[0]
                        
                        # diagonal elements
                        Pk_diagval[k, i] = -rho
                        Pk_diagval[k, index] = rho
                        
                        # off-diagonal elements
                        Pk_offdiagrowindex[k, count] = i
                        Pk_offdiagcolindex[k, count] = index
                        Pk_offdiagval[k, count] = np.sqrt(1.0-rho**2)
                        count += 1
        
        offdiagcounter[k] = count
    
    P = [None] * (n-1)
    
    for k in range(0, n-1):
        Pkdiag = scipy.sparse.diags(Pk_diagval[k], offsets=0, shape=(NY, NY))
        Pkoffdiag = scipy.sparse.csr_matrix(
                        (Pk_offdiagval[k, 0:offdiagcounter[k]], 
                        (Pk_offdiagrowindex[k, 0:offdiagcounter[k]], 
                         Pk_offdiagcolindex[k, 0:offdiagcounter[k]])), 
                        shape=(NY, NY))    
        P[k] = Pkdiag + Pkoffdiag + Pkoffdiag.transpose()
    '''
    
    return P



class SUNFundamental:
    
    def __init__(self, Ns, N, alpha, lattice):
        
        if not np.sum(alpha)==Ns:
            sys.exit('Problem: number of boxes in alpha must match Ns')
        
        if not lattice.Ns==Ns:
            sys.exit('lattice object does not have the correct number of sites.')
        
        self.N = N
        self.Ns = Ns
        self.alpha = np.copy(alpha)
        self.n = np.sum(self.alpha)
        self.falpha = multiplicity(self.alpha)
        self.lattice = lattice
        self.Y = get_SYT(self.alpha)
        self.CY = get_column(self.Y)

    def sun_hamiltonian(self):
        """
        Compute the Hamiltonian
        """
        
        listtranspositions, nbtranspositions = get_transpositions(self.lattice.links)
        
        P = get_adjacent_transposition_matrices(self.alpha, self.Y, self.CY)
        
        H = scipy.sparse.csr_matrix((self.falpha, self.falpha))
        
        for j in range(0, self.lattice.nlinks):
            Hj = scipy.sparse.eye(self.falpha)
            for l in range(nbtranspositions[j]-1, -1, -1):
                k = listtranspositions[j][l]
                Hj = P[k] @ Hj
            H += Hj
        
        #H = 0.5*(H + H.transpose())
        
        return H



class SUNSymmetric:
    
    def __init__(self, Ns, N, m, alpha, lattice):
        
        if not np.sum(alpha)==Ns*m:
            sys.exit('Problem: number of boxes in alpha must match Ns*m')
        
        if not lattice.Ns==Ns:
            sys.exit('lattice object does not have the correct number of sites.')
        
        if m>3:
            sys.exit('Code not yet implemented for m>3')
        
        self.N = N
        self.Ns = Ns
        self.m = m
        self.alpha = np.copy(alpha)
        self.n = np.sum(self.alpha)
        self.lattice = lattice
        self.Y = get_SYT_symm(self.alpha, self.m)
        self.CY = get_column(self.Y)
        self.NY = self.Y.shape[0]
        
    
    def sun_hamiltonian(self):
        # Compute the matrix of the SU(N) Heisenberg model
        # 
        
        H = scipy.sparse.csr_matrix((self.NY, self.NY))
        
        nlinks = self.lattice.nlinks
        
        for j in range(0, nlinks):
            
            Hj = scipy.sparse.csr_matrix((self.NY, self.NY))
            
            for i in range(0, self.NY):
                
                y = np.copy(self.Y[i, :])
                cy = np.copy(self.CY[i, :])
                
                y2, cy2, coeff2 = developp_symmetric(self.alpha, y, cy, self.m, self.lattice.links[j][0], self.lattice.links[j][1])
                ny2 = len(coeff2)
                
                row = np.full(shape=(ny2,), fill_value=i, dtype=int)
                col = np.zeros(shape=(ny2,), dtype=int)
                
                for t in range(0, ny2):
                    yfriend = np.copy(y2[t, :])
                    # search index
                    index = np.argwhere(np.sum(abs(self.Y - yfriend), axis=1) < 1e-13).flatten()[0]
                    
                    col[t] = index
                    
                # end for t
                
                Hj += scipy.sparse.csr_matrix( (coeff2, (row, col)), shape=(self.NY, self.NY))
                
            # end for i
            
            H += Hj
        
        # end for j
        
        H = 0.5 * (H + H.transpose())
        
        return H



class SUNAntiSymmetric:
    
    def __init__(self, Ns, N, m, alpha, lattice):
        
        if not np.sum(alpha)==Ns*m:
            sys.exit('Problem: number of boxes in alpha must match Ns*m')
        
        if not lattice.Ns==Ns:
            sys.exit('lattice object does not have the correct number of sites.')
        
        if m>2:
            sys.exit('Code not yet implemented for m>2.')
        
        self.N = N
        self.Ns = Ns
        self.m = m
        self.alpha = np.copy(alpha)
        self.n = np.sum(self.alpha)
        self.lattice = lattice
        self.Y = get_SYT_antisymm(self.alpha, self.m)
        self.CY = get_column(self.Y)
        self.NY = self.Y.shape[0]
        
    
    def sun_hamiltonian(self):
        # Compute the matrix of the SU(N) Heisenberg model
        # 
        
        H = scipy.sparse.csr_matrix((self.NY, self.NY))
        
        nlinks = self.lattice.nlinks
        
        for j in range(0, nlinks):
            
            Hj = scipy.sparse.csr_matrix((self.NY, self.NY))
            
            for i in range(0, self.NY):
                
                y = np.copy(self.Y[i, :])
                cy = np.copy(self.CY[i, :])
                
                y2, cy2, coeff2 = developp_antisymmetric(self.alpha, y, cy, self.m, self.lattice.links[j][0], self.lattice.links[j][1])
                ny2 = len(coeff2)
                
                row = np.full(shape=(ny2,), fill_value=i, dtype=int)
                col = np.zeros(shape=(ny2,), dtype=int)
                
                for t in range(0, ny2):
                    yfriend = np.copy(y2[t, :])
                    # search index
                    index = np.argwhere(np.sum(abs(self.Y - yfriend), axis=1) < 1e-13).flatten()[0]
                    
                    col[t] = index
                    
                # end for t
                
                Hj += scipy.sparse.csr_matrix( (coeff2, (row, col)), shape=(self.NY, self.NY))
                
            # end for i
            
            H += Hj
        
        # end for j
        
        H = 0.5 * (H + H.transpose())
        
        return H
