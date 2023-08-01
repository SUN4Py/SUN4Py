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
    
    for i in range(0, nlinks):
        nbtranspositions[i] = 2*abs(links[i][1]-links[i][0]) - 1
    
    listtranspositions = [None] * nlinks
    
    for i in range(0, nlinks):
        mini = np.min(links[i])
        maxi = np.max(links[i])
        
        listtranspositions[i] = np.zeros((nbtranspositions[i], ), dtype=int)
        listtranspositions[i][0:((nbtranspositions[i]-1)/2+1).astype(int)] = np.arange(mini, maxi)
        listtranspositions[i][((nbtranspositions[i]-1)/2+1).astype(int):] = np.arange(maxi-2, mini-1, -1)
    
    return listtranspositions, nbtranspositions


def get_axial_distance(y, cy, i, j):
    # Compute axial distance from i to j in SYT y (and columns cy).
    # 
    
    ad = cy[i] - y[i] - cy[j] + y[j];
    
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
    # Compute the dimension of the irrep alpha of SU(N).
    #
    
    nl = len(np.argwhere(alpha>0).flatten())
    if nl>N:
        sys.exit('Problem: alpha has more rows than N.')
    alphap = np.zeros((N,), dtype=int)
    alphap[0:nl] = alpha[0:nl]
    alpha = alphap
    alpha = alpha - np.full(shape=(N,), fill_value=alpha[N-1], dtype=int) # remove columns with N boxes
    
    # Numerator
    arnum = np.full(shape=(nl*alpha[0]), fill_value=1, dtype=int)
    cpt = int(0)
    
    for i in range(0, nl):
        for j in range(0, alpha[i]):
            arnum[cpt] = N-i+j
            cpt += 1
    
    arnum = arnum[0:cpt]
    
    # Denominator: product of Hook lengths
    m = alpha[0]
    alphafull = np.zeros((nl, m), dtype=int)
    for i in range(0, nl):
        for j in range(0, alpha[i]):
            alphafull[i,j] = 1
    
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
    
    d = num//denom
    
    return d


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



def print_to_latex(y, sh, sc):
    # Print a SYT to text in LaTeX format, using \ytableau
    # 
    
    nl = np.max(y) + 1 # number of rows in the shape
    
    print('\\shsc{' + '{:.1f}'.format(sh) + 'pt}{' + '{:.1f}'.format(sc) + '}{\\begin{ytableau}')
    
    for j in range(0, nl):
        # find all numbers situated in the j-th row
        indj = np.argwhere(y==j).flatten()
        
        # print all numbers of the j-th row
        line = ''
        for k in range(0, len(indj)-1):
            line += str(indj[k]+1) + ' & '
        
        line += str(indj[-1]+1) + ' \\\\ '
        print(line)
    
    print('\\end{ytableau}}\n')
    
    return



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
        # Compute the matrix of the SU(N) Heisenberg model
        # 
        
        NH = self.falpha
        
        listtranspositions, nbtranspositions = get_transpositions(self.lattice.links)
        
        Pdiagk = np.zeros((self.n-1, NH)) # value of diagonal elements
        Lnondiagk = np.zeros((self.n-1, NH)) # indices des lignes des éléments hors-diagonaux
        Cnondiagk = np.zeros((self.n-1, NH)) # indices des colonnes des éléments hors-diagonaux
        Valnondiag = np.zeros((self.n-1, NH)) # valeurs des indices hors-diagonaux
    
        numbernondiag = np.zeros((self.n-1, ), dtype=int); # number of off-diag elements for each transposition
        
        for k in range(0, self.n-1):
            
            count = 0
            
            for i in range(0, NH):
                
                y = np.copy(self.Y[i, :])
                cy = np.copy(self.CY[i, :])
                
                if Pdiagk[k, i]==0:
                    if y[k]==y[k+1]:
                        Pdiagk[k, i] = 1
                    else:
                        
                        c1 = cy[k]
                        c2 = cy[k+1]
                        
                        if c1==c2:
                            Pdiagk[k, i] = -1
                        else:
                            rho = 1.0/get_axial_distance(y, cy, k, k+1)
                            yfriend = np.copy(y)
                            yfriend[k] = y[k+1]
                            yfriend[k+1] = y[k]
                            
                            # index = np.argwhere(np.sum(abs(self.Y - yfriend), axis=1) < 1e-13).flatten()[0]
                            index = i + np.argwhere(np.sum(abs(self.Y[i:,:] - yfriend), axis=1)<1.0e-13).flatten()[0]
                            
                            Pdiagk[k, i] = -rho;
                            Pdiagk[k, index] = rho;
                    
                            # non diagonal element
                            Lnondiagk[k, count] = i # index of row
                            Cnondiagk[k, count] = index # index of column
                            Valnondiag[k, count] = np.sqrt(1.0-rho**2) # non-diagonal value
                            count += 1
                        # end if c1==c2
                    # end if y[k]==y[k+1]
                # end if Pdiagk[k,i]==0
            # end for i
            
            numbernondiag[k] = count;
            
        # end for k
        
        H = scipy.sparse.csr_matrix((NH, NH))
        
        nlinks = self.lattice.nlinks
        
        for j in range(0, nlinks):
            
            Hj = scipy.sparse.eye(NH)
            
            for l in range(nbtranspositions[j]-1, -1, -1):
                
                k = listtranspositions[j][l]
                
                Uk = scipy.sparse.diags(Pdiagk[k,:], offsets=0, shape=(NH,NH))
                
                Ukoffdiag = scipy.sparse.csr_matrix((Valnondiag[k, 0:numbernondiag[k]], (Lnondiagk[k, 0:numbernondiag[k]], Cnondiagk[k, 0:numbernondiag[k]])), shape=(NH, NH))
                
                Uk += Ukoffdiag
                Uk += Ukoffdiag.transpose()
                
                Hj = Uk @ Hj
            
            H = H + Hj
        
        H = 0.5*(H + H.transpose())
        
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
