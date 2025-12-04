# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np
import scipy
import scipy.sparse

import sun



def set_overall_phase(COEFF_REF):
    """
    Set the overall phase of a series of SDCs
    
    Parameters
    ----------
    COEFF_REF : numpy array
        set of SDCs (stored in the columns)
    
    Returns
    -------
    COEFF_REF : numpy array
        rotated SDCs with correct phase convention
    IND_REF : numpy array
        positions of non-zero elements in COEFF_REF
    """
    
    nu1nu2nu = COEFF_REF.shape[1]
    
    if (nu1nu2nu==1):
        # The coefficient of the first (in increasing order of the LLOS) non-zero
        # term must be positive
        IND_REF = np.argwhere(abs(COEFF_REF[:, 0])>1.0e-12)
        if (COEFF_REF[IND_REF[0, 0], 0]<0):
            COEFF_REF *= -1.0
    else:
        # find if some coefficients==0
        ind_a = np.argwhere( abs(COEFF_REF[:,0])<1.0e-12 ).flatten()
        ind_b = np.argwhere( abs(COEFF_REF[:,1])<1.0e-12 ).flatten()
        
        if (len(ind_a)>0) | (len(ind_b)>0):
            sys.exit('Might cause a problem here. Need to treat this case.')
        
        # We apply a rotation in order to set to 0 the last component of one of
        # the vectors
        a = COEFF_REF[-1, 0]
        b = COEFF_REF[-1, 1]
        ii = COEFF_REF.shape[0] # size(COEFF_REF,1);
        
        while ( (abs(a**2+b**2)<1.0e-12) & (ii>0) ): # necessary to avoid possible division by 0 in eta
            ii -= 1
            a = COEFF_REF[ii, 0]
            b = COEFF_REF[ii, 1]
        
        eta = np.sqrt( b**2/(a**2+b**2) )
        
        c1 = eta*COEFF_REF[:,0] + np.sqrt(1.0-eta**2)*COEFF_REF[:,1]
        c2 = np.sqrt(1.0-eta**2)*COEFF_REF[:,0] - eta*COEFF_REF[:,1]
        
        if abs(c1[-1])>1.0e-12:
            # we made the wrong choice of phase in lambdaa ==> need to correct
            c1 = eta*COEFF_REF[:,0] - np.sqrt(1.0-eta**2)*COEFF_REF[:,1]
            c2 = np.sqrt(1.0-eta**2)*COEFF_REF[:,0] + eta*COEFF_REF[:,1]
        
        ind_c1 = np.argwhere(abs(c1)>1.0e-12)
        if (c1[ind_c1[0]]<0):
            c1 *= -1
        ind_c2 = np.argwhere(abs(c2)>1.0e-12)
        if (c2[ind_c2[0]]<0):
            c2 *= -1
        
        COEFF_REF[:, 0] = c1
        COEFF_REF[:, 1] = c2
        
        IND_REF = np.stack((ind_c1, ind_c2))
    
    return COEFF_REF, IND_REF



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
    COEFF_FINAL : numpy array
        SDCs. COEFF_FINAL is of dimension (NY, NY2, Ntau) where
            NY   : is the number of SYTs for irrep nu with n1 first particles fixed for nu1
            NY2  : is the number of SYTs in Y2 if Y2 is provided. Otherwise, it is the number
                   of SYTs for nu2
            Ntau : is the multiplicity of nu in the tensor product of nu1 with nu2,
                   defining two sequences of SDCs
    
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
    Y, CY, sdcs = get_SDC(N, nu, nu1, nu2)
    
    This reproduces the 2nd, 3rd and 4th lines of Table 4.18 2a page 187 of
    Ref. [1].
    
    Example 2:
    N = int(3)
    nu = np.array([4, 4, 3], dtype=int)
    nu1 = np.array([3, 3, 1], dtype=int)
    nu2 = np.array([2, 1, 1], dtype=int)
    Y, CY, sdcs = get_SDC(N, nu, nu1, nu2)
    
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
    Y = sun.get_subSYT(nu, nu1, order='LLOS')
    NY = Y.shape[0]
    
    # Fill all obtained SYTs with numbers from 0 to n1-1
    Y = np.hstack((np.matlib.repmat(y1, NY, 1), Y))
    CY = sun.get_column(Y)
    
    # Compute the matrices of the adjacent transpositions for S_{n_2}
    MatAdjaTranspo = []
    for k in range(n1, n-1):
        MatAdjaTranspo.append( sun.get_adjacent_transposition_matrix(nu, Y, CY, k) )
    
    # Compute Casimir (eigenvalue of CSCO-II) for the canonical chain
    # associated to S_{n_2}
    casimir = casimir_canonical_chain(nu2, y2_ref)
    
    # construct projection operator
    M = build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, casimir, ordering='math')
    
    # compute kernel of projection operator
    DD, COEFF_REF = scipy.sparse.linalg.eigsh(M, k=min(4, NY-1), which='SA')
    # keep only eigenvectors with eigenvalue==0
    ind = np.argwhere(abs(DD)<1.0e-12).flatten()
    assert len(ind)==nu1nu2nu
    COEFF_REF = COEFF_REF[:, ind]
    
    # Fix the overall phase
    COEFF_REF, IND_REF = set_overall_phase(COEFF_REF)
    
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

    COEFF_FINAL = np.zeros(shape=(NY2, NY, nu1nu2nu), dtype=float)
    # indices for COEFF_FINAL are (m2, m, tau)

    # Compute SDCs
    
    for i2 in range(0, NY2):
    
        y2 = Y2[i2]
        
        deal_with_phase = False
        
        if (np.sum(abs(y2-y2_ref))!=0):
            deal_with_phase = True
        
        if deal_with_phase==False:
            
            COEFF_FINAL[i2, :, :] = COEFF_REF
            
        else:
            
            # find the sequence of operations which transfrom y2_ref into y2
            sigma, rho = SYT_to_target(y2_ref, y2)
            # y2 = sigma[-1] ... sigma[0] y2_ref
            
            # shift sigma by n1 to go from local to global numbering
            sigma = n1 + sigma            
            
            # apply the sequence of transpositions
            
            for tau in range(0, nu1nu2nu):
                ydev1, cydev1, coeff1 = apply_transpositions(
                                                nu, 
                                                sigma, 
                                                rho, 
                                                np.copy(Y[IND_REF[:, tau]]), 
                                                np.copy(CY[IND_REF[:, tau]]), 
                                                np.copy(COEFF_REF[IND_REF[:, tau]].flatten()))
                
                # now the SYTs in ydev1 are not necessarily stored in ascending 
                # order of LLOS
                # sort SYTs according to Y (--> LLOS)
                COEFF_FINAL[i2, :, tau] = sort_development(Y, ydev1, coeff1)
    
    return Y, CY, COEFF_FINAL




def casimir_canonical_chain(nu, y):
    """
    Get the eigenvalues of the quadratic Casimir operator for the canonical 
    group chain of nu defined by SYT y.
    
    Parameters
    ----------
    nu : numpy array
        irrep
    y : numpy array
        SYT
    
    Returns
    -------
    casimir : numpy array
        eigenvalues of CSCO-II of y of nu
    
    References
    ----------
    [1]         Group Representation Theory for Physicists
                Jin-Quan Chen, Hialun Ping and Fan Wang
                World Scientific, 2nd edition, (2002)
    """
    
    n = np.sum(nu)
    nup = np.copy(nu)
    casimir = np.zeros(shape=(n,), dtype=float)
    for j in range(n-1, -1, -1):
        casimir[j] = sun.casimir_quadratic(nup)
        nup[y[j]] -= 1
    
    return casimir


'''
def build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, casimir):
    
    M = scipy.sparse.csr_matrix((NY, NY))
    
    for j in range(1, n-n1):
        Mj = scipy.sparse.csr_matrix((NY, NY))    
        for k in range(0, j):
            transpo = np.array([n1+k, n1+j], dtype=int)
            sigma = sun.transposition_to_adjacent_transpositions(transpo)
            Mtemp = scipy.sparse.eye(NY)    
            for atr in sigma:
                Mtemp = Mtemp @ MatAdjaTranspo[atr-n1]
            Mj += Mtemp
        val = -(casimir[j] - casimir[j-1])
        Mj = Mj + val*scipy.sparse.eye(NY)
        M += (Mj @ Mj)
    
    return M
'''


def build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, casimir, ordering='dmrg'):
    """
    Build projection operator of the CSCO-II associated with input casimir values
    """
    
    M = scipy.sparse.csr_matrix((NY, NY))
    
    if ordering=='math':
        f = lambda j, k : (n1+k, n1+j)
    elif ordering=='dmrg':
        f = lambda j, k : (n-1-j, n-1-k)
    
    for j in range(1, n-n1):
        Mj = scipy.sparse.csr_matrix((NY, NY))    
        for k in range(0, j):
            transpo = np.array(f(j, k))
            sigma = sun.transposition_to_adjacent_transpositions(transpo)
            Mtemp = scipy.sparse.eye(NY)    
            for atr in sigma:
                Mtemp = Mtemp @ MatAdjaTranspo[atr-n1]
            Mj += Mtemp
        val = -(casimir[j] - casimir[j-1])
        Mj = Mj + val*scipy.sparse.eye(NY)
        M += (Mj @ Mj)
    
    return M



def get_SDC_dmrg(N, nu, nu1, l1, nu2, l2, ref2firstLLOS=True, ref1firstLLOS=True):
    """
    Compute the permutational Subduction Coefficients (SDCs) in the convention 
    useful for DMRG.
    
    Parameters
    ----------
    N : int
        SU(N)
    nu : numpy array
        global irrep with n particles
    nu1 : numpy array
        irrep with n1 particles
    l1 : int
        bottom corner position of last particle in nu1
    nu2 : numpy array
        irrep with n2 particles, n=n1+n2
    l2 : int
        bottom corner position of last particle in nu2
    ref1firstLLOS : bool
        reference SYT used for y1
    ref2firstLLOS : bool
        if True, the first SYT in the ascending order of the LLOS is used as 
        reference for fixing the overall phase convention.  If False, the last 
        SYT is used (namely the first in the descending order of LLOS)
    
    Returns
    -------
    Y : numpy array
        SYTs for the irrep nu, with particle n1-1 situated at bottom corner l1 
        and n2 last particles satisfying symmetries imposed by (nu2, l2)
    CY : numpy array
        associated column positions
    COEFF_FINAL : numpy array
        SDCs. COEFF_FINAL is of dimension (NY, Ntau) where
            NY   : is the number of SYTs in the output Y
            Ntau : is the multiplicity of nu in the tensor product of nu1 with nu2,
                   defining two sequences of SDCs
    
    Remarks
    -------
    - ref1firstLLOS has no influence on the SDCs. It does have an influence on
      the SYTs in the output Y.
    
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
    
    n = np.sum(nu)
    n1 = np.sum(nu1)
    n2 = np.sum(nu2)
    assert n==n1+n2
    
    # Construct tableau S1 with the last number situated at bottom corner l1
    # Remark: the exact form of S1 is not important. The two only important
    # things are:
    # (1) it is a valid SYT
    # (2) the last number is situated in the bottom corner at row l1
    # 
    # In particular, that means that all other particles can be placed
    # arbitrarily, as long as they satisfy (1) above.
    
    # One possible choice: First SYT in LLOS for particles 0 -> n1-2
    nu1p = np.copy(nu1)
    nu1p[l1] -= 1
    if ref1firstLLOS==True:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='LLOS')
    else:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='iLLOS')
    y1 = np.hstack((y1, l1))
    
    # construct reference SYT for nu2
    if ref2firstLLOS==True:
        # the reference is the first SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='LLOS')
    else:
        # the reference is the last SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='iLLOS')

    # Construct all SYTs associated to the shape nu-nu1, in increasing order of LLOS
    Y = sun.get_subSYT(nu, nu1, order='LLOS')
    NY = Y.shape[0]

    # Fill all obtained SYTs with numbers from 1 to n1
    Y = np.hstack((np.matlib.repmat(y1, NY, 1), Y))
    CY = sun.get_column(Y)
    
    # Compute the matrices of the adjacent transpositions for S_{n_2}
    MatAdjaTranspo = []
    for k in range(n1, n-1):
        MatAdjaTranspo.append( sun.get_adjacent_transposition_matrix(nu, Y, CY, k) )
    
    # Compute Casimir (eigenvalue of CSCO-II) for the canonical chain
    # associated to S_{n_2}
    casimir = casimir_canonical_chain(nu2, y2_ref)
    
    # construct projection operator
    M = build_canonical_chain_projector(NY, n, n1, MatAdjaTranspo, casimir, ordering='dmrg')
    
    # compute kernel of projection operator
    if M.shape[0]==1:
        assert abs(M[0,0])<1.0e-12
        COEFF_REF = np.array([[1.0]], dtype=float)
        DD = np.array([0.0], dtype=float)
    elif M.shape[0]<100:
        DD, COEFF_REF = np.linalg.eigh(M.todense())
        COEFF_REF = np.asarray(COEFF_REF)
    else:
        DD, COEFF_REF = scipy.sparse.linalg.eigsh(M, k=min(4, NY), which='SA')
    
    # keep only eigenvectors with eigenvalue==0
    ind = np.argwhere(abs(DD)<1.0e-12).flatten()
    assert len(ind)==nu1nu2nu
    COEFF_REF = COEFF_REF[:, ind]
    
    # Fix the overall phase
    COEFF_REF, IND_REF = set_overall_phase(COEFF_REF)
    
    deal_with_phase = False
    if (abs(l2-y2_ref[-1])>0):
        deal_with_phase = True
    
    # Compute SDCs
    
    if deal_with_phase==0:
        # the last particle in y2_ref is situated at bottom corner l2 in nu2, 
        # thus no need to do any more phase fixing
        COEFF_FINAL = COEFF_REF
    else:
        # one needs to find the sequence of transpositions which brings 
        # local particle n2-1 (namely n-1 in global numbering) at row l2 in nu2
        
        # construct SYT y2 with bottom corner at l2
        nu2p = np.copy(nu2)
        nu2p[l2] -= 1
        if ref2firstLLOS==True:
            y2 = sun.index_to_SYT(0, alpha=nu2p, order='LLOS')
        else:
            y2 = sun.index_to_SYT(0, alpha=nu2p, order='iLLOS')
        y2 = np.hstack((y2, l2))
        
        # find sequence of transpositions which brings y2_ref to y2
        sigma, rho = SYT_to_target(y2_ref, y2)
        
        # y2 = sigma[-1] sigma[-2] ... sigma[1] sigma[0] y2_ref
        
        # IMPORTANT CORRESPONDANCE
        # y2_ref[i] == row of number i, 0<=i<n2
        # <==>
        # y2_ref[i] == row of number n-i-1, 0<=i<n2 in the global SYT
        # same for y2
        
        # go from local to global numbering
        sigma = n - sigma - 2
        
        COEFF_FINAL = np.zeros(shape=COEFF_REF.shape, dtype=float)
        
        # apply the sequence of transpositions
        
        for tau in range(0, nu1nu2nu):
            ydev1, cydev1, coeff1 = apply_transpositions(
                                            nu, 
                                            sigma, 
                                            rho, 
                                            np.copy(Y[IND_REF[:, tau]]), 
                                            np.copy(CY[IND_REF[:, tau]]), 
                                            np.copy(COEFF_REF[IND_REF[:, tau]].flatten()))
            
            # now the SYTs in ydev1 are not necessarily stored in ascending 
            # order of LLOS
            # sort SYTs according to Y (--> LLOS)
            COEFF_FINAL[:, tau] = sort_development(Y, ydev1, coeff1)
        
    return Y, CY, COEFF_FINAL



def apply_transpositions(nu, sigma, rho, ydev1, cydev1, coeffdev1):
    """
    Apply a sequence of transposition operators defined by (sigma, rho) to a 
    development
    
    Let T[i] = (P_{sigma[i], sigma[i]+1} + rho[i])/sqrt(1-rho[i]**2) for i=0, ..., len(sigma)-1
    
    Then we compute
        |output development> = T[-1] ... T[1] T[0] |input development>
    
    where
    
        |input development> = coeffdev1[0] |ydev1[0]> + ... + coeffdev1[-1] |ydev1[-1]>
    
    Parameters
    ----------
    nu : numpy array
        irrep
    sigma : numpy array
        sequence of adjacent transpositions
    rho : numpy array
        inverses of axial distances
    ydev1 : numpy array
        SYT development
    cydev1 : numpy array
        associated column positions
    coeffdev1 : numpy array
        associated coefficients
    
    Returns
    -------
    ydev1, cydev1, coeffdev1 : updated development
    """
    
    for ll in range(0, len(sigma)):
        
        ydev2, cydev2, coeffdev2 = sun.develop_consecutive_number(
                                        nu, 
                                        ydev1, cydev1, coeffdev1, 
                                        k=sigma[ll])
        
        ydev2, cydev2, coeffdev2 = sun.sum_develop(ydev1, cydev1, rho[ll]*coeffdev1, 
                                                   ydev2, cydev2, coeffdev2)
        
        ydev2, cydev2, coeffdev2 = sun.fullsimplify_development(ydev2, cydev2, coeffdev2)
        
        coeffdev2 = coeffdev2/np.sqrt(1.0-rho[ll]**2)
        
        ydev1 = ydev2
        cydev1 = cydev2
        coeffdev1 = coeffdev2
    
    return ydev1, cydev1, coeffdev1



def apply_transpositions_v2(nu, sigma, rho, ydev, cydev, coeffdev, sort=False):
    """
    Apply a sequence of transposition operators defined by (sigma, rho) to a 
    development
    
    Let T[i] = (P_{sigma[i], sigma[i]+1} + rho[i])/sqrt(1-rho[i]**2) for i=0, ..., len(sigma)-1
    
    Then we compute
        |output development> = T[-1] ... T[1] T[0] |input development>
    
    where
    
        |input development> = coeffdev1[0] |ydev1[0]> + ... + coeffdev1[-1] |ydev1[-1]>
    
    Parameters
    ----------
    nu : numpy array
        irrep
    sigma : numpy array
        sequence of adjacent transpositions
    rho : numpy array
        inverses of axial distances
    ydev : numpy array
        SYT development
    cydev : numpy array
        associated column positions
    coeffdev : numpy array
        associated coefficients
    sort : bool [optional], default: False
        if True, sort the output SYTs in ascending order of LLOS
    
    Returns
    -------
    ydev1, cydev1, coeffdev1 : updated development
    """
    
    for ll in range(0, len(sigma)):
        ydev, cydev, coeffdev = sun.t_operator(ydev, cydev, coeffdev, 
                                               k=sigma[ll], 
                                               rho=rho[ll])
    
    if sort==True:
         ydev, ind = sun.sort_SYT(ydev, order='LLOS')
         cydev = cydev[ind, :]
         coeffdev = coeffdev[ind]
    
    return ydev, cydev, coeffdev


def apply_transpositions_v3(nu, sigma, rho, ydev, cydev, coeffdev, ifd, sort=False):
    """
    Apply a sequence of transposition operators defined by (sigma, rho) to a 
    development
    
    Let T[i] = (P_{sigma[i], sigma[i]+1} + rho[i])/sqrt(1-rho[i]**2) for i=0, ..., len(sigma)-1
    
    Then we compute
        |output development> = T[-1] ... T[1] T[0] |input development>
    
    where
    
        |input development> = coeffdev1[0] |ydev1[0]> + ... + coeffdev1[-1] |ydev1[-1]>
    
    Parameters
    ----------
    nu : numpy array
        irrep
    sigma : numpy array
        sequence of adjacent transpositions
    rho : numpy array
        inverses of axial distances
    ydev : numpy array
        SYT development
    cydev : numpy array
        associated column positions
    coeffdev : numpy array
        associated coefficients
    
    Returns
    -------
    ydev1, cydev1, coeffdev1 : updated development
    """
    print('CAUTION: apply_transpositions_v3 is EXPERIMENTAL. Currently, it is \n' 
          + 'most of the time better to use apply_transpositions_v2.')
    
    n = ydev.shape[1]
    
    # find the number of different patterns in the last ifd particles
    _, ia, ic = np.unique(ydev[:, n-ifd:], 
                          return_index=True, 
                          return_inverse=True, 
                          axis=0)
    N_disjoint_groups = len(ia)
    
    max_N_groups = int(40)
        
    # define the actual number of groups to deal with
    N_groups = min(N_disjoint_groups, max_N_groups)
    
    q_groups = N_disjoint_groups // N_groups # number of true groups into an effective group
    r_groups = N_disjoint_groups % N_groups #  remaining groups
    
    ind_groups = []
    for p in range(0, N_groups):
        ind_groups.append( np.argwhere(( ic >= p*q_groups ) & ( ic < (p+1)*q_groups )).flatten() )
    
    # we distribute the <r_groups> remaining groups among the first
    # <r_groups> formed above
    for p in range(0, r_groups):
        icp = np.argwhere(ic==(p+q_groups*N_groups)).flatten()
        ind_groups[p] = np.hstack((ind_groups[p], icp))
    
    # apply transpositions on each  group
    ydev_groups = []
    cydev_groups = []
    coeffdev_groups = []
    
    for p in range(0, N_groups):
        ydev_p = np.copy(ydev[ind_groups[p]])
        cydev_p = np.copy(cydev[ind_groups[p]])
        coeffdev_p = np.copy(coeffdev[ind_groups[p]])
        
        ydev_p, cydev_p, coeffdev_p = apply_transpositions_v2(
                                nu, 
                                sigma, 
                                rho, 
                                ydev_p, 
                                cydev_p, 
                                coeffdev_p,
                                sort=sort)
        
        ydev_groups.append(ydev_p)
        cydev_groups.append(cydev_p)
        coeffdev_groups.append(coeffdev_p)
    
    # merge all groups
    ydev_out = ydev_groups[0]
    cydev_out = cydev_groups[0]
    coeffdev_out = coeffdev_groups[0]
    for p in range(1, N_groups):
        ydev_out = np.vstack((ydev_out, ydev_groups[p]))
        cydev_out = np.vstack((cydev_out, cydev_groups[p]))
        coeffdev_out = np.hstack((coeffdev_out, coeffdev_groups[p]))
    
    if sort==True:
         ydev_out, ind = sun.sort_SYT(ydev_out, order='LLOS')
         cydev_out = cydev_out[ind, :]
         coeffdev_out = coeffdev_out[ind]
    
    return ydev_out, cydev_out, coeffdev_out


def sort_development(Y, ydev, coeffdev):
    """
    Sort a development according to a collection of SYTs
    
    Parameters
    ----------
    Y : numpy array
        collection of SYTs (stored in the rows)
    ydev : numpy array
        SYT development
    coeffdev : numpy array
        coefficients of the SYTs in the development
    
    Returns
    -------
    coeff_sorted : numpy array (of dimension (Y.shape[0], ))
        coefficients of development, expressed in the order defined by Y
    
    Details
    -------
    
    """
    coeff_sorted = np.zeros(shape=(Y.shape[0],), dtype=float)
    
    for t in range(0, ydev.shape[0]):
        ind = np.argwhere( np.sum(abs(Y - ydev[t]), axis=1)==0 ).flatten()
        assert len(ind)==1
        ind = ind[0]
        coeff_sorted[ind] = coeffdev[t]
    
    return coeff_sorted



def SYT_to_target(y, ytarget):
    """
    Find the sequence of operations which brings an SYT to a target SYT
    
    Parameters
    ----------
    y : numpy array
        SYT to transfrom
    ytarget : numpy array
        target SYT
    
    Returns
    -------
    sigma : numpy array
        sequence of transpositions
    rho : numpy array
        array of inverses of axial distances
    
    Remark
    ------
    - The sequence of operations needs to be read from first to last element:
        y --------> y' --------> y'' ... --------> ytarget
          sigma[0]    sigma[1]           sigma[-1]
    - The operations necessary to obtain y from ytarget are obtained by flipping 
      the two output arrays and changing the sign of all elements of rho:
        sigma = np.flip(sigma)
        rho = np.flip(-rho)
    
    Details
    -------
    In terms of SYTs, we have:
        
        ytarget = sigma[-1] sigma[-2] ... sigma[1] sigma[0] y
    
    In terms of states:
        
        |ytarget> = (sigma[-1]+rho[-1])/sqrt(1-rho[-1])**2  ... (sigma[0]+rho[r0])/sqrt(1-rho[0]**2) |y>
    
    Examples
    --------
    Example 1:
              0 2       0 1                   0 2            0 1
    ytarget = 1     y = 2       --> ytarget = 1     = (1, 2) 2    = (1, 2) y
              3         3                     3              3
    
    and |ytarget> = (P_{(1,2)} + 1/2)/(sqrt(3)/2) |y>
    
    thus sigma = np.array([1]) and rho = np.array([1/2])
    
    Example 2:
              0 1       0 3                   0 1            0 2                 0 3
    ytarget = 2     y = 1       --> ytarget = 2     = (1, 2) 1   = (1, 2) (2, 3) 1    = (1, 2) (2, 3) y
              3         2                     3              3                   2
    
    thus sigma = [2, 1] and rho = [-1/3, -1/2]
    """
    
    assert len(y) == len(ytarget)
    n = len(y)
    cy = sun.get_column(y)
    cytarget = sun.get_column(ytarget)
    
    yp = np.copy(y)
    cyp = np.copy(cy)
    
    sigma = np.zeros(shape=(n*n,), dtype=int) # list of adjacent transpositions
    rho = np.zeros(shape=(n*n,), dtype=float) # list of inverse of axial distances
    
    cpt = int(0)
    
    for k in range(0, n):
        if yp[k]==ytarget[k]:
            # k is at good location
            pass
        else:
            row_target = ytarget[k] # row that k should occupy
            col_taret = cytarget[k] # column that k should occupy
            
            # find index of particle that occupies the location that k SHOULD occupy
            yx = np.stack((yp, cyp), axis=1)
            yf = np.array([row_target, col_taret])
            kprime = np.argwhere( np.sum( abs( yx - yf ), axis=1 )==0 ).flatten()
            assert len(kprime)==1
            kprime = kprime[0]
            
            for ll in range(kprime-1, k-1, -1):
                sigma[cpt] = ll
                axtemp = sun.get_axial_distance(yp, cyp, ll, ll+1)
                rho[cpt] = 1.0/axtemp
                
                y_new = np.copy(yp)
                cy_new = np.copy(cyp)
                
                y_new[ll] = yp[ll+1]
                y_new[ll+1] = yp[ll]
                cy_new[ll] = cyp[ll+1]
                cy_new[ll+1] = cyp[ll]
                
                yp = y_new
                cyp = cy_new
                
                cpt += 1
    
    sigma = sigma[:cpt]
    rho = rho[:cpt]
    assert np.sum(abs(yp-ytarget))==0
    assert np.sum(abs(cyp-cytarget))==0
    
    return sigma, rho


def get_descendants(alpha, N):
    """
    Get all descendants shapes
    """
    assert len(alpha<=N)
    
    alpha = np.hstack((alpha, np.zeros(shape=(N-len(alpha)), dtype=int)))
    
    alpha_desc = np.zeros(shape=(N+1, N), dtype=int)
    box_pos = np.zeros(shape=(N+1,), dtype=int)

    # first descendant is obtained by adding a box in the first row
    alpha_desc[0] = alpha
    alpha_desc[0][0] += 1
    box_pos[0] = 0
    
    # all others descendants are obtained by putting a box in a bottom corner
    bc_vec = sun.get_bottom_corner(alpha)
    bc_vec = bc_vec[bc_vec!=N-1]
    
    cpt = int(1)
    for bc in bc_vec:
        alpha_desc[cpt] = alpha
        alpha_desc[cpt][bc+1] += 1
        box_pos[cpt] = bc+1
        cpt += 1
    
    alpha_desc = alpha_desc[0:cpt]
    box_pos = box_pos[0:cpt]
    
    return alpha_desc, box_pos

