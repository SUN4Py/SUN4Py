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



def get_SDC(N, nu, nu1, l1, nu2, l2, ref2firstLLOS=True, ref1firstLLOS=True):
    """
    Compute subduction coefficients in DMRG framework using the shortcut in Chen's
    method, developing irreps as products of their columns or rows
    
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
    ref2firstLLOS : bool [OPTIONAL][DEFAULT: True]
        if True, the first SYT in the ascending order of the LLOS is used as 
        reference for fixing the overall phase convention. This corresponds to
        a development accross rows
        if False, the last SYT is used (namely the first in the descending 
        order of LLOS). This corresponds to a development accross columns
    ref1firstLLOS : bool [OPTIONAL][DEFAULT: True]
        reference SYT used for y1
    
    Returns
    -------
    ydev_final : numpy array
        SYTs
    ydev_final : numpy array
        associated columns
    coeffdev_final : numpy array
        coefficients
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

    n = np.sum(nu)
    n1 = np.sum(nu1)
    n2 = np.sum(nu2)
    assert n==n1+n2
    
    nu1nu2nu = sun.multiplicity_irrep_mixed(nu, np.array([nu1, nu2], dtype=int), N)
    assert(nu1nu2nu==1)
    # currently, we have not dealt with the case of multiplicities in this 
    # advanced method
    
    # construct SYT for nu1
    nu1p = np.copy(nu1)
    nu1p[l1] -= 1
    if ref1firstLLOS==True:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='LLOS')
    else:
        y1 = sun.index_to_SYT(0, alpha=nu1p, order='iLLOS')
    y1 = np.hstack((y1, l1))
    
    # construct reference SYT for nu2
    
    if ref2firstLLOS==True:
        # we will develop wrt rows
        constraint = 'symm'
        symmetry = 'rows'
        si = 1.0
        nu2_rc = np.copy(nu2)
        nr_nu2 = len(np.argwhere(nu2>0).flatten()) # number of non-vanishing rows
        nb_rc_nu2 = nr_nu2
        # the reference is the first SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='LLOS')
    elif ref2firstLLOS==False:
        # we will develop wrt columns
        constraint = 'antisymm'
        symmetry = 'cols'
        si = -1.0
        nu2_col = sun.transpose_shape(nu2)
        nu2_rc = np.copy(nu2_col)
        nc_nu2 = nu2[0] # number of columns in nu2
        nb_rc_nu2 = nc_nu2
        # the reference is the last SYT in the increasing order of LLOS
        y2_ref = sun.index_to_SYT(0, alpha=nu2, order='iLLOS')        
    else:
        sys.exit('Problem: undefined ref2firstLLOS')
    
    # Construct all SYTs associated to the shape nu-nu1, in increasing order of LLOS
    Y = get_sSYT_constraint(nu, nu1, nu2, constraint=constraint,
                            fill_nu1=True, fill_y1=y1)
    NY = Y.shape[0]
    assert NY>0
    
    if symmetry=='rows':
        # need to compute the column positions
        CY = sun.get_column(Y)
    else:
        CY = Y
    
    if NY==1:
        coeffdev_r = np.array([1.0], dtype=float)
    else:
        
        Proj = np.zeros(shape=(NY, NY), dtype=float)
        
        for k in range(0, nb_rc_nu2-1):
            
            # all numbers of the k-th row/col of y2_ref, numbered with global DMRG numbering
            rc1 = np.flip( np.arange(n-1 - np.sum(nu2_rc[0:k]), n-1 - np.sum(nu2_rc[0:k+1]), -1) )
            lrc1 = len(rc1)
            
            # all numbers of the (k+1)-th row/col of y2_ref, numbered with global DMRG numbering
            rc2 = np.flip(np.arange(n-1 - np.sum(nu2_rc[0:k+1]), n-1 - np.sum(nu2_rc[0:k+2]), -1))
            
            # extract number situated in first position of the (k+1)-th row/col
            nkp1 = rc2[-1] # last element because we have flipped rc2
            assert(nkp1+1==rc1[0])
            
            # all numbers of nu2 which are not in the k-th and (k+1)-th rows/cols
            # of y2_ref (global numbering)
            orc = np.setdiff1d(np.arange(n1, n), np.hstack((rc1, rc2)))
            
            Hk_diag = np.zeros(shape=(NY,1))
            Hk_offdiag = np.zeros(shape=(NY,NY))
            
            for j in range(0, NY):
                
                # find state which is symmetric wrt row k and row k+1
                # respectively antisymmetric wrt col k and col k+1
                ydev_j, coeffdev_j = develop_symmetry_two_ensembles(nu, Y[j], rc1, rc2, symmetry=symmetry)
                cydev_j = sun.get_column(ydev_j)
                Nydev_j = ydev_j.shape[0]
                
                # apply permutation which interchanges the last number of
                # row/col k with the first number of row/col k+1
                
                # diagonal part
                
                valdiag = 0.0
                
                for p in range(0, Nydev_j):                
                    if (ydev_j[p, nkp1] == ydev_j[p, nkp1+1]):
                        valdiag += coeffdev_j[p]**2
                    else:
                        if (cydev_j[p, nkp1]==cydev_j[p, nkp1+1]):
                            valdiag -= coeffdev_j[p]**2
                        else:
                            ax = sun.get_axial_distance(ydev_j[p], cydev_j[p], nkp1, nkp1+1)
                            rho = - 1.0/ax
                            valdiag += rho*coeffdev_j[p]**2
                
                Hk_diag[j] = valdiag
                
                # Off-diagonal part
                
                if (j<NY-1):
                    
                    for i in range(j+1, NY):
                        
                        # check if all "other particles" (not in rows/cols k 
                        # and k+1) are situated in the same places in equivalence 
                        # class i than in equivalence class j
                        condition1 = ( np.sum( abs( Y[i, orc] - Y[j, orc] ) ) == 0 )
                        
                        '''
                        # extract rows of numbers of row k and row k+1 for
                        # each class
                        tempi = np.sort(Y[i, np.hstack((rc1, rc2))])
                        tempj = np.sort(Y[j, np.hstack((rc1, rc2))])
                        condition1prime = ( np.sum( abs(tempi-tempj) ) == 0 )
                        
                        # condition1prime is redundant
                        if condition1==True:
                            assert(condition1prime==True)
                        
                        condition1 *= condition1prime
                        '''
                        
                        if condition1==True:
                            
                            # recall that for column development, CY is just 
                            # and alias to Y
                            cyrow1_j = CY[j, rc1]
                            cyrow1_i = CY[i, rc1]
                            C1 = np.intersect1d(cyrow1_j, cyrow1_i)
                            condition2_rc1 = ( len(C1) == len(cyrow1_j) - 1 )
                            
                            cyrow2_j = CY[j, rc2]
                            cyrow2_i = CY[i, rc2]
                            C2 = np.intersect1d(cyrow2_j, cyrow2_i)
                            condition2_rc2 = ( len(C2) == len(cyrow2_j)-1 )
                            
                            condition2 = condition2_rc1 * condition2_rc2
                            
                            if (condition2==True):
                                
                                p1 = np.setdiff1d(cyrow1_j, C1).flatten()
                                p2 = np.setdiff1d(cyrow1_i, C1).flatten()
                                assert(len(p1)==1)
                                assert(len(p2)==1)
                                p2p1 = np.array([p2[0], p1[0]], dtype=int)
                                
                                # find state of class i which is symmetric 
                                # wrt row k and row k+1
                                # respectively antisymmetric wrt col k and col k+1
                                ydev_i, coeffdev_i = develop_symmetry_two_ensembles(nu, Y[i], rc1, rc2, symmetry=symmetry)
                                if symmetry=='rows':
                                    cydev_i = sun.get_column(ydev_i)
                                
                                if symmetry=='rows':
                                    ind_j = np.argwhere(np.sum(abs(cydev_j[:, nkp1:nkp1+2] - p2p1), axis=1)==0).flatten()
                                else:
                                    ind_j = np.argwhere( np.sum(abs(ydev_j[:, nkp1:nkp1+2] - p2p1) , axis=1)==0 ).flatten()
                                
                                ydev_j_p = ydev_j[ind_j, :]
                                cydev_j_p = cydev_j[ind_j, :]
                                coeffdev_j_p = coeffdev_j[ind_j]
                                
                                if symmetry=='rows':
                                    ind_i = np.argwhere(np.sum(abs(cydev_i[:, nkp1:nkp1+2] - np.flip(p2p1)), axis=1)==0).flatten()
                                else:
                                    ind_i = np.argwhere( np.sum( abs(ydev_i[:, nkp1:nkp1+2] - np.flip(p2p1)), axis=1)==0 ).flatten()
                                
                                ydev_i_p = ydev_i[ind_i, :]
                                if symmetry=='rows':
                                    cydev_i_p = cydev_i[ind_i, :]
                                coeffdev_i_p = coeffdev_i[ind_i]
                                
                                if symmetry=='rows':
                                    cydev_j_p[:, nkp1] = p1
                                    cydev_j_p[:, nkp1+1] = p2
                                    # get row positions from column positions
                                    ydev_j_p_0 = sun.get_column(cydev_j_p[0]) # sun.get_column is an involution
                                else:
                                    ydev_j_p[:, nkp1] = p1
                                    ydev_j_p[:, nkp1+1] = p2
                                    cydev_j_p_0 = sun.get_column(ydev_j_p[0, :])
                                
                                if symmetry=='rows':
                                    rho = 1.0/sun.get_axial_distance(ydev_j_p_0, cydev_j_p[0, :], nkp1, nkp1+1)
                                else:
                                    rho = 1.0/sun.get_axial_distance(ydev_j_p[0, :], cydev_j_p_0, nkp1, nkp1+1)
                                
                                coeffdev_j_p = np.sqrt( 1.0 - rho**2 ) * coeffdev_j_p
                                
                                if symmetry=='rows':
                                    valij = sun.overlap(cydev_i_p, coeffdev_i_p, 
                                                        cydev_j_p, coeffdev_j_p, 
                                                        need_fullsimplify=False)
                                else:
                                    valij = sun.overlap(ydev_i_p, coeffdev_i_p, 
                                                        ydev_j_p, coeffdev_j_p, 
                                                        need_fullsimplify=False)
                                
                                Hk_offdiag[i, j] = valij
                        
            Projk = np.diagflat( Hk_diag ) + Hk_offdiag + Hk_offdiag.transpose()
            Projk += si*np.diagflat(np.full(shape=(NY,), fill_value=1/lrc1, dtype=float))
            Proj += Projk @ Projk
            
        # Compute kernel of operator
        D, V = np.linalg.eigh(Proj)
        ind = np.argwhere(abs(D)<1e-10).flatten()
        if len(ind)==0:
            sys.exit('Problem: the kernel is empty.')
        V = V[:, ind]
        coeffdev_r = V.flatten()
    
    # Expand each class
    ydev_ref = np.copy(Y[0])
    cydev_ref = sun.get_column(ydev_ref)
    coeffdev_ref = np.array([0.0], dtype=float)
    
    for t in range(0, NY):
        
        ydev_t = Y[t]
        coeffdev_t = coeffdev_r[t]
        Nt = int(1)
        
        for k in range(nb_rc_nu2-1, -1, -1):
            
            # symmetrize wrt row/col k
            st = n1 + np.sum(nu2_rc[k+1:])
            ed = n1 + np.sum(nu2_rc[k:])
            rc1 = np.arange(st, ed)
            
            y_temp, coeff_temp = develop_symmetry_one_ensemble(nu, Y[t], rc1, symmetry=symmetry)
            N_temp = len(coeff_temp)
            
            ydev_t = np.matlib.repmat(ydev_t, N_temp, 1)
            y_temp = np.repeat(y_temp[:, rc1], repeats=Nt, axis=0)
            ydev_t[:, rc1] = y_temp
            coeffdev_t = np.multiply( np.matlib.repmat(coeffdev_t, 1, N_temp).flatten(), 
                                      np.repeat(coeff_temp, Nt) )
            
            Nt *= N_temp
        
        # TO DO need to define a sum_develop without column positions
        cydev_t = sun.get_column(ydev_t)
        ydev_ref, cydev_ref, coeffdev_ref = sun.sum_develop(ydev_ref, cydev_ref, coeffdev_ref, 
                                                            ydev_t, cydev_t, coeffdev_t)
    
    # order SYTs in ascending order of LLOS
    ydev_ref, ind = sun.sort_SYT(ydev_ref, order='LLOS')
    coeffdev_ref = coeffdev_ref[ind]
    coeffdev_ref[np.argwhere(abs(coeffdev_ref)<1.0e-12).flatten()] = 0.0
    
    # Deal with global phase
    coeffdev_ref, ind_ref = sdcutils.set_overall_phase(
                                np.reshape(coeffdev_ref, (len(coeffdev_ref), 1)))
    
    cydev_ref = sun.get_column(ydev_ref)
    
    # generate y2
    nu2p = np.copy(nu2)
    nu2p[l2] -= 1
    if ref2firstLLOS==True:
        y2 = sun.index_to_SYT(0, alpha=nu2p, order='LLOS')
    else:
        y2 = sun.index_to_SYT(0, alpha=nu2p, order='iLLOS')
    y2 = np.hstack((y2, l2))
    
    ifd = np.argwhere(abs(y2_ref-y2)>0).flatten()
    
    if (len(ifd)==0):
        ydev_final = ydev_ref
        cydev_final = cydev_ref
        coeffdev_final = coeffdev_ref.flatten()
    else:
        
        # find sequence of transpositions which brings y2_ref to y2
        sigma, rho = sun.SYT_to_target(y2_ref, y2)
        
        # go from local to global numbering
        sigma = n - sigma - 2
        
        # double-check sequence of transpositions
        ifd = ifd[0]
        Nop = n2 - ifd - 1
        assert(Nop==len(sigma))
        sigmap = n - np.arange(ifd, n2-1) - 2
        assert(np.linalg.norm(sigmap==sigma))
        
        ydev_ref_copy = np.copy(ydev_ref[ind_ref[:, 0]])
        cydev_ref_copy = np.copy(cydev_ref[ind_ref[:, 0]])
        coeffdev_ref_copy = np.copy(coeffdev_ref[ind_ref[:, 0]]).flatten()
        
        '''
        # base code - slow
        ydev_final, cydev_final, coeffdev_final = subduction.apply_transpositions(
                                        nu, 
                                        sigma, 
                                        rho, 
                                        ydev_ref_copy, 
                                        cydev_ref_copy, 
                                        coeffdev_ref_copy)
        '''
        
        ydev_final, cydev_final, coeffdev_final = sun.apply_transpositions_v2(
                                    nu, 
                                    sigma, 
                                    rho, 
                                    ydev_ref_copy, 
                                    cydev_ref_copy, 
                                    coeffdev_ref_copy)
        
    # sort SYTs in ascending order of LLOS
    ydev_final, ind = sun.sort_SYT(ydev_final, order='LLOS')
    cydev_final = cydev_final[ind, :]
    coeffdev_final = coeffdev_final[ind]
        
    return ydev_final, cydev_final, coeffdev_final


def develop_symmetry_one_ensemble(nu, y, xvec, symmetry):
    """
    Find symmetric or antisymmetric state of equivalence class wrt a row or 
    column, respectively
    
    Parameters
    ----------
    nu : numpy array
        irrep
    y : numpy array
        SYT of equivalence class
    xvec : numpy array
        particles of the column/row to antisymmetrize/symmetrize
    symmetry : str
        'symmetric', 'row' ---> find symmetric state
        'antisymmetric', 'col', 'column' ---> find antisymmetric state
    
    Returns
    -------
    Y : numpy array
        SYTs
    V : numpy array
        coefficients of state (development)
    """
    
    n = np.sum(nu)
    assert len(y)==n
    if len(xvec):
        m = np.min(xvec)
        M = np.max(xvec)
        assert (np.linalg.norm(xvec - np.arange(m, M+1))==0)
    
    nu_M = np.copy(nu)
    for i in range(n-1, M, -1):
        nu_M[y[i]] -= 1
    
    nu_m = np.copy(nu_M)
    if (M>0):
        for i in range(M, m-1, -1):
            nu_m[y[i]] -= 1
    else:
        sys.exit('Problem ?')
    
    Y, V = __develop_symmetry(y, nu_M, nu_m, M, m, symmetry)
    
    return Y, V


def develop_symmetry_two_ensembles(nu, y, xvec1, xvec2, symmetry):
    """
    Find symmetric or antisymmetric state of equivalence class wrt two rows or 
    columns, respectively
    
    Parameters
    ----------
    nu : numpy array
        irrep
    y : numpy array
        SYT of equivalence class
    xvec1 : numpy array
        particles of the first column/row to antisymmetrize/symmetrize
    xvec2 : numpy array
        particles of the second column/row to antisymmetrize/symmetrize
    symmetry : str
        'symmetric', 'row' ---> find symmetric state
        'antisymmetric', 'col', 'column' ---> find antisymmetric state
    
    Returns
    -------
    Y : numpy array
        SYTs
    V : numpy array
        coefficients of state (development)
    """
    if xvec2[0]<xvec1[0]:
        xvec1, xvec2 = xvec2, xvec1
    
    y1, V1 = develop_symmetry_one_ensemble(nu, y, xvec1, symmetry)
    y2, V2 = develop_symmetry_one_ensemble(nu, y, xvec2, symmetry)
    
    Ny1 = y1.shape[0]
    Ny2 = y2.shape[0]
    
    # merge states
    n = np.sum(nu)
    m1 = xvec1[0]
    M2 = xvec2[-1]
    
    NY = Ny1 * Ny2
    Y = np.full(shape=(NY, n), fill_value=-1, dtype=int)    
    
    Y[:, 0:m1] = np.matlib.repmat(y[0:m1], NY, 1)
    if M2<n-1:
        Y[:, M2+1:] = np.matlib.repmat(y[M2+1:], NY, 1)
    
    Y[:, xvec1] = np.repeat(y1[:, xvec1], repeats=Ny2, axis=0)
    Y[:, xvec2] = np.matlib.repmat(y2[:, xvec2], Ny1, 1)
    coeffY = np.kron(V1, V2)
    
    '''
    # other possible ordering - not a great choice
    Y[:, xvec1] = np.matlib.repmat(y1[:, xvec1], Ny2, 1)
    Y[:, xvec2] = np.repeat(y2[:, xvec2], repeats=Ny1, axis=0)
    coeffY = np.kron(V2, V1)
    ''' 
    
    return Y, coeffY


def __develop_symmetry(y, nu_M, nu_m, M, m, symmetry):
    """
    Helper function
    """
    assert(symmetry in ['symmetric', 'antisymmetric', 'row', 'rows', 'column', 'columns', 'col', 'cols'])
    
    si = 1.0
    if symmetry in ['antisymmetric', 'column', 'columns', 'col', 'cols']:
        si = -1.0
    
    y2 = sun.get_subSYT(nu_M, nu_m, order='LLOS')
    
    Ny2 = y2.shape[0]
    y_lowpart = np.matlib.repmat(y[0:m], Ny2, 1)
    y_highpart = np.matlib.repmat(y[M+1:], Ny2, 1)
    y2 = np.hstack((y_lowpart, y2, y_highpart))
    cy2 = sun.get_column(y2)
    
    nu2 = nu_M - nu_m
    npa = M - m + 1
    assert(npa==np.sum(nu2))
    assert(len(y)==y2.shape[1])
    
    if Ny2==1:
        V2 = np.array([1.0], dtype=float)
    else:
        
        # create projector
        Proj = scipy.sparse.csr_matrix((Ny2, Ny2))
        
        for k in range(m, M):
            bool_lignes = (y2[:, k]==y2[:, k+1]) # k and k+1 in the same row
            bool_col = (cy2[:, k]==cy2[:, k+1]) # k and k+1 in the same column
            row_ind = np.argwhere(bool_lignes==True).flatten()
            col_ind = np.argwhere(bool_col==True).flatten()
            
            ind_off = np.argwhere( np.multiply(1-bool_lignes, 1-bool_col) ).flatten()
            # ind_off --> neither in same row nor same column
            
            # compute axial distance from k to k+1
            ax_vec = y2[ind_off, k+1] - cy2[ind_off, k+1] + cy2[ind_off, k] - y2[ind_off, k]
            rho_vec = 1.0/ax_vec
            
            # perform interchange for all positions where k and k+1 are neither
            # in the same row nor in the same column
            yoff2 = np.copy(y2[ind_off])
            yoff2[:, k] = y2[ind_off, k+1]
            yoff2[:, k+1] = y2[ind_off, k]
            
            # find the index of each SYT in yoff2 in the array y2
            ind_off2 = np.zeros(shape=(yoff2.shape[0],), dtype=int)
            for i, yo2 in enumerate(yoff2):
                temp_ind = np.argwhere(np.sum(abs(y2 - yo2), axis=1)==0).flatten()
                assert len(temp_ind)==1
                ind_off2[i] = temp_ind[0]
            
            # diagonal elements
            Hkrow = scipy.sparse.csr_matrix((np.full(shape=(len(row_ind),), fill_value=1.0, dtype=float), 
                                            (row_ind, row_ind)), 
                                            shape=(Ny2, Ny2))
            
            Hkcol = - scipy.sparse.csr_matrix((np.full(shape=(len(col_ind),), fill_value=1.0, dtype=float), 
                                                     (col_ind, col_ind)), 
                                                     shape=(Ny2, Ny2))
            # off-diag elements
            Hk_offdiag1 = scipy.sparse.csr_matrix( (-rho_vec, (ind_off, ind_off)), 
                                                    shape=(Ny2, Ny2))
            
            Hk_offdiag2 = scipy.sparse.csr_matrix( (np.sqrt(1-np.multiply(rho_vec, rho_vec)), 
                                                   (ind_off, ind_off2) ), 
                                                   shape=(Ny2, Ny2))
            
            # matrix of adjacent transposition P_{k, k+1}
            Pk = Hkrow + Hkcol + Hk_offdiag1 + Hk_offdiag2
            
            # now, ensure that we will extract a state which is ANTISYMMETRIC
            # when finding the kernel of Proj
            Projk = Pk - si*scipy.sparse.eye(Ny2)
            
            Proj += Projk
        
        # Find kernel of projector ==> antisymmetric state
        if Ny2<1000:
            DD, VV = np.linalg.eigh(Proj.todense())     
        else:
            DD, VV = scipy.sparse.linalg.eigsh(Proj, k=4, which='SA')
        ind = np.argwhere(abs(DD)<1.0e-12).flatten()
        
        if len(ind)==1:
            V2 = VV[:, ind[0]]
            V2 = np.asarray(V2).flatten()
            ind = np.argwhere(abs(V2)>1.0e-12).flatten()
            if V2[ind[0]]<0:
                V2 *= (-1)
        else:
            stri = 'Problem [develop_symmetry]: found ' + len(ind) + ' vanishing eigenvalues.'
            sys.exit(stri)
    
    return y2, V2


def get_sSYT_constraint(nu, nu1, nu2, constraint='antisymm', **kwargs):
    """
    Compute all SYTs (or sub_SYTs) for the irrep n (respectively nu-nu1)
    satisfying internal constraints imposed by irrep nu2 viewed as a product of
    antisymmetric [default] or symmetric irreps
    
    Parameters
    ----------
    nu : numpy array
        global irrep
    nu1 : numpy array
        irrep
    nu2 : numpy array
        irrep
    constraint : str [optional]
        [default] 'antisymm' or 'symm'
    fill_nu1 : bool [optional]
        if True, alocate space for nu1 in SYTs
    fill_y1 : numpy array [optional]
        if provided in conjunction with fill_nu1=True, fill the boxes corresponding
        to nu1 with this SYT
    order_y1 : str [optional]
        'LLOS' or 'iLLOS'
        if fill_y1 is not provided, fill SYT with y1 of nu1 corresponding to the 
        first SYT defined by this order
    
    Returns
    -------
    Y : numpy array
        collection of SYTs
    
    Details
    -------
    constraint='antisymm'
        Numbers in each column of nu2 appear in strictly ascending rows in the
        output SYTs
    constraint='symm'
        Numbers in each row of nu2 appear in strictly ascending columns in the
        output SYTs
    """
    
    if not constraint in ['symm', 'antisymm']:
        sys.exit('Undefined constraint ', constraint)
    
    n = np.sum(nu)
    n1 = np.sum(nu1)
    n2 = np.sum(nu2)
    assert n==n1+n2
    
    if 'fill_nu1' in kwargs:
        fill_nu1 = kwargs['fill_nu1']
    else:
        fill_nu1 = False
    
    m = n2
    if fill_nu1==True:
        m = n

    Y = np.full(shape=(1, m), fill_value=-1, dtype=int)
    nbsyt = int(1)
    
    if 'fill_y1' in kwargs:
        fill_y1 = kwargs['fill_y1']
    else:
        if fill_nu1==True:
            if 'order_y1' in kwargs:
                order_y1 = kwargs['order_y1']
            else:
                order_y1 = 'LLOS'
            fill_y1 = sun.index_to_SYT(0, alpha=nu1, order=order_y1)
    
    if fill_nu1==True:
        Y[0, 0:n1] = fill_y1
    
    if constraint=='antisymm':
        
        nc_nu2 = nu2[0] # number of columns in nu2
        nu2T = sun.transpose_shape(nu2) # lengths of columns of nu2
        nu2_col = nu2T
        
        for j in range(0, nc_nu2): # iteration over the columns of nu2
            for k in range(0, nu2T[j]): # iteration over the boxes of the j-th column of nu2
                
                cpt = int(0)
                Ynew = np.full(shape=(0, m), fill_value=-1, dtype=int)
                
                # particle to place (local numbering n2-1, ..., 1, 0)
                # here it means that particles are ordered column-wise in the reference
                nbv = m - ( np.sum(nu2_col[0:j]) + k ) - 1
                
                # place number nbv in SYT
                for q in range(0, nbsyt):
                    
                    nup = np.hstack((np.copy(nu), 0))
                    for p in range(m-1, nbv, -1):
                        nup[Y[q, p]] -= 1
                    
                    # find at which rows we can still place numbers, based on shape nu1
                    nudiff = nup - np.hstack((nu1, 0))
                    rows = np.argwhere(nudiff>0).flatten()
                    
                    for row in rows:
                        
                        # check if there is a bottom corner at row
                        is_bc = ( nup[row]>nup[row+1] )
                        
                        # verify that nbv will appear in a row above nbv+1
                        if ( (k>0) & (is_bc==True) ):
                            is_bc = is_bc * (row<Y[q, nbv+1])
                        
                        if (is_bc==True):
                            ytmp = np.copy(Y[q, :])
                            ytmp[nbv] = row
                            Ynew = np.vstack([Ynew, ytmp])
                            cpt += 1
                
                Y = np.copy(Ynew)
                nbsyt = cpt
        Y = Y[0:nbsyt, :]
    
    elif constraint=='symm':
        
        nr_nu2 = len(np.argwhere(nu2>0).flatten()) # number of rows in nu2
        
        CY = np.full(shape=(1, m), fill_value=-1, dtype=int)
        
        if fill_nu1==True:
            CY[0, 0:n1] = sun.get_column(Y[0, 0:n1]).flatten()
        
        for j in range(0, nr_nu2): # iteration over the rows of nu2
            for k in range(0, nu2[j]): # iteration over the boxes of the j-th row of nu2
                
                cpt = int(0)
                Ynew = np.full(shape=(0, m), fill_value=-1, dtype=int)
                CYnew = np.full(shape=(0, m), fill_value=-1, dtype=int)
                
                # particle to place (local numbering n2-1, ..., 1, 0)
                # here it means that particles are ordered row-wise in the reference
                nbv = m - ( np.sum(nu2[0:j]) + k ) - 1
                
                # place number nbv in SYT
                for q in range(0, nbsyt):
                    
                    nup = np.hstack((np.copy(nu), 0))
                    for p in range(m-1, nbv, -1):
                        nup[Y[q, p]] -= 1
                    
                    # find at which rows we can still place numbers, based on shape nu1
                    nudiff = nup - np.hstack((nu1, 0))
                    rows = np.argwhere(nudiff>0).flatten()
                    
                    for row in rows:
                        
                        # check if there is a bottom corner at row
                        is_bc = ( nup[row]>nup[row+1] )
                        
                        # extract column position corresponding to row
                        col = nup[row]
                        
                        # verify that nbv will appear in a column on the left of nbv+1
                        if ( (k>0) & (is_bc==True) ):
                            is_bc = is_bc * (col<CY[q, nbv+1])
                        
                        if (is_bc==True):
                            ytmp = np.copy(Y[q, :])
                            ytmp[nbv] = row
                            cytmp = np.copy(CY[q, :])
                            cytmp[nbv] = col
                            Ynew = np.vstack([Ynew, ytmp])
                            CYnew = np.vstack([CYnew, cytmp])
                            cpt += 1
                
                Y = np.copy(Ynew)
                CY = np.copy(CYnew)
                nbsyt = cpt
        Y = Y[0:nbsyt, :]
    
    return Y
