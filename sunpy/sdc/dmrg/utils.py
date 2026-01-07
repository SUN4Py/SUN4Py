# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import sys
import numpy as np
import scipy.sparse

from sunpy.sun import sun



def get_transpositions(nu2, l2, y2_ref, ref2firstLLOS):
    """
    Get the sequence of transpositions which bring y2_ref onto the target SYT
    for the irrep nu2 with m last particles in bottom corners defined by l2, 
    and first n2-m particles ordered according to ref2firstLLOS
    
    Parameters
    ----------
    nu2 : numpy array
        irrep
    l2 : numpy array
        position(s) of bottom corner(s) of m last particles
    y2_ref : numpy array
        reference SYT for nu2
    ref2firstLLOS : bool
        if True, the first SYT in the ascending order of the LLOS is used to
        fill the first n2-m particles in the SYT. If False, the last 
        SYT is used (namely the first in the descending order of LLOS)
    
    Returns
    -------
    sigma : numpy array
        sequence of transpositions
    rho : numpy array
        sequence of inverse of axial distances
    y2_final : numpy array
        target SYT
    
    Remark
    ------
    The transpositions (and inverse of axial distances) are ordered such that:
        y2_final = sigma[-1] ... sigma[1] sigma[0] y2_ref
    
    """
    
    n2 = np.sum(nu2)
    m = len(l2)
    y2_all = np.zeros(shape=(m+1, n2), dtype=int)
    y2_all[0] = y2_ref
    
    nu2p = np.copy(nu2)
    cpt = int(1)
    for q in range(m-1, -1, -1):
        nu2p[l2[q]] -= 1
        if ref2firstLLOS==True:
            y2 = sun.index_to_SYT(0, alpha=nu2p, order='LLOS')
        else:
            y2 = sun.index_to_SYT(0, alpha=nu2p, order='iLLOS')
        y2 = np.hstack((y2, l2[q:]))
        assert(len(y2)==n2)
        y2_all[cpt] = y2
        cpt += 1
    
    y2_final = np.copy(y2_all[-1])
    
    y2_temp = np.copy(y2_ref)
    cy2_temp = sun.get_column(y2_temp)
    
    sigma = np.array([], dtype=int)
    rho = np.array([])
    
    for q in range(1, m+1):
        
        y2 = y2_all[q]    
        ifd = np.argwhere(abs(y2_all[q-1]-y2)>0).flatten()
        
        if (len(ifd)==0):
            pass
        else:
            # find sequence of transpositions which brings y2_all[q-1] to y2
            ifd = ifd[0]
            k = n2 - ifd - 1
            Nop = k - q + 1
            
            sigmaq = np.zeros(shape=(Nop,), dtype=int)
            rhoq = np.zeros(shape=(Nop,))
            
            for s in range(0, Nop):
                sigmaq[s] = ifd + s
                rhoq[s] = 1./sun.get_axial_distance(y2_temp, cy2_temp, sigmaq[s], sigmaq[s]+1)
                
                y2_temp_copy = np.copy(y2_temp)
                cy2_temp_copy = np.copy(cy2_temp)
                
                y2_temp[sigmaq[s]] = y2_temp_copy[sigmaq[s]+1]
                y2_temp[sigmaq[s]+1] = y2_temp_copy[sigmaq[s]]
                cy2_temp[sigmaq[s]] = cy2_temp_copy[sigmaq[s]+1]
                cy2_temp[sigmaq[s]+1] = cy2_temp_copy[sigmaq[s]]
            
            sigmap, rhop = sun.SYT_to_target(y2_all[q-1], y2)
            
            assert(np.sum(abs(sigmap-sigmaq))==0)
            assert(np.sum(abs(rhop-rhoq))<1.0e-12)
            
            sigma = np.hstack((sigma, sigmaq))
            rho = np.hstack((rho, rhoq))
    
    assert(np.sum(abs(y2_temp-y2_final))==0)
    
    return sigma, rho, y2_final


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


def project_symmetry(y1, y2, Y, CY, COEFF_FINAL, m, symmetry, sort=True):
    """
    Apply projection operator on a development to ensure that local constraints
    (symmetric/antisymmetric) are respected for the last m particles of y1 and
    of y2
    
    Parameters
    ----------
    Y : numpy array
        SYTs
    CY : numpy array
        associated column positions
    COEFF_FINAL : numpy array
        coefficients of development
    m : int
        number of particles per site
    symmetry : str
        symmetry constraint ('symmetric' or 'antisymmetric')
    sort : bool [optional][default: True]
        if True, sort the output development in ascending order of LLOS
    
    Returns
    -------
    Y : numpy array
        SYTs
    CY : numpy array
        associated column positions
    COEFF_FINAL : numpy array
        coefficients of development
    """
    
    n1 = len(y1)
    n2 = len(y2)
    NY = Y.shape[0]
    
    assert(symmetry in ['symmetric', 'symm', 'antisymmetric', 'antisymm'])
    
    recompute_CY = False
        
    if symmetry in ['symmetric', 'symm']:
    
        if m==2:
            '''
            # previous implementation
            if not y1[n1-2]==y1[n1-1]:
                # symmetrize n1-2 and n1-1
                Y = np.matlib.repmat(Y, 2, 1)
                COEFF_FINAL = np.matlib.repmat(COEFF_FINAL, 1, 2).flatten()
                Y[NY:, n1-2] = Y[:NY, n1-1]
                Y[NY:, n1-1] = Y[:NY, n1-2]
                recompute_CY = True
                cy1 = sun.get_column(y1)
                rho = 1.0/sun.get_axial_distance(y1, cy1, n1-2, n1-1)
                COEFF_FINAL[:NY] *= np.sqrt((1 - rho)/2)
                COEFF_FINAL[NY:] *= np.sqrt((1 + rho)/2)
            
            if not y2[n2-2]==y2[n2-1]:
                # symmetrize n1 and n1+1 (in global numbering)
                cy2 = sun.get_column(y2)
                rho = 1.0/sun.get_axial_distance(y2, cy2, n2-2, n2-1)
                if recompute_CY==True:
                    CY = sun.get_column(Y)
                    recompute_CY = False
                ydev, cydev, coeffdev = sun.t_operator(Y, CY, COEFF_FINAL, n1, rho)
                coeffdev *= np.sqrt((1 + rho)/2.)
                COEFF_FINAL *= np.sqrt((1 - rho)/2.)
                Y, CY, COEFF_FINAL = sun.sum_develop(ydev, cydev, coeffdev, 
                                                     Y, CY, COEFF_FINAL)
            '''
            
            cy1 = sun.get_column(y1)
            rho = 1.0/sun.get_axial_distance(y1, cy1, n1-2, n1-1)
            
            Y = np.matlib.repmat(Y, 2, 1)
            COEFF_FINAL = np.matlib.repmat(COEFF_FINAL, 1, 2).flatten()
            st = NY
            
            if not y1[n1-2]==y1[n1-1]:
                # symmetrize n1-2 and n1-1
                Y[st:st+NY, n1-2] = Y[:NY, n1-1]
                Y[st:st+NY, n1-1] = Y[:NY, n1-2]
                COEFF_FINAL[st:st+NY] *= np.sqrt((1 + rho)/2)
                recompute_CY = True
                st += NY
            
            COEFF_FINAL[:NY] *= np.sqrt((1 - rho)/2)
            
            Y = Y[:st]
            COEFF_FINAL = COEFF_FINAL[:st]
            if recompute_CY:
                CY = sun.get_column(Y)
                recompute_CY = False
            
            cy2 = sun.get_column(y2)
            rho = 1.0/sun.get_axial_distance(y2, cy2, n2-2, n2-1)
            
            add_ydev = False
            if not y2[n2-2]==y2[n2-1]:
                # symmetrize n1 and n1+1 (in global numbering)
                ydev, cydev, coeffdev = sun.t_operator(Y, CY, COEFF_FINAL, n1, rho)
                coeffdev *= np.sqrt((1 + rho)/2.)
                add_ydev = True
                
            COEFF_FINAL *= np.sqrt((1 - rho)/2.)
            
            if add_ydev==True:
                Y, CY, COEFF_FINAL = sun.sum_develop(ydev, cydev, coeffdev, 
                                                     Y, CY, COEFF_FINAL)
            
        elif m==3:
            
            # symmetrize wrt m=3 last particles of y1
            cy1 = sun.get_column(y1)
            
            rhox = 1.0/sun.get_axial_distance(y1, cy1, n1-3, n1-2)
            rhoy = 1.0/sun.get_axial_distance(y1, cy1, n1-3, n1-1)
            rhoz = 1.0/sun.get_axial_distance(y1, cy1, n1-2, n1-1)
            assert(abs(1./rhox+1./rhoz-1./rhoy)<1.0e-12)
            
            Y = np.matlib.repmat(Y, 6, 1)
            COEFF_FINAL = np.matlib.repmat(COEFF_FINAL, 1, 6).flatten()
            
            st = NY
            
            if not y1[n1-3]==y1[n1-2]:
                # apply T1
                Y[st:st+NY, n1-3] = Y[:NY, n1-2]
                Y[st:st+NY, n1-2] = Y[:NY, n1-3]
                COEFF_FINAL[st:st+NY] *= np.sqrt(1+rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                st += NY
                
                # apply T4
                Y[st:st+NY, n1-3] = Y[:NY, n1-2]
                Y[st:st+NY, n1-2] = Y[:NY, n1-1]
                Y[st:st+NY, n1-1] = Y[:NY, n1-3]
                COEFF_FINAL[st:st+NY] *= np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                recompute_CY = True
                st += NY
            
            if not y1[n1-2]==y1[n1-1]:                    
                # apply T2
                Y[st:st+NY, n1-2] = Y[:NY, n1-1]
                Y[st:st+NY, n1-1] = Y[:NY, n1-2]
                COEFF_FINAL[st:st+NY] *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                st += NY
                
                # apply T3
                Y[st:st+NY, n1-3] = Y[:NY, n1-1]
                Y[st:st+NY, n1-2] = Y[:NY, n1-3]
                Y[st:st+NY, n1-1] = Y[:NY, n1-2]
                COEFF_FINAL[st:st+NY] *= np.sqrt(1-rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                st += NY
                
                if not y1[n1-3]==y1[n1-2]:
                    # apply T5
                    Y[st:st+NY, n1-3] = Y[:NY, n1-1]
                    Y[st:st+NY, n1-1] = Y[:NY, n1-3]
                    COEFF_FINAL[st:st+NY] *= np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                    st += NY
                
                recompute_CY = True
            
            # apply T0
            COEFF_FINAL[:NY] *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
            
            Y = Y[:st]
            COEFF_FINAL = COEFF_FINAL[:st]
            if recompute_CY:
                CY = sun.get_column(Y)
            
            # symmetrize wrt m=3 last particles of y2
            cy2 = sun.get_column(y2)
            rhox = 1.0/sun.get_axial_distance(y2, cy2, n2-3, n2-2)
            rhoy = 1.0/sun.get_axial_distance(y2, cy2, n2-3, n2-1)
            rhoz = 1.0/sun.get_axial_distance(y2, cy2, n2-2, n2-1)
            
            add_ydev1 = False
            add_ydev2 = False
            
            if not y2[n2-3]==y2[n2-2]:
                ydev, cydev, coeffdev = sun.t_operator(Y, CY, COEFF_FINAL, n1+1, rhox)
                ydevt1, cydevt1, coeffdevt1 = sun.t_operator(ydev, cydev, coeffdev, n1, rhoy)
                
                # apply T1
                coeffdev *= np.sqrt(1+rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                # apply T4
                coeffdevt1 *= np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                
                ydev1, cydev1, coeffdev1 = sun.sum_develop(ydev, cydev, coeffdev,
                                                           ydevt1, cydevt1, coeffdevt1)
                add_ydev1 = True
            
            if not y2[n2-2]==y2[n2-1]:
                ydev, cydev, coeffdev = sun.t_operator(Y, CY, COEFF_FINAL, n1, rhoz)
                ydevt1, cydevt1, coeffdevt1 = sun.t_operator(ydev, cydev, coeffdev, n1+1, rhoy)
                
                add_ydevt2 = False
                
                if not y2[n2-3]==y2[n2-2]:
                    # apply T5
                    ydevt2, cydevt2, coeffdevt2 = sun.t_operator(ydevt1, cydevt1, coeffdevt1, n1, rhox)
                    coeffdevt2 *= np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                    add_ydevt2 = True
                
                # apply T2
                coeffdev *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                # apply T3
                coeffdevt1 *= np.sqrt(1-rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                
                ydev2, cydev2, coeffdev2 = sun.sum_develop(ydev, cydev, coeffdev,
                                                           ydevt1, cydevt1, coeffdevt1)
                if add_ydevt2:
                    ydev2, cydev2, coeffdev2 = sun.sum_develop(ydev2, cydev2, coeffdev2,
                                                               ydevt2, cydevt2, coeffdevt2)
                add_ydev2 = True
            
            # apply T0
            COEFF_FINAL *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
            
            if add_ydev1:
                Y, CY, COEFF_FINAL = sun.sum_develop(Y, CY, COEFF_FINAL,
                                                     ydev1, cydev1, coeffdev1)
            if add_ydev2:
                Y, CY, COEFF_FINAL = sun.sum_develop(Y, CY, COEFF_FINAL,
                                                     ydev2, cydev2, coeffdev2)
            
        elif m>3:
            sys.exit('get_SDC: symmetric case m>3 not yet implemented.')
    
    else:
        # anti-symmetric irrep at each site
        
        if m==2:
            
            cy1 = sun.get_column(y1)
            
            if not cy1[n1-2]==cy1[n1-1]:
                # antisymmetrize n1-2 and n1-1
                CY = np.matlib.repmat(CY, 2, 1)
                COEFF_FINAL = np.matlib.repmat(COEFF_FINAL, 1, 2).flatten()
                CY[NY:, n1-2] = CY[:NY, n1-1]
                CY[NY:, n1-1] = CY[:NY, n1-2]
                Y = sun.get_column(CY)
                rho = 1.0/sun.get_axial_distance(y1, cy1, n1-2, n1-1)
                COEFF_FINAL[:NY] *= np.sqrt((1 + rho)/2)
                COEFF_FINAL[NY:] *= -np.sqrt((1 - rho)/2)
            
            cy2 = sun.get_column(y2)
            
            if not cy2[n2-2]==cy2[n2-1]:
                # antisymmetrize n1 and n1+1 (in global numbering)
                rho = 1.0/sun.get_axial_distance(y2, cy2, n2-2, n2-1)
                ydev, cydev, coeffdev = sun.t_operator(Y, CY, COEFF_FINAL, n1, rho)
                coeffdev *= -np.sqrt((1 - rho)/2.)
                COEFF_FINAL *= np.sqrt((1 + rho)/2.)
                Y, CY, COEFF_FINAL = sun.sum_develop(ydev, cydev, coeffdev, 
                                                     Y, CY, COEFF_FINAL)
            
        elif m==3:
            
            # anti-symmetrize wrt m=3 last particles of y1
            
            rhox = 1.0/sun.get_axial_distance(y1, cy1, n1-3, n1-2)
            rhoy = 1.0/sun.get_axial_distance(y1, cy1, n1-3, n1-1)
            rhoz = 1.0/sun.get_axial_distance(y1, cy1, n1-2, n1-1)
            assert(abs(1./rhox+1./rhoz-1./rhoy)<1.0e-12)
            
            Y = np.matlib.repmat(Y, 6, 1)
            COEFF_FINAL = np.matlib.repmat(COEFF_FINAL, 1, 6).flatten()
            
            st = NY
            
            if not y1[n1-3]==y1[n1-2]:
                # apply T1
                Y[st:st+NY, n1-3] = Y[:NY, n1-2]
                Y[st:st+NY, n1-2] = Y[:NY, n1-3]
                COEFF_FINAL[st:st+NY] *= - np.sqrt(1-rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                st += NY
                
                # apply T4
                Y[st:st+NY, n1-3] = Y[:NY, n1-2]
                Y[st:st+NY, n1-2] = Y[:NY, n1-1]
                Y[st:st+NY, n1-1] = Y[:NY, n1-3]
                COEFF_FINAL[st:st+NY] *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                recompute_CY = True
                st += NY
            
            if not y1[n1-2]==y1[n1-1]:                    
                # apply T2
                Y[st:st+NY, n1-2] = Y[:NY, n1-1]
                Y[st:st+NY, n1-1] = Y[:NY, n1-2]
                COEFF_FINAL[st:st+NY] *= -np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                st += NY
                
                # apply T3
                Y[st:st+NY, n1-3] = Y[:NY, n1-1]
                Y[st:st+NY, n1-2] = Y[:NY, n1-3]
                Y[st:st+NY, n1-1] = Y[:NY, n1-2]
                COEFF_FINAL[st:st+NY] *= np.sqrt(1+rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                st += NY
                
                if not y1[n1-3]==y1[n1-2]:
                    # apply T5
                    Y[st:st+NY, n1-3] = Y[:NY, n1-1]
                    Y[st:st+NY, n1-1] = Y[:NY, n1-3]
                    COEFF_FINAL[st:st+NY] *= -np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                    st += NY
                
                recompute_CY = True
            
            # apply T0
            COEFF_FINAL[:NY] *= np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
            
            Y = Y[:st]
            COEFF_FINAL = COEFF_FINAL[:st]
            if recompute_CY:
                CY = sun.get_column(Y)
            
            # symmetrize wrt m=3 last particles of y2
            
            rhox = 1.0/sun.get_axial_distance(y2, cy2, n2-3, n2-2)
            rhoy = 1.0/sun.get_axial_distance(y2, cy2, n2-3, n2-1)
            rhoz = 1.0/sun.get_axial_distance(y2, cy2, n2-2, n2-1)
            
            add_ydev1 = False
            add_ydev2 = False
            
            if not y2[n2-3]==y2[n2-2]:
                ydev, cydev, coeffdev = sun.t_operator(Y, CY, COEFF_FINAL, n1+1, rhox)
                ydevt1, cydevt1, coeffdevt1 = sun.t_operator(ydev, cydev, coeffdev, n1, rhoy)
                
                # apply T1
                coeffdev *= -np.sqrt(1-rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                # apply T4
                coeffdevt1 *= np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
                
                ydev1, cydev1, coeffdev1 = sun.sum_develop(ydev, cydev, coeffdev,
                                                           ydevt1, cydevt1, coeffdevt1)
                add_ydev1 = True
            
            if not y2[n2-2]==y2[n2-1]:
                ydev, cydev, coeffdev = sun.t_operator(Y, CY, COEFF_FINAL, n1, rhoz)
                ydevt1, cydevt1, coeffdevt1 = sun.t_operator(ydev, cydev, coeffdev, n1+1, rhoy)
                
                add_ydevt2 = False
                
                if not y2[n2-3]==y2[n2-2]:
                    # apply T5
                    ydevt2, cydevt2, coeffdevt2 = sun.t_operator(ydevt1, cydevt1, coeffdevt1, n1, rhox)
                    coeffdevt2 *= -np.sqrt(1-rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                    add_ydevt2 = True
                
                # apply T2
                coeffdev *= -np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                # apply T3
                coeffdevt1 *= np.sqrt(1+rhox) * np.sqrt(1-rhoy) * np.sqrt(1-rhoz) / np.sqrt(6)
                
                ydev2, cydev2, coeffdev2 = sun.sum_develop(ydev, cydev, coeffdev,
                                                           ydevt1, cydevt1, coeffdevt1)
                if add_ydevt2:
                    ydev2, cydev2, coeffdev2 = sun.sum_develop(ydev2, cydev2, coeffdev2,
                                                               ydevt2, cydevt2, coeffdevt2)
                add_ydev2 = True
            
            # apply T0
            COEFF_FINAL *= np.sqrt(1+rhox) * np.sqrt(1+rhoy) * np.sqrt(1+rhoz) / np.sqrt(6)
            
            if add_ydev1:
                Y, CY, COEFF_FINAL = sun.sum_develop(Y, CY, COEFF_FINAL,
                                                     ydev1, cydev1, coeffdev1)
            if add_ydev2:
                Y, CY, COEFF_FINAL = sun.sum_develop(Y, CY, COEFF_FINAL,
                                                     ydev2, cydev2, coeffdev2)
            
        else:
            sys.exit('get_SDC: anti-symmetric case m>3 not yet implemented.')
    
    if recompute_CY==True:
        CY = sun.get_column(Y)
    
    if sort==True:
        Y, ind = sun.sort_SYT(Y)
        CY = CY[ind]
        COEFF_FINAL = COEFF_FINAL[ind]
    
    return Y, CY, COEFF_FINAL

