# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import numpy as np



def find_row(A, a):
    """
    Search array A for rows equal to a and return indices of matching rows
    """
    indices = np.argwhere(np.sum(abs(A - a), axis=1)==0).flatten()
    return indices


def find_col(A, a):
    """
    Search array A for columns equal to a and return indices of matching columns
    """
    indices = np.argwhere(np.sum(abs(A.transpose() - a), axis=1)==0).flatten()
    return indices


def intersect_row(A, B):
    """
    Find rows which appear in A AND in B
    
    Source - https://stackoverflow.com/a
    Posted by Joe Kington
    Retrieved 2025-11-16, License - CC BY-SA 3.0
    """
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}
    
    C, indA, indB = np.intersect1d(A.view(dtype), B.view(dtype), 
                                   assume_unique=True, return_indices=True)
    C = C.view(A.dtype).reshape(-1, ncols)
    
    return C, indA, indB


def sortrows(A):
    """
    Equivalent to Matlab's sortrows.
    
    Parameters
    ----------
    A : numpy array
    
    Returns
    -------
    B : numpy array
      sorted array of rows
    index :  numpy array 
        indices such that B == A[index, :]
    
    Remark
    ------
    See https://ch.mathworks.com/help/matlab/ref/double.sortrows.html
    """
    
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
