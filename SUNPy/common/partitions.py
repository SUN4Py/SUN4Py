#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

Reference:
https://www.ams.org/journals/mcom/1982-39-159/S0025-5718-1982-0658226-5/S0025-5718-1982-0658226-5.pdf

@author: sgozel
"""

import math



def p1(n):
    # number of partitions of n in 1 part
    return int(1)


def p2(n):
    # number of partitions of n in 2 parts
    p2 = n//2
    return int(p2)


def p3(n):
    # number of partitions of n in 3 parts
    p3 = (n**2+3)//(3*2*2)
    return int(p3)


def p4(n):
    # number of partitions of n in 4 parts
    p4 = (n**3 + 3*n**2 + (9*n*(-1)**n-9*n)/2 + 32)//(math.factorial(4)*math.factorial(3))
    return int(p4)


def pnm(n, m):
    # number of partitions of n in exactly m parts
    if m<=0:
        return int(0)
    if n<=0:
        return int(0)
    
    if n==m:
        return int(1)
    elif m>n:
        return int(0)
    
    if m==1:
        return p1(n)
    elif m==2:
        return p2(n)
    elif m==3:
        return p3(n)
    elif m==4:
        return p4(n)
    else:
        return pnm(n-m, m)+pnm(n-1, m-1)


def pstarnm(n, m):
    # number of partitions of n in at most m parts
    ps = int(0)
    for k in range(1, m+1):
        ps += pnm(n, k)
    return ps


def p(n):
    # number of partitions of n
    if n<=0:
        return int(0)
    pn = int(0)
    for k in range(1, n+1):
        pn += pnm(n, k)
    return pn
