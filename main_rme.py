# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

from sunpy.dmrg.rme import rmefund

N = int(3)
num_irreps = int(16)

rme_engine = rmefund.RMEEngine(N, num_irreps, 
                               restarting=False, 
                               checkpointing=False, 
                               chkpt_method='log2')

rme_engine.run(tech='shortcut_cols')
