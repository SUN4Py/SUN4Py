# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

from sunpy.dmrg.rme import rmefund
#from sunpy.dmrg.rme import rmesymm


N = int(3)
num_irreps = int(16)

rme_engine = rmefund.RMEEngineFund(N, num_irreps, 
                                   target='GS', 
                                   tech='shortcut_cols', 
                                   restarting=True, 
                                   checkpointing=False, 
                                   chkpt_method='log2')

rme_engine.run()


########################################
# SYMMETRIC IRREPS
########################################
'''
N = int(3)
m = int(3)
num_irreps = int(6)

rme_engineSymm = rmesymm.RMEEngineSymm(N, m, num_irreps, 
                                   #target='ES300_idmrg_edgeAdjoint',
                                   target='ES330_idmrg_edgeAdjoint',
                                   tech='shortcut_rows', 
                                   restarting=False, 
                                   checkpointing=False)

rme_engineSymm.run()
'''