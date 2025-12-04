# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import dmrg_rme

N = int(3)
num_irreps = int(12)

rme_engine = dmrg_rme.RMEEngine(N, num_irreps, restarting=True, checkpointing=True)

rme_engine.run(tech='shortcut_cols')
