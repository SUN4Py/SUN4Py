#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2023 Samuel GOZEL, GNU GPLv3

@author: sgozel
"""

import pytest
from sunpy.common import partitions



@pytest.mark.parametrize("n, expected_result", [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 5),
    (5, 7),
    (6, 11),
    (7, 15),
    (8, 22),
    (9, 30),
    (10, 42),
    (20, 627),
    (30, 5604)
])
def test_p(n, expected_result):
    assert partitions.p(n) == expected_result
