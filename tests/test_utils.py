"""
test_utils.py
tests for utils.py

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

from bayes_yplus import utils


def test_gaussian():
    x = np.linspace(-10.0, 10.0, 101)
    y = utils.gaussian(x, 0.0, 1.0).eval()
    assert not np.any(np.isnan(y))
