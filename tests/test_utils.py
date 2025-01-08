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


def test_lorentzian():
    x = np.linspace(-10.0, 10.0, 101)
    y = utils.lorentzian(x, 0.0, 1.0)
    assert not np.any(np.isnan(y))


def test_calc_psuedo_voight():
    velocity_axis = np.linspace(-100.0, 100.0, 101)
    velocity = np.array([-10.0, 0.0, 10.0])
    fwhm = np.array([25.0, 30.0, 35.0])
    fwhm_L = 1.0
    line_profile = utils.calc_pseudo_voigt(velocity_axis, velocity, fwhm, fwhm_L).eval()
    assert line_profile.shape == (101, 3)
