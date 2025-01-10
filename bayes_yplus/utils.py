"""
utils.py
Model Utilities

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytensor.tensor as pt
import numpy as np


def gaussian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Gaussian function

    Parameters
    ----------
    x : float
        Position at which to evaluate
    center : float
        Gaussian centroid
    fwhm : float
        Gaussian FWHM line width

    Returns
    -------
    float
        Gaussian evaluated at x
    """
    return pt.exp(-4.0 * np.log(2.0) * (x - center) ** 2.0 / fwhm**2.0) * pt.sqrt(
        4.0 * np.log(2.0) / (np.pi * fwhm**2.0)
    )
