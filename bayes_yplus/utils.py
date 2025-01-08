"""
utils.py
Model Utilities

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

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


def lorentzian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Lorentzian function

    Parameters
    ----------
    x : float
        Position at which to evaluate
    center : float
        Centroid
    fwhm : float
        FWHM

    Returns
    -------
    float
        Lorentzian evaluated at x
    """
    return fwhm / (2.0 * np.pi) / ((x - center) ** 2.0 + (fwhm / 2.0) ** 2.0)


def calc_pseudo_voigt(
    velocity_axis: Iterable[float],
    velocity: Iterable[float],
    fwhm: Iterable[float],
    fwhm_L: Iterable[float],
) -> Iterable[float]:
    """Evaluate a pseudo Voight profile in order to aid in posterior exploration
    of the parameter space. This parameterization includes a latent variable fwhm_L, which
    can be conditioned on zero to analyze the posterior. We also consider the spectral
    channelization. We do not perform a full boxcar convolution, rather
    we approximate the convolution by assuming an equivalent FWHM for the
    boxcar kernel of 4 ln(2) / pi * channel_width ~= 0.88 * channel_width

    Parameters
    ----------
    velocity_axis : Iterable[float]
        Observed velocity axis (km s-1; length S)
    velocity : Iterable[float]
        Cloud center velocity (km s-1; length N)
    fwhm : Iterable[float]
        Cloud FWHM line widths (km s-1; length N)
    fwhm_L : Iterable[float]
        Latent pseudo-Voigt profile Lorentzian FWHM (km s-1)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x N)
    """
    channel_size = pt.abs(velocity_axis[1] - velocity_axis[0])
    channel_fwhm = 4.0 * np.log(2.0) * channel_size / np.pi
    fwhm_conv = pt.sqrt(fwhm**2.0 + channel_fwhm**2.0 + fwhm_L**2.0)
    fwhm_L_frac = fwhm_L / fwhm_conv
    eta = 1.36603 * fwhm_L_frac - 0.47719 * fwhm_L_frac**2.0 + 0.11116 * fwhm_L_frac**3.0

    # gaussian component
    gauss_part = gaussian(velocity_axis[:, None], velocity[None, :], fwhm_conv[None, :])

    # lorentzian component
    lorentz_part = lorentzian(velocity_axis[:, None], velocity[None, :], fwhm_conv[None, :])

    # linear combination
    return eta * lorentz_part + (1.0 - eta) * gauss_part
