"""
yplus_model.py
YPlusModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional

import pymc as pm
import numpy as np

import pytensor.tensor as pt

from bayes_spec import BaseModel
from bayes_yplus import utils


class YPlusModel(BaseModel):
    """
    Definition of the YPlusModel
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new YPlusModel instance

        Parameters
        ----------
        *args : Additional arguments passed to BaseModel
        **kwargs : Additional arguments passed to BaseModel
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Select features used for posterior clustering
        self._cluster_features += [
            "H_area",
            "H_center",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "H_area": r"$\int T_{L, \rm H} dV$ (K km s$^{-1}$)",
                "H_center": r"$V_{\rm LSR, H}$ (km s$^{-1}$)",
                "H_fwhm": r"$\Delta V_{\rm H}$ (km s$^{-1}$)",
                "He_H_fwhm_ratio": r"$\Delta V_{\rm He}/\Delta V_{\rm H}$",
                "yplus": r"$y^+$",
                "H_amplitude": r"$T_{L,\rm H}$",
                "He_amplitude": r"$T_{L,\rm He}$",
                "He_center": r"$V_{\rm LSR, He}$ (km s$^{-1}$)",
                "He_fwhm": r"$\Delta V_{\rm He}$ (km s$^{-1}$)",
                "He_area": r"$\int T_{L, \rm He} dV$ (K km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_H_area: float = 1000.0,
        prior_H_center: Iterable[float] = [0.0, 25.0],
        prior_H_fwhm: Iterable[float] = [25.0, 10.0],
        prior_He_H_fwhm_ratio: Iterable[float] = [1.0, 0.1],
        prior_yplus: float = 0.05,
        prior_rms: Optional[dict[str, float]] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_H_area : float, optional
            Prior distribution on H RRL line area (mK km s-1), by default 1000.0, where
            H_area ~ HalfNormal(sigma=prior)
        prior_H_center : Iterable[float], optional
            Prior distribution on H RRL center velocity (km s-1), by default [0, 25.0], where
            H_center ~ Normal(mu=prior[0], sigma=prior[1])
        prior_H_fwhm : Iterable[float], optional
            Prior distribution on H RRL FWHM line width (km s-1), by default [25.0, 10.0], where
            H_fwhm ~ Gamma(mu=prior[0], sigma=prior[1])
        prior_He_H_fwhm_ratio : Iterable[float], optional
            Prior distribution on He/H RRL FWHM line width ratio, by default [1.0, 0.1], where
            He_H_fwhm_ratio ~ Gamma(mu=prior[0], sigma=prior[1])
        prior_yplus : float, optional
            Prior distribution on y+, by default 0.05, where
            yplus ~ HalfNormal(sigma=prior_yplus)
        prior_rms : Optional[dict[str, float]], optional
            Prior distribution on spectral rms (K), by default None, where
            rms ~ HalfNormal(sigma=prior)
            Keys are dataset names and values are priors. If None, then the spectral rms is taken
            from dataset.noise and not inferred.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Prior distribution on the normalized baseline polynomial coefficients, by default None.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset.
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False.
            If True, the prior distribution on the velocity becomes
            H_center(cloud = n) ~
                prior[0] + sum_i(H_center[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # H center velocity (km s-1)
            if ordered:
                H_center_norm = pm.Gamma("H_center_norm", alpha=2.0, beta=1.0, dims="cloud")
                H_center_offset = prior_H_center[1] * H_center_norm
                H_center = pm.Deterministic(
                    "H_center",
                    prior_H_center[0] + pm.math.cumsum(H_center_offset),
                    dims="cloud",
                )
            else:
                H_center_norm = pm.Normal(
                    "H_center_norm",
                    mu=0.0,
                    sigma=1.0,
                    dims="cloud",
                )
                H_center = pm.Deterministic(
                    "H_center",
                    prior_H_center[0] + prior_H_center[1] * H_center_norm,
                    dims="cloud",
                )

            # H line area (mK km s-1)
            H_area_norm = pm.HalfNormal("H_area_norm", sigma=1.0, dims="cloud")
            H_area = pm.Deterministic("H_area", prior_H_area * H_area_norm, dims="cloud")

            # H FWHM line width (km s-1)
            H_fwhm = pm.Gamma("H_fwhm", mu=prior_H_fwhm[0], sigma=prior_H_fwhm[1], dims="cloud")

            # He/H FWHM line width ratio
            He_H_fwhm_ratio = pm.Gamma(
                "He_H_fwhm_ratio", mu=prior_He_H_fwhm_ratio[0], sigma=prior_He_H_fwhm_ratio[1], dims="cloud"
            )

            # y+
            yplus_norm = pm.HalfNormal("yplus_norm", sigma=1.0, dims="cloud")
            yplus = pm.Deterministic("yplus", prior_yplus * yplus_norm, dims="cloud")

            # Spectral rms (K)
            if prior_rms is not None:
                for label in self.data.keys():
                    rms_norm = pm.HalfNormal(f"rms_{label}_norm", sigma=1.0)
                    _ = pm.Deterministic(f"rms_{label}", rms_norm * prior_rms[label])

            # H amplitude (mK)
            H_amplitude = pm.Deterministic(
                "H_amplitude",
                H_area / H_fwhm / np.sqrt(np.pi / (4.0 * np.log(2.0))),
                dims="cloud",
            )

            # He amplitude (mK)
            He_amplitude = pm.Deterministic("He_amplitude", H_amplitude * yplus / He_H_fwhm_ratio, dims="cloud")

            # He center velocity (km s-1)
            _ = pm.Deterministic("He_center", H_center - 122.15, dims="cloud")

            # He FWHM line width (km s-1)
            He_fwhm = pm.Deterministic("He_fwhm", H_fwhm * He_H_fwhm_ratio, dims="cloud")

            # He line area (mK km s-1)
            _ = pm.Deterministic("He_area", He_amplitude * He_fwhm * np.sqrt(np.pi / (4.0 * np.log(2.0))), dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model."""
        # Predict baseline models
        baseline_models = self.predict_baseline()

        # Predict all spectra
        for label, dataset in self.data.items():
            # Evaluate line profiles
            H_profile = utils.gaussian(dataset.spectral[:, None], self.model["H_center"], self.model["H_fwhm"])
            He_profile = utils.gaussian(dataset.spectral[:, None], self.model["He_center"], self.model["He_fwhm"])

            # Evaluate spectrum
            H_spectrum = self.model["H_area"][None, :] * H_profile
            He_spectrum = self.model["He_area"][None, :] * He_profile
            predicted_line = pt.sum(H_spectrum + He_spectrum, axis=1)

            # Add baseline model
            predicted = predicted_line + baseline_models[label]

            with self.model:
                sigma = dataset.noise
                if f"rms_{label}" in self.model:
                    sigma = self.model[f"rms_{label}"]

                # Evaluate likelihood
                _ = pm.Normal(
                    label,
                    mu=predicted,
                    sigma=sigma,
                    observed=dataset.brightness,
                )
