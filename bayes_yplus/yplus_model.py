"""
yplus_model.py
YPlusModel definition

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from typing import Iterable, Optional

import pymc as pm
import numpy as np

from bayes_spec import BaseModel
from bayes_spec.utils import gaussian


class YPlusModel(BaseModel):
    """
    Definition of the YPlusModel
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new YPlusModel instance

        :param `*args`: Arguments passed to :class:`BaseModel`
        :param `**kwargs`: Keyword arguments passed to :class:`BaseModel`
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
                "rms_observation": r"rms (K)",
            }
        )

    def add_priors(
        self,
        prior_H_area: float = 1000.0,
        prior_H_center: Iterable[float] = [0.0, 25.0],
        prior_H_fwhm: float = 20.0,
        prior_He_H_fwhm_ratio: float = 0.1,
        prior_yplus: float = 0.1,
        prior_rms: float = 1.0,
        prior_baseline_coeffs: Optional[Iterable[float]] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        :param prior_H_area: Prior distribution on H line area (mK km s-1), where
            H_area = Gamma(alpha=2.0, beta=1.0/prior_H_area)
            defaults to 1000.0
        :type prior_H_area: float, optional
        :param prior_H_center: Prior distribution on H line centroid velocity (km s-1), where
            H_center ~ Normal(mu=prior_H_center[0], sigma=prior_H_center[1]) if :param:ordered is `False`
            H_center(cloud=N) ~ prior_H_center[0] + sum(H_center(cloud<N)) + Gamma(alpha=2.0, beta=1.0/prior_H_center[1]) if :param:ordered is `True`
            defaults to [0.0, 25.0]
        :type prior_H_center: Iterable[float], optional
        :param prior_H_fwhm: Prior distribution on H FWHM line width (km s-1), where
            H_fwhm ~ Gamma(alpha=3.0, beta=2.0/prior_H_fwhm)
            defaults to 20.0
        :type prior_H_fwhm: float, optional
        :param prior_He_H_fwhm_ratio: Prior distribution on He/H FWHM line width ratio, where
            He_H_fwhm_ratio ~ Normal(mu=1.0, sigma=prior_He_H_fwhm_ratio)
            defaults to 0.1
        :type prior_He_H_fwhm_ratio: float, optional
        :param prior_yplus: Prior distribution on y+, He/H abundance by number, where
            yplus ~ Gamma(alpha=3.0, beta=2.0/prior_yplus)
            defaults to 0.1
        :type prior_yplus: float, optional
        :param prior_rms: Prior distribution on spectral rms (mK), where
            rms ~ HalfNormal(sigma=prior_rms)
            defaults to 1.0
        :type prior_rms: float, optional
        :param prior_baseline_coeffs: Width of normal prior distribution on the normalized baseline polynomial
            coefficients. If None, use `[1.0]*(baseline_degree+1)`, defaults to None
        :type prior_baseline_coeff: float, optional
        :param ordered: If True, assume ordered velocities, defaults to False
        :type ordered: bool
        """
        if prior_baseline_coeffs is not None:
            prior_baseline_coeffs = {"observation": prior_baseline_coeffs}

        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # H line area (mK km s-1)
            H_area_norm = pm.Gamma("H_area_norm", alpha=2.0, beta=1.0, dims="cloud")
            H_area = pm.Deterministic("H_area", prior_H_area * H_area_norm, dims="cloud")

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
                    initval=np.linspace(-1.0, 1.0, self.n_clouds),
                )
                H_center = pm.Deterministic(
                    "H_center",
                    prior_H_center[0] + prior_H_center[1] * H_center_norm,
                    dims="cloud",
                )

            # H FWHM line width (km s-1)
            H_fwhm_norm = pm.Gamma("H_fwhm_norm", alpha=3.0, beta=2.0, dims="cloud")
            H_fwhm = pm.Deterministic("H_fwhm", prior_H_fwhm * H_fwhm_norm, dims="cloud")

            # He/H FWHM line width ratio
            He_H_fwhm_ratio_norm = pm.Normal(
                "He_H_fwhm_ratio_norm",
                mu=0.0,
                sigma=1.0,
                dims="cloud",
            )
            He_H_fwhm_ratio = pm.Deterministic(
                "He_H_fwhm_ratio",
                1.0 + prior_He_H_fwhm_ratio * He_H_fwhm_ratio_norm,
                dims="cloud",
            )

            # y+
            yplus_norm = pm.Gamma(
                "yplus_norm",
                alpha=3.0,
                beta=2.0,
                dims="cloud",
            )
            yplus = pm.Deterministic(
                "yplus",
                prior_yplus * yplus_norm,
                dims="cloud",
            )

            # Spectral rms (mK)
            rms_observation_norm = pm.HalfNormal("rms_observation_norm", sigma=1.0)
            _ = pm.Deterministic("rms_observation", prior_rms * rms_observation_norm)

            # H amplitude (mK)
            H_amplitude = pm.Deterministic(
                "H_amplitude",
                H_area / H_fwhm / np.sqrt(np.pi / (4.0 * np.log(2.0))),
                dims="cloud",
            )

            # He amplitude (mK)
            _ = pm.Deterministic("He_amplitude", H_amplitude * yplus / He_H_fwhm_ratio, dims="cloud")

            # He center velocity (km s-1)
            _ = pm.Deterministic("He_center", H_center - 122.15, dims="cloud")

            # He FWHM line width (km s-1)
            _ = pm.Deterministic("He_fwhm", H_fwhm * He_H_fwhm_ratio, dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model. Data key must be "observation"."""
        # Predict spectrum (mK)
        predicted_H = gaussian(
            self.data["observation"].spectral[:, None],
            self.model["H_amplitude"],
            self.model["H_center"],
            self.model["H_fwhm"],
        ).sum(axis=1)
        predicted_He = gaussian(
            self.data["observation"].spectral[:, None],
            self.model["He_amplitude"],
            self.model["He_center"],
            self.model["He_fwhm"],
        ).sum(axis=1)
        predicted_line = predicted_H + predicted_He

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["observation"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "observation",
                mu=predicted,
                sigma=self.model["rms_observation"],
                observed=self.data["observation"].brightness,
            )
