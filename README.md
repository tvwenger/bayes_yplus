# bayes_yplus <!-- omit in toc -->

![publish](https://github.com/tvwenger/bayes_yplus/actions/workflows/publish.yml/badge.svg)
![tests](https://github.com/tvwenger/bayes_yplus/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/bayes-yplus/badge/?version=latest)](https://bayes-yplus.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/tvwenger/bayes_yplus/graph/badge.svg?token=8C0SU7JR3N)](https://codecov.io/gh/tvwenger/bayes_yplus)

A Bayesian Model of Radio Recombination Line Emission

`bayes_yplus` implements models to infer the helium abundance (`y+`) from radio recombination line (RRL) observations.

- [Installation](#installation)
  - [Basic Installation](#basic-installation)
  - [Development Installation](#development-installation)
- [Notes on Physics \& Radiative Transfer](#notes-on-physics--radiative-transfer)
- [Models](#models)
  - [`YPlusModel`](#yplusmodel)
  - [`ordered`](#ordered)
- [Syntax \& Examples](#syntax--examples)
- [Issues and Contributing](#issues-and-contributing)
- [License and Copyright](#license-and-copyright)


# Installation

## Basic Installation

Install with `pip` in a `conda` virtual environment:
```
conda create --name bayes_yplus -c conda-forge pymc pip
conda activate bayes_yplus
pip install bayes_yplus
```

## Development Installation

Alternatively, download and unpack the [latest release](https://github.com/tvwenger/bayes_yplus/releases/latest), or [fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and contribute to the development of `bayes_yplus`!

Install in a `conda` virtual environment:
```
cd /path/to/bayes_yplus
conda env create -f environment.yml
conda activate bayes_yplus-dev
pip install -e .
```

# Notes on Physics & Radiative Transfer

All models in `bayes_yplus` assume the emission is optically thin. The helium RRL is assumed to have a fixed centroid velocity -122.15 km/s from that of the hydrogen RRL.

# Models

The models provided by `bayes_yplus` are implemented in the [`bayes_spec`](https://github.com/tvwenger/bayes_spec) framework. `bayes_spec` assumes that the source of spectral line emission can be decomposed into a series of "clouds", each of which is defined by a set of model parameters. Here we define the models available in `bayes_yplus`.

## `YPlusModel`

The basic model is `YPlusModel`. The model assumes that the emission can be explained by hydrogen and helium RRL emission from discrete clouds. The following diagram demonstrates the relationship between the free parameters (empty ellipses), deterministic quantities (rectangles), model predictions (filled ellipses), and observations (filled, round rectangles). Many of the parameters are internally normalized (and thus have names like `_norm`). The subsequent tables describe the model parameters in more detail.

![hfs model graph](examples/yplus_model.png)

| Cloud Parameter<br>`variable` | Parameter                  | Units       | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}`                  | Default<br>`prior_{variable}` |
| :---------------------------- | :------------------------- | :---------- | :------------------------------------------------------------------------ | :---------------------------- |
| `H_area`                      | H RRL line area            | `mK km s-1` | $\int T_{B, \rm H} dV \sim {\rm Gamma}(\alpha=2.0, \beta=1.0/p)$          | `1000.0`                      |
| `H_center`                    | H RRL center velocity      | `km s-1`    | $V_{\rm LSR, H} \sim {\rm Normal}(\mu=p_0, \sigma=p_1)$                   | `[0.0, 25.0]`                 |
| `H_fwhm`                      | H RRL FWHM line width      | `km s-1`    | $\Delta V_{\rm H} \sim {\rm Gamma}(\alpha=3.0, \beta=2.0/p)$              | `20.0`                        |  |
| `He_H_fwhm_ratio`             | He/H FWHM line width ratio | ``          | $\Delta V_{\rm He}/\Delta V_{\rm H} \sim {\rm Normal}(\mu=1.0, \sigma=p)$ | `0.1`                         |
| `yplus`                       | He abundance by number     | ``          | $y^+ \sim {\rm Gamma}(\alpha=3.0, \beta=2.0/p)$                           | `0.1`                         |

| Hyper Parameter<br>`variable` | Parameter                                   | Units | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}` | Default<br>`prior_{variable}` |
| :---------------------------- | :------------------------------------------ | :---- | :------------------------------------------------------- | :---------------------------- |
| `rms`                         | Spectral rms noise                          | `mK`  | ${\rm rms} \sim {\rm HalfNormal}(\sigma=p)$              | `0.01`                        |
| `baseline_coeffs`             | Normalized polynomial baseline coefficients | ``    | $\beta_i \sim {\rm Normal}(\mu=0.0, \sigma=p_i)$         | `[1.0]*baseline_degree`       |

## `ordered`

An additional parameter to `set_priors` for these models is `velocity`. By default, this parameter is `False`, in which case the order of the clouds is arbitrary. Sampling from these models can be challenging due to the labeling degeneracy: if the order of clouds does not matter (i.e., the emission is optically thin), then each Markov chain could decide on a different, equally-valid order of clouds.

If we assume that the emission is optically thin, then we can set `ordered=True`, in which case the order of clouds is restricted to be increasing with velocity. This assumption can *drastically* improve sampling efficiency. When `ordered=True`, the `velocity` prior is defined differently:

| Cloud Parameter<br>`variable` | Parameter             | Units    | Prior, where<br>($p_0, p_1, \dots$) = `prior_{variable}`                 | Default<br>`prior_{variable}` |
| :---------------------------- | :-------------------- | :------- | :----------------------------------------------------------------------- | :---------------------------- |
| `H_center`                    | H RRL center velocity | `km s-1` | $V_i \sim p_0 + \sum_0^{i-1} V_i + {\rm Gamma}(\alpha=2, \beta=1.0/p_1)$ | `[0.0, 25.0]`                 |

# Syntax & Examples

See the various notebooks under [examples](https://github.com/tvwenger/bayes_yplus/tree/main/examples) for demonstrations of these models.

# Issues and Contributing

Anyone is welcome to submit issues or contribute to the development
of this software via [Github](https://github.com/tvwenger/bayes_yplus).

# License and Copyright

Copyright (c) 2024 Trey Wenger

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