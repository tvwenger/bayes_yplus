"""
test_yplus_model.py - tests for YPlusModel

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

import numpy as np

from bayes_spec import SpecData
from bayes_yplus import YPlusModel


def test_yplus_model():
    velocity = np.linspace(-100.0, 100.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(velocity, brightness, 1.0)}
    model = YPlusModel(data, 2, baseline_degree=1)
    model.add_priors(prior_baseline_coeffs=[1.0, 1.0])
    model.add_likelihood()
    assert model._validate()


def test_yplus_model_ordered():
    velocity = np.linspace(-100.0, 100.0, 1000)
    brightness = np.random.randn(1000)
    data = {"observation": SpecData(velocity, brightness, 1.0)}
    model = YPlusModel(data, 2, baseline_degree=1)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()
