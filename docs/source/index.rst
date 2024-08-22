bayes_yplus
===========

``bayes_yplus`` is a Bayesian model of radio recombination line emission. Written in the ``bayes_spec`` Bayesian modeling framework, ``bayes_yplus`` implements a model to infer the helium abundance by number, ``yplus``, from radio recombination line observations. The ``bayes_spec`` framework provides methods to fit this model to data using Monte Carlo Markov Chain methods.

Useful information can be found in the `bayes_yplus Github repository <https://github.com/tvwenger/bayes_yplus>`_, the `bayes_spec Github repository <https://github.com/tvwenger/bayes_spec>`_, and in the tutorials below.

============
Installation
============
.. code-block::

    conda create --name bayes_yplus -c conda-forge pymc pip
    conda activate bayes_yplus
    pip install bayes_yplus

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/tutorial
   notebooks/optimization

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules
