
Fractional MOD16 (MOD16-fET) Model
==================================

**This implementation of the MOD16 terrestrial evapotranspiration (ET) algorithm computes terrestrial ET for fractional land-cover.** It is adapted to produce high-resolution (30-m) estimates of ET for dynamic, fractional land-cover at regional-to-continental extent. Specifically, MOD16-fET implements the following changes to MOD16:

- Daily, instantaneous ET estimation, instead of separate nighttime and daytime estimates
- Improved ground heat flux estimation based on Santanello & Friedl (2003)
- Weighted average of ET across fractional land-cover types


Installation and Tests
----------------------

It's recommended that you install the package in "editable" mode using `pip`. From the root of the repository:

```sh
pip install -e .
```

If you want to install additional libraries needed for calibrating MOD16:

```sh
pip install -e .[calibration]
```

Tests can be run by:

```sh
python tests/tests.py
```

The MOD16 library depends on the [MOD17 library.](https://github.com/arthur-e/MOD17)



References
--------------

- Santanello, J. A., and M. A. Friedl. 2003. Diurnal covariation in soil heat flux and net radiation. *Journal of Applied Meteorology* 42 (6):851–862.
