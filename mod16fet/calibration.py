'''
Calibration of MOD16 against a representative, global eddy covariance (EC)
flux tower network. The model calibration is based on Markov-Chain Monte
Carlo (MCMC). Example use:

    # For a single run with the configured number of chains
    python calibration.py tune --pft=1 --config="my_config.yaml"

    # For a 3-folds cross-validation
    python calibration.py tune --pft=1 --config="my_config.yaml" --k-folds=3

The general calibration protocol used here involves:

1. Check how well the chain(s) are mixing by running short, e.g., only 5000
draws from the posterior:
`python calibration.py tune 1 --draws=5000`
2. If any chain is "sticky," run a short chain while tuning the jump scale:
`python calibration.py tune 1 --tune=scaling --draws=5000`
3. Using the trace plot from Step (2) as a reference, experiment with
different jump scales to try and achieve the same (optimal) mixing when
tuning on `lambda` (default) instead, e.g.:
`python calibration.py tune 1 --scaling=1e-2 --draws=5000`
4. When the right jump scale is found, run a chain at the desired length.

Instead of changing `draws` and `scaling` at the command line, as above, you
could change these parameters in the configuration file.

Once a good mixture is obtained, it is necessary to prune the samples to
eliminate autocorrelation, e.g., from the command line:

    python calibration.py plot-autocorr --pft=1 --burn=1000 --thin=10

Or in Python (use `--ipdb` to get an interactive session):

    # sampler is already available if you used --ipdb, otherwise:
    sampler = MOD16StochasticSampler(...)
    sampler.plot_autocorr(burn = 1000, thin = 10)
    trace = sampler.get_trace(burn = 1000, thin = 10)

A thinned posterior can be exported from the command line, e.g.:

    python calibration.py export-posterior ET <parameter_name>
        output.h5 --burn=1000 --thin=10

NOTE: If using k-folds cross-validation, add the following option to any
command, where K is the number of folds:

    --k-folds=K

**The Cal-Val dataset** is a single HDF5 file that contains all the input
variables necessary to drive MOD16 as well as the observed latent heat fluxes
against which the model is calibrated. The HDF5 file specification is as
follows, where the shape of multidimensional arrays is given in terms of
T, the number of time steps (days); N, the number of tower sites; L, the
number of land-cover types (PFTs); and P, a sub-grid of MODIS pixels
surrounding a tower:

    FLUXNET/
      SEB               -- (T x N) Surface energy balance, from tower data
      air_temperature   -- (T x N) Air temperatures reported at the tower
      *latent_heat      -- (T x N) Observed latent heat flux [W m-2]
      validation_mask   -- (L x T x N) Indicates what site-days are reserved

    *MERRA2/
      LWGNT             -- (T x N) Net long-wave radiation, 24-hr mean [W m-2]
      LWGNT_daytime     -- (T x N) ... for daytime hours only
      LWGNT_nighttime   -- (T x N) ... for nighttime hours only
      PS                -- (T x N) Surface air pressure [Pa], 24-hr mean
      PS_daytime        -- (T x N) ... for daytime hours only
      PS_nighttime      -- (T x N) ... for nighttime hours only
      QV10M             -- (T x N) Water vapor mixing ratio at 10-meter height
      QV10M_daytime     -- (T x N) ... for daytime hours only
      QV10M_nighttime   -- (T x N) ... for nighttime hours only
      SWGDN             -- (T x N) Down-welling short-wave radiation [W m-2]
      SWGDN_daytime     -- (T x N) ... for daytime hours only
      SWGDN_nighttime   -- (T x N) ... for nighttime hours only
      T10M              -- (T x N) Air temperature at 10-meter height [deg K]
      T10M_daytime      -- (T x N) ... for daytime hours only
      T10M_nighttime    -- (T x N) ... for nighttime hours only
      Tmin              -- (T x N) Daily minimum air temperature [deg K]

    *MODIS/
      *MCD43GF_black_sky_sw_albedo
          -- (T x N x P) Short-wave albedo under black-sky conditions
      *MOD15A2HGF_LAI
          -- (T x N x P) Leaf area index in scaled units (10 * [m3 m-3])
      *MOD15A2HGF_fPAR
          -- (T x N x P) Fraction of photosynthetically active radiation [%]

    coordinates/
      lng_lat       -- (2 x N) Longitude, latitude coordinates of each tower

    state/
      *PFT          -- (N x P) The plant functional type (PFT) of each pixel
      *PFT_annual   -- (T x N x P) Same as "PFT" but with the option to change
                        PFT at every time step T; optional, but needed if the
                        land-cover classes (PFT classes) are dynamic
      elevation_m   -- (N) The elevation in meters above sea level

     site_id        -- (N) Unique identifier for each site, e.g., "US-BZS"
    time            -- (T x 3) The Year, Month, Day of each daily time step
    weights         -- (N) A number between 0 and 1 used to down-weight towers


NOTE: A star, `*`, indicates that this dataset or group's name can be changed
in the configuration file. All others are currently required to match this
specification exactly.

NOTE: For now, constraints (like an annual precipitation constraint) cannot be
used unless `classes_are_dynamic = True` in the configuration file. This is
because the only supported constraint (annual precipitation) requires a
`(T x N)` data structure.

NOTE: The biggest improvement needed here is a way for users to specify not
just the values of the prior and which parameters are fixed but also the
functional form of the prior; currently, this is hard-coded into the
`compile_et_model()` function.
'''

import datetime
import yaml
import os
import warnings
import numpy as np
import h5py
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import mod16fet
from collections import OrderedDict
from functools import partial
from statistics import mode
from pathlib import Path
from typing import Callable, Sequence
from scipy import signal
from matplotlib import pyplot
from mod17.calibration import AbstractSampler, BlackBoxLikelihood
from mod16fet import MOD16_FET, PFT_VALID, latent_heat_vaporization
from mod16fet.utils import restore_bplut, pft_dominant, flatten_params_dict

MOD16_DIR = os.path.dirname(mod16fet.__file__)
DRIVER_NAMES = (
    'lw_net', 'lw_net_day', 'lw_net_night', 'sw_rad', 'sw_rad_day', 'sw_albedo',
    'tmean', 'tmin', 'tmax', 'vpd', 'rhumidity', 'pressure',
    'fpar', 'lai'
)


class SimultaneousStochasticSampler(AbstractSampler):
    '''
    A Markov Chain-Monte Carlo (MCMC) sampler for MOD16-fET. The
    specific sampler used is the Differential Evolution (DE) MCMC algorithm
    described by Ter Braak (2008), though the implementation is specific to
    the PyMC3 library.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    model : Callable
        The function to call (with driver data and parameters); this function
        should take driver data as positional arguments and the model
        parameters as a `*Sequence`; it should require no external state.
    params_vector : Sequence or None
        TODO FIXME
    backend : str or None
        Path to a NetCDF4 file backend (Default: None)
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    constraints : Sequence or None
        Sequence of one or more Callables (function) that return a competing
        value of the objective function (e.g., an RMSE). If there is more than
        one Callable, they are each called and the largest value is retained.
        If the (final) return value is greater than the value of the original
        objective function, than that value is returned instead. This is a way
        to tell the sampler that certain conditions are associated, e.g., with
        very high RMSE. Each Callable should take one argument: a vector of
        model predictions.
    '''
    required_parameters = {
        'ET': MOD16_FET.required_parameters
    }
    required_drivers = {
        'ET': DRIVER_NAMES
    }

    def __init__(
            self, config: dict, model: Callable, params_vector: Sequence = None,
            backend: str = None, weights: Sequence = None,
            constraints: Sequence = None):
        self.backend = backend
        self.config = config
        self.constraints = constraints
        self.model = model
        if hasattr(model, '__name__'):
            self.name = model.__name__.strip('_').upper() # "_gpp" = "GPP"
        self.params = params_vector
        # Set the model's prior distribution assumptions and any fixed values
        self.prior = dict()
        self.weights = weights
        assert os.path.exists(os.path.dirname(backend))

    def compile_et_model(
            self, observed: Sequence, drivers: Sequence) -> pm.Model:
        '''
        Creates a new ET model based on the prior distribution. Model can be
        re-compiled multiple times, e.g., for cross validation.

        There are two attributes that are set on the sampler when it is
        initialized that could be helpful here:

            self.priors
            self.bounds

        `self.priors` is a dict with a key for each parameter that has
        informative priors. For parameters with a non-informative (Uniform)
        prior, `self.bounds` is a similar dict (with a key for each parameter)
        that describes the lower and upper bounds of the Uniform prior, but
        this is deprecated.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function

        Returns
        -------
        pm.Model
        '''
        # A function to extract the values of a sequence-valued dictionary
        repack = lambda d, pft: dict([(k, v[pft]) for k, v in d.items()])
        # Define the objective/ likelihood function
        log_likelihood = BlackBoxLikelihood(
            self.model, observed, x = drivers, weights = self.weights,
            objective = self.config['optimization']['objective'],
            constraints = self.constraints)
        # Prepare to add parameters for each PFT
        n_params = len(MOD16_FET.required_parameters)
        n_pft = len(PFT_VALID)
        # Get the start, end indices of the parameters for each PFT
        starts = np.arange(0, n_params * n_pft, n_params)
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            params_list = []
            # (Stochstic) Priors for unknown model parameters
            for j, idx in enumerate(zip(starts, starts + n_params)):
                i0, i1 = idx
                pft = PFT_VALID[j] # Just in case PFT codes start at int > 0
                # NOTE: Getting the parameters for *this* PFT class;
                #   params[i] below will refer to the ith parameter of the
                #   MOD16_FET.required_parameters vector
                params = self.params[i0:i1]
                # NOTE: tmin_close, tmin_open, and vpd_open are just copied from
                #   the original parameters table
                tmin_close = params[0]
                tmin_open = params[1]
                vpd_open = params[2]
                vpd_close =   pm.Uniform(
                    f'vpd_close{pft}', **repack(self.prior['vpd_close'], pft))
                gl_sh =       pm.LogNormal(
                    f'gl_sh{pft}', **repack(self.prior['gl_sh'], pft))
                gl_wv =       pm.LogNormal(
                    f'gl_wv{pft}', **repack(self.prior['gl_wv'], pft))
                g_cuticular = pm.LogNormal(
                    f'g_cuticular{pft}', **repack(self.prior['g_cuticular'], pft))
                csl =         pm.LogNormal(
                    f'csl{pft}', **repack(self.prior['csl'], pft))
                rbl_min =     pm.Triangular(
                    f'rbl_min{pft}', **repack(self.prior['rbl_min'], pft))
                rbl_max =     pm.Triangular(
                    f'rbl_max{pft}', **repack(self.prior['rbl_max'], pft))
                beta =        pm.Uniform(
                    f'beta{pft}', **repack(self.prior['beta'], pft))
                params_list.extend([
                    tmin_close, tmin_open, vpd_open, vpd_close, gl_sh, gl_wv,
                    g_cuticular, csl, rbl_min, rbl_max, beta])

            # Convert model parameters to a tensor vector
            params = pt.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        return model

    def run(
            self, observed: Sequence, drivers: Sequence,
            draws = 1000, chains = 3, tune = 'lambda', scaling: float = 1e-3,
            prior: dict = dict(), fixed: dict = dict(),
            check_shape: bool = False, save_fig: bool = False,
            show_fig: bool = True, var_names: Sequence = None) -> None:
        '''
        Fits the model using DE-MCMCz approach. `tune="lambda"` (default) is
        recommended; lambda is related to the scale of the jumps learned from
        other chains, but epsilon ("scaling") controls the scale directly.
        Using a larger value for `scaling` (Default: 1e-3) will produce larger
        jumps and may directly address "sticky" chains.

        Parameters
        ----------
        observed : Sequence
            The observed data the model will be calibrated against
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function
        draws : int
            Number of samples to draw (on each chain); defaults to 1000
        chains : int
            Number of chains; defaults to 3
        tune : str or None
            Which hyperparameter to tune: Defaults to 'lambda', but can also
            be 'scaling' or None.
        scaling : float
            Initial scale factor for epsilon (Default: 1e-3)
        prior : dict
            Dictionary of parameters and their prior values;
            should be of the form `{parameter: value}`
        fixed : dict
            Dictionary of parameters for which a fixed value should be used;
            should be of the form `{parameter: value}`
        check_shape : bool
            True to require that input driver datasets have the same shape as
            the observed values (Default: False)
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        show_fig: bool
            True to show the trace plot at the end of a run (Default: True)
        var_names : Sequence
            One or more variable names to show in the plot
        '''
        assert not check_shape or drivers[0].shape == observed.shape,\
            'Driver data should have the same shape as the "observed" data'
        assert len(drivers) == len(self.required_drivers[self.name]),\
            'Did not receive expected number of driver datasets!'
        assert tune in ('lambda', 'scaling') or tune is None
        self.prior.update(prior) # Update prior assumptions
        # Generate an initial goodness-of-fit score
        predicted = self.model(self.params, *drivers)
        if self.weights is not None:
            score = np.sqrt(
                np.nanmean(((predicted - observed) * self.weights) ** 2))
        else:
            score = np.sqrt(np.nanmean(((predicted - observed)) ** 2))
        print('-- RMSD at the initial point: %.3f' % score)
        print('Compiling model...')
        try:
            compiler = getattr(self, 'compile_%s_model' % self.name.lower())
        except AttributeError:
            raise AttributeError('''Could not find a compiler for model named
            "%s"; make sure that a function "compile_%s_model()" is defined on
             this class''' % (model_name, model_name))
        with compiler(observed, drivers) as model:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                step_func = pm.DEMetropolisZ(tune = tune, scaling = scaling)
                trace = pm.sample(
                    draws = draws, step = step_func, cores = chains,
                    chains = chains, idata_kwargs = {'log_likelihood': True})
            if self.backend is not None:
                print('Writing results to file...')
                trace.to_netcdf(self.backend)
            if var_names is None:
                az.plot_trace(trace, var_names = ['~log_likelihood'])
            else:
                az.plot_trace(trace, var_names = var_names)
            if save_fig:
                pyplot.savefig('.'.join(self.backend.split('.')[:-1]) + '.png')
            elif show_fig:
                pyplot.show()


class CalibrationAPI(object):
    '''
    Convenience class for calibrating the MOD16 ET model. Meant to be used
    at the command line, in combination with the option to specify a
    configuration file:

        --config=my_configuration.yaml

    For example, to run calibration for PFT 1, you would write:

        python calibration.py tune --pft=1 --config=my_configuration.yaml

    If `--config` is not provided, the default configuration file,
    `mod16/MOD16_calibration_config.yaml` will be used.
    '''

    def __init__(self, config = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                MOD16_DIR, 'data/MOD16_calibration_config.yaml')
        print(f'Using configuration file: "{config_file}"')
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.hdf5 = self.config['data']['file']

    def _filter(self, raw: Sequence, size: int):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        return raw # Or, revert to the raw data

    def _load_data(self, exceptions: dict = None):
        'Read in driver datasets from the HDF5 file, structured by years'
        constraints = dict()
        with h5py.File(self.hdf5, 'r') as hdf:
            sites = hdf['site_id'][:].tolist()
            if hasattr(sites[0], 'decode'):
                sites = [s.decode('utf-8') for s in sites]
            # Figure out the calibration reference period (defaults to all
            #   data available)
            time = hdf['time'][:]
            t0 = 0
            if 'date_start' in self.config['data'].keys():
                date_start = self.config['data']['date_start']
                ds = datetime.datetime.strptime(date_start, '%Y-%m-%d')
                t0 = int(np.argwhere(
                    np.logical_and(
                        np.logical_and(time[:,0] == ds.year,
                            time[:,1] == ds.month),
                        time[:,2] == ds.day)).ravel()[0])
            # Number of time steps
            nsteps = time.shape[0]
            # In case some tower sites should not be used
            blacklist = self.config['data']['sites_blacklisted']

            pft_map = hdf[self.config['data']['class_map']]
            # Also, ensure the blacklist matches the shape of this mask;
            #   i.e., blacklisted sites should NEVER be used
            if blacklist is not None:
                if len(blacklist) > 0:
                    blacklist = np.array(blacklist)
                    site_mask = (~np.in1d(sites, blacklist))

            # Get tower weights, for when towers are too close together
            weights = hdf['weights'][:]
            # If only a single value is given for each site, repeat the weight
            #   along the time axis
            if weights.ndim == 1:
                weights = weights[None,...].repeat(nsteps, axis = 0)
            weights = weights[:,site_mask]

            # Get a (P x T x N) array of PFT fractions; there's a much easier
            #   way to do this with indexing that *used* to work but no longer:
            #   pft_map = pft_map[:,(time[:,0] - time[:,0].min())]
            year_idx = (time[:,0] - time[:,0].min())
            new_pft_map = []
            for idx in np.unique(year_idx):
                n_repeats = np.in1d(year_idx, idx).sum()
                new_pft_map.append(
                    pft_map[:,idx][:,np.newaxis].repeat(n_repeats, axis = 1))
            pft_map = np.concatenate(new_pft_map, axis = 1)
            pft_map = pft_map[:,t0:]
            # Subset to just those sites we'll be using
            pft_map = pft_map[...,site_mask]

            # After subsetting the PFT map, time now starts at this start date
            time = time[t0:]

            # Read in tower observations; we select obs of interest in three
            #   steps because we want *only* matching tower-day observations
            #   but we'll want driver data for a full year if that year
            #   contains *any* matching tower-day observations
            print('Masking out validation data...')
            tower_obs = hdf[self.config['data']['target_observable']][t0:,site_mask]

            # Read in driver datasets
            print('Loading driver datasets...')
            lookup = self.config['data']['datasets']
            # Allow exceptions to the configuration file's datasets to be
            #   specified here; i.e., use a different driver
            if exceptions is not None:
                lookup.update(exceptions)

            lw_net = hdf[lookup['LWGNT'][0]][t0:][:,site_mask]
            lw_net_day = hdf[lookup['LWGNT'][1]][t0:][:,site_mask]
            lw_net_night = hdf[lookup['LWGNT'][2]][t0:][:,site_mask]
            sw_rad = hdf[lookup['SWGDN'][0]][t0:][:,site_mask]
            sw_rad_day = hdf[lookup['SWGDN'][1]][t0:][:,site_mask]
            sw_albedo = hdf[lookup['albedo']][t0:][:,site_mask]
            tmean = hdf[lookup['Tmean']][t0:][:,site_mask]
            tmax = hdf[lookup['Tmax']][t0:][:,site_mask]
            tmin = hdf[lookup['Tmin']][t0:][:,site_mask]
            vpd = hdf[lookup['VPD']][t0:][:,site_mask]
            if tmin.min() < 0 or tmin.max() < 100:
                print("WARNING: Temperatures are expected in deg K but may actually be in deg C")

            # Compute relative humidity
            rhumidity = MOD16_FET.rhumidity(tmean, vpd)

            # After VPD is calculated, air pressure is based solely
            #   on elevation
            elevation = hdf[lookup['elevation']][:]
            elevation = elevation[np.newaxis,:]\
                .repeat(nsteps, axis = 0)[:,site_mask]
            # Assumed to be (T x N) or (T x N x ...)
            if elevation.ndim == 3:
                # If there is a site sub-grid...
                elevation = elevation.mean(axis = -1)
            pressure = MOD16_FET.air_pressure(elevation)[t0:]

            # Read in fPAR, LAI
            fpar = hdf[lookup['fPAR']][t0:][:,site_mask]
            lai = hdf[lookup['LAI']][t0:][:,site_mask]

            # If a heterogeneous sub-grid is used at each tower (i.e., there
            #   is a third axis to these datasets), then average over that
            #   sub-grid
            if sw_albedo.ndim == 3 and fpar.ndim == 3 and lai.ndim == 3:
                sw_albedo = np.nanmean(sw_albedo, axis = -1)
                fpar = np.nanmean(fpar, axis = -1)
                lai = np.nanmean(lai, axis = -1)
            # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR and LAI
            fpar /= 100
            lai /= 10

        drivers = dict(zip(DRIVER_NAMES, [
            lw_net, lw_net_day, lw_net_night, sw_rad, sw_rad_day, sw_albedo,
            tmean, tmin, tmax, vpd, rhumidity, pressure, fpar, lai]))
        # Clean the tower observations
        tower_obs = self.clean_observed(tower_obs)
        return (tower_obs, drivers, weights, pft_map)

    def clean_observed(
            self, raw: Sequence, filter_length: int = 2) -> Sequence:
        '''
        Cleans observed tower flux data according to a prescribed protocol.
        NOT intended to be called from the command line.

        Parameters
        ----------
        raw : Sequence
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data

        Returns
        -------
        Sequence
        '''
        # Read in the observed data and apply smoothing filter; then mask out
        #   negative latent heat observations
        assert raw.ndim == 2, 'Expected 2D input raw observations'
        obs = self._filter(raw, filter_length)
        # Filtering out negative latent heat fluxes, as these generally arise
        #   during stable conditions, when EC tower measurements are less
        #   reliable (see https://doi.org/10.5194/bg-21-2051-2024)
        return np.where(obs < 0, np.nan, obs)

    def export_posterior(
            self, model: str, param: str, output_path: str, thin: int = 10,
            burn: int = 1000, k_folds: int = 1):
        '''
        Exports posterior distribution for a parameter, for each PFT to HDF5.

        Parameters
        ----------
        model : str
            The name of the model (only "ET" is supported)
        param : str
            The model parameter to export
        output_path : str
            The output HDF5 file path
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        k_folds : int
            The number of k-folds used in cross-calibration/validation;
            if more than one (default), the folds for each PFT will be
            combined into a single HDF5 file
        '''
        params_dict = restore_bplut(self.config['BPLUT'][model])
        bplut = params_dict.copy()
        # Filter the parameters to just those for the PFT of interest
        post = []
        for pft in self.config['data']['classes']:
            params = dict([(k, v[pft]) for k, v in params_dict.items()])
            post_by_fold = []
            for fold in range(1, k_folds + 1):
                backend = self.config['optimization']['backend_template'] %\
                    (model, pft)
                if k_folds > 1:
                    backend = backend[:backend.rfind('.')] + f'-k{fold}' + backend[backend.rfind('.'):]
                # NOTE: This value was hard-coded in the extant version of MOD16
                if 'beta' not in params:
                    params['beta'] = 250
                sampler = MOD16StochasticSampler(
                    self.config, getattr(MOD16, '_%s' % model.lower()), params,
                    backend = backend)
                trace = sampler.get_trace()
                fit = trace.sel(draw = slice(burn, None, thin))['posterior']
                if param in fit:
                    post_by_fold.append(
                        az.extract_dataset(fit, combined = True)[param].values)
                else:
                    # In case there is, e.g., a parameter that takes on a
                    #   constant value for a specific PFT
                    if k_folds > 1:
                        post_by_fold.append(
                            np.ones((1, post[-1].shape[-1])) * np.nan)
                    else:
                        a_key = list(fit.keys())[0]
                        post_by_fold.append(
                            np.ones(fit[a_key].values.shape) * np.nan)
            if k_folds > 1:
                post.append(np.vstack(post_by_fold))
            else:
                post.extend(post_by_fold)
        # If not every PFT's posterior has the same number of samples (e.g.,
        #   when one set of chains was run longer than another)...
        if not all([p.shape == post[0].shape for p in post]):
            max_len = max([p.shape for p in post])[0]
            # ...Reshape all posteriors to match the greatest sample size
            post = [
                np.pad(
                    p.astype(np.float32), (0, max_len - p.size),
                    mode = 'constant', constant_values = (np.nan,))
                for p in post
            ]
        with h5py.File(output_path, 'a') as hdf:
            post = np.stack(post)
            ts = datetime.date.today().strftime('%Y-%m-%d') # Today's date
            dataset = hdf.create_dataset(
                f'{param}_posterior', post.shape, np.float32, post)
            dataset.attrs['description'] = 'CalibrationAPI.export_posterior() on {ts}'

    def plot_autocorr(self, pft: int, k_folds: int = 1, **kwargs):
        '''
        Plot the autocorrelation in the trace for each parameter.

        Parameters
        ----------
        pft : int
            The numeric PFT code
        '''
        # Filter the parameters to just those for the PFT of interest
        params_dict = restore_bplut(self.config['BPLUT']['ET'])
        params_dict = dict([(k, v[pft]) for k, v in params_dict.items()])
        backend = self.config['optimization']['backend_template'] % ('ET', pft)
        # Use a different naming scheme for the backend
        if k_folds > 1:
            for fold in range(1, k_folds + 1):
                sampler = MOD16StochasticSampler(
                    self.config, mod16fet._et, params_dict,
                    backend = backend[:backend.rfind('.')] + f'-k{fold}' + backend[backend.rfind('.'):])
                sampler.plot_autocorr(**kwargs, title = f'Fold {fold} of {k_folds}')
        else:
            sampler = MOD16StochasticSampler(
                self.config, mod16fet._et, params_dict, backend = backend)
            sampler.plot_autocorr(**kwargs)

    def tune(
            self, plot_trace: bool = False, ipdb: bool = False,
            save_fig: bool = False, **kwargs):
        '''
        Run the MOD16 ET calibration.

        Parameters
        ----------
        plot_trace : bool
            True to plot the trace for a previous calibration run; this will
            also NOT start a new calibration (Default: False)
        ipdb : bool
            True to drop the user into an ipdb prompt, prior to and instead of
            running calibration
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        **kwargs
            Additional keyword arguments passed to
            `MOD16StochasticSampler.run()`

        NOTE that `MOD16StochasticSampler` inherits methods from the `mod17`
        module, including [run()](https://arthur-e.github.io/MOD17/calibration.html#mod17.calibration.StochasticSampler).
        '''
        # Pass configuration parameters to MOD16StochasticSampler.run()
        for key in ('chains', 'draws', 'tune', 'scaling'):
            if key in self.config['optimization'].keys():
                kwargs[key] = self.config['optimization'][key]
        # Load the params dict
        params_dict = restore_bplut(self.config['BPLUT']['ET'])
        # NOTE: This value was hard-coded in the extant version of MOD16
        if np.isnan(params_dict['beta']).all():
            params_dict['beta'] = 250
        # Convert to the vectorized form expected in the new model
        params_vector = flatten_params_dict(params_dict)

        # Load the data
        tower_obs, drivers, weights, constr = self._load_data()

        print('Initializing sampler...')
        backend = self.config['optimization']['backend']
        sampler = MOD16StochasticSampler(
            self.config, mod16fet._et, params_dict, backend = backend,
            weights = weights, constraints = constraints)

        # Either: Enter diagnostic mode or run the sampler
        if plot_trace or ipdb:
            # This matplotlib setting prevents labels from overplotting
            pyplot.rcParams['figure.constrained_layout.use'] = True
            trace = sampler.get_trace()
            if ipdb:
                import ipdb
                ipdb.set_trace()#FIXME
            az.plot_trace(trace, var_names = mod16fet.required_parameters)
            pyplot.show()
            return

        # Get (informative) priors for just those parameters that have them
        with open(self.config['optimization']['prior'], 'r') as file:
            prior = yaml.safe_load(file)
        prior_params = list(filter(
            lambda p: p in prior.keys(), sampler.required_parameters['ET']))
        prior = dict([
            (p, dict([(k, v[pft]) for k, v in prior[p].items()]))
            for p in prior_params
        ])

        # Determine whether any parameters are fixed
        fixed = []
        for name in mod16fet.required_parameters:
            if self.config['optimization']['fixed'] is None:
                break
            if name in self.config['optimization']['fixed'].keys():
                fixed.append(
                    (name, self.config['optimization']['fixed'][name][pft]))
        fixed = dict(fixed)

        # Set var_names to tell ArviZ to plot only the free parameters; i.e.,
        #   those with priors and which are not fixed
        var_names = list(filter(
            lambda x: x in prior.keys(), mod16fet.required_parameters))
        # Remove any random variables that have fixed values from the list
        #   of variables to be plotted
        for key in fixed.keys():
            if fixed[key] is not None and key in var_names:
                var_names.remove(key)
        kwargs.update({'var_names': var_names})

        # TODO Someday, MOD17 will be updated to allow "drivers" to be a
        #   dictionary instead of a sequence; until then: drivers.values()
        sampler.run( # Only show the trace plot if not using k-folds
            tower_obs, drivers.values(), prior = prior, fixed = fixed,
            save_fig = save_fig, show_fig = (k_folds == 1), **kwargs)


if __name__ == '__main__':
    import fire
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
