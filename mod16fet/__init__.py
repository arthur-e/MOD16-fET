r'''
    0: Bare Ground/ Litter
    1: Annual Forbs and Grasses
    2: Perennial Forbs and Grasses
    3: Shrubs
    4: Deciduous Broadleaf Trees
    5: Evergreen Needleleaf Trees
    6: Evergreen Broadleaf Trees
    7: Wetlands
'''

__version__ = 'v0.1.0'

import warnings
import numpy as np
from collections import Counter
from typing import Iterable, Sequence, Tuple
from numbers import Number
from mod17 import linear_constraint

PFT_VALID = (0,1,2,3,4,5,6,7)
STEFAN_BOLTZMANN = 5.67e-8 # Stefan-Boltzmann constant, W m-2 K-4
SPECIFIC_HEAT_CAPACITY_AIR = 1013 # J kg-1 K-1, Monteith & Unsworth (2001)
# Ratio of molecular weight of water vapor to that of dry air (ibid.)
MOL_WEIGHT_WET_DRY_RATIO_AIR = 0.622
TEMP_LAPSE_RATE = 0.0065 # Standard temperature lapse rate [-(deg K) m-1]
GRAV_ACCEL = 9.80665 # Gravitational acceleration [m s-2]
GAS_LAW_CONST = 8.3143 # Ideal gas law constant [m3 Pa (mol)-1 K-1]
AIR_MOL_WEIGHT = 28.9644e-3 # Molecular weight of air [kg (mol)-1]
STD_TEMP_K = 288.15 # Standard temperature at sea level [deg K]
STD_PRESSURE_PASCALS = 101325.0 # Standard pressure at sea level [Pa]
# A pre-determined quantity, not physically meaningful, used in air_pressure()
AIR_PRESSURE_RATE = GRAV_ACCEL / (TEMP_LAPSE_RATE * (GAS_LAW_CONST / AIR_MOL_WEIGHT))

# Calculate the latent heat of vaporization (J kg-1)
latent_heat_vaporization = lambda temp_k: (2.501 - 0.002361 * (temp_k - 273.15)) * 1e6


class MOD16(object):
    r'''
    The MODIS MxD16 Evapotranspiration model. The required model parameters are:

    - `tmin_close`: Temperature at which stomata are almost completely
        closed due to (minimum) temperature stress (deg C)
    - `tmin_open`: Temperature at which stomata are completely open, i.e.,
        there is no effect of temperature on transpiration (deg C)
    - `vpd_open`: The VPD at which stomata are completely open, i.e.,
        there is no effect of water stress on transpiration (Pa)
    - `vpd_close`: The VPD at which stomata are almost completely closed
        due to water stress (Pa)
    - `gl_sh`: Leaf conductance to sensible heat per unit LAI
        (m s-1 LAI-1);
    - `gl_wv`: Leaf conductance to evaporated water per unit LAI
        (m s-1 LAI-1);
    - `g_cuticular`: Leaf cuticular conductance (m s-1);
    - `csl`: Mean potential stomatal conductance per unit leaf area (m s-1);
    - `rbl_min`: Minimum atmospheric boundary layer resistance (s m-1);
    - `rbl_max`: Maximum atmospheric boundary layer resistance (s m-1);
    - `beta`: Factor in soil moisture constraint on potential soil
        evaporation, i.e., (VPD / beta); from Bouchet (1963)

    Parameters
    ----------
    params : dict
        Dictionary of model parameters
    '''
    required_parameters = [
        'tmin_close', 'tmin_open', 'vpd_open', 'vpd_close', 'gl_sh', 'gl_wv',
        'g_cuticular', 'csl', 'rbl_min', 'rbl_max', 'beta'
    ]

    def __init__(self, params: dict):
        self.params = params
        for key in self.required_parameters:
            setattr(self, key, params[key])

    @staticmethod
    def _et(
            parameters, lw_net, lw_net_day, lw_net_night, sw_rad, sw_rad_day,
            sw_albedo, tmean, tmin, tmax, mat, vpd, rhumidity, pressure, fpar,
            lai, f_wet = None, tiny = 1e-7, r_corr = None
        ) -> Number:
        '''
        Optimized ET code, intended for use in model calibration ONLY.
        Returns combined day and night ET.

        Parameters
        ----------
        params : list
            A list of arrays, each array representing a different parameter,
            in the order specified by `MOD16.required_parameters`. Each array
            should be a (1 x N) array, where N is the number of sites/pixels.
        *drivers
            Every subsequent argument is a separate (T x N) where T is the
            number of time steps and N is the number of sites/pixels.

        Returns
        -------
        numpy.ndarray
            The total latent heat flux [W m-2] for each site/pixel
        '''
        day, night = MOD16._evapotranspiration(
            parameters, lw_net, lw_net_day, lw_net_night, sw_rad, sw_rad_day,
            sw_albedo, tmean, tmin, tmax, mat, vpd, rhumidity, pressure, fpar,
            lai, f_wet = None, tiny = 1e-7, r_corr = r_corr)
        return np.add(day, night)

    @staticmethod
    def _evapotranspiration(
            parameters, lw_net, lw_net_day, lw_net_night, sw_rad, sw_rad_day,
            sw_albedo, tmean, tmin, tmax, mat, vpd, rhumidity, pressure, fpar,
            lai, f_wet = None, tiny = 1e-7, r_corr = None
        ) -> Iterable[Tuple[Sequence, Sequence]]:
        '''
        Optimized ET code, intended for use in model calibration ONLY. The
        `params` are expected to be given in the order specified by
        `MOD16.required_parameters`. NOTE: total ET values returned are in
        [W m-2], for comparison to tower ET values. Divide by the latent heat
        of vaporization (J kg-1) to obtain a mass flux (kg m-2 s-1).

        Parameters
        ----------
        params : list
            A list of arrays, each array representing a different parameter,
            in the order specified by `MOD16.required_parameters`. Each array
            should be a (1 x N) array, where N is the number of sites/pixels.
        *drivers
            Every subsequent argument is a separate (T x N) where T is the
            number of time steps and N is the number of sites/pixels.

        Returns
        -------
        numpy.ndarray
            The ET associated with a given PFT class; a (P x ...) array,
            for P total PFT classes
        '''
        # Net radiation to surface, based on down-welling short-wave
        #   radiation and net long-wave radiation
        rad_net = sw_rad * (1 - sw_albedo) + lw_net
        rad_net_day = sw_rad_day * (1 - sw_albedo) + lw_net_day
        # Ground heat flux
        g_soil = MOD16.ground_heat_flux(rad_net_day, lw_net_night, tmin, tmax)
        # Radiation received by the soil, see Section 2.2 of User Guide
        rad_soil = (1 - fpar) * (rad_net - g_soil)
        # Radiation intercepted by the canopy
        rad_canopy = fpar * rad_net

        # Compute wet surface fraction and other quantities
        _svp = svp(tmean)
        f_wet = np.where(rhumidity < 0.7, 0, np.power(rhumidity, 4))
        # Slope of saturation vapor pressure curve
        s = svp_slope(tmean, _svp)
        # Latent heat of vaporization (J kg-1)
        lhv = latent_heat_vaporization(tmean)
        # Psychrometric constant
        gamma = psychrometric_constant(pressure, tmean)
        # Correction for atmospheric temperature and pressure
        #   (Equation 13, MOD16 C6.1 User's Guide)
        if r_corr is None:
            r_corr = (101300 / pressure) * (tmean / 293.15)**1.75

        # Resistance to radiative heat transfer through air ("rrc")
        rho = MOD16.air_density(tmean, pressure, rhumidity) # Air density
        r_r = (rho * SPECIFIC_HEAT_CAPACITY_AIR) / (
            4 * STEFAN_BOLTZMANN * tmean**3)

        e_canopy = list()
        e_soil = list()
        transpiration = list()
        et_total = list()
        for i, pft in enumerate(PFT_VALID):
            # NOTE: Getting the parameters for *this* PFT class
            params = parameters[i]
            # Anticipate warnings because, in some cases, f_wet may be zero,
            #   which would lead to zero in the denominator of r_h and r_e
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # -- Wet canopy resistance to sensible heat ("rhc")
                r_h = 1 / (params[4] * lai * f_wet)
                # -- Wet canopy resistance to evaporated water on the surface
                #   ("rvc")
                r_e = 1 / (params[5] * lai * f_wet)
                # -- Aerodynamic resistance to evaporated water on the wet
                #   canopy surface ("rhrc")
                r_a_wet = np.divide(r_h * r_r, r_h + r_r) # (s m-1)

            # EVAPORATION FROM WET CANOPY
            e = np.divide(
                f_wet * ((s * rad_canopy) + (
                    rho * SPECIFIC_HEAT_CAPACITY_AIR * fpar * vpd * 1/r_a_wet
                )),
                s + ((pressure * SPECIFIC_HEAT_CAPACITY_AIR * r_e) *\
                    1/(lhv * MOL_WEIGHT_WET_DRY_RATIO_AIR * r_a_wet)))
            e_canopy.append(np.where(lai * f_wet <= tiny, 0, e))

            # Surface conductance
            m_tmin = linear_constraint(params[0], params[1])
            m_vpd = linear_constraint(params[2], params[3], 'reversed')
            g_surf = (params[7] * m_tmin(tmin - 273.15) * m_vpd(vpd))
            g_surf /= r_corr
            g_cuticular = params[6] / r_corr
            # -- Canopy conductance, should be zero when LAI or f_wet are zero;
            #   updated calculation for conductance to sensible heat, see User
            #   Guide's Equation 15
            gl_sh = params[4] * lai * (1 - f_wet)
            g = ((gl_sh * (g_surf + g_cuticular)) / (
                gl_sh + g_surf + g_cuticular))
            g_canopy = np.where(
                np.logical_and(lai > 0, (1 - f_wet) > 0), g, tiny)
            # -- Aerodynamic resistance to heat, water vapor from dry canopy
            #   surface into the air (Equation 16, MOD16 C6.1 User's Guide)
            r_a_dry = (1/params[4] * r_r) / (1/params[4] + r_r) # (s m-1)

            # PLANT TRANSPIRATION
            if np.any(g_surf > 0):
                t = (1 - f_wet) * ((s * rad_canopy) +\
                    (rho * SPECIFIC_HEAT_CAPACITY_AIR * fpar * (vpd / r_a_dry)))
                t /= (s + gamma * (1 + (1 / g_canopy) / r_a_dry))
            else:
                t = 0
            transpiration.append(t)

            # BARE SOIL EVAPORATION
            # -- Total aerodynamic resistance as a function of VPD and the
            #   atmospheric boundary layer resistance...
            # VERIFIED 2024-02, as VPD is really a proxy for soil moisture
            #   here; hence "boundary-layer resistance" (really surface
            #   resistance) should be low when VPD is low (soil is "wet")
            r_tot = np.where(vpd <= params[2], params[8], # rbl_min
                np.where(vpd >= params[3], params[9], # rbl_max
                params[9] - (
                    (params[9] - params[8]) * (params[3] - vpd))\
                        / (params[3] - params[2])))
            # ...CORRECTED for atmospheric temperature, pressure
            r_tot = r_tot / r_corr
            # -- Aerodynamic resistance at the soil surface
            r_as = (r_tot * r_r) / (r_tot + r_r)
            # -- Terms common to evaporation and potential evaporation
            numer = (s * rad_soil) +\
                (rho * SPECIFIC_HEAT_CAPACITY_AIR * (1 - fpar) * (vpd / r_as))
            denom = (s + gamma * (r_tot / r_as))
            # -- Evaporation from "wet" soil (saturated fraction)
            evap_sat = (numer * f_wet) / denom
            # -- (Potential) Evaporation from unsaturated fraction
            evap_unsat = (numer * (1 - f_wet)) / denom
            # -- Finally, apply the soil moisture constraint from Fisher et al.
            #   (2008); see MOD16 C6.1 User's Guide, pp. 9-10
            e = evap_sat + evap_unsat * rhumidity**(vpd / params[10])
            e_soil.append(e)

            # Result is the sum of the three components
            et_total.append((transpiration[i] + e_canopy[i] + e_soil[i]))

        return np.stack(et_total, axis = 0)

    @staticmethod
    def air_density(
            temp_k: Number, pressure: Number, rhumidity: Number
        ) -> Number:
        '''
        NIST simplified air density formula with buoyancy correction from:

        - National Physical Laboratory (2021),
            ["Buoyancy Correction and Air Density Measurement."](http://resource.npl.co.uk/docs/science_technology/mass_force_pressure/clubs_groups/instmc_weighing_panel/buoycornote.pdf)


        Parameters
        ----------
        temp_k : int or float or numpy.ndarray
            Air temperature (degrees K)
        pressure : int or float or numpy.ndarray
            Air pressure (Pa)
        rhumidity : int or float or numpy.ndarray
            Relative humidity, on the interval [0, 1]

        Returns
        -------
        int or float or numpy.ndarray
            Air density (kg m-3)
        '''
        return np.divide( # Convert Pa to mbar, RH to RH% (percentage)
            0.348444 * (pressure / 100) - (rhumidity * 100) *\
                (0.00252 * (temp_k - 273.15) - 0.020582),
            temp_k) # kg m-3

    @staticmethod
    def air_pressure(elevation_m: Number) -> Number:
        r'''
        Atmospheric pressure as a function of elevation. From the discussion on
        atmospheric statics (p. 168) in:

        - Iribane, J.V., and W.L. Godson, 1981. Atmospheric Thermodynamics
            2nd Edition. D. Reidel Publishing Company, Dordrecht,
            The Netherlands.

        It is calculated:

        $$
        P_a = P_{\text{std}}\times \left(
            1 - \ell z T_{\text{std}}^{-1}
          \right)^{5.256}
        $$

        Where \(\ell\) is the standard temperature lapse rate (0.0065 deg K
        per meter), \(z\) is the elevation in meters, and \(P_{\text{std}}\),
        \(T_{\text{std}}\) are the standard pressure (101325 Pa) and
        temperature (288.15 deg K) at sea level.

        Parameters
        ----------
        elevation_m : Number

        Returns
        -------
        Number
            Air pressure in Pascals
        '''
        temp_ratio = 1 - ((TEMP_LAPSE_RATE * elevation_m) / STD_TEMP_K)
        return STD_PRESSURE_PASCALS * np.power(temp_ratio, AIR_PRESSURE_RATE)

    @staticmethod
    def ground_heat_flux(
            rad_net_day: Number, lw_net_night: Number,
            tmin: Number, tmax: Number
        ) -> Iterable[Tuple[Sequence, Sequence]]:
        r'''
        Ground heat flux [MJ m-2], based on Santanello & Friedl (2002):

        $$
        \text{max}\left(\frac{G}{A}\right) = 0.0074\,\Delta T + 0.088
        $$

        The ratio is the same for night or day, but we compute a total daily
        $G$ as the sum of separate day and night quantities:

        $$
        G_{\text{day}} = (G^*/A) \times \left(R_S(1 - \alpha) +
            R_{L,\text{day}}\right)
        $$
        $$
        G_{\text{night}} = (G^*/A) \times R_{L,\text{night}}
        $$
        $$
        G = G_{\text{day}} + G_{\text{night}}
        $$

        Parameters
        ----------
        rad_net_day : int or float or numpy.ndarray
            Net radiation received at the earth's sruface during daytime hours;
            i.e., integrated while the sun is up, (J m-2 s-1) or (W m-2)
        lw_net_night : int or float or numpy.ndarray
            Net downward long-wave radiation during nighttime hours; i.e.,
            integrated while the sun is down, (J m-2 s-1) or (W m-2)
        tmin : int or float or numpy.ndarray
            Minimum daily temperature (degrees K)
        tmax : int or float or numpy.ndarray
            Maximum daily temperature (degrees K)

        Returns
        -------
        Number or numpy.ndarray
        '''
        ratio = 0.0074 * (tmax - tmin) + 0.088
        g_day = ratio * rad_net_day # i.e., SW and LW radiation during the day
        g_night = ratio * lw_net_night
        return g_day + g_night

    @staticmethod
    def vpd(qv10m: Number, pressure: Number, tmean: Number) -> Number:
        r'''
        Computes vapor pressure deficit (VPD) from surface meteorology:

        $$
        \text{VPD} = \left(610.7\times \text{exp}\left(
          \frac{17.38\times T}{239 + T}
        \right) - \text{AVP}\right)
        $$

        Where \(T\) is the temperature in deg K and the actual vapor pressure
        (AVP) is given:

        $$
        \text{AVP} = \frac{\text{QV10M}\times
            \text{P}}{0.622 + 0.379\times \text{QV10M}}
        $$

        Where P is the air pressure in Pascals and QV10M is the water vapor
        mixing ratio at 10-meter height.

        Parameters
        ----------
        qv10m : int or float or numpy.ndarray
            Water vapor mixing ratio at 10-meter height (Pa)
        pressure : int or float or numpy.ndarray
            Atmospheric pressure (Pa)
        tmean : int or float or numpy.ndarray
            Mean daytime temperature (degrees K)

        Returns
        -------
        int or float or numpy.ndarray
        '''
        temp_c = tmean - 273.15
        # Actual vapor pressure (Gates 1980, Biophysical Ecology, p.311)
        avp = (qv10m * pressure) / (0.622 + (0.379 * qv10m))
        # Saturation vapor pressure (similar to FAO formula)
        svp = 610.7 * np.exp((17.38 * temp_c) / (239 + temp_c))
        return svp - avp

    @staticmethod
    def rhumidity(temp_k: Number, vpd: Number) -> Number:
        r'''
        Calculates relative humidity as the difference between the saturation
        vapor pressure (SVP) and VPD, normalized by the SVP:

        $$
        \text{RH} = \frac{\text{SVP} - \text{VPD}}{\text{SVP}}
        $$

        Parameters
        ----------
        temp_k : int or float or numpy.ndarray
            Temperature in degrees K
        vpd : int or float or numpy.ndarray
            Vapor pressure deficit (Pa)

        Returns
        -------
        int or float or numpy.ndarray
        '''
        # Requiring relative humidity to be calculated from VPD:
        #   VPD = VPsat - VPactual; RH = VPactual / VPsat
        #     --> RH = (VPsat - VPD) / VPsat
        esat = svp(temp_k)
        avp = esat - vpd
        rh = avp / esat
        return np.where(avp < 0, 0, np.where(rh > 1, 1, rh))


def psychrometric_constant(pressure: Number, temp_k: Number) -> Number:
    r'''
    The psychrometric constant, which relates the vapor pressure to the air
    temperature. Calculation derives from the "Handbook of Hydrology" by D.R.
    Maidment (1993), Section 4.2.28.

    $$
    \gamma = \frac{C_p \times P}{\lambda\times 0.622}
    $$

    Where \(C_p\) is the specific heat capacity of air, \(P\) is air pressure,
    and \(\lambda\) is the latent heat of vaporization. The \(C_p\) of air
    varies with its saturation, so the value 1.013e-3 [MJ kg-1 (deg C)-1]
    reflects average atmospheric conditions.

    Parameters
    ----------
    pressure : float or numpy.ndarray
        The air pressure in Pascals
    temp_k : float or numpy.ndarray
        The air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
        The psychrometric constant at this pressure, temperature (Pa K-1)
    '''
    lhv = latent_heat_vaporization(temp_k) # Latent heat of vaporization (J kg-1)
    return (SPECIFIC_HEAT_CAPACITY_AIR * pressure) /\
        (lhv * MOL_WEIGHT_WET_DRY_RATIO_AIR)


def svp(temp_k: Number) -> Number:
    r'''
    The saturation vapor pressure, based on [
    the Food and Agriculture Organization's (FAO) formula, Equation 13
    ](http://www.fao.org/3/X0490E/x0490e07.htm).

    $$
    \mathrm{SVP} = 1\times 10^3\left(
    0.6108\,\mathrm{exp}\left(
      \frac{17.27 (T - 273.15)}{T - 273.15 + 237.3}
      \right)
    \right)
    $$

    This is also referred to as the August-Roche-Magnus equation.

    Parameters
    ----------
    temp_k : float or numpy.ndarray
        The air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
    '''
    temp_c = temp_k - 273.15
    # And convert from kPa to Pa
    return 1e3 * 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def svp_slope(temp_k: Number, s: Number = None) -> Number:
    r'''
    The slope of the saturation vapour pressure curve, which describes the
    relationship between saturation vapor pressure and temperature. This
    approximation is taken from the MOD16 C source code. An alternative is
    based on [the Food and Agriculture Organization's (FAO) formula,
    Equation 13 ](http://www.fao.org/3/X0490E/x0490e07.htm).

    $$
    \Delta = 4098\times [\mathrm{SVP}]\times (T - 273.15 + 237.3)^{-2}
    $$

    Parameters
    ----------
    temp_k : float or numpy.ndarray
        The air temperature in degrees K
    s : float or numpy.ndarray or None
        Saturation vapor pressure, if already known (Optional)

    Returns
    -------
    float or numpy.ndarray
        The slope of the saturation vapor pressure curve in Pascals per
        degree K (Pa degK-1)
    '''
    if s is None:
        s = svp(temp_k)
    return 17.38 * 239.0 * s / (239.0 + temp_k - 273.15)**2
