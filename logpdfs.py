import numpy as np
from scipy import interpolate

import schechter
import sample_nf_probability_density as snpd

# SFMS continuation constants
MC_GAP = 0.3
Z_CUTOFF = 0.25
logphi = schechter.log_schechter


def lnprior(mass, sfr, mthreshold):
    """Apply prior constraints on mass and SFR.

    Parameters
    ----------
    mass : float
        Galaxy stellar mass in log10(M_solar)
    sfr : float
        Star formation rate in log10(M_solar/yr)
    mthreshold : float
        Mass threshold at given redshift

    Returns
    -------
    float
        0 if parameters within bounds, -inf otherwise
    """
    if not mthreshold < mass < snpd.mmax or not snpd.sfrmin < sfr < snpd.sfrmax:
        return -np.inf
    return 0


def lnprior_continuity(mass, sfr, mthreshold, ztarget):
    """Apply continuity model prior on mass and SFR.
    
    Accounts for redshift-specific slope and applies Schechter function
    correction below mass threshold.

    Parameters
    ----------
    mass : float
        Galaxy stellar mass in log10(M_solar)
    sfr : float
        Star formation rate in log10(M_solar/yr)
    mthreshold : float
        Mass completeness threshold at given redshift
    ztarget : float
        Target redshift for determining slope

    Returns
    -------
    tuple
        (prior_value, corrected_sfr)
        prior_value: float, log prior probability
        corrected_sfr: float, SFR after continuity correction
    """
    if not snpd.mmin < mass < snpd.mmax or not snpd.sfrmin < sfr < snpd.sfrmax:
        return -np.inf, sfr
    if mass < mthreshold + MC_GAP:
        sfr_corrected = sfr - slope(ztarget) * (mass - mthreshold - MC_GAP)
        prior_adjustment = logphi(mass, ztarget) - logphi(mthreshold + MC_GAP, ztarget)
        return prior_adjustment, sfr_corrected
    return 0, sfr


def lnposterior(mass, sfr, logpdf_interpolator, mthreshold, ztarget, continuity=False, redshift_evolution=True):
    """Compute total log-posterior (lnprior + lnlikelihood).

    Parameters
    ----------
    mass : float
        Galaxy stellar mass in log10(M_solar)
    sfr : float
        Star formation rate in log10(M_solar/yr)
    logpdf_interpolator : callable
        Interpolation function for log probability density
    mthreshold : float
        Mass threshold at given redshift
    ztarget : float
        Target redshift
    continuity : bool, optional
        Whether to apply continuity model, by default False

    Returns
    -------
    float
        Log posterior probability
    """
    if continuity:
        prior, sfr_corrected = lnprior_continuity(mass, sfr, mthreshold, ztarget)
        if prior == -np.inf:
            return -np.inf  # mass or SFR out of bounds    
        if prior != 0:
            # mass < mthreshold + MC_GAP in lnprior_continuity
            mass = mthreshold + MC_GAP 
        
        if redshift_evolution and ztarget < Z_CUTOFF:
            sfr_corrected += sfms_ridge(mass, ztarget) - sfms_ridge(mass, Z_CUTOFF)
            prior += logphi(mass, Z_CUTOFF) - logphi(mass, ztarget)
        
        likelihood = logpdf_interpolator(mass, sfr_corrected)[0]
        return prior + likelihood
    
    else:
        prior = lnprior(mass, sfr, mthreshold)
        if prior == -np.inf:
            return -np.inf
        likelihood = logpdf_interpolator(mass, sfr)[0]
        return prior + likelihood


def get_logpdf(ztarget, prob_density, continuity=True, redshift_evolution=True):
    """
    Return a log-posterior function for a given redshift and probability density.
    Supports optional continuity model.
    Parameters:
    ----------
    ztarget (float): The target redshift.
    prob_density (numpy.ndarray): 3D array representing the probability density over mass, SFR, and redshift.
                                    Obtained from snpd.sample_density.
    continuity (bool, optional): Whether to include continuity model prior. Default is True.

    Returns:
    ----------
    function: A log-posterior function that takes a tuple (mass, sfr) as input and returns the log-posterior value.

    Notes:
    ----------
    - For z<Z_CUTOFF, the function assumes z=Z_CUTOFF for calculations.
    - The log-posterior function is created using a RectBivariateSpline interpolator over the mass and SFR grids.
    - The mass completeness threshold is determined using snpd.cosmos15_mass_completeness.

    Example:
    ----------
    >>> logpdf_func = get_logpdf(0.5, prob_density)
    >>> logpdf_value = logpdf_func((10.5, 1.0))
    """
    z_for_leja = ztarget if ztarget > Z_CUTOFF else Z_CUTOFF
    zidx = np.abs(snpd.zgrid - z_for_leja).argmin()
    density = prob_density[:, :, zidx]
    mthreshold = snpd.cosmos15_mass_completeness(z_for_leja)

    logpdf_interpolator = interpolate.RectBivariateSpline(
        snpd.mgrid, snpd.sfrgrid, np.log(density)
    )

    def logposterior_function(theta):
        mass, sfr = theta
        return lnposterior(mass, sfr, logpdf_interpolator, mthreshold, ztarget,
                           continuity, redshift_evolution)

    return logposterior_function


def get_weighted_logpdf(logpdf, a, b):
    """Create weighted logpdf function combining mass and SFR terms.
    
    Parameters
    ----------
    logpdf : callable
        Base log probability density function
    a : float
        Mass weighting coefficient
    b : float
        SFR weighting coefficient
    
    Returns
    -------
    callable
        Weighted logpdf function that returns ln(p * (a*m + b*sfr))
        where p is the original probability density
    
    Notes
    -----
    Input masses and SFRs should be in log10 scale
    Returns -inf for invalid parameter combinations
    """
    def weighted_logpdf(theta):
        m, sfr = theta
        if np.isinf(logpdf(theta)):
            return -np.inf
        return np.log(np.exp(logpdf(theta)) * (a* 10**m + b * 10**sfr))
    return weighted_logpdf


def slope(z):
    """Calculate star-forming main sequence slope for low masses (from Leja 2022).
    Numbers are taken from Table 1 of Leja et al. 2022 for ridge mode, for parameter b.
    The slope is calculated in Eq (10) of Leja et al. 2022
    
    Parameters
    ----------
    z : float
        Redshift
    
    Returns
    -------
    float
        SFMS slope at given redshift using quadratic fit
    
    References
    ----------
    Leja et al. 2020
    """
    return 0.9605 + 0.04990*z - 0.05984*z**2


def calc_params(z):
    """Calculate parameters for the SFMS ridge at a given redshift,
    using Leja et al. 2022 Table 1 and Eq (10).
    
    Parameters
    ----------
    z : float
        Redshift
    
    Returns
    -------
    tuple
        (a, b, c, Mt) parameters for the SFMS ridge model
    """
    # Coefficients for the quadratic fit
    coeffs = {
        'a': (0.03746, 0.3448, -0.1156),
        'b': (0.9605, 0.04990, -0.05984),
        'c': (0.2516, 1.118, -0.2006),
        'Mt': (10.22, 0.3826, -0.04491)
    }
    
    return tuple(co[0] + co[1]*z + co[2]*z**2 for co in coeffs.values())


def sfms_ridge(m, z):
    """
    Calculate the star-forming main sequence ridge at a given mass and redshift.
    Using Leja et al. 2022 Table 1 for the parameters and"""
    
    a, b, c, Mt = calc_params(z)

    if m > Mt:
        return a * (m - Mt) + c
    else:
        return b * (m - Mt) + c


def create_z_to_logpdf(host_galaxies, prob_density, a, b):
    """Create mapping from redshift to weighted log PDF functions.
    
    Parameters
    ----------
    host_galaxies : pandas.DataFrame
        Host galaxy data containing 'z' column
    prob_density : callable
        Base probability density function
    a : float
        Mass weighting coefficient
    b : float
        SFR weighting coefficient
    
    Returns
    -------
    dict
        Maps redshift to corresponding weighted log PDF functions
    """
    z_to_logpdf = {z: 
                   get_weighted_logpdf(
                       get_logpdf(z, prob_density), a, b)
                   for z in host_galaxies['z']} 
    return z_to_logpdf

