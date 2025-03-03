import numpy as np
from scipy import interpolate

import schechter
import sample_nf_probability_density as snpd

# SFMS continuation constants
MC_GAP = 0.3
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
        Mass threshold at given redshift
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


def lnposterior(mass, sfr, logpdf_interpolator, mthreshold, ztarget, continuity=False):
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
            return -np.inf    
        if prior != 0:
            mass = mthreshold + MC_GAP 
        likelihood = logpdf_interpolator(mass, sfr_corrected)[0]
        return prior + likelihood
    else:
        prior = lnprior(mass, sfr, mthreshold)
        if prior == -np.inf:
            return -np.inf
        likelihood = logpdf_interpolator(mass, sfr)[0]
        return prior + likelihood


def get_logpdf(ztarget, prob_density, continuity=True):
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
    - The function uses a minimum redshift value of 0.25 for calculations.
    - The log-posterior function is created using a RectBivariateSpline interpolator over the mass and SFR grids.
    - The mass completeness threshold is determined using snpd.cosmos15_mass_completeness.

    Example:
    ----------
    >>> logpdf_func = get_logpdf(0.5, prob_density)
    >>> logpdf_value = logpdf_func((10.5, 1.0))
    """
    z_for_leja = ztarget if ztarget > 0.25 else 0.25
    zidx = np.abs(snpd.zgrid - z_for_leja).argmin()
    density = prob_density[:, :, zidx]
    mthreshold = snpd.cosmos15_mass_completeness(z_for_leja)

    logpdf_interpolator = interpolate.RectBivariateSpline(
        snpd.mgrid, snpd.sfrgrid, np.log(density)
    )

    def logposterior_function(theta):
        mass, sfr = theta
        return lnposterior(mass, sfr, logpdf_interpolator, mthreshold, ztarget,
                           continuity)

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
    """Calculate star-forming main sequence slope from Leja 2020.
    
    Parameters
    ----------
    z : float
        Redshift
    
    Returns
    -------
    float
        SFMS slope at given redshift using quadratic fit:
        0.9387 + 0.004599*z - 0.02751*z^2
    
    References
    ----------
    Leja et al. 2020
    """
    return 0.9387 + 0.004599*z - 0.02751*z**2


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

