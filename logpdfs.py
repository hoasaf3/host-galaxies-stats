import numpy as np
from scipy import interpolate

import schechter
import sample_nf_probability_density as snpd

# SFMS continuation constants
MC_GAP = 0.3
logphi = schechter.log_schechter


def lnprior(mass, sfr, mthreshold):
    """
    Apply prior constraints on mass and sfr.
    Returns -inf if the parameters are out of bounds, otherwise 0.
    :param mass: float, the mass parameter.
    :param sfr: float, the star formation rate parameter.
    :param mthreshold: float, the mass threshold for the given redshift.
    """
    if not mthreshold < mass < snpd.mmax or not snpd.sfrmin < sfr < snpd.sfrmax:
        return -np.inf
    return 0


def lnprior_continuity(mass, sfr, mthreshold, ztarget):
    """
    Apply continuity model prior on mass and sfr, accounting for redshift-specific slope.
    :param mass: float, the mass parameter.
    :param sfr: float, the star formation rate parameter.
    :param mthreshold: float, the mass threshold for the given redshift.
    :param ztarget: float, the target redshift for determining the slope.
    """
    if not snpd.mmin < mass < snpd.mmax or not snpd.sfrmin < sfr < snpd.sfrmax:
        return -np.inf, sfr
    if mass < mthreshold + MC_GAP:
        sfr_corrected = sfr - slope(ztarget) * (mass - mthreshold - MC_GAP)
        prior_adjustment = logphi(mass, ztarget) - logphi(mthreshold + MC_GAP, ztarget)
        return prior_adjustment, sfr_corrected
    return 0, sfr


def lnposterior(mass, sfr, logpdf_interpolator, mthreshold, ztarget,
                continuity=False):
    """
    Compute the total log-posterior (lnprior + lnlikelihood).
    Supports optional continuity model prior.
    :param mass: float, the mass parameter.
    :param sfr: float, the star formation rate parameter.
    :param logpdf_interpolator: callable, interpolates log-likelihood.
    :param mthreshold: float, the mass threshold for the given redshift.
    :param ztarget: float, the target redshift for continuity model.
    :param continuity: bool, whether to use continuity model prior.
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


def get_logpdf(ztarget, prob_density, continuity=False):
    """
    Return a log-posterior function for a given redshift and probability density.
    Supports optional continuity model.
    :param ztarget: float, the target redshift.
    :param prob_density: 3D array, the probability density over mass, sfr, and
                         redshift. Obtained from snpd.sample_density
    :param continuity: bool, whether to include continuity model prior.
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


def get_weighted_logpdf(simple_logpdf, a, b):
    ''' Wrapper for a simple logpdf function that returns weighted likelihood function
    :simple_logpdf: the logpdf fucntion (density of mass, sfr)
    :a: coefficient for mass
    :b: coefficient for sfr
    :return a function f([m, sfr]) -> ln(p * (a*m+b*sfr) )
    '''
    def weighted_logpdf(theta):
        m, sfr = theta
        if np.isinf(simple_logpdf(theta)):
            return -np.inf
        return np.log(np.exp(simple_logpdf(theta)) * (a* 10**m + b * 10**sfr))
    return weighted_logpdf


def slope(z):
    ''' Calc the slope star forming main sequence from Leja 2020'''
    return 0.9387 + 0.004599*z - 0.02751*z**2


def create_z_to_logpdf(a, b, host_galaxies, prob_density):
    ''' create map of redshift to weighted logpdf function
    :a: coefficient for mass
    :b: coefficient for sfr
    '''
    z_to_logpdf = {z: get_weighted_logpdf(get_logpdf(z, prob_density), a, b)
                      for z in host_galaxies['z']} 
    return z_to_logpdf

