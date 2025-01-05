import numpy as np
from scipy import interpolate

import schechter
import sample_nf_probability_density as snpd

def lnprior(logpdf, mthreshold):
    '''Wrapping a logpdf with lnprior limits
    Limits sfr to (-3, 1) and mass to (mthreshold, 12) otherwise -inf.
    :param logpdf: Original logpdf function of some redshift that takes mass,sfr as arguments
    :param mthreshold: float, the mass threshold for the redshift of logpdf
    Returns a logpdf function that takes mass,sfr as arguments'''
    def logpdf_with_prior(mass, sfr):
        if np.isnan(logpdf(mass, sfr)):
            return -np.inf
        if not mthreshold < mass < snpd.mmax or not snpd.sfrmin < sfr < snpd.sfrmax:
            return -np.inf  # out of bounds
        return logpdf(mass, sfr)[0]  # RectBivariateSpline returns [[result]]
    return logpdf_with_prior


def get_logpdf(ztarget, prob_density):
    '''Get 2D interpolation of ln(density) at target redshift'''
    zidx = np.abs(snpd.zgrid-ztarget).argmin()
    density = prob_density[:,:,zidx]
    mthreshold = snpd.cosmos15_mass_completeness(ztarget)
    return lnprior(interpolate.RectBivariateSpline(snpd.mgrid, snpd.sfrgrid, np.log(density)), mthreshold)


def get_weighted_logpdf(simple_logpdf, a, b):
    ''' Wrapper for a simple logpdf function that returns "smart" likelihood function
    :simple_logpdf: the logpdf fucntion (density of mass, sfr)
    :a: coefficient for mass
    :b: coefficient for sfr
    :return a function f(m, sfr) -> ln(p * (a*m+b*sfr) )
    '''
    def smart_logpdf(m, sfr):
        if np.isinf(simple_logpdf(m, sfr)):
            return -np.inf
        return np.log(np.exp(simple_logpdf(m, sfr)) * (a* 10**m + b * 10**sfr))
    return smart_logpdf


# SFMS continuation constants
MC_GAP = 0.3
logphi = schechter.log_schechter

def slope(z):
    ''' Calc the slope star forming main sequence from Leja 2020'''
    return 0.9387 + 0.004599*z - 0.02751*z**2

def lnprior_extension(logpdf, mthreshold, ztarget):
    ''' Same as lnprior but with continuity model
    :param ztarget: the redshift, to pick the right slope for continuity model
    '''
    def logpdf_with_prior(mass, sfr):
        if mass < snpd.mmin or sfr < snpd.sfrmin:  # max is already enforced by lnprior
            return -np.inf
        if mass < mthreshold + MC_GAP:
            return logpdf(mthreshold+MC_GAP, sfr-slope(ztarget) * (mass-mthreshold-MC_GAP))\
                    + logphi(mass, ztarget) - logphi(mthreshold+MC_GAP, ztarget)
        return logpdf(mass, sfr)
    return logpdf_with_prior


def get_logpdf_w_extension(ztarget, prob_density, optical_bias=True, metallicity_fix=False):
    '''
    Same as get_logpdf but with extension
    optical_bias: if True, also includes optical selection bias
    '''
    z_for_leja = ztarget if ztarget > 0.25 else 0.25
    mthreshold = snpd.cosmos15_mass_completeness(z_for_leja)
    lnprior_pdf = get_logpdf(z_for_leja, prob_density)
    lnprior_ext = lnprior_extension(lnprior_pdf, mthreshold, z_for_leja)
    # if optical_bias:
    #     return lnprior_optical_bias(lnprior_ext, z_to_selection_factor[ztarget], metallicity_fix)
    return lnprior_ext


def create_z_to_logpdf(a, b, host_galaxies, prob_density):
    ''' create map of redshift to weighted logpdf function
    :a: coefficient for mass
    :b: coefficient for sfr
    '''
    #z_to_logpdf = {z: get_weighted_logpdf(get_logpdf(z), a, b) for z in filtered_frb['zsp']}           
    z_to_logpdf = {z: get_weighted_logpdf(get_logpdf_w_extension(z, prob_density), a, b) for z in host_galaxies['z']}           
    return z_to_logpdf

