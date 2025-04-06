import emcee
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF

import logpdfs
from sample_nf_probability_density import sfrmin, sfrmax, mmin, mmax
from helpers import gen_values_w_halfnorms

# Global variables for MCMC sampling
NDIM = 2  # Dimension of parameter space (mass, SFR)
NWALKERS = 32  # Number of MCMC walkers
NSAMPLES_PER_Z = 1000  # Samples per redshift
TQDM_DISABLE = False  # Flaf for disabling tqdm


def get_samples(logpdf, p0=None, steps=5000):
    """Generate MCMC samples using emcee sampler.

    Parameters
    ----------
    logpdf : callable
        Log probability density function to sample from
    p0 : ndarray, optional
        Initial positions of walkers, shape (NWALKERS, NDIM)
    steps : int, optional
        Number of MCMC steps after burn-in, by default 5000

    Returns
    -------
    ndarray
        Flattened chain of MCMC samples, shape (steps * NWALKERS, NDIM)

    Notes
    -----
    - Uses 100 steps burn-in period
    - Default initial positions span mass=[9,10], SFR=[-1,0]
    """
    if not p0:
        p0 = np.random.rand(NWALKERS, NDIM)  * np.array(((10-9), (0-(-1)))) + np.array((10, -0.5))
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, logpdf)
    state = sampler.run_mcmc(p0, 100) # burn in
    sampler.reset()

    # run sampler 5k times (x32 walkers)
    sampler.run_mcmc(state, steps)
    samples = sampler.get_chain(flat=True)
    return samples


def get_weighted_samples(host_galaxies, z_to_logpdf, nsamples_per_z=1000):
    """Generate MCMC samples for each host galaxy redshift.

    Parameters
    ----------
    host_galaxies : pandas.DataFrame
        DataFrame containing host galaxy data with 'z' column
    z_to_logpdf : dict
        Maps redshift to corresponding log PDF functions
    nsamples_per_z : int, optional
        Number of samples per redshift, by default 1000

    Returns
    -------
    ndarray
        Array of samples, shape (n_galaxies, nsamples_per_z * NWALKERS, 3)
        Each sample contains (mass, sfr, redshift)

    Notes
    -----
    Uses tqdm progress bar unless TQDM_DISABLE is True
    """
    # create empty array to fill samples for each redshift
    all_samples = np.empty((len(host_galaxies), nsamples_per_z * NWALKERS, 3)) # each sample has 3 dimensions - mass,sfr,z
    
    #for i,z in enumerate(host_galaxies['z']):
    for i, z in tqdm(enumerate(host_galaxies['z']), 
                    total=len(host_galaxies['z']),
                    desc="Generating samples for each redshift",
                    unit="hosts",
                    disable=TQDM_DISABLE):

        logpdf = z_to_logpdf[z]
        samples_per_z = get_samples(logpdf, steps=nsamples_per_z)
        
        # add a column with z
        samples_per_z = np.append(samples_per_z,
                                  np.ones((samples_per_z.shape[0], 1)) * z, 1)
        all_samples[i] = samples_per_z
    return all_samples


def calc_weighted_likelihood(z_to_logpdf, values):
    """Calculate total log likelihood across multiple redshifts.

    Parameters
    ----------
    z_to_logpdf : dict
        Maps redshift to corresponding logpdf function
    values : ndarray
        Array of (mass, sfr, redshift) tuples, where:
        - mass: log10(stellar mass)
        - sfr: log10(star formation rate)
        - redshift: host galaxy redshift

    Returns
    -------
    float
        Sum of log probabilities across all values

    Notes
    -----
    Checks for out-of-bounds mass/SFR values and prints warning
    """
    weighted_likelihood = 0
    for mass, sfr, z in values:
        logpdf = z_to_logpdf[z]
        p = logpdf([mass, sfr])
        if type(p) == float:  # should be a list
            print(p, mass, sfr, z)
            print('probably received mass/sfr out of bounds ({} - {}/{} - {})'.format(mmin, mmax, sfrmin, sfrmax))
        weighted_likelihood += p[0]
    return weighted_likelihood


def calc_pnom_of_samples(samples, z_to_logpdf, host_galaxies):
    """Calculate p-nominal value for MCMC samples.

    Parameters
    ----------
    samples : ndarray
        Array of shape (n_galaxies, n_samples, 3) containing:
        - mass: log10(stellar mass)
        - sfr: log10(star formation rate)
        - redshift: host galaxy redshift
    z_to_logpdf : dict
        Maps redshift to corresponding logpdf function
    host_galaxies : pandas.DataFrame
        Host galaxy data containing Mstar and SFR columns

    Returns
    -------
    float
        Probability of random sample set having lower weighted 
        likelihood than observed sample set
    """
    hosts_values = np.vstack([np.log10(host_galaxies['Mstar']),
                              np.log10(host_galaxies['SFR']),
                              host_galaxies['z']])

    hosts_weighted_likelihood = calc_weighted_likelihood(z_to_logpdf, hosts_values.T)
    print('Likelihood of host galaxies: ', hosts_weighted_likelihood)
    choices = [samples[:,i,:] for i in range(samples.shape[1])] # list of (ngalaxies,3) arrays
    
    # calc weighted likelihood for each tuple len(host_galaxies) samples
    weighted_likelihood = np.array([calc_weighted_likelihood(z_to_logpdf, c) for c in choices])

    ecdf_weighted_lklhd = ECDF(weighted_likelihood)
    p_value = ecdf_weighted_lklhd(hosts_weighted_likelihood)

    return p_value


def plot_from_samples(samples):
    """Plot density distribution of MCMC samples using Gaussian KDE.

    Parameters
    ----------
    samples : ndarray
        Array of shape (n_samples, 2) containing:
        - mass: log10(stellar mass)
        - sfr: log10(star formation rate)

    Returns
    -------
    tuple
        (ax, Z, extent) where:
        - ax: matplotlib Axes object
        - Z: 2D array of KDE values
        - extent: Plot limits [xmin, xmax, ymin, ymax]
    """
    m1 = samples[:,0]  # masses
    m2 = samples[:,1]  # SFRs

    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    extent = [xmin, xmax, ymin, ymax]

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])  # 2x10000
    values = np.vstack([m1, m2]) # 2xN_samples
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)  # 100x100

    _, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=extent, animated=True)
    return ax, Z, extent


def calc_weighted_pnom(host_galaxies, prob_density, a, b):
    z_to_logpdf = logpdfs.create_z_to_logpdf(host_galaxies, prob_density, a, b)
    print('Generating samples for a =', a, 'b =', b, '...')
    samples = get_weighted_samples(host_galaxies, z_to_logpdf)
    p_value = calc_pnom_of_samples(samples, z_to_logpdf, host_galaxies)
    return p_value


def calc_p90(host_galaxies, z_to_logpdf, nominal_likelihood=None, n_vals=10000):
    '''
    Calculate the p<90% confidence and p nominal
    :param n_vals: number of values to generate for mass, sfr per host

    :nominal_likelihood: float, the nominal likelihood of transients. For example:
        frbs_likelihood = calc_weighted_likelihood(z_to_logpdf, frb_values.T)
        If not provided, p_nom will return as
    :return: Tuple of (p90, p_nom, cdf of likelihoods, frbs_likelihood_cdf)
    '''
    samples = get_weighted_samples(host_galaxies, z_to_logpdf)
    
    # list of (n_hosts, 3) arrays - list of (mass, sfr, z) for every host
    emcee_values = [samples[:,i,:] for i in range(samples.shape[1])]
    
    weighted_likelihoods = np.array([calc_weighted_likelihood(z_to_logpdf, v)
                                     for v in emcee_values])
    weighted_likelihood_cdf = ECDF(weighted_likelihoods)

    mass_values = gen_values_w_halfnorms(host_galaxies, val_key='Mstar',
                                         lowerr_key='Mstar_lowerr',
                                         uperr_key='Mstar_uperr',
                                         n_vals=n_vals)
    
    sfr_values = gen_values_w_halfnorms(host_galaxies, val_key='SFR',
                                        lowerr_key='SFR_lowerr',
                                        uperr_key='SFR_uperr',
                                        n_vals=n_vals)
    
    hosts_likelihoods = []

    for _ in range(n_vals):
        
        random_values = []
        for i in host_galaxies.index:
            mass = np.log10(random.choice(mass_values[i]))
            with np.errstate(invalid='ignore'):
                sfr = np.log10(random.choice(sfr_values[i]))
                while np.isnan(sfr) or sfr < sfrmin or sfr > sfrmax:
                    # retry if sfr_values[i]<0 or sfr not within plot range
                    sfr = np.log10(random.choice(sfr_values[i]))
                
            z = host_galaxies['z'][i]
            random_values.append([mass, sfr, z])
        
        hosts_likelihoods.append(calc_weighted_likelihood(z_to_logpdf, random_values))
    hosts_likelihood_cdf = ECDF(hosts_likelihoods)
    ninetieth_conf = np.percentile(hosts_likelihood_cdf.x, [90])[0]
    
    p_90 = weighted_likelihood_cdf(ninetieth_conf)
    p_nom = None
    if nominal_likelihood is not None:
        p_nom = weighted_likelihood_cdf(nominal_likelihood)
    
    return p_90, p_nom, weighted_likelihood_cdf, hosts_likelihood_cdf