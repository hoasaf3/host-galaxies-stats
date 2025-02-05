''' emcee utils '''
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

NDIM = 2
NWALKERS = 32
NSAMPLES_PER_Z = 1000


def get_samples(logpdf, p0=None, steps=5000,):
    '''Get samples from a sampler with a given logpdf
    :p0: Initial state for walkers
    '''
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
    '''
    :param z_to_logpdf: dict of z->logdpf function, to generate samples
    :return: array of all samples - shape is (nfiltered, NSAMPLES_PER_Z * NWALKERS, 3)
    each sample has 3 dimensiosn - mass, sfr, z
    '''

    # create empty array to fill samples for each redshift
    all_samples = np.empty((len(host_galaxies), nsamples_per_z * NWALKERS, 3)) # each sample has 3 dimensions - mass,sfr,z
    
    #for i,z in enumerate(host_galaxies['z']):
    for i, z in tqdm(enumerate(host_galaxies['z']), 
                    total=len(host_galaxies['z']),
                    desc="Generating samples for each redshift",
                    unit="hosts"):

        logpdf = z_to_logpdf[z]
        samples_per_z = get_samples(logpdf, steps=nsamples_per_z)
        
        # add a column with z
        samples_per_z = np.append(samples_per_z,
                                  np.ones((samples_per_z.shape[0], 1)) * z, 1)
        all_samples[i] = samples_per_z
    return all_samples


def calc_weighted_likelihood(z_to_logpdf, values):
    '''
    :z_to_logpdf: dict, redshift -> logpdf function
    :values: list of (log10(mass), log10(sfr), redshift)
    :return: the likelihood of values - sum(log(p(m_i,sfr_i,z_i))) for m_i,sfr_i,z_i in values
    '''
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
    '''
    Calculate p-nominal value of given samples,
    :samples: array of all samples with shape (number of host galaxies, NSAMPLES_PER_Z * NWALKERS, 3)
              each sample has 3 dimensiosn - mass, sfr, z
    :z_to_logpdf: dict redshift -> logpdf function
    
    :return: the probability of a a random set having lower weighted likelihood than the weighted likelihood of the given samples.
    '''
    hosts_values = np.vstack([np.log10(host_galaxies['Mstar']),
                              np.log10(host_galaxies['SFR']),
                              host_galaxies['z']])

    frb_weighted_likelihood = calc_weighted_likelihood(z_to_logpdf, hosts_values.T)
    print('Likelihood of FRBs: ', frb_weighted_likelihood)
    choices = [samples[:,i,:] for i in range(samples.shape[1])] # list of (ngalaxies,3) arrays
    
    # calc weighted likelihood for each tuple len(host_galaxies) samples
    weighted_likelihood = np.array([calc_weighted_likelihood(z_to_logpdf, c) for c in choices])

    ecdf_weighted_lklhd = ECDF(weighted_likelihood)
    p_value = ecdf_weighted_lklhd(frb_weighted_likelihood)

    return p_value


def plot_from_samples(samples):
    """Plot density distribution of MCMC samples using Gaussian KDE.
    
    Creates a 2D density plot of mass-SFR samples using Gaussian kernel 
    density estimation. The plot shows the probability density of the samples
    with a color gradient.

    Parameters
    ----------
    samples : numpy.ndarray
        Array of shape (n_samples, 2) containing the MCMC samples.
        First column should be masses, second column SFRs.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object containing the plot for further customization.

    Notes
    -----
    - Uses gaussian_kde from scipy.stats for density estimation
    - Color scheme uses gist_earth_r colormap
    - Plot is oriented with mass on x-axis and SFR on y-axis
    - Density is estimated on a 100x100 grid
    
    Example
    -------
    >>> samples = get_mcmc_samples()  # Shape (1000, 2)
    >>> ax = plot_from_samples(samples)
    >>> ax.set_xlabel('log Mass')
    >>> plt.show()
    """
    m1 = samples[:,0]  # masses
    m2 = samples[:,1]  # SFRs

    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    _, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax], animated=True)
    return ax


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

    for _ in tqdm(range(n_vals),
                  desc="Calculating likelihoods based on halfnorms values"):
        
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