''' emcee utils '''
import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF

from sample_nf_probability_density import sfrmin, sfrmax, mmin, mmax


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


def get_weighted_samples(z_to_logpdf, host_galaxies, nsamples_per_z=1000):
    '''
    :param z_to_logpdf: dict of z->logdpf function, to generate samples
    :return: array of all samples - shape is (nfiltered, NSAMPLES_PER_Z * NWALKERS, 3)
    each sample has 3 dimensiosn - mass, sfr, z
    '''

    # create empty array to fill samples for each redshift
    all_samples = np.empty((len(host_galaxies), nsamples_per_z * NWALKERS, 3)) # each sample has 3 dimensions - mass,sfr,z
    
    for i,z in enumerate(host_galaxies['z']):
        logpdf = z_to_logpdf[z]
        samples_per_z = get_samples(logpdf, steps=nsamples_per_z)

        # add a column with z
        samples_per_z = np.append(samples_per_z, np.ones((samples_per_z.shape[0],1)) * z, 1)
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


def calc_pnom_of_samples(samples, z_to_logpdf, host_galaxies, plot=True):
    '''
    Calculate p-nominal value of given samples,
    :samples: array of all samples with shape (number of host galaxies, NSAMPLES_PER_Z * NWALKERS, 3)
              each sample has 3 dimensiosn - mass, sfr, z
    :z_to_logpdf: dict redshift -> logpdf function
    :plot: if True, plots the cummulative probability graph
    
    :return: the probability of a a random set having lower weighted likelihood than the weighted likelihood of the given samples.
    '''
    hosts_values = np.vstack([np.log10(host_galaxies['Mstar']),
                              np.log10(host_galaxies['SFR']),
                              host_galaxies['z']])

    frb_weighted_likelihood = calc_weighted_likelihood(z_to_logpdf, hosts_values.T)
    print('Likelihood of FRBs: ', frb_weighted_likelihood)
    choices = [samples[:,i,:] for i in range(samples.shape[1])] # list of (ngalaxies,3) arrays
    
    # calc weighted likelihood for each tuple len(host_galaxies) samples
    print("Calculating weighted likelihood of samples...")
    weighted_likelihood = np.array([calc_weighted_likelihood(z_to_logpdf, c) for c in choices])

    ecdf_weighted_lklhd = ECDF(weighted_likelihood)
    p_value = ecdf_weighted_lklhd(frb_weighted_likelihood)
    print("P<frb_likelihood: %.4f" % (p_value,))
    
    if plot:
        xmin = np.min(weighted_likelihood)
        xmax = np.max(weighted_likelihood)
        x = np.linspace(xmin, xmax, 1000)
        plt.plot(x, ecdf_weighted_lklhd(x), color='purple')
        plt.yscale('log', nonpositive='clip')
        ax = plt.gca()
        plt.axvline(frb_weighted_likelihood, color='red', linestyle='--')

        plt.title("%s of %s galaxies based on 10k samples" % (weight_eq_text, len(host_galaxies)))
        plt.xlabel('Likelihood [unitless]', fontsize=16)
        plt.ylabel('Probability [unitless]', fontsize=16)
        plt.legend(['Cummulative distribution of MCMC samples', 'FRB likelihood'],
                  loc='upper left')
        plt.text(0.1, 0.5, "P<frb_likelihood: %.4f" % (p_value,),
                verticalalignment='bottom', horizontalalignment='left', transform = ax.transAxes)
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