''' emcee utils '''
import emcee
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

from sample_nf_probability_density import sfrmin, sfrmax, mmin, mmax


NDIM = 2
NWALKERS = 32
NSAMPLES_PER_Z = 1000


def logpdf_wrapper(logpdf):
    '''A logpdf wrapper to get arguments as array instead of 2 parameters - mass, sfr (used for emcee)
        :logpdf: function that receives mass, sfr as 2 arguments: logpdf(mass, sfr)
        :return: function that receives an array: log_wrapper(logpdf)([mass, sfr])
        Usage:
        wrapped_logpdf = wrap_logpdf(logpdf)
        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, wrapped_logpdf)
    '''
    def logpdf_with_array_args(x):
        return logpdf(x[0], x[1])
    return logpdf_with_array_args


def get_samples(logpdf, p0=None, steps=5000, logpdf_args_as_array=True):
    '''Get samples from a sampler with a given logpdf
    :p0: Initial state for walkers
    :logpdf_args_as_array: bool. If True, logpdf can be called with logpdf([mass, sfr]). False means 
    it should be called with logpdf(mass, sfr)
    '''
    if not p0:
        p0 = np.random.rand(NWALKERS, NDIM)  * np.array(((10-9), (0-(-1)))) + np.array((10, -0.5))
    if not logpdf_args_as_array:
        logpdf = logpdf_wrapper(logpdf)
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
        samples_per_z = get_samples(logpdf, steps=nsamples_per_z, logpdf_args_as_array=False)

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
    smart_likelihood = 0
    for mass, sfr, z in values:
        logpdf = z_to_logpdf[z]
        if type(logpdf(mass, sfr)) == float:
            print(logpdf(mass,sfr), mass, sfr, z)
            print('probably received mass/sfr out of bounds ({} - {}/{} - {})'.format(mmin, mmax, sfrmin, sfrmax))
        smart_likelihood += logpdf(mass, sfr)[0]
    return smart_likelihood


def calc_pnom_of_samples(samples, z_to_logpdf, host_galaxies, weight_eq_text, plot=True):
    '''
    Calculate p-nominal value of given samples,
    :samples: array of all samples with shape (number of host galaxies, NSAMPLES_PER_Z * NWALKERS, 3)
              each sample has 3 dimensiosn - mass, sfr, z
    :z_to_logpdf: dict redshift -> logpdf function
    :weight_eq_text: Used for title of graph. For example: "ln(p * (0.2M + 0.8SFFR) )"
    :plot: if True, plots the cummulative probability graph
    
    :return: the probability of a a random set having lower weighted likelihood than the weighted likelihood of the given samples.
    '''
    hosts_values = np.vstack([np.log10(host_galaxies['Mstar']),
                              np.log10(host_galaxies['SFR']),
                              host_galaxies['z']])

    frb_weighted_likelihood = calc_weighted_likelihood(z_to_logpdf, hosts_values.T)
    print('Likelihood of FRBs: ', frb_weighted_likelihood)
    choices = [samples[:,i,:] for i in range(samples.shape[1])] # list of (ngalaxies,3) arrays
    
    # calc smart likelihood for each tuple of 18 samples
    print("Calculating smart likelihood of samples...")
    smart_likelihood = np.array([calc_weighted_likelihood(z_to_logpdf, c) for c in choices])

    ecdf_smart_lklhd = ECDF(smart_likelihood)
    p_value = ecdf_smart_lklhd(frb_weighted_likelihood)
    print("P<frb_likelihood: %.4f" % (p_value,))
    
    if plot:
        xmin = np.min(smart_likelihood)
        xmax = np.max(smart_likelihood)
        x = np.linspace(xmin, xmax, 1000)
        plt.plot(x, ecdf_smart_lklhd(x), color='purple')
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
        #plt.savefig('Figures/frb_samples_cdf.pdf', transparent=True, bbox_inches='tight')  
    return p_value