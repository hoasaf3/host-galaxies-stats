import numpy as np
import scipy.stats as stats
 
 
def gen_values_w_halfnorms(host_galaxies, val_key, lowerr_key, uperr_key, n_vals=1000):
    """
    Generate values from a Gaussian distribution with different standard deviations on the left and right sides.

    Parameters
    ----------
    host_galaxies : pandas.DataFrame
        The DataFrame containing host galaxy data.
    val_key : str
        The key name for the central value (e.g., 'Mstar', 'SFR').
    lowerr_key : str
        The key name for the lower error.
    uperr_key : str
        The key name for the upper error.
    n_vals : int, optional
        The total number of values to generate. Default is 1000.

    Returns
    -------
    numpy.ndarray
        A 2D array where each row contains the generated values for a host galaxy.
    """
    sfr_values = np.ndarray((len(host_galaxies), n_vals))

    for i in host_galaxies.index:
        left_scale, right_scale = host_galaxies[lowerr_key][i], host_galaxies[uperr_key][i]  
        mu = host_galaxies[val_key][i]

        left_norm = stats.halfnorm(loc=mu, scale=left_scale)
        right_norm = stats.halfnorm(loc=mu, scale=right_scale)

        left_samples = mu + mu - left_norm.rvs(n_vals//2)
        right_samples = right_norm.rvs(n_vals//2)

        sfr_values[i] = np.concatenate((left_samples, right_samples))
        
    return sfr_values
