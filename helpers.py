import numpy as np
import scipy.stats as stats
 
 
def gen_values_w_halfnorms(host_galaxies, val_key, lowerr_key, uperr_key, n_vals=1000):
    '''
    param frbs: pandas DataFrame, the list of frbs
    :param val_key: str, key name for the value (for example 'Mstar', 'SFR')
    :param lowerr_key: str, key name of lower err
    :param uperr_key: str, key name of up error
    :param n_vals: int, total number of values to generate
    return: list of generated values from a gaussian with uperr std on the right and lowerr std on the left
    '''
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
