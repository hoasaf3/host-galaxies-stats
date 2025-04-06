import numpy as np
import matplotlib.pyplot as plt


def Mstar_of_Mh(Mh):
    ''' Stellar Mass - Halo Mass function from Behroozi et al. (2019) '''
    M1 = 10**12.040
    epsilon = -1.430
    alpha = 1.973
    beta = 0.473
    delta = 0.407
    gamma = 10**(-1.088)
    x = np.log10(Mh/M1)
    logMstar_over_M1 = epsilon - np.log10(10**(-alpha*x)+10**(-beta*x)) + gamma*np.exp(-0.5*(x/delta)**2)
    return M1*10**logMstar_over_M1

Mh_intrp = np.logspace(8,18,300)
Mstar_intrp = Mstar_of_Mh(Mh_intrp)

def Mh_of_Mstar(Mstar):
    ''' inverse of Stellar Mass - Halo Mass function '''
    return 10**np.interp( np.log10(Mstar), np.log10(Mstar_intrp),np.log10(Mh_intrp) )

def hurdle_correction(Mstar):
    ''' Eadie et al. (2022) lognormal hurdle model correction '''
    beta0 = -10.83
    beta1 = 1.59
    return (1e0+np.exp(-(beta0+beta1*np.log10(Mstar))))**(-1)

def MGC_of_Mstar(Mstar,hurdle=False):
    ''' Globular Cluster mass as a function of galaxy stellar mass '''
    Mh = Mh_of_Mstar(Mstar) # calculate galaxy halo mass
    eta = 10**(-4.54) # best-fit value from Harris et al. (2017)
    M_GC = eta*Mh # assume GC mass proportional to halo mass
    if hurdle:
        M_GC = M_GC**hurdle_correction(Mstar) # potential correction at low Mstar
    return M_GC


def plot_MGC_vs_Mstar():
    plt.figure()
    flt1 = Mh_intrp<10**10.5
    flt2 = Mh_intrp>10**15
    plt.loglog(Mh_intrp[(~flt1)*(~flt2)],Mstar_intrp[(~flt1)*(~flt2)],'-k',linewidth=5)
    plt.loglog(Mh_intrp[flt1],Mstar_intrp[flt1],'--k',linewidth=3)
    plt.loglog(Mh_intrp[flt2],Mstar_intrp[flt2],'--k',linewidth=3)
    plt.xlabel(r'$M_{\rm halo} \,\,\,\,\, (M_\odot)$',fontsize=12)
    plt.ylabel(r'$M_\star \,\,\,\,\, (M_\odot)$',fontsize=12)

    plt.figure()
    Mstar = np.logspace(7,12,150)
    plt.loglog(Mstar,MGC_of_Mstar(Mstar),'-k',linewidth=5)
    plt.loglog(Mstar,MGC_of_Mstar(Mstar,hurdle=False),'-',color='indianred',linewidth=3)
    plt.xlabel(r'$M_\star \,\,\,\,\, (M_\odot)$',fontsize=12)
    plt.ylabel(r'$M_{\rm GC} \,\,\,\,\, (M_\odot)$',fontsize=12)

    plt.show()


def get_gc_weighted_logpdf(logpdf):
    ''' Wrapper for a simple logpdf function that returns weighted likelihood function
    :simple_logpdf: the logpdf fucntion (density of mass, sfr)
    :return a function f(m, sfr) -> ln(p(m,sfr) * m_gc(m))
    '''
    def gc_weighted_logpdf(theta):
        m, _ = theta
        if np.isinf(logpdf(theta)):
            return -np.inf
        return np.log(np.exp(logpdf(theta)) * MGC_of_Mstar(10 ** m, hurdle=False))
    return gc_weighted_logpdf
