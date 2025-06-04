# Leja et al. 2020 Appendix B

# Usage example:
#
#   z0 = 0.3
#   logm = np.linspace(7, 12, 100) # log10(M) grid
#   plt.figure()
#   plt.loglog(10**logm,np.exp(log_schechter(logm,z0)),'-k',linewidth=5)
#   plt.xlabel(r'$M \,\,\,\,\, (M_\star)$',fontsize=12)
#   plt.ylabel(r'$\Phi \left( \frac{N}{log(M)} \cdot Mpc^{-3} \cdot dex^{-1} \right)$',fontsize=12)
#   plt.show()

import numpy as np

def schechter(logm, logphi, logmstar, alpha):
  """
  Generate a Schechter function (in dlogm).
  """
  phi = ( (10**logphi) * np.log(10) * 10**((logm-logmstar) * (alpha + 1)) * np.exp(-10**(logm-logmstar)) )
  return phi

def parameter_at_z0(y,z0,z1=0.2,z2=1.6,z3=3.0):
  """
  Compute parameter at redshift ‘z0‘ as a function
  of the polynomial parameters ‘y‘ and the
  redshift anchor points ‘z1‘, ‘z2‘, and ‘z3‘.
  """
  y1, y2, y3 = y
  a = (((y3-y1) + (y2-y1) / (z2-z1) * (z1-z3)) / (z3**2-z1**2 + (z2**2-z1**2) / (z2-z1) * (z1-z3)))
  b = ((y2-y1) - a * (z2**2-z1**2)) / (z2-z1)
  c = y1-a * z1**2-b * z1
  return a * z0**2 + b * z0 + c

def log_schechter(logm,z):
    """
    Compute log of best-fitting Schechter function (in dlogm) from Leja+20.
    """
    # Continuity model median parameters + 1-sigma uncertainties.
    pars = { 'logphi1': [-2.44, -3.08, -4.14],
        'logphi1_err': [0.02, 0.03, 0.1],
        'logphi2': [-2.89, -3.29, -3.51],
        'logphi2_err': [0.04, 0.03, 0.03],
        'logmstar': [10.79,10.88,10.84],
        'logmstar_err': [0.02, 0.02, 0.04],
        'alpha1': [-0.28],
        'alpha1_err': [0.07],
        'alpha2': [-1.48],
        'alpha2_err': [0.1] }

    logmstar = parameter_at_z0(pars['logmstar'],z)
    logphi1 = parameter_at_z0(pars['logphi1'],z)
    phi1 = schechter(logm, logphi1, logmstar, pars['alpha1'][0]) # primary component
    logphi2 = parameter_at_z0(pars['logphi2'],z)
    phi2 = schechter(logm, logphi2, logmstar, pars['alpha2'][0]) # secondary component
    phi = phi1 + phi2 # combined mass function
    return np.log(phi)
