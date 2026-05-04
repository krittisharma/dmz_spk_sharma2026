import numpy as np
from astropy import constants as con
from scipy.integrate import trapezoid
import pyccl as ccl
import glob

import warnings
warnings.filterwarnings("ignore")

redshifts = np.linspace(0, 3, 21)

# Constants
c = con.c.to('km/s').value
sigma_T = 6.65e-25 # cm^2
m_e = 9.10938e-28  # g
c_cgs = con.c.cgs.value
G = con.G.cgs.value; m_p = con.m_p.cgs.value; pc = con.pc.cgs.value; Mpc = 1e6*pc
f_d = 0.9; chi_e = 1 - (0.2453 / 2)

class CosmologyCalculator:
    """
    A helper class for basic cosmological calculations used in the sensitivity
    analysis of baryon probes and FRB observables.

    Wraps a pyccl.Cosmology object and provides convenient methods for computing
    the Hubble parameter H(z), comoving radial distance chi(z), and the radial
    kernel for the mean cosmic dispersion measure d<DM>/dz. Supports extensions
    to the standard LCDM cosmology including massive neutrinos and a time-varying
    dark energy equation of state (w0, wa).

    Parameters
    ----------
    H0_kms_per_Mpc : float
        Hubble constant in km/s/Mpc.
    Ob0 : float
        Baryon density parameter Omega_b.
    Om0 : float
        Total matter density parameter Omega_m.
    sigma8 : float
        RMS amplitude of linear matter fluctuations smoothed on 8 h^-1 Mpc scales.
    m_nu : float, optional
        Sum of neutrino masses in eV. Default is 0 (massless neutrinos).
    w0 : float, optional
        Dark energy equation-of-state parameter at z=0. Default is -1 (Lambda).
    wa : float, optional
        Time derivative of the dark energy equation-of-state. Default is 0.

    Attributes
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    Ob0 : float
        Baryon density parameter Omega_b.
    Om0 : float
        Total matter density parameter Omega_m.
    sigma8 : float
        sigma_8 amplitude of linear matter fluctuations.
    h : float
        Dimensionless Hubble parameter H0/100.
    cosmo : ccl.Cosmology
        Underlying pyccl Cosmology object initialized with linear matter
        power spectrum.
    """

    def __init__(self, H0_kms_per_Mpc, Ob0, Om0, sigma8, m_nu=0, w0=-1, wa=0):
        self.H0 = float(H0_kms_per_Mpc)   # Hubble constant in km/s/Mpc
        self.Ob0 = Ob0                    # Baryon density
        self.Om0 = Om0                    # Matter density
        self.sigma8 = sigma8
        self.m_nu = m_nu
        self.w0 = w0
        self.wa = wa
        self.c = c
        self.h = self.H0 / 100.0
        self.chi_e = chi_e
        self.f_d = f_d
        self.G = G
        self.m_p = m_p
        self.cosmo = ccl.Cosmology(Omega_c = Om0-Ob0, 
                                   Omega_b = Ob0, 
                                   h = self.h, 
                                   sigma8 = sigma8, 
                                   m_nu = m_nu,
                                   w0 = w0,
                                   wa = wa,
                                   n_s = 0.9665, 
                                   matter_power_spectrum='linear',
                                #    extra_parameters={"camb": {"dark_energy_model": 'ppf'}}
                                 )

    def H_z(self, z):
        """
        Computes the Hubble parameter H(z) at redshift z, assuming a flat
        LCDM cosmology.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate H(z).

        Returns
        -------
        H : float or np.ndarray
            Hubble parameter in km/s/Mpc at the given redshift(s).
        """
        return self.H0 * np.sqrt(self.Om0 * (1 + z)**3 + (1 - self.Om0))

    def chi(self, z):
        """
        Computes the comoving radial distance chi(z) in Mpc, using pyccl's
        comoving_radial_distance method.

            chi(z) = integral_0^z c / H(z') dz'

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the comoving distance.

        Returns
        -------
        chi : float or np.ndarray
            Comoving radial distance in Mpc at the given redshift(s).
        """
        return self.cosmo.comoving_radial_distance(1/(1+z))
    
    def dDM_dz(self, z):
        """
        Computes the redshift derivative of the mean cosmic dispersion measure,
        d<DM>/dz, at redshift z. This is the integrand of equation 2 of the
        paper (the DM radial kernel W_DM(chi)), and is given by:

            d<DM>/dz = (3 * c * chi_e * Omega_b * H0 * f_d * (1+z))
                       / (8 * pi * G * m_p * H(z)/H0)

        where chi_e is the free electron fraction, f_d is the diffuse baryon
        fraction, and H(z)/H0 = h_over_h0(z) is computed via pyccl. The result
        is in units of pc/cm^3 per unit redshift.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate d<DM>/dz.

        Returns
        -------
        dDM_dz : float or np.ndarray
            Redshift derivative of the mean cosmic DM in units of pc/cm^3
            per unit redshift at the given redshift(s).
        """
        return (1/pc) * 3 * (self.c*1e5) * self.chi_e * self.Ob0 * (self.H0*1e5/Mpc) * self.f_d * (1 + z) / (8 * np.pi * self.G * self.m_p * self.cosmo.h_over_h0(1/(1+z))) # in pc/cc


class SourceDistribution:
    """
    Handles the loading, normalization, and computation of source redshift
    distributions n(z) and line-of-sight projection kernels W(z) for multiple
    observational probes used in the sensitivity and Fisher forecast analyses.

    Specifically, computes the kernels for:
    - Weak gravitational lensing convergence (W_kappa): the lensing efficiency
      kernel for two tomographic source bins, integrated over the source
      redshift distribution behind each lens plane.
    - CMB lensing convergence (W_kappa_CMB): the lensing efficiency kernel
      with the CMB as the source at z ~ 1100.
    - Thermal Sunyaev-Zel'dovich effect (W_tSZ): proportional to (1+z)^2,
      tracing the thermal pressure of hot gas in halos.
    - Galaxy number density (W_g): the normalized galaxy selection function
      for two lens tomographic bins.
    - FRB dispersion measure (W_DM): the DM radial kernel W_DM(chi) weighted
      by the FRB redshift distribution, used to compute sigma^2[DM_cosmic(z_s)]
      via equation 3 of the paper.

    All kernels are evaluated on a uniform redshift grid z_grid = linspace(zmin,
    zmax, 1000), and stored as instance attributes for use in angular power
    spectrum computations.

    Parameters
    ----------
    cosmo_calc : CosmologyCalculator
        An initialized CosmologyCalculator instance providing H(z), chi(z),
        and d<DM>/dz.
    nz_file_k : str
        Path to the file containing the weak lensing source redshift distributions
        n(z) for the tomographic bins. Expected format: first column is redshift,
        subsequent columns are n(z) for each tomographic bin.
    nz_file_g : str
        Path to the file containing the galaxy lens redshift distributions n(z)
        for the tomographic bins. Same format as nz_file_k.
    nz_file_f : str, optional
        Path to an optional FRB redshift distribution file. If None, the FRB
        n(z) is modeled analytically as z^2 * exp(-3.5*z), representative of
        next-generation FRB surveys. Default is None.
    zmin : float, optional
        Minimum redshift of the z_grid. Default is 1e-4.
    zmax : float, optional
        Maximum redshift of the z_grid. Default is 3.0.
    zbin_k_idx1 : int, optional
        Column index in nz_file_k for the first weak lensing tomographic bin.
        Default is 4.
    zbin_k_idx2 : int, optional
        Column index in nz_file_k for the second weak lensing tomographic bin.
        Default is 4.
    zbin_g_idx1 : int, optional
        Column index in nz_file_g for the first galaxy tomographic bin.
        Default is 1.
    zbin_g_idx2 : int, optional
        Column index in nz_file_g for the second galaxy tomographic bin.
        Default is 1.
    Nfrb : int, optional
        Total number of FRBs in the survey, used to normalize the 3D FRB
        number density n3d_frb. Default is 100.
    omega_sky : float, optional
        Sky fraction covered by the survey, used to normalize the 3D galaxy
        and FRB number densities. Default is 0.1.
    Ngal1 : float, optional
        Total number of galaxies in the first lens tomographic bin, used to
        normalize n3d_g1. Default is 1e8.
    Ngal2 : float, optional
        Total number of galaxies in the second lens tomographic bin, used to
        normalize n3d_g2. Default is 1e8.

    Attributes
    ----------
    W_k_wt1 : np.ndarray, shape (nz,)
        Weak lensing convergence kernel for the first tomographic source bin.
    W_k_wt2 : np.ndarray, shape (nz,)
        Weak lensing convergence kernel for the second tomographic source bin.
    W_k_CMB_wt : np.ndarray, shape (nz,)
        CMB lensing convergence kernel with source at z ~ 1100.
    W_tSZ_wt : np.ndarray, shape (nz,)
        tSZ weight function, proportional to (1+z)^2.
    W_g_wt1 : np.ndarray, shape (nz,)
        Normalized galaxy selection kernel for the first lens tomographic bin.
    W_g_wt2 : np.ndarray, shape (nz,)
        Normalized galaxy selection kernel for the second lens tomographic bin.
    W_DM_wt : np.ndarray, shape (nz,)
        FRB dispersion measure radial kernel W_DM(chi), weighted by the FRB
        redshift distribution and d<DM>/dz, used to compute DM variance via
        equation 3 of the paper.
    """

    def __init__(self, cosmo_calc, nz_file_k, nz_file_g, nz_file_f=None, 
                 zmin=1e-4, zmax=3.0, zbin_k_idx1=4, zbin_k_idx2=4, zbin_g_idx1=1, zbin_g_idx2=1, 
                 Nfrb=100, omega_sky=0.1, Ngal1=1e8, Ngal2=1e8):
        self.cosmo = cosmo_calc
        self.zmin = zmin
        self.zmax = zmax
        self.zbin_k_idx1 = zbin_k_idx1
        self.zbin_k_idx2 = zbin_k_idx2
        self.zbin_g_idx1 = zbin_g_idx1
        self.zbin_g_idx2 = zbin_g_idx2
        self.z_grid = np.linspace(zmin, zmax, 1000)
        
        self.Nfrb = Nfrb
        self.Ngal1 = Ngal1
        self.Ngal2 = Ngal2
        self.omega_sky = omega_sky

        self._load_nz(nz_file_k, nz_file_g, nz_file_f)
        self._compute_W_z()
        

    def _load_nz(self, nz_file_k, nz_file_g, nz_file_f):
        """
        Loads, interpolates, and normalizes the source and lens redshift
        distributions n(z) for weak lensing, galaxy clustering, and FRB probes
        onto the internal z_grid.

        For weak lensing source bins (k1, k2): loads dn2d/dz from nz_file_k,
        clips negative values, and normalizes to unit integral over z_grid.

        For galaxy lens bins (g1, g2): loads dn2d/dz from nz_file_g, clips,
        normalizes to unit integral, converts to 3D number density n3d via:

            n3d(z) = H(z) / (c * chi^2) * dn2d/dz

        and rescales so that the total galaxy count within the survey volume
        matches Ngal1 or Ngal2.

        For FRBs: models dn2d/dz analytically as z^2 * exp(-3.5*z),
        normalizes to unit integral, converts to 3D number density n3d_frb,
        and rescales to match the total FRB count Nfrb within the survey volume.

        Parameters
        ----------
        nz_file_k : str
            Path to the weak lensing source n(z) file.
        nz_file_g : str
            Path to the galaxy lens n(z) file.
        nz_file_f : str or None
            Path to an optional FRB n(z) file (currently unused; FRB n(z)
            is always modeled analytically).
        """
        
        self.Hz = self.cosmo.H_z(self.z_grid)
        self.chi = self.cosmo.chi(self.z_grid)
        
        data = np.loadtxt(nz_file_k)
        zbins = data[:, 0]
        self.dn2d_dz_k1 = np.interp(self.z_grid, zbins, data[:, self.zbin_k_idx1])
        self.dn2d_dz_k2 = np.interp(self.z_grid, zbins, data[:, self.zbin_k_idx2])
        self.dn2d_dz_k1 = np.clip(self.dn2d_dz_k1, 0.0, None)
        self.dn2d_dz_k2 = np.clip(self.dn2d_dz_k2, 0.0, None)
        self.dn2d_dz_k1 /= trapezoid(self.dn2d_dz_k1, self.z_grid)
        self.dn2d_dz_k2 /= trapezoid(self.dn2d_dz_k2, self.z_grid)
        
        data = np.loadtxt(nz_file_g)
        zbins = data[:, 0]
        self.dn2d_dz_g1 = np.interp(self.z_grid, zbins, data[:, self.zbin_g_idx1])
        self.dn2d_dz_g1 = np.clip(self.dn2d_dz_g1, 0.0, None)
        self.dn2d_dz_g1 /= trapezoid(self.dn2d_dz_g1, self.z_grid)
        self.n3d_g1 = (self.Hz / (self.cosmo.c * self.chi**2)) * self.dn2d_dz_g1
        norm = trapezoid(self.n3d_g1 * (self.chi**2) * (self.omega_sky) * self.cosmo.c / self.Hz, self.z_grid)
        self.n3d_g1 *= self.Ngal1 / norm
        
        self.dn2d_dz_g2 = np.interp(self.z_grid, zbins, data[:, self.zbin_g_idx2])
        self.dn2d_dz_g2 = np.clip(self.dn2d_dz_g2, 0.0, None)
        self.dn2d_dz_g2 /= trapezoid(self.dn2d_dz_g2, self.z_grid)
        self.n3d_g2 = (self.Hz / (self.cosmo.c * self.chi**2)) * self.dn2d_dz_g2
        norm = trapezoid(self.n3d_g2 * (self.chi**2) * (self.omega_sky) * self.cosmo.c / self.Hz, self.z_grid)
        self.n3d_g2 *= self.Ngal2 / norm
        
        self.dn2d_dz_frb = (self.z_grid**2)*np.exp(-3.5*self.z_grid)
        self.dn2d_dz_frb /= trapezoid(self.dn2d_dz_frb, self.z_grid)
        
        self.n3d_frb = (self.Hz / (self.cosmo.c * self.chi**2)) * self.dn2d_dz_frb
        norm = trapezoid(self.n3d_frb * (self.chi**2) * (self.omega_sky) * self.cosmo.c / self.Hz, self.z_grid)
        self.n3d_frb *= self.Nfrb / norm


    def _compute_W_z(self):
        """
        Computes the line-of-sight projection kernels W(z) for all probes
        on the internal z_grid, using the normalized redshift distributions
        loaded by _load_nz.

        The following kernels are computed and stored as instance attributes:

        W_k_wt1, W_k_wt2 : Weak lensing convergence kernels for the two
            tomographic source bins, computed as:

                W_kappa(chi) = (3/2) * Omega_m * (H0/c)^2 * chi/a
                               * integral_{chi}^{chi_max} n(chi') * (chi'-chi)/chi' dchi'

            Only sources behind the lens (z' >= z) contribute to the integral.

        W_k_CMB_wt : CMB lensing convergence kernel, computed using the same
            lensing prefactor with the CMB at chi_CMB = chi(z=1100) as the
            single source plane:

                W_kappa_CMB(chi) = (3/2) * Omega_m * (H0/c)^2 * chi/a
                                   * (chi_CMB - chi) / chi_CMB

        W_tSZ_wt : tSZ weight function, proportional to (1+z)^2, tracing
            the thermal pressure of hot ionized gas in halos along the
            line of sight.

        W_g_wt1, W_g_wt2 : Normalized galaxy selection kernels for the two
            lens tomographic bins, computed as:

                W_g(z) = n3d(z) * chi^2 * c / (H(z) * (1+z))

            normalized to unit integral over z, and multiplied by H(z)/c
            to convert to a per-redshift weight.

        W_frb_wt_z : Normalized FRB redshift weight function, computed
            analogously to W_g but using n3d_frb.

        W_DM_wt : FRB dispersion measure radial kernel W_DM(chi), computed as:

                W_DM(chi) = d<DM>/dz * H(z)/c
                            * integral_{z}^{z_max} W_frb(z') dz'

            This is the kernel that enters equation 3 of the paper for
            computing sigma^2[DM_cosmic(z_s)], weighted by the FRB source
            redshift distribution beyond each lens redshift.
        """
        z = self.z_grid              # shape (nz,)
        chi = self.chi               # comoving distance at z, shape (nz,)
        Hz = self.Hz                 # H(z) at z, shape (nz,)
        n_k_z1 = self.dn2d_dz_k1     # n(z') per unit redshift, shape (nz,)  (or (nbin,nz))
        n_k_z2 = self.dn2d_dz_k2
        chi_CMB = self.cosmo.chi(1100)
        
        # scale factor a(z)
        a = 1.0 / (1.0 + z)

        # prefactor: 3 H0^2 Omega_m chi(z) / (2 c^2 a(z))
        H0_over_c = (self.cosmo.H0 / self.cosmo.c)
        prefactor = 1.5 * self.cosmo.Om0 * (H0_over_c**2) * chi / a

        # make 2D grids: source (chi', z') along axis=0, lens (chi, z) along axis=1
        chi_s, chi_l = np.meshgrid(chi, chi, indexing='ij')
        z_s,   z_l   = np.meshgrid(z, z, indexing='ij')    

        n_s_z_interp1 = np.interp(z_s, z, n_k_z1)  # shape (nz, nz)
        n_s_z_interp2 = np.interp(z_s, z, n_k_z2)

        # geometry factor: (chi' - chi)/chi'
        eps = 1e-30
        geom = (chi_s - chi_l) / np.where(chi_s == 0.0, eps, chi_s)

        # only sources behind lens contribute: use z' >= z
        integrand1 = n_s_z_interp1 * geom
        integrand1 = np.where(z_s >= z_l, integrand1, 0.0)
        integrand2 = n_s_z_interp2 * geom
        integrand2 = np.where(z_s >= z_l, integrand2, 0.0)

        # integrate over z' (trapz uses z spacing) -> no extra Jacobian factor
        # final lensing kernel
        self.W_k_wt1 = prefactor * trapezoid(integrand1, z, axis=0)
        self.W_k_wt2 = prefactor * trapezoid(integrand2, z, axis=0)

        self.W_tSZ_wt = (1+z)**2 # tSZ weight function
        
        self.W_k_CMB_wt = prefactor * (chi_CMB - chi)/chi_CMB

        W_g_wt_z = self.n3d_g1 * chi**2 * self.cosmo.c / (Hz * (1 + z))
        self.n2d_mean_g1 = trapezoid(W_g_wt_z, z)
        W_g_wt_z /= self.n2d_mean_g1
        self.W_g_wt1 = W_g_wt_z * Hz / self.cosmo.c
        
        W_g_wt_z = self.n3d_g2 * chi**2 * self.cosmo.c / (Hz * (1 + z))
        self.n2d_mean_g2 = trapezoid(W_g_wt_z, z)
        W_g_wt_z /= self.n2d_mean_g2
        self.W_g_wt2 = W_g_wt_z * Hz / self.cosmo.c
        
        self.W_frb_wt_z = self.n3d_frb * chi**2 * self.cosmo.c / (Hz * (1 + z))
        self.n2d_mean_frb = trapezoid(self.W_frb_wt_z, z)
        self.W_frb_wt_z /= self.n2d_mean_frb

        dDM_dz = self.cosmo.dDM_dz(z)
        valid_mask = (z[:, None] >= z) & (z[:, None] <= self.zmax)
        masked_integrand = np.where(valid_mask, self.W_frb_wt_z[:, None], 0)
        W_frb_int = trapezoid(masked_integrand, z, axis=0)
        self.W_DM_wt = dDM_dz * W_frb_int * Hz / self.cosmo.c


class HMCalculator_NoCorrection(ccl.halos.halo_model.HMCalculator):
    """
    A subclass of pyccl's HMCalculator that overrides the mass function
    integration methods to exclude the standard mass-to-total-mass correction
    (Mdelta_to_Mtot) applied in the parent class.

    In the standard pyccl HMCalculator, integrals over the halo mass function
    include a correction that accounts for the difference between the halo
    mass definition (e.g. M200c) and the total halo mass. This subclass removes
    that correction by integrating directly over the halo mass function (mf)
    and the halo mass function times halo bias (mf * bf), without any additional
    mass conversion factor. This is appropriate for the sensitivity analysis
    in this module, where the one-halo power spectra are computed for halos
    truncated at a fixed mass threshold log10M_max, used to isolate the
    contribution of halos above a given mass scale to the matter power spectrum
    suppression (as illustrated in Fig. 1c of the paper).

    Methods
    -------
    _integrate_over_mf(array_2)
        Integrates array_2 weighted by the halo mass function over log-mass,
        without mass-to-total-mass correction.
    _integrate_over_mbf(array_2)
        Integrates array_2 weighted by the halo mass function times halo bias
        over log-mass, without mass-to-total-mass correction.
    """

    def _integrate_over_mf(self, array_2):
        i1 = self._integrator(self._mf * array_2, self._lmass)
        return i1
    
    def _integrate_over_mbf(self, array_2):
        i1 = self._integrator(self._mf * self._bf * array_2, self._lmass)
        return i1
    

def power_spectra(cosmo, log10M_max):
    """
    Computes the one-halo gas-matter (P_gm), gas-gas (P_gg), and matter-matter
    (P_mm) power spectra at each redshift in the module-level redshifts array,
    using the BCEMU (Schneider+2019) baryonic feedback prescription at fiducial
    parameter values.

    This function is used in the sensitivity analysis to compute the halo mass
    and scale sensitivity of various baryon probes, as illustrated in Fig. 1 of
    the paper. By varying the upper halo mass threshold log10M_max, the
    contribution of halos above a given mass scale to the one-halo power spectra
    can be isolated, enabling the computation of the matter power spectrum
    suppression sensitivity shown in Fig. 1c.

    The gas profile is normalized to the mean baryon density, and the total
    matter profile (DMB) is normalized to the mean matter density. Both profiles
    are computed using BaryonForge's Schneider19 prescription at fixed fiducial
    feedback parameter values (log_Mc=13.5, theta_ej=6, mu_beta=0.94, eta=0.22,
    eta_delta=0.21, gamma=2.79, delta=7.33). The HMCalculator_NoCorrection
    subclass is used to integrate over halos up to log10M_max without
    mass-to-total-mass correction.

    Parameters
    ----------
    cosmo : ccl.Cosmology
        A pyccl Cosmology object defining the background cosmology.
    log10M_max : float
        Upper halo mass threshold in log10(M_sun) up to which the one-halo
        power spectra are integrated. Varying this parameter isolates the
        contribution of halos below this mass to the power spectra, enabling
        the halo mass sensitivity analysis of Fig. 1c.

    Returns
    -------
    redshifts : np.ndarray, shape (21,)
        Redshifts at which the power spectra are evaluated, linearly spaced
        between 0 and 3.
    P_gm : np.ndarray, shape (21, len(k_fields))
        One-halo gas-matter cross power spectrum at each redshift, with
        negative values clipped to 1e-100 for numerical stability.
    P_gg : np.ndarray, shape (21, len(k_fields))
        One-halo gas auto power spectrum at each redshift.
    P_mm : np.ndarray, shape (21, len(k_fields))
        One-halo matter auto power spectrum at each redshift.
    """
    P_gm, P_gg, P_mm = [], [], []
    
    for j in range(len(redshifts)):
        rho  = ccl.rho_x(cosmo, 1/(1+redshifts[j]), 'matter', is_comoving = True)
        
        par = dict(theta_ej=6, theta_co=0.1, M_c=(10**13.5)/cosmo.cosmo.params.h, mu_beta=0.94, 
                   eta=0.22, eta_delta=0.21, tau=-1.5, tau_delta=0, A=0.055/2, 
                   M1=2.5e11/cosmo.cosmo.params.h, epsilon_h=0.015, a=0.3, n=2, epsilon=4, p=0.3, 
                   q=0.707, gamma=2.79, delta=7.33)
        
        DMO = bfg.Profiles.Schneider19.DarkMatter(**par, r_min_int=1e-3, r_max_int=1e2, r_steps=500, mass_def=mass_def, c_M_relation=c_M_relation)
        GAS = bfg.Profiles.Schneider19.Gas(**par, r_min_int=1e-3, r_max_int=1e2, r_steps=500, mass_def=mass_def, c_M_relation=c_M_relation)
        STR = bfg.Profiles.Schneider19.Stars(**par, r_min_int=1e-6, r_max_int=5, r_steps=500, mass_def=mass_def, c_M_relation=c_M_relation)
        CLM = bfg.Profiles.Schneider19.CollisionlessMatter(**par, max_iter=2, reltol=5e-2, r_steps=500, mass_def=mass_def, c_M_relation=c_M_relation)
        DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**par,
                                                        gas=GAS, stars=STR, collisionlessmatter=CLM, darkmatter=DMO,
                                                        twohalo=bfg.Profiles.misc.Zeros(), 
                                                        r_steps=500, mass_def=mass_def, c_M_relation=c_M_relation)    
        DMB = DMB / rho
        GAS = GAS / (rho*cosmo.cosmo.params.Omega_b/(cosmo.cosmo.params.Omega_b+cosmo.cosmo.params.Omega_c))
        for p in [DMB, GAS]:
            p.update_precision_fftlog(**fft_precision)

        HMC  = HMCalculator_NoCorrection(mass_function = 'Tinker08', halo_bias = 'Tinker10', 
                                         mass_def = ccl.halos.massdef.MassDef200c, 
                                         log10M_min = 6, log10M_max = log10M_max, nM = 100)
        
        P_gm.append(ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k_fields, 1/(1+redshifts[j]), GAS, prof2=DMB, get_2h=False))
        P_gg.append(ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k_fields, 1/(1+redshifts[j]), GAS, get_2h=False))
        P_mm.append(ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k_fields, 1/(1+redshifts[j]), DMB, get_2h=False))
    
    P_gm = np.array(P_gm)
    P_gm[P_gm<0] = 1e-100
    return redshifts, np.array(P_gm), np.array(P_gg), np.array(P_mm)
        


def precomputed_power_spectra(dir_name, field):
    """
    Loads precomputed power spectra from disk for a given field type and
    directory, across all redshifts in the module-level redshifts array.

    Power spectrum files are expected to be stored as individual .npy files,
    one per redshift, following the naming convention:

        data/forecasts/precomputed_power_spectra/{dir_name}/P_{field}_z=*.npy

    Files are sorted alphanumerically to ensure consistent redshift ordering.
    This function is used to efficiently load precomputed gas, matter, or
    cross power spectra during the sensitivity and Fisher forecast analyses,
    avoiding repeated expensive halo model computations.

    Parameters
    ----------
    dir_name : str
        Name of the subdirectory under
        data/forecasts/precomputed_power_spectra/ containing the power
        spectrum files, typically identifying the feedback model or
        parameter configuration.
    field : str
        Field identifier string used in the filename, e.g. "gm" for
        gas-matter, "gg" for gas-gas, or "mm" for matter-matter.

    Returns
    -------
    redshifts : np.ndarray, shape (21,)
        Redshifts at which the power spectra are evaluated, linearly spaced
        between 0 and 3.
    Pk_fields : np.ndarray, shape (21, len(k_fields))
        Power spectra loaded from disk at each redshift in the redshifts array.
    """
    Pk_files = np.sort(glob.glob("data/forecasts/precomputed_power_spectra/{}/P_{}_z=*.npy".format(dir_name, field)))
    Pk_fields = []
    for i in range(len(redshifts)):
        Pk_fields.append(np.load(Pk_files[i]))
    return redshifts, np.array(Pk_fields)

