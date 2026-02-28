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
    Helper class for basic cosmology calculations.
    Provides H(z), chi(z), and dimensionless h.
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
        """Hubble parameter at redshift z in km/s/Mpc"""
        return self.H0 * np.sqrt(self.Om0 * (1 + z)**3 + (1 - self.Om0))

    def chi(self, z):
        """
        Comoving radial distance in Mpc.
        Uses numerical integration: chi(z) = ∫ c / H(z) dz
        """
        return self.cosmo.comoving_radial_distance(1/(1+z))
    
    def dDM_dz(self, z):
        """
        d<DM>/dz
        """
        return (1/pc) * 3 * (self.c*1e5) * self.chi_e * self.Ob0 * (self.H0*1e5/Mpc) * self.f_d * (1 + z) / (8 * np.pi * self.G * self.m_p * self.cosmo.h_over_h0(1/(1+z))) # in pc/cc


class SourceDistribution:
    """
    Handles source redshift distribution n(z) and computes kernels W(z).
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
        """Load n(z) from file, interpolate, and normalize."""
        
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
        Compute kernels.
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
    def _integrate_over_mf(self, array_2):
        i1 = self._integrator(self._mf * array_2, self._lmass)
        return i1
    def _integrate_over_mbf(self, array_2):
        i1 = self._integrator(self._mf * self._bf * array_2, self._lmass)
        return i1
    
def power_spectra(cosmo, log10M_max):
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
        Pk_files = np.sort(glob.glob("data/forecasts/precomputed_power_spectra/{}/P_{}_z=*.npy".format(dir_name, field)))
        Pk_fields = []
        for i in range(len(redshifts)):
            Pk_fields.append(np.load(Pk_files[i]))
        return redshifts, np.array(Pk_fields)

