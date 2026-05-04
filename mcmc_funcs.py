import scipy
import pyccl as ccl
import BaryonForge as bfg
from joblib import Parallel, delayed
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pandas as pd
from astropy import constants as con
from scipy.stats import lognorm
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings("ignore")
import pickle
import os

G = con.G.cgs.value
m_p = con.m_p.cgs.value
c = con.c.cgs.value
pc = con.pc.cgs.value; Mpc = 1e6*pc
chi_e = 1 - (0.2453 / 2)
k_fields = np.geomspace(1e-4, 1e2, 1000)

Pk_model = os.getenv("PK_MODEL")
method   = os.getenv("METHOD")
N_frb    = int(os.getenv("N_FRB"))
N_params = int(os.environ["N_PARAMS"])
HMcode_mode = 'MatterPressure'
epsilon_trunc = 100

# Read in the emulator
if "emulator" in method:
    from tensorflow.keras.models import load_model
    model_dm = load_model(
        "emulators/{}/DMvar_emulator_{}.h5".format(Pk_model, Pk_model[:6]), 
        compile=False)
    print("emulators/{}/SPk_emulator_{}.h5".format(Pk_model, Pk_model[:6]))
    model_spk = load_model(
        "emulators/{}/SPk_emulator_{}.h5".format(Pk_model, Pk_model[:6]), 
        compile=False)
    from joblib import load
    scaler_y_dm = load(
        "emulators/{}/scaler_y_dm_{}.joblib".format(Pk_model, Pk_model[:6]))
    with open("emulators/{}/pca_objects_{}.pkl".format(Pk_model, Pk_model[:6]), 
              "rb") as f:
        pca_dict = pickle.load(f)
    ipca_dm  = pca_dict["ipca_dm"]
    ipca_spk  = pca_dict["ipca_spk"]
else:
    print("Not using emulators.")

# Read in the FRB sample
frb_sample = pd.read_csv('{}/{}.csv'.format(os.environ["DIR_PATH"], 
                                            os.environ["SAMPLE"]))
if os.environ["SAMPLE_EXTENDED"] == "Yes":
    frb_sample = frb_sample[np.logical_or(
        frb_sample["sharma_sample"] == "yes", 
        frb_sample["sharma_sample"] == "unclear")].reset_index()
else:
    frb_sample = frb_sample[frb_sample["sharma_sample"] == "yes"].reset_index()
frb_sample["DMexgal"] = [float(frb_sample["DMexgal"][i]) for i in range(len(frb_sample["DMexgal"]))]

dm_obs_list = np.array(list(frb_sample.DMexgal))
z_list = np.array(list(frb_sample.z_sample))

# Build age interpolator to save time during MCMC
H0_grid = np.linspace(50, 90, 50)
Om0_grid = np.linspace(0.2, 0.4, 50)
z_grid_temp   = np.linspace(0, 2, 50)
ages_grid = np.zeros((len(H0_grid), len(Om0_grid), len(z_grid_temp)))
for i, H0 in enumerate(H0_grid):
    for j, Om0 in enumerate(Om0_grid):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        ages_grid[i, j, :] = cosmo.age(z_grid_temp).value
age_interp = RegularGridInterpolator(
    (H0_grid, Om0_grid, z_grid_temp), ages_grid, bounds_error=False, fill_value=None
)


# Functions to compute f_diffuse(z)

def rho_Mstar(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, Pk_model, z, mode = HMcode_mode):
    """
    Computes the mean stellar mass density (rho_Mstar) and total baryon density (rho_baryon)
    at redshift(s) z, for an array of walkers with different cosmological and feedback parameters.

    For each walker, the stellar mass density is computed by integrating the stellar profile
    (central + satellite stars) over halo mass using the halo model framework, with gas profiles
    constructed from either the HMcode (Mead+2020) or BCEMU (Schneider+2019) prescription
    depending on Pk_model. The computation across walkers is parallelized using joblib.

    Parameters
    ----------
    H0_arr : array_like, shape (nwalkers,)
        Hubble constant values in km/s/Mpc for each walker.
    Ob0_arr : array_like, shape (nwalkers,)
        Baryon density parameter Omega_b for each walker.
    Om0_arr : array_like, shape (nwalkers,)
        Total matter density parameter Omega_m for each walker.
    s8_arr : array_like, shape (nwalkers,)
        sigma_8 (RMS amplitude of linear matter fluctuations) for each walker.
    feedback_params_list : list of dict, length nwalkers
        List of feedback parameter dictionaries, one per walker. 
        For HMcode, expects key "Tagn".
        For BCEmu, expects keys: "theta_ej", "log_Mc", "mu_beta", "eta", "eta_delta",
        "tau", "A", "gamma", "delta".
    Pk_model : str
        Power spectrum model identifier. Must be either "HMcode" or contain "BCEmu".
    z : array_like
        Redshift(s) at which to evaluate the stellar and baryon mass densities.
    mode : str, optional
        HMcode mode string passed to BaryonForge profile constructors.
        Defaults to HMcode_mode (module-level constant).

    Returns
    -------
    rho_Mstar_arr : np.ndarray, shape (nwalkers, len(z))
        Mean stellar mass density (central + satellite galaxies) in comoving units,
        integrated over the halo mass function, for each walker and redshift.
    rho_baryon_arr : np.ndarray, shape (nwalkers, len(z))
        Total comoving baryon mass density for each walker and redshift,
        computed as rho_matter * (Omega_b / Omega_m).
    """
    def rho_Mstar_element(H0, Ob0, Om0, s8, feedback_params, Pk_model, z, HMcode_mode):
        a = 1/(1+z)
        h = H0/(100*1e5/Mpc)
        cosmo = ccl.Cosmology(Omega_c = Om0-Ob0, Omega_b = Ob0, h = H0/(100*1e5/Mpc), 
                              sigma8 = s8, n_s = 0.9665, matter_power_spectrum='linear')
        def stellar_profile_integral(M, a):
            if Pk_model == "HMcode":
                par = bfg.Profiles.Mead20.Tagn2pars(feedback_params["Tagn"], mode = mode)
                STR = bfg.Profiles.Mead20.Stars(**par)+bfg.Profiles.Mead20.SatelliteStars(**par)
            elif "BCEmu" in Pk_model:
                par = dict(theta_ej = feedback_params["theta_ej"], theta_co = 0.1,
                        M_c = (10**feedback_params["log_Mc"])/h,
                        mu_beta = feedback_params["mu_beta"],
                        eta = feedback_params["eta"],
                        eta_delta = feedback_params["eta_delta"], tau = feedback_params["tau"],
                        tau_delta = 0, A = feedback_params["A"], M1 = 2.5e11/h,
                        epsilon_h = 0.015, a = 0.3, n = 2, epsilon = 4, p = 0.3,
                        q = 0.707, gamma = feedback_params["gamma"],
                        delta = feedback_params["delta"])
                STR = bfg.Profiles.Schneider19.Stars(**par, r_min_int = 1e-8, r_max_int = 1e2, r_steps = 100)+bfg.Profiles.Schneider19.SatelliteStars(**par, r_min_int = 1e-8, r_max_int = 1e2, r_steps = 100)
            else:
                print("Unknown Pk model!") 
            R200c = ccl.halos.massdef.MassDef200c.get_radius(cosmo, M, a)/a # in comoving Mpc/h
            R = np.geomspace(1e-5 * R200c.min(), R200c.max(), 100) # For stellar profiles going out to R200c is sufficient
            STR = STR.real(cosmo, M=M, r=R, a=a)
            return scipy.integrate.simpson(STR*4*np.pi*R**2, x=R)

        rho_matter  = ccl.rho_x(cosmo, a, 'matter', is_comoving = True)
        f_b = cosmo['Omega_b']/cosmo['Omega_m']
        rho_baryon = rho_matter*f_b
        HMC  = ccl.halos.halo_model.HMCalculator(mass_function = 'Tinker08', halo_bias = 'Tinker10', 
                                                 mass_def = ccl.halos.massdef.MassDef200c, 
                                                 log10M_min = 9, log10M_max = 16, nM = 100)
        rho_Mstar = []
        for this_a in np.atleast_1d(a):
            int_star = lambda M: stellar_profile_integral(M, this_a)
            mean_star = HMC.integrate_over_massfunc(int_star, cosmo, this_a)
            rho_Mstar.append(mean_star)
        return rho_Mstar, rho_baryon


    def run_one(idx):
        H0, Ob0, Om0, s8, feedback_params = H0_arr[idx], Ob0_arr[idx], Om0_arr[idx], s8_arr[idx], feedback_params_list[idx]
        return rho_Mstar_element(H0, Ob0, Om0, s8, feedback_params, Pk_model, z, HMcode_mode)

    output = Parallel(n_jobs=12)(delayed(run_one)(i) for i in range(len(H0_arr)))
    rho_Mstar_arr, rho_baryon_arr = np.array(output)[:, 0, :], np.array(output)[:, 1, :]
    return rho_Mstar_arr, rho_baryon_arr
    

def rho_H2(z):
    """
    Computes the mean molecular hydrogen (H2) mass density as a function of redshift,
    following the fitting formula of Boylan-Kolchin et al. (2025, A&A, 695, A163).
    See: https://ui.adsabs.harvard.edu/abs/2025A%26A...695A.163B/abstract

    The redshift evolution is modeled as a double power-law with a peak at z ~ 2,
    consistent with the observed cosmic star formation history.

    Parameters
    ----------
    z : float or array_like
        Redshift(s) at which to evaluate the H2 mass density.

    Returns
    -------
    rho_H2 : float or np.ndarray
        Molecular hydrogen mass density in units of M_sun / Mpc^3 at the given redshift(s).
    """
    return 5.3e6*((1+z)**3.6)/(1+((1+z)/2.1)**5.9)

def rho_H2_rho_HI(z):
    """
    Returns the ratio of molecular hydrogen (H2) to neutral hydrogen (HI) mass densities
    as a function of redshift, by interpolating over a precomputed table.

    The tabulated rho_H2/rho_HI values are drawn from:
    - Boylan-Kolchin et al. (2025, A&A, 695, A163) for H2:
      https://ui.adsabs.harvard.edu/abs/2025A%26A...695A.163B/abstract
    - Peroux & Howk (2020, ARA&A, 58, 363) for HI:
      https://ui.adsabs.harvard.edu/abs/2020ARA%26A..58..363P/abstract

    This ratio is used in the computation of f_diffuse to subtract the cold gas
    (H2 + HI) contribution from the total baryon budget, alongside the stellar
    mass density.

    Parameters
    ----------
    z : float or array_like
        Redshift(s) at which to evaluate the H2/HI density ratio.

    Returns
    -------
    rho_H2_rho_HI : float or np.ndarray
        The ratio rho_H2 / rho_HI at the given redshift(s), interpolated from
        the precomputed data table.
    """
    df = pd.read_csv("/Users/krittisharma/Desktop/research/frb_cosmo/dmz_spk/dmz_spk_sharma2026/data/f_diffuse/rhoH2_rhoHI.csv")
    return np.interp(z, df.z, df.rhoH2_rhoHI)


def f_diffuse(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, Pk_model, z, mode = HMcode_mode):
    """
    Computes the diffuse baryon fraction f_diffuse(z) for each walker at redshift(s) z.

    The diffuse baryon fraction is defined as the fraction of baryons not locked in
    stellar mass or cold gas (molecular H2 + neutral HI), i.e.:

        f_diffuse(z) = 1 - (rho_Mstar + rho_H2 + rho_HI) / rho_baryon

    This quantity enters directly into the computation of the mean cosmic dispersion
    measure <DM_cosmic(z)> (equation 2 of the paper), where it modulates the
    contribution of diffuse ionized gas along the line of sight. It encodes the
    fraction of baryons available to contribute to free electrons in the IGM and
    intervening halos.

    Parameters
    ----------
    H0_arr : array_like, shape (nwalkers,)
        Hubble constant values in km/s/Mpc for each walker.
    Ob0_arr : array_like, shape (nwalkers,)
        Baryon density parameter Omega_b for each walker.
    Om0_arr : array_like, shape (nwalkers,)
        Total matter density parameter Omega_m for each walker.
    s8_arr : array_like, shape (nwalkers,)
        sigma_8 for each walker.
    feedback_params_list : list of dict, length nwalkers
        List of feedback parameter dictionaries, one per walker.
    Pk_model : str
        Power spectrum model identifier. Must be either "HMcode" or contain "BCEmu".
    z : array_like
        Redshift(s) at which to evaluate f_diffuse.
    mode : str, optional
        HMcode mode string passed to BaryonForge profile constructors.
        Defaults to HMcode_mode (module-level constant).

    Returns
    -------
    f_d_arr : np.ndarray, shape (nwalkers, len(z))
        Diffuse baryon fraction for each walker and redshift.
    """
    rho_Mstar_arr, rho_baryons_arr = rho_Mstar(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, Pk_model, z, HMcode_mode)
    rho_H2_ = rho_H2(z)
    rho_HI = rho_H2_/rho_H2_rho_HI(z)
    f_d_arr = 1 - ((rho_Mstar_arr+rho_H2_[None, :]+rho_HI[None, :])/rho_baryons_arr)
    return f_d_arr


def DM_cosmic_mean(Om0_arr, Ob0_arr, H0_arr, s8_arr, feedback_params_list, Pk_model, z):
    """
    Computes the mean cosmic dispersion measure <DM_cosmic(z)> for each walker
    at the observed FRB redshift(s) z, following equation 2 of the paper:

        <DM_cosmic(z_s)> = integral_0^{z_s} [ 3*c*chi_e*Omega_b*H0 / (8*pi*G*m_p) ]
                           * f_diffuse(z) * (1+z) / sqrt(Omega_m*(1+z)^3 + Omega_Lambda) dz

    The diffuse baryon fraction f_diffuse(z) is first evaluated on a coarse redshift
    grid and then interpolated onto a finer grid before numerical integration using
    the cumulative trapezoid rule. The result is interpolated to the observed FRB
    redshifts z.

    Parameters
    ----------
    Om0_arr : array_like, shape (nwalkers,)
        Total matter density parameter Omega_m for each walker.
    Ob0_arr : array_like, shape (nwalkers,)
        Baryon density parameter Omega_b for each walker.
    H0_arr : array_like, shape (nwalkers,)
        Hubble constant values in cm/s/Mpc (pre-scaled by H0 * 1e5/Mpc) for each walker.
    s8_arr : array_like, shape (nwalkers,)
        sigma_8 for each walker.
    feedback_params_list : list of dict, length nwalkers
        List of feedback parameter dictionaries, one per walker.
    Pk_model : str
        Power spectrum model identifier. Must be either "HMcode" or contain "BCEmu".
    z : array_like, shape (nz,)
        Observed FRB redshift(s) at which to evaluate <DM_cosmic>.

    Returns
    -------
    DM_mean : np.ndarray, shape (nwalkers, nz)
        Mean cosmic dispersion measure <DM_cosmic(z)> in units of pc/cm^3,
        for each walker and observed redshift.
    """
    nwalkers = len(H0_arr)
    nz = len(z)
    z_prime = np.linspace(0, np.max(z), 50)
    z_prime_sub = np.linspace(0, np.max(z), 10)

    f_d = f_diffuse(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, Pk_model, z_prime_sub)  # (nwalkers, len(z_prime_sub))
    f_d = np.array([np.interp(z_prime, z_prime_sub, f_d[i]) for i in range(nwalkers)])

    integrand = f_d * (1+z_prime)[None,:] / np.sqrt(Om0_arr[:,None]*(1+z_prime)[None,:]**3 + (1-Om0_arr)[:,None])
    integral = cumulative_trapezoid(integrand, z_prime, axis=1, initial=0)

    DM_mean = np.zeros((nwalkers, nz))
    prefactor = 3*c*Ob0_arr*H0_arr*chi_e/(8*np.pi*G*m_p)
    for i in range(nwalkers):
        DM_mean[i] = prefactor[i] * np.interp(z, z_prime, integral[i]) / pc
    return DM_mean


def SPk_HMcode(H0, Ob0, Om0, s8, Tagn, mode = HMcode_mode):
    """
    Computes the matter power spectrum suppression ratio P_hydro(k) / P_gravity(k) at z=0
    using the HMcode (Mead+2020) baryonic feedback prescription, for a single set of
    cosmological and feedback parameters.

    The suppression ratio is defined as the ratio of the halo model power spectrum
    computed with baryonic feedback (dark matter + baryons, DMB) to the gravity-only
    (dark matter only, DMO) power spectrum. The AGN feedback strength is controlled
    by the single parameter Tagn, which is mapped to the full set of HMcode profile
    parameters via BaryonForge.

    Two computation modes are supported, selected via the module-level environment
    variable METHOD:
    - "exact"     : Directly computes the halo model power spectra using BaryonForge
                    and pyccl, with high-precision FFTLog settings.
    - "emulators" : Uses a pre-trained neural network emulator to predict the suppression
                    ratio from the input parameters, and saves the posterior samples
                    and summary statistics (median, 16th, 84th percentiles) to disk.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    Ob0 : float
        Baryon density parameter Omega_b.
    Om0 : float
        Total matter density parameter Omega_m.
    s8 : float
        sigma_8 (RMS amplitude of linear matter fluctuations).
    Tagn : float
        AGN heating temperature parameter controlling the strength of baryonic
        feedback in the HMcode prescription. Prior range: [7.6, 8.0].
    mode : str, optional
        HMcode mode string passed to BaryonForge profile constructors.
        Defaults to HMcode_mode (module-level constant).

    Returns
    -------
    SPk : np.ndarray, shape (len(k_fields),)
        Ratio P_DMB(k) / P_DMO(k) evaluated at the module-level k_fields array,
        returned only when method == "exact". When method == "emulators", results
        are saved to disk and nothing is returned.
    """
    if method == "exact":
        k_fields = np.geomspace(1e-4, 1e2, 1000)
        fft_precision = dict(padding_lo_fftlog = 1e-8, padding_hi_fftlog = 1e8, n_per_decade = 500)
        cosmo = ccl.Cosmology(Omega_c = Om0-Ob0, 
                              Omega_b = Ob0, 
                              h = H0/(1e5/Mpc)/100, 
                              sigma8 = s8, 
                              n_s = 0.9665, 
                              matter_power_spectrum='linear'
                             )
        rho  = ccl.rho_x(cosmo, 1/(1+0), 'matter', is_comoving = True)
        par = bfg.Profiles.Mead20.Tagn2pars(Tagn, mode = mode)
        DMB = bfg.Profiles.Mead20.DarkMatterBaryon(**par) / rho
        DMO = bfg.Profiles.Mead20.DarkMatter(**par) / rho
        for p in [DMO, DMB]: p.update_precision_fftlog(**fft_precision)
        HMC  = ccl.halos.halo_model.HMCalculator(mass_function = 'Tinker08', halo_bias = 'Tinker10', 
                                                 mass_def = ccl.halos.massdef.MassDef200c, 
                                                 log10M_min = 9, log10M_max = 16, nM = 100)
        P_DMB = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k_fields, 1/(1+0), DMB, 
                                                        suppress_1h = lambda k : 1e-2)
        P_DMO = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k_fields, 1/(1+0), DMO, 
                                                        suppress_1h = lambda k : 1e-2)
        return P_DMB/P_DMO
    
    elif method == "emulators":
        X = np.array([H0/(1e5/Mpc), Ob0, Om0, s8, Tagn])
        X = X.astype("float32").T
        pred_spk_pca = model_spk.predict(X, verbose=0)
        emulated_SPks = np.array(ipca_spk.inverse_transform(pred_spk_pca))
        np.save('data/frb_results/spk_HMcode_posterior_samples_feedbackonly{}.npy'.format(os.environ["SUFFIX"]), np.array(emulated_SPks))

        def get_band(x, qlo=16, qhi=84):
            med = np.median(x, axis=0)
            lo  = np.percentile(x, qlo, axis=0)
            hi  = np.percentile(x, qhi, axis=0)
            return med, lo, hi
        
        ratio_spk_med, ratio_spk_lo, ratio_spk_hi = get_band(emulated_SPks)
        np.save('data/frb_results/spk_HMcode{}.npy'.format(os.environ["SUFFIX"]), np.array([ratio_spk_lo, ratio_spk_med, ratio_spk_hi]))

    else:
        print("Unknown method!")
        return None


def Pgas_HMcode(H0, Ob0, Om0, s8, Tagn, z_arr, mode = HMcode_mode):
    """
    Computes the gas power spectrum Pgas(k, z) at each redshift in z_arr using the
    HMcode (Mead+2020) baryonic feedback prescription, for a single set of cosmological
    and feedback parameters.

    The gas profile is constructed using BaryonForge's Mead20.Gas profile, normalized
    by the mean baryon density (rho_matter * Omega_b / Omega_m) at each redshift. The
    halo model power spectrum is then computed using pyccl's halomod_power_spectrum
    with the Tinker+2008 mass function and Tinker+2010 halo bias, integrated over
    halo masses in the range 10^9 -- 10^16 M_sun. High-precision FFTLog settings are
    applied to ensure accurate small-scale power spectrum computation.

    Pgas(k, z) enters the computation of the DM variance sigma^2[DM_cosmic(z_s)]
    through equation 3 of the paper, where it is integrated over k and z with the
    DM radial kernel W_DM(chi) to yield the sightline-to-sightline variance in
    the cosmic dispersion measure.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    Ob0 : float
        Baryon density parameter Omega_b.
    Om0 : float
        Total matter density parameter Omega_m.
    s8 : float
        sigma_8 (RMS amplitude of linear matter fluctuations).
    Tagn : float
        AGN heating temperature parameter controlling the strength of baryonic
        feedback in the HMcode prescription. Prior range: [7.6, 8.0].
    z_arr : array_like
        Redshifts at which to evaluate the gas power spectrum.
    mode : str, optional
        HMcode mode string passed to BaryonForge profile constructors.
        Defaults to HMcode_mode (module-level constant).

    Returns
    -------
    k_fields : np.ndarray, shape (1000,)
        Wavenumbers in units of h/Mpc at which the power spectrum is evaluated,
        log-spaced between 1e-4 and 1e2 h/Mpc.
    P_GAS : list of np.ndarray, length len(z_arr)
        Gas power spectrum Pgas(k) at each redshift in z_arr, in units of (Mpc/h)^3.
    """
    fft_precision = dict(padding_lo_fftlog = 1e-8, padding_hi_fftlog = 1e8, n_per_decade = 500)
    cosmo = ccl.Cosmology(Omega_c = Om0-Ob0, 
                          Omega_b = Ob0, 
                          h = H0/(1e5/Mpc)/100, 
                          sigma8 = s8, 
                          n_s = 0.9665, 
                          matter_power_spectrum='linear'
                         )
    rho  = ccl.rho_x(cosmo, 1/(1+z_arr), 'matter', is_comoving = True)
    par = bfg.Profiles.Mead20.Tagn2pars(Tagn, mode = mode)
    GAS = [bfg.Profiles.Mead20.Gas(**par) / (rho[i]*(Ob0/Om0)) for i in range(len(z_arr))]
    for p in GAS: p.update_precision_fftlog(**fft_precision)
    HMC  = ccl.halos.halo_model.HMCalculator(mass_function = 'Tinker08', halo_bias = 'Tinker10', 
                                             mass_def = ccl.halos.massdef.MassDef200c, 
                                             log10M_min = 9, log10M_max = 16, nM = 100)
    P_GAS = []
    for i in range(len(z_arr)):
        P_GAS.append(ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC, k_fields, 1/(1+z_arr[i]), GAS[i], suppress_1h = lambda k : 1e-2))
    return k_fields, P_GAS


def SPk_BCEmu(H0, Ob0, Om0, s8, feedback_params, m_nu=0, w0=-1, wa=0):
    """
    Computes the matter power spectrum suppression ratio P_hydro(k) / P_gravity(k) at z=0
    using the flexible analytical BCEMU (Schneider+2019) baryonic feedback prescription,
    for a single set of cosmological and feedback parameters.

    The suppression ratio is defined as the ratio of the halo model power spectrum
    computed with baryonic feedback (dark matter + baryons, DMB) to the gravity-only
    (dark matter only, DMO) power spectrum. The BCEMU prescription models gas profiles
    analytically with free parameters that are not tied to any specific subgrid feedback
    implementation, making it more flexible than HMcode. The full set of baryonic
    components (dark matter, gas, stars, collisionless matter) are modeled separately
    and combined into the total DMB profile. A truncation at epsilon_trunc * R200c is
    applied to both DMO and DMB profiles to account for gas ejection beyond the halo
    boundary. Mass-to-total-mass corrections (Mdelta_to_Mtot) are applied separately
    to the DMO and DMB profiles to ensure consistent halo mass definitions.

    Supports optional extensions to the standard LCDM cosmology, including massive
    neutrinos (m_nu), and a time-varying dark energy equation of state (w0, wa).

    Two computation modes are supported, selected via the module-level environment
    variable METHOD:
    - "exact"     : Directly computes the halo model power spectra using BaryonForge
                    and pyccl, with high-precision FFTLog settings.
    - "emulators" : Uses a pre-trained neural network emulator to predict the suppression
                    ratio from the input parameters (supporting BCEmu1, BCEmu7, BCEmu8
                    model variants), and saves the posterior samples and summary statistics
                    (median, 16th, 84th percentiles) to disk.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    Ob0 : float
        Baryon density parameter Omega_b.
    Om0 : float
        Total matter density parameter Omega_m.
    s8 : float
        sigma_8 (RMS amplitude of linear matter fluctuations).
    feedback_params : dict
        Dictionary of BCEMU feedback parameters. Expected keys:
        - "log_Mc"    : log10 of characteristic halo mass scale below which gas
                        profile becomes shallower than the NFW profile.
        - "theta_ej"  : Maximum radius of gas ejection in units of R200c.
        - "mu_beta"   : Mass dependence of the inner gas profile slope beta.
        - "eta"       : High-mass end slope of the stellar-to-halo mass relation.
        - "eta_delta" : Partition of stellar content between central and satellite galaxies.
        - "tau"       : Low-mass end slope of the stellar-to-halo mass relation.
        - "A"         : Normalization of the stellar-to-halo mass relation.
        - "gamma"     : Outer slope of the bound gas profile.
        - "delta"     : Outer slope of the bound gas profile beyond truncation.
    m_nu : float, optional
        Sum of neutrino masses in eV. Default is 0 (massless neutrinos).
    w0 : float, optional
        Dark energy equation-of-state parameter at z=0. Default is -1 (Lambda).
    wa : float, optional
        Time derivative of the dark energy equation-of-state parameter. Default is 0.

    Returns
    -------
    SPk : np.ndarray, shape (len(k_fields),)
        Ratio P_DMB(k) / P_DMO(k) evaluated at the module-level k_fields array,
        returned only when method == "exact". When method == "emulators", results
        are saved to disk and nothing is returned.
    """
    k_fields = np.geomspace(1e-4, 1e2, 1000)
    if method == "exact":
        fft_precision = dict(padding_lo_fftlog = 1e-8, padding_hi_fftlog = 1e8, n_per_decade = 500)
        h = H0/(1e5/Mpc)/100

        if wa != 0:
            cosmo = ccl.Cosmology(Omega_c = Om0-Ob0, 
                                  Omega_b = Ob0, 
                                  h = h, 
                                  sigma8 = s8, 
                                  n_s = 0.9665, 
                                  matter_power_spectrum='linear',
                                  m_nu=m_nu,
                                  w0=w0,
                                  wa=wa,
                                #   extra_parameters={"camb": {"dark_energy_model": "ppf"}},
                     )
        else:
            cosmo = ccl.Cosmology(Omega_c = Om0-Ob0, 
                      Omega_b = Ob0, 
                      h = h, 
                      sigma8 = s8, 
                      n_s = 0.9665, 
                      matter_power_spectrum='linear',
                      m_nu=m_nu,
                      w0=w0,
                      wa=wa
         )

        rho  = ccl.rho_x(cosmo, 1/(1+0), 'matter', is_comoving = True)
        par = dict(theta_ej = feedback_params["theta_ej"], theta_co = 0.1,
                    M_c = (10**feedback_params["log_Mc"])/h, 
                    mu_beta = feedback_params["mu_beta"], 
                    eta = feedback_params["eta"], 
                    eta_delta = feedback_params["eta_delta"], tau = feedback_params["tau"], 
                    tau_delta = 0, A = feedback_params["A"], M1 = 2.5e11/h, 
                    epsilon_h = 0.015, a = 0.3, n = 2, epsilon = 4, p = 0.3, 
                    q = 0.707, gamma = feedback_params["gamma"], 
                    delta = feedback_params["delta"])

        DMO = bfg.Profiles.Schneider19.DarkMatter(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
        GAS = bfg.Profiles.Schneider19.Gas(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
        STR = bfg.Profiles.Schneider19.Stars(**par, r_min_int = 1e-6, r_max_int = 1e2, r_steps = 500)
        CLM = bfg.Profiles.Schneider19.CollisionlessMatter(**par, max_iter = 2, reltol = 5e-2, 
                                                           r_steps = 500, r_min_int = 1e-6, r_max_int = 1e2)
        DMB = bfg.Profiles.Schneider19.DarkMatterBaryon(**par,
                                                        gas = GAS, stars = STR,
                                                        collisionlessmatter = CLM, darkmatter = DMO,
                                                        twohalo = bfg.Profiles.misc.Zeros(),
                                                        r_min_int = 1e-6, r_max_int = epsilon_trunc, r_steps = 500)
        T   = bfg.Profiles.misc.Truncation(epsilon_trunc = epsilon_trunc)
        M_2_Mtot_DMO = bfg.Profiles.misc.Mdelta_to_Mtot(DMO * T, r_min = 1e-6, r_max = epsilon_trunc, N_int = 100)
        M_2_Mtot_DMB = bfg.Profiles.misc.Mdelta_to_Mtot(DMB * T, r_min = 1e-6, r_max = epsilon_trunc, N_int = 100)
        DMO = (DMO / rho) * T; DMB = (DMB / rho) * T

        for p in [DMO, DMB]: p.update_precision_fftlog(**fft_precision)
        HMC_flex_DMO = bfg.utils.FlexibleHMCalculator(mass_function = 'Tinker08', halo_bias = 'Tinker10', 
                                                      halo_m_to_mtot = M_2_Mtot_DMO,
                                                      mass_def = ccl.halos.massdef.MassDef200c,
                                                      log10M_min = 9, log10M_max = 16, nM = 100)

        HMC_flex_DMB = bfg.utils.FlexibleHMCalculator(mass_function = 'Tinker08', halo_bias = 'Tinker10', 
                                                      halo_m_to_mtot = M_2_Mtot_DMB,
                                                      mass_def = ccl.halos.massdef.MassDef200c,
                                                      log10M_min = 9, log10M_max = 16, nM = 100)

        P_DMO = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC_flex_DMO, k_fields, 1/(1+0), DMO, 
                                                        suppress_1h = lambda k : 1e-2)
        P_DMB = ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC_flex_DMB, k_fields, 1/(1+0), DMB, 
                                                        suppress_1h = lambda k : 1e-2)
        return P_DMB/P_DMO
    
    elif method == "emulators":
        if "BCEmu1" in Pk_model:
            X = np.array([H0/(1e5/Mpc), Ob0, Om0, s8, 
                [fp["log_Mc"] for fp in feedback_params]])
            
        elif "BCEmu7" in Pk_model:
            X = np.array([H0/(1e5/Mpc), Ob0, Om0, s8,
                [fp["log_Mc"] for fp in feedback_params],
                [fp["theta_ej"] for fp in feedback_params],
                [fp["eta_delta"] for fp in feedback_params],
                [fp["mu_beta"] for fp in feedback_params],
                [fp["gamma"] for fp in feedback_params],
                [fp["delta"] for fp in feedback_params],
                [fp["eta"] for fp in feedback_params]])

        elif "BCEmu8" in Pk_model:
            X = np.array([H0/(1e5/Mpc), Ob0, Om0, s8,
                [fp["log_Mc"] for fp in feedback_params],
                [fp["theta_ej"] for fp in feedback_params],
                [fp["eta_delta"] for fp in feedback_params],
                [fp["mu_beta"] for fp in feedback_params],
                [fp["gamma"] for fp in feedback_params],
                [fp["delta"] for fp in feedback_params],
                [fp["eta"] for fp in feedback_params],
                [fp["A"] for fp in feedback_params]])

        else:
            raise ValueError("Unknown Pk_model!")

        X = X.astype("float32").T
        pred_spk_pca = model_spk.predict(X, verbose=0)
        emulated_SPks = np.array(ipca_spk.inverse_transform(pred_spk_pca))
        print("SPk emulation done!")
        np.save('data/frb_results/spk_BCEmu{}_posterior_samples_feedbackonly{}.npy'.format(
            N_params,
            os.environ["SUFFIX"]), np.array(emulated_SPks))

        def get_band(x, qlo=16, qhi=84):
            med = np.median(x, axis=0)
            lo  = np.percentile(x, qlo, axis=0)
            hi  = np.percentile(x, qhi, axis=0)
            return med, lo, hi

        ratio_spk_med, ratio_spk_lo, ratio_spk_hi = get_band(emulated_SPks)
        np.save('data/frb_results/spk_BCEmu{}{}.npy'.format(
            N_params, 
            os.environ["SUFFIX"]), 
            np.array([ratio_spk_lo, ratio_spk_med, ratio_spk_hi]))
    else:
        print("Unknown method!")
        return None
    

def Pgas_BCEmu(H0, Ob0, Om0, s8, feedback_params, z_arr):
    """
    Computes the gas power spectrum Pgas(k, z) at each redshift in z_arr using the
    flexible analytical BCEMU (Schneider+2019) baryonic feedback prescription, for
    a single set of cosmological and feedback parameters.

    The gas profile is constructed using BaryonForge's Schneider19.Gas profile,
    normalized by the mean baryon density (rho_matter * Omega_b / Omega_m) at each
    redshift, and truncated at epsilon_trunc * R200c to account for gas ejection
    beyond the halo boundary. A mass-to-total-mass correction (Mdelta_to_Mtot) is
    applied to ensure consistent halo mass definitions when computing the halo model
    power spectrum. The halo model power spectrum is then computed using pyccl's
    halomod_power_spectrum with the Tinker+2008 mass function and Tinker+2010 halo
    bias, integrated over halo masses in the range 10^9 -- 10^16 M_sun. High-precision
    FFTLog settings are applied to ensure accurate small-scale power spectrum computation.

    Pgas(k, z) enters the computation of the DM variance sigma^2[DM_cosmic(z_s)]
    through equation 3 of the paper, where it is integrated over k and z with the
    DM radial kernel W_DM(chi) to yield the sightline-to-sightline variance in
    the cosmic dispersion measure.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    Ob0 : float
        Baryon density parameter Omega_b.
    Om0 : float
        Total matter density parameter Omega_m.
    s8 : float
        sigma_8 (RMS amplitude of linear matter fluctuations).
    feedback_params : dict
        Dictionary of BCEMU feedback parameters. Expected keys:
        - "log_Mc"    : log10 of characteristic halo mass scale below which gas
                        profile becomes shallower than the NFW profile.
        - "theta_ej"  : Maximum radius of gas ejection in units of R200c.
        - "mu_beta"   : Mass dependence of the inner gas profile slope beta.
        - "eta"       : High-mass end slope of the stellar-to-halo mass relation.
        - "eta_delta" : Partition of stellar content between central and satellite galaxies.
        - "tau"       : Low-mass end slope of the stellar-to-halo mass relation.
        - "A"         : Normalization of the stellar-to-halo mass relation.
        - "gamma"     : Outer slope of the bound gas profile.
        - "delta"     : Outer slope of the bound gas profile beyond truncation.
    z_arr : array_like
        Redshifts at which to evaluate the gas power spectrum.

    Returns
    -------
    k_fields : np.ndarray, shape (1000,)
        Wavenumbers in units of h/Mpc at which the power spectrum is evaluated,
        log-spaced between 1e-4 and 1e2 h/Mpc.
    P_GAS : list of np.ndarray, length len(z_arr)
        Gas power spectrum Pgas(k) at each redshift in z_arr, in units of (Mpc/h)^3,
        normalized to the mean baryon density.
    """
    fft_precision = dict(padding_lo_fftlog = 1e-8, padding_hi_fftlog = 1e8, n_per_decade = 500)
    h = H0/(1e5/Mpc)/100
    
    cosmo = ccl.Cosmology(Omega_c = Om0-Ob0, 
                          Omega_b = Ob0, 
                          h = h, 
                          sigma8 = s8, 
                          n_s = 0.9665, 
                          matter_power_spectrum='linear'
                         )
    
    P_GAS = []
    for z in z_arr:
        rho  = ccl.rho_x(cosmo, 1/(1+z), 'matter', is_comoving = True)
        par = dict(theta_ej = feedback_params["theta_ej"], theta_co = 0.1,
                   M_c = (10**feedback_params["log_Mc"])/h, 
                   mu_beta = feedback_params["mu_beta"], 
                   eta = feedback_params["eta"], 
                   eta_delta = feedback_params["eta_delta"], tau = feedback_params["tau"], 
                   tau_delta = 0, A = feedback_params["A"], M1 = 2.5e11/h, 
                   epsilon_h = 0.015, a = 0.3, n = 2, epsilon = 4, p = 0.3, 
                   q = 0.707, gamma = feedback_params["gamma"], 
                   delta = feedback_params["delta"])
        DMO = bfg.Profiles.Schneider19.DarkMatter(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
        GAS = bfg.Profiles.Schneider19.Gas(**par, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)
        M_2_Mtot = bfg.Profiles.misc.Mdelta_to_Mtot(DMO, r_min = 1e-6, r_max = 1e2, N_int = 100)
        GAS = GAS / (rho*(Ob0/Om0))
        T   = bfg.Profiles.misc.Truncation(epsilon_trunc = 100)
        GAS = GAS * T
        for p in [GAS]: p.update_precision_fftlog(**fft_precision)
        HMC_flex = bfg.utils.FlexibleHMCalculator(mass_function = 'Tinker08', halo_bias = 'Tinker10', 
                                                  halo_m_to_mtot = M_2_Mtot,
                                                  mass_def = ccl.halos.massdef.MassDef200c, 
                                                  log10M_min = 9, log10M_max = 16, nM = 100)

        P_GAS.append(ccl.halos.pk_2pt.halomod_power_spectrum(cosmo, HMC_flex, k_fields, 1/(1+z), GAS, 
                                                             suppress_1h = lambda k : 1e-2))
    return k_fields, P_GAS


if "high_z" in Pk_model:
    zbins = np.linspace(0, 5, 15)
else:
    zbins = np.linspace(0, 1, 15)


def get_DM_variance(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, z_obs, Pk_model):
    """
    Computes the standard deviation of the cosmic dispersion measure,
    sigma[DM_cosmic(z_s)], for each walker at the observed FRB redshift(s) z_obs,
    following equation 3 of the paper:

        sigma^2[DM_cosmic(z_s)] = integral_0^{chi_s} dchi * W_DM^2(chi)
                                  * integral_0^inf dk * k / (2*pi) * Pgas(k, z(chi))

    where W_DM(chi) is the DM radial kernel (equation 2), and Pgas(k, z) is the
    feedback-dependent gas power spectrum. The sightline-to-sightline variance in
    DM_cosmic arises from fluctuations in the ionized gas distribution within
    intervening collapsed structures, and is directly sensitive to the strength
    of baryonic feedback through Pgas(k, z).

    Two computation modes are supported, selected via the module-level environment
    variable METHOD:
    - "emulators" : Uses a pre-trained neural network emulator to predict
                    sigma[DM_cosmic(z)] as a function of redshift for each walker,
                    supporting HMcode, BCEmu1, BCEmu7, and BCEmu8 model variants.
                    The emulated DM variance is interpolated to the observed FRB
                    redshifts z_obs.
    - "exact"     : Directly computes sigma[DM_cosmic(z_s)] by numerically integrating
                    the gas power spectrum Pgas(k, z) over k using Simpson's rule,
                    and then integrating over redshift using the cumulative trapezoid
                    rule with the appropriate DM radial kernel weighting. The
                    redshift integration grid is set by the module-level zbins array
                    (0 to 1 for standard runs, 0 to 5 for high-z runs).

    Parameters
    ----------
    H0_arr : array_like, shape (nwalkers,)
        Hubble constant values in km/s/Mpc for each walker.
    Ob0_arr : array_like, shape (nwalkers,)
        Baryon density parameter Omega_b for each walker.
    Om0_arr : array_like, shape (nwalkers,)
        Total matter density parameter Omega_m for each walker.
    s8_arr : array_like, shape (nwalkers,)
        sigma_8 for each walker.
    feedback_params_list : list of dict, length nwalkers
        List of feedback parameter dictionaries, one per walker.
    z_obs : array_like, shape (nz,)
        Observed FRB redshift(s) at which to evaluate sigma[DM_cosmic].
    Pk_model : str
        Power spectrum model identifier. Must be "HMcode", "BCEmu1", "BCEmu7",
        or "BCEmu8".

    Returns
    -------
    DM_var_out : np.ndarray, shape (nwalkers, nz)
        Standard deviation sigma[DM_cosmic(z_s)] in units of pc/cm^3,
        for each walker and observed FRB redshift.
    """
    nwalkers = len(H0_arr)
    z_obs = np.atleast_1d(z_obs)
    nz = len(z_obs)

    DM_var_out = np.zeros((nwalkers, nz))

    if "emulator" in method:
        # Build batch input for all walkers
        if "HMcode" in Pk_model:
            X = np.column_stack([H0_arr/(1e5/Mpc), Ob0_arr, Om0_arr, s8_arr,
                                 [fp["Tagn"] for fp in feedback_params_list]])
        elif "BCEmu1" in Pk_model:
            X = np.column_stack([H0_arr/(1e5/Mpc), Ob0_arr, Om0_arr, s8_arr,
                                 [fp["log_Mc"] for fp in feedback_params_list]])
            
        elif "BCEmu7" in Pk_model:
            X = np.column_stack([H0_arr/(1e5/Mpc), Ob0_arr, Om0_arr, s8_arr,
                                 [fp["log_Mc"] for fp in feedback_params_list],
                                 [fp["theta_ej"] for fp in feedback_params_list],
                                 [fp["eta_delta"] for fp in feedback_params_list],
                                 [fp["mu_beta"] for fp in feedback_params_list],
                                 [fp["gamma"] for fp in feedback_params_list],
                                 [fp["delta"] for fp in feedback_params_list],
                                 [fp["eta"] for fp in feedback_params_list]
                                 ])

        elif "BCEmu8" in Pk_model:
            X = np.column_stack([H0_arr/(1e5/Mpc), Ob0_arr, Om0_arr, s8_arr,
                                 [fp["log_Mc"] for fp in feedback_params_list],
                                 [fp["theta_ej"] for fp in feedback_params_list],
                                 [fp["eta_delta"] for fp in feedback_params_list],
                                 [fp["mu_beta"] for fp in feedback_params_list],
                                 [fp["gamma"] for fp in feedback_params_list],
                                 [fp["delta"] for fp in feedback_params_list],
                                 [fp["eta"] for fp in feedback_params_list],
                                 [fp["A"] for fp in feedback_params_list]
                                 ])
        
        else:
            raise ValueError("Unknown Pk_model!")

        X = X.astype("float32")
        pred_dm_pca = model_dm.predict(X, verbose=0)
        pred_dm_pca = scaler_y_dm.inverse_transform(pred_dm_pca)
        for i in range(nwalkers):
            emulated_DMvars = np.array([0] + list(10**ipca_dm.inverse_transform(pred_dm_pca[i])))
            if "high_z" in Pk_model:
                DM_var_out[i] = np.interp(z_obs, np.linspace(0, 5, 100), emulated_DMvars)
            else:
                DM_var_out[i] = np.interp(z_obs, np.linspace(0, 1, 100), emulated_DMvars)
        
    elif method == "exact":
        # exact computation requires a for loop over walkers
        for i in range(nwalkers):
            H0, Ob0, Om0, s8, fp = H0_arr[i], Ob0_arr[i], Om0_arr[i], s8_arr[i], feedback_params_list[i]
            if Pk_model == "HMcode":
                k_fields, integrand_zk = Pgas_HMcode(H0, Ob0, Om0, s8, fp["Tagn"], zbins)
            elif "BCEmu" in Pk_model:
                k_fields, integrand_zk = Pgas_BCEmu(H0, Ob0, Om0, s8, fp, zbins)
            else:
                raise ValueError("Unknown Pk_model!")
            
            integrand_z = (simpson((k_fields * integrand_zk / (2 * np.pi)), k_fields, axis=1)) * Mpc
            Hz = H0 * np.sqrt(Om0*(1 + zbins)**3 + (1 - Om0))
            integrand_z = ((1 + zbins)**2 / Hz) * integrand_z 
            # Note that we do not multiply by f_d^2 because Pgas was originally normalized to mean matter density, 
            # which we converted to normalization to mean baryon density, so it already has an f_d^2 in there!

            rho_c0 = 3 * (H0**2) / (8 * np.pi * G)
            ne0_bar = Ob0 * rho_c0 * chi_e / m_p

            integrand_z *= c * (ne0_bar**2)
            integral = cumulative_trapezoid(integrand_z, zbins, initial=0)
            DM_variance = np.interp(z_obs, zbins, integral)
            DM_var_out[i] = ((np.array(DM_variance)/(pc**2))**0.5)

    else:
        raise ValueError("Unknown method!")

    return DM_var_out  # (nwalkers, len(z_obs))


def p_cosmic(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, z_obs, DM_obs, Pk_model):
    """
    Computes the conditional probability distribution p(DM_cosmic | z_s) of the
    cosmic dispersion measure for each walker, at each observed FRB redshift and
    DM value, following equation 1 of the paper.

    The cosmic DM distribution is modeled as a log-normal, parameterized by its
    first two moments: the mean <DM_cosmic(z_s)> (equation 2) and the variance
    sigma^2[DM_cosmic(z_s)] (equation 3). The log-normal parameters mu_log and
    sigma_log are computed from these moments as:

        mu_log    = log( <DM_cosmic>^2 / sqrt(<DM_cosmic>^2 + sigma^2) )
        sigma_log = sqrt( log(1 + sigma^2 / <DM_cosmic>^2) )

    The mean and variance are computed by calls to DM_cosmic_mean and
    get_DM_variance respectively, and depend on the feedback-dependent gas power
    spectrum Pgas(k, z) through sigma^2[DM_cosmic]. The resulting PDF is sensitive
    to the efficiency of gas expulsion from halos through its variance, as described
    in the paper.

    Parameters
    ----------
    H0_arr : array_like, shape (nwalkers,)
        Hubble constant values in km/s/Mpc for each walker.
    Ob0_arr : array_like, shape (nwalkers,)
        Baryon density parameter Omega_b for each walker.
    Om0_arr : array_like, shape (nwalkers,)
        Total matter density parameter Omega_m for each walker.
    s8_arr : array_like, shape (nwalkers,)
        sigma_8 for each walker.
    feedback_params_list : list of dict, length nwalkers
        List of feedback parameter dictionaries, one per walker.
    z_obs : array_like, shape (nz,)
        Observed FRB redshift(s) at which to evaluate p(DM_cosmic | z_s).
    DM_obs : array_like, shape (nz, n_DM_host)
        Cosmic DM values at which to evaluate the PDF, for each FRB redshift.
        These are typically the DM_cosmic values obtained by subtracting the host
        and Milky Way halo contributions from the observed extragalactic DM.
    Pk_model : str
        Power spectrum model identifier. Must be "HMcode", "BCEmu1", "BCEmu7",
        or "BCEmu8".

    Returns
    -------
    pdf : np.ndarray, shape (nwalkers, nz, n_DM_host)
        Log-normal probability density p(DM_cosmic | z_s) evaluated at each
        DM_obs value, for each walker and observed FRB redshift.
    """
    H0_arr = np.atleast_1d(H0_arr)
    z_obs = np.atleast_1d(z_obs)
    DM_obs = np.atleast_1d(DM_obs)

    # Vectorized DM mean and variance
    DM_mean = DM_cosmic_mean(Om0_arr, Ob0_arr, H0_arr, s8_arr, feedback_params_list, Pk_model, z_obs)  # (nwalkers, nz)
    DM_var = get_DM_variance(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, z_obs, Pk_model)**2  # (nwalkers, nz)

    # Lognormal parameters
    mu_log = np.log(DM_mean**2 / np.sqrt(DM_mean**2 + DM_var))  # (nwalkers, nz)
    sigma_log = np.sqrt(np.log(1 + DM_var / DM_mean**2))         # (nwalkers, nz)
    
    # Reshape mu_log and sigma_log for broadcasting
    mu_log = mu_log[:, :, np.newaxis]       # (nwalkers, nz, 1)
    sigma_log = sigma_log[:, :, np.newaxis] # (nwalkers, nz, 1)
    DM_obs = DM_obs[np.newaxis, :, :]  # (1, nz, n_DM_host)

    # Compute lognormal PDF
    pdf = lognorm.pdf(DM_obs, s=sigma_log, scale=np.exp(mu_log))  # (nwalkers, nz, n_DM_obs_flat)
    return pdf

def madau_dickinson_factor(z):
    """
    Computes the Madau-Dickinson star formation rate (SFR) factor as a function
    of redshift, following the fitting formula of Madau & Dickinson (2014).

    This factor describes the redshift evolution of the cosmic star formation rate
    density, and is used to optionally model a redshift evolution of the FRB host
    galaxy DM contribution, DMhost(z). If FRBs preferentially trace star-forming
    galaxies, DMhost is expected to evolve with redshift in a manner tracking the
    cosmic SFR history. The factor is normalized to its z=0 value in likelihood_frb
    to yield a multiplicative correction to the median host DM.

    Parameters
    ----------
    z : float or array_like
        Redshift(s) at which to evaluate the Madau-Dickinson SFR factor.

    Returns
    -------
    sfr_factor : float or np.ndarray
        The Madau-Dickinson SFR factor (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6)
        at the given redshift(s), unnormalized.
    """
    return (1 + z)**2.7 / (1 + ((1 + z)/2.9)**5.6)

def likelihood_frb(DM_obs, z_obs, H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list,
                   mu_host_arr, sigma_host_arr, Pk_model, n_host_bins=400): 
    """
    Computes the total log-likelihood of the observed FRB extragalactic dispersion
    measures given the model parameters, for each walker simultaneously.

    For each FRB, the likelihood is computed by marginalizing over the unknown host
    galaxy DM contribution, DMhost, following equation 1 of the paper:

        p(DM_exgal | z_s) = integral_0^{DM_exgal} p(DM_cosmic | z_s) * p(DMhost | z_s) dDMhost

    where DM_cosmic = DM_exgal - DMhost - dm_halos, with dm_halos = 50 pc/cm^3
    accounting for the Milky Way halo contribution. The host DM distribution
    p(DMhost | z_s) is modeled as a log-normal in the rest frame, converted to
    the observer frame by a factor of (1 + z_s). The cosmic DM distribution
    p(DM_cosmic | z_s) is evaluated via p_cosmic. The marginalization integral
    over DMhost is performed numerically using a per-FRB adaptive grid of n_host_bins
    bins spanning [1e-2, DM_exgal - dm_halos].

    Optionally, a redshift evolution of the median host DM is applied following
    the Madau-Dickinson star formation rate history, controlled by the environment
    variable HOST_Z_EVOLUTION.

    Parameters
    ----------
    DM_obs : array_like, shape (n_FRB,)
        Observed extragalactic dispersion measures DMexgal in pc/cm^3,
        after subtracting the Milky Way ISM contribution.
    z_obs : array_like, shape (n_FRB,)
        Spectroscopic redshifts of the FRB host galaxies.
    H0_arr : array_like, shape (nwalkers,)
        Hubble constant values in cm/s/Mpc (pre-scaled) for each walker.
    Ob0_arr : array_like, shape (nwalkers,)
        Baryon density parameter Omega_b for each walker.
    Om0_arr : array_like, shape (nwalkers,)
        Total matter density parameter Omega_m for each walker.
    s8_arr : array_like, shape (nwalkers,)
        sigma_8 for each walker.
    feedback_params_list : list of dict, length nwalkers
        List of feedback parameter dictionaries, one per walker.
    mu_host_arr : array_like, shape (nwalkers,)
        Log-normal location parameter for the rest-frame host DM distribution,
        i.e. the mean of log(DMhost) for each walker.
    sigma_host_arr : array_like, shape (nwalkers,)
        Log-normal scale parameter for the rest-frame host DM distribution,
        i.e. the standard deviation of log(DMhost) for each walker.
    Pk_model : str
        Power spectrum model identifier. Must be "HMcode", "BCEmu1", "BCEmu7",
        or "BCEmu8".
    n_host_bins : int, optional
        Number of bins used to numerically integrate over the host DM distribution.
        Default is 400, sufficient for FRB samples extending to z ~ 4. For samples
        limited to z_max ~ 1.5 or z_max ~ 3, values of 100 or 200 are sufficient.

    Returns
    -------
    log_likelihood : np.ndarray, shape (nwalkers,)
        Total log-likelihood summed over all FRBs, for each walker.
    """
    # need to tune n_host_bins based on maximum frb redshift in the analysis
    # n_host_bins=100 is sufficient for z_max = 1.5
    # n_host_bins=200 is sufficient for z_max = 3
    # n_host_bins=400 is sufficient for z_max = 4

    DM_obs = np.atleast_1d(DM_obs)
    z_obs = np.atleast_1d(z_obs)

    dm_halos = 50.0  # pc/cm^3

    # For each FRB, build host bin grid from 1 --> max_host_i = max(1, DM_obs[i] - dm_halos)
    max_host = np.maximum(1.0, DM_obs - dm_halos)   # (n_FRB,)
    frac = np.linspace(0.0, 1.0, n_host_bins)      # normalized fraction
    dm_host_obsframes = 1e-2 + (max_host - 1e-2)[:, None] * frac[None, :]   # (n_FRB, n_host_bins)

    # bin width per FRB (uniform per-FRB grid)
    dm_host_bin = (max_host - 1e-2) / n_host_bins   # (n_FRB,)
    dm_host_bin = np.maximum(dm_host_bin, 1e-12)          # avoid zero

    # Host PDF (evaluate in host rest frame if mu_host/sigma_host are in rest frame)
    # Convert to rest-frame: DM_host_rest = DM_host_obs * (1 + z)
    dm_host_rest = dm_host_obsframes[None, :, :] * (1.0 + z_obs)[None, :, None]   # (1, nFRB, n_host_bins)
    # mu_host_arr and sigma_host_arr are per-walker: (nwalkers,)
    
    if os.environ["HOST_Z_EVOLUTION"] == "Yes":
        f_MD = madau_dickinson_factor(z_obs)/madau_dickinson_factor(0)
        log_f_MD = np.log(f_MD)
        p_host_vals = lognorm.pdf(dm_host_rest,
                                  s=sigma_host_arr[:, None, None],
                                  scale=np.exp(mu_host_arr[:, None, None] + log_f_MD[None, :, None])) * (1.0 + z_obs)[None, :, None] 
    else:
        p_host_vals = lognorm.pdf(dm_host_rest,
                                  s=sigma_host_arr[:, None, None],
                                  scale=np.exp(mu_host_arr[:, None, None])) * (1.0 + z_obs)[None, :, None] 
        # (nwalkers, nFRB, n_host_bins)
        

    # Cosmic PDF: (nwalkers, n_FRB, n_host_bins)
    DM_cosmic_vals = (DM_obs[:, None] - dm_host_obsframes - dm_halos)  # (n_FRB, n_host_bins)
    p_cosmic_vals = p_cosmic(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, z_obs, DM_cosmic_vals, Pk_model)
    # Ensure numerical stability
    p_cosmic_vals = np.nan_to_num(p_cosmic_vals, nan=0.0)
    p_cosmic_vals = np.clip(p_cosmic_vals, 1e-300, None)

    # Combine host and cosmic PDFs and integrate over host DM bins
    joint_probability = p_host_vals * p_cosmic_vals * dm_host_bin[None, :, None] # (nwalkers, n_FRB, n_host_bins)
    integral = np.sum(joint_probability, axis=2)  # integrate over host DM bins -> (nwalkers, n_FRB)

    # Total log-likelihood over all FRBs
    log_likelihood = np.sum(np.log(integral), axis=1)  # (nwalkers,)

    return log_likelihood


def log_posterior(params_arr, dm_obs_list, z_list, Pk_model):
    """
    Computes the log-posterior probability for each walker, combining the FRB
    log-likelihood with optional informative priors on the stellar-to-halo mass
    (SHM) relation parameters.

    For each walker, the feedback parameter dictionary is first populated from
    the MCMC parameter array params_arr, with the number of free parameters
    determined by the module-level N_params variable. Parameters not included
    in the free parameter set are held fixed at their fiducial values, read
    from environment variables. Cosmological parameters (H0, Ob0, Om0, s8) are
    always fixed to their Planck18 values from environment variables.

    The function supports the following model configurations:
    - HMcode  : 1 feedback parameter (Tagn).
    - BCEmu1  : 1 feedback parameter (log_Mc).
    - BCEmu4  : 4 feedback parameters (log_Mc, theta_ej, mu_beta, delta).
    - BCEmu5  : 5 feedback parameters (log_Mc, theta_ej, mu_beta, delta, eta).
    - BCEmu7  : 7 feedback parameters (log_Mc, theta_ej, eta_delta, mu_beta,
                gamma, delta, eta).
    - BCEmu8  : 8 feedback parameters (log_Mc, theta_ej, eta_delta, mu_beta,
                gamma, delta, eta, A).

    In all cases, mu_host and sigma_host (the log-normal host DM distribution
    parameters) are also free parameters included in params_arr.

    An optional informative prior on the high-mass end slope of the SHM relation
    (eta) is applied via the environment variable ETA_PRIOR, which can take values:
    - "Chandra" : Prior from joint fit to Chandra X-ray + SDSS stellar mass
                  measurements of galaxy clusters (Kravtsov+2014).
    - "SPT"     : Prior from joint fit to SPT SZ + DES/WISE/Spitzer stellar mass
                  measurements of galaxy clusters (Chiu+2018).
    - "No"      : No informative prior on eta; flat prior only.

    The eta prior is computed as the log-likelihood of the stellar mass fraction
    data given the current model parameters, evaluated in parallel across walkers
    using joblib. This is the physically motivated prior on the SHM relation
    described in Extended Data Fig. 5 of the paper.

    Parameters
    ----------
    params_arr : np.ndarray, shape (nwalkers, n_params_total)
        Array of MCMC parameter values for each walker. The parameter ordering
        depends on Pk_model and N_params, as described above. Always includes
        mu_host and sigma_host as the last two columns.
    dm_obs_list : array_like, shape (n_FRB,)
        Observed extragalactic dispersion measures DMexgal in pc/cm^3.
    z_list : array_like, shape (n_FRB,)
        Spectroscopic redshifts of the FRB host galaxies.
    Pk_model : str
        Power spectrum model identifier. Must be "HMcode" or contain "BCEmu".

    Returns
    -------
    log_post : np.ndarray, shape (nwalkers,)
        Log-posterior probability for each walker, equal to the sum of the
        FRB log-likelihood and any informative log-prior contributions.
    """
    nwalkers = params_arr.shape[0]
    log_post = np.zeros(nwalkers)

    if "HMcode" in Pk_model:
        Tagn_arr, mu_host_arr, sigma_host_arr = params_arr.T
        feedback_params_list = [{"Tagn": t} for t in Tagn_arr]

    elif "BCEmu" in Pk_model:
        feedback_params_list = [
                {
                    "log_Mc": float(os.environ["log_Mc"]), 
                    "theta_ej": float(os.environ["theta_ej"]), 
                    "eta_delta": float(os.environ["eta_delta"]), 
                    "delta": float(os.environ["delta"]), 
                    "mu_beta": float(os.environ["mu_beta"]), 
                    "gamma": float(os.environ["gamma"]), 
                    "eta": float(os.environ["eta"]), 
                    "A": float(os.environ["A"]),
                    "tau": float(os.environ["tau"])
                    } for i in range(nwalkers)]
        
        if N_params==1:
            log_Mc_arr, mu_host_arr, sigma_host_arr = params_arr.T
            for i in range(nwalkers):
                feedback_params_list[i]["log_Mc"] = log_Mc_arr[i]

        elif N_params==4:
            log_Mc_arr, theta_ej_arr, mu_beta_arr, delta_arr, mu_host_arr, sigma_host_arr = params_arr.T
            for i in range(nwalkers):
                feedback_params_list[i]["log_Mc"] = log_Mc_arr[i]
                feedback_params_list[i]["theta_ej"] = theta_ej_arr[i]
                feedback_params_list[i]["mu_beta"] = mu_beta_arr[i]
                feedback_params_list[i]["delta"] = delta_arr[i]

        elif N_params==5:
            log_Mc_arr, theta_ej_arr, mu_beta_arr, delta_arr, eta_arr, mu_host_arr, sigma_host_arr = params_arr.T
            for i in range(nwalkers):
                feedback_params_list[i]["log_Mc"] = log_Mc_arr[i]
                feedback_params_list[i]["theta_ej"] = theta_ej_arr[i]
                feedback_params_list[i]["delta"] = delta_arr[i]
                feedback_params_list[i]["mu_beta"] = mu_beta_arr[i]
                feedback_params_list[i]["eta"] = eta_arr[i]

        elif N_params==7:
            log_Mc_arr, theta_ej_arr, eta_delta_arr, mu_beta_arr, gamma_arr, delta_arr, eta_arr, mu_host_arr, sigma_host_arr = params_arr.T
            for i in range(nwalkers):
                feedback_params_list[i]["log_Mc"] = log_Mc_arr[i]
                feedback_params_list[i]["theta_ej"] = theta_ej_arr[i]
                feedback_params_list[i]["delta"] = delta_arr[i]
                feedback_params_list[i]["gamma"] = gamma_arr[i]
                feedback_params_list[i]["eta"] = eta_arr[i]
                feedback_params_list[i]["mu_beta"] = mu_beta_arr[i]
                feedback_params_list[i]["eta_delta"] = eta_delta_arr[i]
    
        elif N_params==8:
            log_Mc_arr, theta_ej_arr, eta_delta_arr, mu_beta_arr, gamma_arr, delta_arr, eta_arr, A_arr, mu_host_arr, sigma_host_arr = params_arr.T
            for i in range(nwalkers):
                feedback_params_list[i]["log_Mc"] = log_Mc_arr[i]
                feedback_params_list[i]["theta_ej"] = theta_ej_arr[i]
                feedback_params_list[i]["delta"] = delta_arr[i]
                feedback_params_list[i]["gamma"] = gamma_arr[i]
                feedback_params_list[i]["eta"] = eta_arr[i]
                feedback_params_list[i]["mu_beta"] = mu_beta_arr[i]
                feedback_params_list[i]["eta_delta"] = eta_delta_arr[i]
                feedback_params_list[i]["A"] = A_arr[i]
        
        else:
            print("Unknown number of parameters!")
    
    else:
        raise ValueError("Unknown Pk_model!")

    Ob0_arr = np.array([float(os.environ["Ob0"]) for i in range(nwalkers)])
    Om0_arr = np.array([float(os.environ["Om0"]) for i in range(nwalkers)])
    H0_arr = np.array([float(os.environ["H0"]) for i in range(nwalkers)])
    s8_arr = np.array([float(os.environ["s8"]) for i in range(nwalkers)])
    
    # Apply H0 scaling as in original
    H0_arr_scaled = H0_arr * (1e5 / Mpc)

    # Vectorized likelihood
    likelihood_arr = likelihood_frb(dm_obs_list, z_list, H0_arr_scaled, Ob0_arr, Om0_arr, s8_arr, feedback_params_list, mu_host_arr, sigma_host_arr, Pk_model)

    # Prior on eta for BCEmu models
    def f_star_eta(Mass, h0, ob0, om0, s8, log_Mc, theta_ej, eta_delta, mu_beta, gamma, delta, eta, A, tau): # Mass is in Msun
        cosmo = ccl.Cosmology(Omega_c = om0-ob0, Omega_b = ob0, h = h0/100, sigma8 = s8, n_s = 0.9665, matter_power_spectrum='linear')

        par = dict(theta_ej = theta_ej, theta_co = 0.1, M_c = (10**log_Mc)/cosmo.cosmo.params.h, 
                   mu_beta = mu_beta, eta = eta, eta_delta = eta_delta, tau = tau, tau_delta = 0, A = A,
                   M1 = 2.5e11/cosmo.cosmo.params.h, epsilon_h = 0.015, a = 0.3, n = 2, 
                   epsilon = 4, p = 0.3, q = 0.707, gamma = gamma, delta = delta)
        mdef  = ccl.halos.massdef.MassDef500c
        CGA = bfg.Profiles.Schneider19.Stars(**par, mass_def = mdef, c_M_relation=ccl.halos.concentration.ConcentrationIshiyama21)
        SAT = bfg.Profiles.Schneider19.SatelliteStars(**par, mass_def = mdef, c_M_relation=ccl.halos.concentration.ConcentrationIshiyama21)
    
        R500c = mdef.get_radius(cosmo, Mass, 1) / (1+0) # Mpc
        out   = np.zeros_like(Mass) + -99
        for i in range(Mass.size):
            r = np.geomspace(R500c[i] * 1e-3, R500c[i] * 1, 1000) # Mpc
            m_cga = np.trapz(4 * np.pi * r**2 * CGA.real(cosmo, r, Mass[i], 1), x = r) # Msun
            m_sat = np.trapz(4 * np.pi * r**2 * SAT.real(cosmo, r, Mass[i], 1), x = r) # Msun
            out[i] = (m_cga+m_sat)/Mass[i] 
        return out

    def single_loglike(x, y, yerr, xerr, h0, ob0, om0, s8,
                    log_Mc, theta_ej, eta_delta, mu_beta, gamma, delta, eta, A, tau):
        model = f_star_eta(x, h0, ob0, om0, s8, log_Mc, theta_ej, eta_delta, mu_beta, gamma, delta, eta, A, tau)
        return -0.5 * np.sum(((y - model)**2 / (yerr**2)) + np.log(2 * np.pi * (yerr**2)))

    def log_likelihood_eta(x, y, yerr, xerr, n_jobs=-1):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(single_loglike)(
                    x, y, yerr, xerr,
                    H0_arr[i], Ob0_arr[i], Om0_arr[i], s8_arr[i], 
                    feedback_params_list[i]["log_Mc"], feedback_params_list[i]["theta_ej"], 
                    feedback_params_list[i]["eta_delta"], feedback_params_list[i]["mu_beta"],  
                    feedback_params_list[i]["gamma"], feedback_params_list[i]["delta"], 
                    feedback_params_list[i]["eta"], feedback_params_list[i]["A"], feedback_params_list[i]["tau"])
                for i in range(len(feedback_params_list))
                )
        return np.array(results)

    # offset_Chabrier_to_Salpeter = +0.24
    # offset_Chabrier_to_Kroupa = +0.05
    
    chandra_mass_med = np.array([7.32588735e+13, 5.28314263e+14])
    chandra_mass_err = np.array([1.16528104e+14, 9.49432406e+14])
    chandra_mfrac_med = np.array([0.025479412434924364, 0.015226384336272327]) # *(10**offset_Chabrier_to_Kroupa)
    chandra_mfrac_err = np.array([0.00805755, 0.00713466]) # *(10**offset_Chabrier_to_Kroupa)

    spt_mass_med = np.array([4.48825144e+14, 7.25423285e+14, 1.13384887e+15])
    spt_mass_err = np.array([9.15864690e+13, 1.36314208e+14, 1.95178992e+14])
    spt_mfrac_med = np.array([0.00960486, 0.00809319, 0.00768941]) # *(10**offset_Chabrier_to_Kroupa)
    spt_mfrac_err = np.array([0.00323631, 0.00276477, 0.00122413]) # *(10**offset_Chabrier_to_Kroupa)

    # Combine log-likelihood and log-priors
    if os.environ["ETA_PRIOR"] == "Chandra":
        # eta_mean, eta_sigma = 0.23, 0.07 # Chandra prior (bfg)
        # eta_arr = np.array([feedback_params_list[i]["eta"] for i in range(len(feedback_params_list))])
        # logprior_eta = -0.5*((eta_arr - eta_mean)/eta_sigma)**2 - np.log(np.sqrt(2*np.pi)*eta_sigma)
        
        # A_arr = np.array([feedback_params_list[i]["A"] for i in range(len(feedback_params_list))])
        # logprior_A = -0.5*((A_arr - float(os.environ["A"]))/0.0001)**2 - np.log(np.sqrt(2*np.pi)*0.0001)
        logprior_eta = log_likelihood_eta(chandra_mass_med, chandra_mfrac_med, chandra_mfrac_err, chandra_mass_err)
        log_post = likelihood_arr + logprior_eta # + logprior_A
    elif os.environ["ETA_PRIOR"] == "SPT":
        # eta_mean, eta_sigma = 0.28, 0.02 # SPT prior (bfg)
        # eta_arr = np.array([feedback_params_list[i]["eta"] for i in range(len(feedback_params_list))])
        # logprior_eta = -0.5*((eta_arr - eta_mean)/eta_sigma)**2 - np.log(np.sqrt(2*np.pi)*eta_sigma)

        # A_arr = np.array([feedback_params_list[i]["A"] for i in range(len(feedback_params_list))])
        # logprior_A = -0.5*((A_arr - float(os.environ["A"]))/0.0001)**2 - np.log(np.sqrt(2*np.pi)*0.0001)
        logprior_eta = log_likelihood_eta(spt_mass_med, spt_mfrac_med, spt_mfrac_err, spt_mass_err)
        log_post = likelihood_arr + logprior_eta # + logprior_A
    else:
        log_post = likelihood_arr
    
    return log_post


if "HMcode" in Pk_model:
    # Tagn, mu_host, sigma_host
    mins = np.array([7, 4, 0.2])
    maxs = np.array([8.4, 6, 2])

elif "BCEmu1" in Pk_model:
    # log_Mc, mu_host, sigma_host
    mins = np.array([11, 4, 0.2])
    maxs = np.array([16, 6, 1])

elif ("BCEmu7" in Pk_model) and (N_params==4):
    # log_Mc, theta_ej, mu_beta, delta, mu_host, sigma_host
    mins = np.array([11, 2, 0, 3, 4, 0.2])
    maxs = np.array([16, 8, 2, 11, 6, 1])

elif ("BCEmu7" in Pk_model) and (N_params==5):
    # log_Mc, theta_ej, mu_beta, delta, eta, mu_host, sigma_host
    if os.environ["ETA_PRIOR"] == "No":
        mins = np.array([11, 2, 0, 3, 0.15, 4, 0.2])
        maxs = np.array([16, 8, 2, 11, 0.3, 6, 1])
    else:
        mins = np.array([11, 2, 0, 3, 0.05, 4, 0.2])
        maxs = np.array([16, 8, 2, 11, 0.4, 6, 1])

elif ("BCEmu7" in Pk_model) and (N_params==7):
    # log_Mc, theta_ej, eta_delta, mu_beta, gamma, delta, eta, mu_host, sigma_host
    mins = np.array([11, 2, 0.05, 0, 1, 3, 0.15, 4, 0.2])
    maxs = np.array([16, 8, 0.4, 2, 4, 11, 0.3, 6, 1])

elif ("BCEmu8" in Pk_model) and (N_params==6):
    # log_Mc, theta_ej, mu_beta, delta, eta, A, mu_host, sigma_host
    mins = np.array([11, 2, 0, 3, 0.05, 0, 4, 0.2])
    maxs = np.array([16, 8, 2, 11, 0.4, 0.2, 6, 1])
                    
else:
    print("Unknown Pk_model!")


def log_prior(theta_arr, mins, maxs):
    """
    Computes the log-prior probability for each walker under a uniform (flat)
    prior over the allowed parameter ranges.

    For each walker, the prior is 0 (log-probability = 0.0) if all parameters
    lie within their respective bounds [mins, maxs], and -inf otherwise. The
    parameter bounds are defined at the module level for each model configuration
    (HMcode, BCEmu1, BCEmu7, BCEmu8) and passed in as mins and maxs.

    Parameters
    ----------
    theta_arr : array_like, shape (nwalkers, n_params)
        Array of MCMC parameter values for each walker.
    mins : array_like, shape (n_params,)
        Lower bounds of the uniform prior for each parameter.
    maxs : array_like, shape (n_params,)
        Upper bounds of the uniform prior for each parameter.

    Returns
    -------
    lp_arr : np.ndarray, shape (nwalkers,)
        Log-prior probability for each walker. Returns 0.0 if all parameters
        are within bounds, -inf otherwise.
    """
    theta_arr = np.atleast_2d(theta_arr)
    valid = np.all((theta_arr >= mins) & (theta_arr <= maxs), axis=1)
    lp_arr = np.where(valid, 0.0, -np.inf)
    return lp_arr


def log_probability(theta_arr, dm_obs_list, z_list, Pk_model, mins, maxs):
    """
    Computes the log-posterior probability for each walker, combining the
    flat log-prior with the log-posterior from log_posterior. This is the
    top-level function called by the MCMC sampler (emcee) at each step.

    The function first evaluates the flat log-prior via log_prior. Walkers
    that fall outside the prior bounds (lp = -inf) are immediately assigned
    log-probability of -inf without evaluating the likelihood, saving
    computational cost. For walkers within the prior bounds, the log-posterior
    is computed via log_posterior and added to the log-prior.

    Parameters
    ----------
    theta_arr : array_like, shape (nwalkers, n_params)
        Array of MCMC parameter values for each walker. The parameter ordering
        depends on Pk_model and N_params as described in log_posterior.
    dm_obs_list : array_like, shape (n_FRB,)
        Observed extragalactic dispersion measures DMexgal in pc/cm^3,
        after subtracting the Milky Way ISM contribution.
    z_list : array_like, shape (n_FRB,)
        Spectroscopic redshifts of the FRB host galaxies.
    Pk_model : str
        Power spectrum model identifier. Must be "HMcode" or contain "BCEmu".
    mins : array_like, shape (n_params,)
        Lower bounds of the uniform prior for each parameter.
    maxs : array_like, shape (n_params,)
        Upper bounds of the uniform prior for each parameter.

    Returns
    -------
    log_prob : np.ndarray, shape (nwalkers,)
        Log-posterior probability for each walker, equal to log_prior +
        log_posterior for walkers within the prior bounds, and -inf for
        walkers outside the prior bounds.
    """
    theta_arr = np.atleast_2d(theta_arr)
    lp_arr = log_prior(theta_arr, mins, maxs)
    
    # If prior is -inf, posterior is -inf
    log_prob = np.copy(lp_arr)
    valid_mask = np.isfinite(lp_arr)
    if np.any(valid_mask):
        log_prob[valid_mask] += log_posterior(theta_arr[valid_mask, :], dm_obs_list, z_list, Pk_model)
    
    return log_prob  # (nwalkers,)

