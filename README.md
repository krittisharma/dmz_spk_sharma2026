<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/dab3a6ee-92f6-4731-81cf-17225c35698e" />

# FRB DM-z Analysis

## Installation Requirements
The only specific requirements for running these scripts are [BaryonForge](https://github.com/DhayaaAnbajagane/BaryonForge) and [CCL](https://ccl.readthedocs.io/en/latest/).

## Description

**Data Directory**
- `data/f_diffuse/rhoH2_rhoHI.csv`: Redshift evolution of $\rho_\mathrm{H2}/\rho_\mathrm{HI}$, as adopted from [Bollo+2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...695A.163B/abstract) and [Peroux+2020](https://ui.adsabs.harvard.edu/abs/2020ARA%26A..58..363P/abstract).
- `data/forecasts` includes
    - `precomputed_data_vectors` ( $C^{\gamma\gamma}$, $C^{\gamma g}$, $C^{gg}$, $C^{\gamma \mathcal{D}}$, $C^{g \mathcal{D}}$, $C^{\mathcal{DD}}$ ) for variations of feedback, cosmology and maximum halo mass, used to conduct fisher forecasts and compute sensitivity for LSST and DSA-2000.
    - `precomputed_power_spectra` ( $P_\mathrm{gg}(z)$, $P_\mathrm{gm}(z)$, $P_\mathrm{mm}(z)$ ) for variations of feedback and cosmology.
    - `catalogue_so_simulated_0.npy` SNR forecasts for Simons Observatory tSZ galaxy clusters from [Zubeldia+2024](https://ui.adsabs.harvard.edu/abs/2024JCAP...11..018Z/abstract).
    - `cov_blocks_1e4FRBs.npz`, `cov_blocks_1e5FRBs.npz`: Full covariance matrix for LSST+DSA-2000 6x2-point analysis assuming $10^4$ and $10^5$ FRBs.
    - `lens_LSSTY1.nz`, `source_LSSTY1.nz`: Lens and source redshift distributions for LSST-Y1, adapted from [CosmoLike](https://github.com/CosmoLike/cocoa_baryons_lssty1/).
    - `sensitivity_DSA2000FRBs_DMvar_MZ_1h.npy`: $\log M_{200}-z$ sensitivity forecast for a DM$-z$ analysis of $10^4$ DSA-2000 FRBs.
    - `sensitivity_LSST_DSA2000_DMxgalaxy12_MZ_lmax1e4_1h.npy`: $\log M_{200}-z$ sensitivity forecast for a DM-galaxy cross-correlation of $10^4$ DSA-2000 FRBs and tomography bins 1-2 of LSST-Y1.
    - `sensitivity_LSST_DSA2000_DMxgalaxy345_MZ_lmax1e4_1h.npy`: $\log M_{200}-z$ sensitivity forecast for a DM-galaxy cross-correlation of $10^4$ DSA-2000 FRBs and tomography bins 3-5 of LSST-Y1.
    - `sensitivity_LSST_cosmic_shear_MZ_lmax1e4_1h.npy`: $\log M_{200}-z$ sensitivity forecast for LSST-Y1 cosmic shear.
- `data/frb_results` includes
    - `frb_sample.csv`: The FRB sample.
    - MCMC posterior samples for BCEmu1 (`emcee_BCEmu1_extended.pkl`), BCEmu4 (`emcee_BCEmu4_extended.pkl`), BCEmu5 with $\eta$-prior from Chandra data (`emcee_BCEmu5_extended_Chandraprior.pkl`), BCEmu5 with $\eta$-prior from SPT data (`emcee_BCEmu5_extended_SPTprior.pkl`), BCEmu7 (`emcee_BCEmu7_extended.pkl`), BCEmu7 with $\eta$-prior from Chandra data (`emcee_BCEmu7_extended_Chandraprior.pkl`), HMcode excluding the CHIME sub-arcmin to arcmin-scale localizations (`emcee_HMcode.pkl`), HMcode (`emcee_HMcode_extended.pkl`) and HMcode assuming redshift evolution of host DM contribution (`emcee_HMcode_extended_MDevolution.pkl`).
    - 68% confidence interval constraints on matter power spectrum suppression from BCEmu1 (`spk_BCEmu1_extended.npy`), BCEmu4 (`spk_BCEmu4_extended.npy`), BCEmu5 with $\eta$-prior from Chandra data (`spk_BCEmu5_extended_Chandraprior.npy`), BCEmu5 with $\eta$-prior from SPT data (`spk_BCEmu5_extended_SPTprior.npy`), BCEmu7 with $\eta$-prior from Chandra data (`spk_BCEmu7_extended_Chandraprior.npy`), HMcode (`spk_HMcode_extended.npy`)
    -  Constraints from the first CPC mode: Posterior samples from BCEmu5 (`spk_BCEmu5_posterior_samples_feedbackonly_extended_Chandraprior.npy`), prior range of BCEmu7 (`spk_BCEmu7_prior_1sigma_vary_all.npy`) and prior range for first CPC mode (`spk_BCEmu7_prior_1sigma_vary_all_but_logMc_thetaej_mubeta_delta.npy`)
    - `extreme_scenario_test`: MCMC posterior samples and 68% confidence interval constraints on matter power spectrum suppression from extreme feedback scenario tests with mock FRBs.
    - `jacknife_sampling_tests`: 68% confidence interval constraints on matter power spectrum suppression from 109 variants of leave-one-out validation tests.
    - `prior`: The full prior volume for BCEmu1, BCEmu4, BCEmu5, BCEmu7 and HMcode.
    - `shm_relation`: MCMC posterior samples, 68% confidence interval constraints on matter power spectrum suppression and f_diffuse/f_star samples for Chabrier and Salpeter IMF variants of the normalization of stellar-to-halo mass relation.
    - `vary_feedback_params_cosmology`: 68% confidence interval constraints on matter power spectrum suppression for variations of feedback and cosmology parameters.
