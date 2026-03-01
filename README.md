<img width="900" height="500" alt="image" src="https://github.com/user-attachments/assets/dab3a6ee-92f6-4731-81cf-17225c35698e" />


# FRB DM-z Analysis

This repository hosts code and data associated with the publication Sharma et al. (2026) on the measurement of matter power spectrum suppression using fast radio bursts. It is a reproduction package of the DM-z analysis and figures in that work.


## Installation

**Python Dependencies**: Our codebase uses the standard Python scientific stack. The only specific requirements for running these scripts are [BaryonForge](https://github.com/DhayaaAnbajagane/BaryonForge) and [CCL](https://ccl.readthedocs.io/en/latest/). Installation procedures using environment files are described below.

**OS Requirements**: The jupyter notebooks contained in this repository were run on `macOS: Sonoma 14.5` and the MCMC inference was conducted on `linux: Red Hat Enterprise Linux 9.3 (Plow)`. The jupyter notebooks require a standard computer with enough RAM to support the in-memory operations. The recommended resources for MCMC inference on linux include a single computer node with 12 CPU cores and 192 GB memory. In principle, MCMC pipeline is also suitable for running on a standard computer with longer run times without parallelization on an HPC.

**Installation/Python Environment**: Upon cloning the git repository using `git clone https://github.com/krittisharma/dmz_spk_sharma2026`, the user can set up the environments for the two aforementioned OS, as follows. This process should not take more than 10 minutes.

MacOS environment:

`conda env create -f environment_mac.yml`

`conda activate dmz_spk_sharma2026`

`pip install git+https://github.com/DhayaaAnbajagane/BaryonForge.git`

`pip install jax==0.4.23 jaxlib==0.4.23 ml_dtypes==0.2.0`

Linux environment:

`conda env create -f environment_linux.yml`

**Demo**: The usage of our scripts is very well described in the jupyter notebooks, python scripts and slurm job examples, which reproduce all the figures and data presented in our work.


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
- `data/hydrosims_spk` includes 68% confidence interval constraints on matter power spectrum suppression from a range of different hydrodynamical simulations: [BAHAMAS](https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.2936M/abstract), [SIMBA](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2827D/abstract), [FLAMINGO](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.4978S/abstract), [IllustrisTNG](https://ui.adsabs.harvard.edu/abs/2019ComAC...6....2N/abstract), [Illustris](https://ui.adsabs.harvard.edu/abs/2015A%26C....13...12N/abstract) and [OWLS-AGN](https://ui.adsabs.harvard.edu/abs/2010MNRAS.406..822M/abstract).
- `data/literature_constraints` includes
    - Gas fraction measurements from [Akino+2022](https://ui.adsabs.harvard.edu/abs/2022PASJ...74..175A/abstract)  (`Akino_pre-Erosita_gasfrac.csv`, `Akino_pre-Erosita_gasfrac_minus.csv`, `Akino_pre-Erosita_gasfrac_plus.csv`), [Bigwood+2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.534..655B/abstract)  (`Bigwood_gasfrac.txt`), [Pandey+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250607432P/abstract) (`Pandey_DESxACT_rho.pkl`), [Popesso+2024](https://ui.adsabs.harvard.edu/abs/2024arXiv241116555P/abstract) (`Popesso_gasfrac_err.csv`, `Popesso_gasfrac_mean.csv`) and [Siegel+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250910455S/abstract) (`Siegel_EROSITA_fgas_high.csv`, `Siegel_EROSITA_fgas_low.csv`, `Siegel_preEROSITA_fgas_high.csv`, `Siegel_preEROSITA_fgas_low.csv`).
    - 68% confidence interval constraints on matter power spectrum suppression (`combined_Sk_k_z0.hdf5`) from [Wright+2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...703A.158W/abstract), [Arico+2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...678A.109A/abstract), [Chen+2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.5340C/abstract), [Terasawa+2024](https://ui.adsabs.harvard.edu/abs/2025PhRvD.111f3509T/abstract), [Anbajagane+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250903582A/abstract), [Xu+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv251025596X/abstract), [Dalal+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250704476D/abstract), [Pandey+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250607432P/abstract), [Bigwood+2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.534..655B/abstract) and [Siegel+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv251202954S/abstract).
- `data/sensitivity_analysis` includes
    - Sample halo mass and redshift distributions for eRASS1 clusters and ACT stacks on DESI BGS and LRG samples from [Siegel+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250910455S/abstract) (`siegel2025`)
    - DES-Y3 covariance matrix (`DESY3_cov_shear.txt`), data vectors (`DESY3_shear.txt`), optimized mask (`DESY3_shear_optimized.mask`) and source redshift distribution (`source_DESY3.txt`) adapted from [Amon+2022](https://ui.adsabs.harvard.edu/abs/2022PhRvD.105b3514A/abstract)
    - DES-Y3 x ACT compton y-map covariance matrix (`DESxACT_gty_xip_xim_DV_ilc_SZ_yy_maskedPS.pk`) adapted from [Pandey+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250607432P/abstract)
    - `YM_sensitivity.npy`: $\log M_{200}-z$ sensitivity of tSZ Y-M relation from [Dalal+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250704476D/abstract)
    - `power_spectra_object_currentFRBs_1h.npy` and `power_spectra_object_DSA2000FRBs_1h.npy`: Precomputed data vectors for DM-z sensitivity calculation of current FRB sample and expected sample from DSA.
    - `power_spectra_object_bins*_1h.npy`: Precomputed data vectors for $\xi_\pm$ and $\xi^{\gamma y}$ sensitivity calculation.
    - `sensitivity_DES_ACT_xi_MK.npy`: $\log M_{200}-k$ sensitivity of $\xi^{\gamma y}$ from DES-Y3 x ACT compton y-map cross-correlation.
    - `sensitivity_DES_ACT_xi_MZ.npy`: $\log M_{200}-z$ sensitivity of $\xi^{\gamma y}$ from DES-Y3 x ACT compton y-map cross-correlation.
    - `sensitivity_DES_xi_pm_MK.npy`: $\log M_{200}-k$ sensitivity of $\xi_\pm$ from DES-Y3.
    - `sensitivity_DES_xi_pm_MZ.npy`: $\log M_{200}-z$ sensitivity of $\xi_\pm$ from DES-Y3.
    - `sensitivity_currentFRBs_DMvar_MK.npy`: $\log M_{200}-k$ sensitivity of FRB DM-z analysis with current FRB sample.
    - `sensitivity_currentFRBs_DMvar_MZ.npy`: $\log M_{200}-z$ sensitivity of FRB DM-z analysis with current FRB sample.
- `data/shm_relation` includes halo stellar mass fraction measurements from [Chiu+2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.3072C/abstract).


**Emulators**
- `BCEmu1_high_z`, `BCEmu7_high_z`, `BCEmu8_high_z`, `HMcode`
- `DMvar_emulator_*.h5`: Pretrained emulator for predicting FRB DM variance using the `*` baryonic feedback model.
- `SPk_emulator_*.h5`: Pretrained emulator for predicting the matter power spectrum suppression for `*` baryonic feedback model.
- `param_grid_*.csv`: Parameter grid used to train the emulators, including ranges for feedback and cosmological parameters.
- `pca_objects_*.pkl`: PCA decomposition objects for dimensionality reduction of the training outputs in the emulator pipeline.
- `scaler_y_dm_*.joblib`: Scaling object to normalize DM and related outputs before training or prediction with the `*` emulator.


**Scripts**
- `mcmc_funcs.py`: Methods to model the cosmic DM contribution to FRBs using cosmological halo-model frameworks such as HMcode and BCEmu. It computes the mean and variance of the intergalactic DM distribution, incorporating baryonic feedback and diffuse gas fractions. The pipeline supports both exact calculations and machine-learning emulators to efficiently evaluate matter power spectrum modifications and gas statistics. It ultimately enables likelihood evaluation of FRB observations for cosmological and baryonic parameter inference.
- `run_mcmc.py`: Performs MCMC parameter inference for FRB DM observations using cosmological baryonic feedback models such as HMcode and BCEmu. It provides a configurable pipeline to sample feedback and host-galaxy parameters under different modeling assumptions and priors. The sampler uses a two-stage burn-in and production strategy with emcee to efficiently explore parameter space and generate posterior chains. The resulting samples are then used to evaluate matter power spectrum modifications for cosmological analysis.
- `sensitivity_funcs.py`: Builds cosmological kernels and halo-model power spectra relevant for FRB, galaxy, weak lensing, and tSZ observables. It provides modular classes to compute background cosmology, redshift distributions, and line-of-sight weighting functions for large-scale structure analyses. Using CCL and halo-model frameworks, it evaluates gas–matter and total matter power spectra across redshift. The outputs enable forward modeling of cross-correlations involving dispersion measure and large-scale structure tracers.
- `slurmjob.sh`: SLURM batch script to run the MCMC pipeline `run_mcmc.py` on a computing cluster. It requests 12 CPU tasks and 192 GB of memory for up to 20 hours. The job environment activates a Conda environment `SPk` and executes the MCMC with the `BCEmu7_high_z` power spectrum model on an extended FRB sample, using 5 free parameters and a Chandra eta prior. Job notifications are sent to specified email address at start, end, and failure.


**Jupyter Notebooks**
- `6x2pt_LSST_DSA_forecast.ipynb`: Fisher forecast for a joint 6x2pt analysis combining LSST Y1 lensing and clustering observables with a DSA-like FRB sample. Computes angular power spectra ($C_\ell$) from pre-computed matter power spectra using redshift distributions for source and lens bins, and models cosmological and baryonic feedback parameters (e.g., $\Omega_m$, $\sigma_8$, $m_\nu$, $w_0$, $w_a$, and $\log M_c$). The pipeline constructs covariance matrices and derives parameter constraints to quantify the improvement from including FRBs in large-scale structure analyses.
- `6x2pt_LSST_DSA_sensitivity_analysis.ipynb`: Performs a sensitivity forecast of the joint LSST + DSA FRB 6x2pt analysis.
- `DES_ACT_sensitivity_calculation.ipynb`: Sensitivity calculation for DES-Y3 cosmic shear and DES-Y3 x ACT compton y-map cross-correlation.
- `compare_sensitivity.ipynb`: Compares our sensitivity calculations for various probes in $\log M_{200}-z$ and $\log M_{200}-k$ space.
- `dmz_sensitivity_analysis.ipynb`: Sensitivity calculation for FRB DM-z analysis for current FRB sample and future DSA-like sample.
- `eta_prior.ipynb`: Imposing observations-informed prior on stellar-to-halo mass relation, specifically the slope on the high mass-end.
- `extreme_scenarios_tests_CPCmodes.ipynb`: (1) Tests the ability of current FRB sample to distinguish between extreme scenarios, such as no feedback vs extremely strong feedback, using mock FRB samples. (2) Computes the information gain from FRBs over the prior volume using KL divergence. (3) Tests robustness to variations of other feedback parameters and cosmology. (4) Visualizes the constraining power from first CPC mode.
- `spk_fgas_profile.ipynb`: (1) Visualizes and compares our FRB constraints on the suppression of matter power spectrum and halo gas mass fractions with literature. (2) Plots the MCMC parameter posterior constraints. (3) Visualizes the results from jackknife resampling.
- `tensiometer_params.ipynb`: Performs CPC analysis.
- `understanding_IMF_impact.ipynb`: Tests the impact of stellar IMF on suppression of matter power spectrum and halo gas mass fraction constraints.


## Example Usage

Our MCMC pipeline is configured to run on an HPC and its runtime is approximately 5-10 hours, depending on the complexity of the model used. Following is an example usage of this script to generate baseline results presented in the main text of our work. 

`python run_mcmc.py --Pk_model BCEmu7_high_z --dir_path data/frb_results --sample frb_sample --N_params 5 --eta_prior Chandra --sample_extended Yes --host_z_evol No`

Here, 
- `Pk_model` is the power spectrum model. Options include `HMcode`, `BCEmu1_high_z`, `BCEmu7_high_z` and `BCEmu8_high_z`.
- `dir_path` is the path to FRB sample and path where results should be saved.
- `sample` is the name of FRB sample file.
- `N_params` is the number of free parameters to use in `BCEmu` model. Options include 1/4/5/6/7.
- `eta_prior` is the choice of stellar-to-halo mass relation prior. Options include `Chandra` and `SPT`.
- `sample_extended` is the flag for FRB data subset to use. Options include `Yes` for extended sample and `No` for fiducial sample.
- `host_z_evol` is the flag to include redshift evolution of host DM contribution consistent with cosmic star-formation rate history of the Universe.

This script should generate the following files: `data/frb_results/emcee_BCEmu5_extended_Chandraprior.pkl`, `data/frb_results/spk_BCEmu5_extended_Chandraprior.npy` and `data/frb_results/spk_BCEmu5_posterior_samples_feedbackonly_extended_Chandraprior.npy`. Using these, `spk_fgas_profile.ipynb` notebook presents the comparison with various literature constraints and produces the following plots:

<img width="2723" height="916" alt="image" src="https://github.com/user-attachments/assets/4ae90cac-3035-41c1-be98-364a0da14bc7" />

<img width="2804" height="1012" alt="image" src="https://github.com/user-attachments/assets/058df25c-1c83-4431-b8c8-8b05bed7f183" />


## License

This project is covered under the Apache 2.0 License.
