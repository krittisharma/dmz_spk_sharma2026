import numpy as np
import pickle
import emcee
import warnings
warnings.filterwarnings("ignore")
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="MCMC configuration")
    parser.add_argument("--Pk_model", type=str, 
                        help="Power spectrum model")
    parser.add_argument("--method", type=str, default="emulators", 
                        help="Method to use")
    parser.add_argument("--N_frb", type=int, 
                        help="Number of FRBs")
    parser.add_argument("--dir_path", type=str,
                        help="Directory path to use")
    parser.add_argument("--sample", type=str, 
                        help="FRB sample")
    parser.add_argument("--sample_extended", type=str,
                        help="FRB sample extended?")
    parser.add_argument("--N_params", type=int,
                        help="Number of free parameters")
    parser.add_argument("--eta_prior", type=str,
                        help="Include eta prior?", default="No")
    parser.add_argument("--host_z_evol", type=str,
                    help="Include host DM redshift evolution?", default="No")
    return parser.parse_args()
args = parse_args()

os.environ["SAMPLE"] = args.sample
os.environ["PK_MODEL"] = args.Pk_model
os.environ["N_FRB"]    = str(args.N_frb)
os.environ["DIR_PATH"] = args.dir_path
os.environ["METHOD"] = args.method
os.environ["N_PARAMS"] = str(args.N_params)
os.environ["SAMPLE_EXTENDED"] = args.sample_extended
os.environ["ETA_PRIOR"] = args.eta_prior
os.environ["HOST_Z_EVOLUTION"] = args.host_z_evol

from astropy.cosmology import Planck18 as cosmo
os.environ["Ob0"] = str(cosmo.Ob0) # +/-(3*0.0008)
os.environ["Om0"] = str(cosmo.Om0) # 0.293; +/-(3*0.0056)
os.environ["H0"] = str(cosmo.H0.value) # +/-(3*0.42)
os.environ["s8"] = str(0.8111) # 0.764; +/-(3*0.0060)

os.environ["log_Mc"] = str(14)
os.environ["theta_ej"] = str(3.5)
os.environ["eta_delta"] = str(0.2) # 0.1, 0.3
os.environ["delta"] = str(7)
os.environ["mu_beta"] = str(1)
os.environ["gamma"] = str(2.5) # 2, 3
os.environ["eta"] = str(0.2)

offset_Chabrier_to_Salpeter = +0.24
offset_Chabrier_to_Kroupa = +0.05

os.environ["A"] = str((0.055/2)) # *(10**offset_Chabrier_to_Salpeter))
os.environ["tau"] = str(-1)

from mcmc_funcs import *
if args.sample_extended == "Yes":
    suffix = "_extended"
else:
    suffix = ""
if "extremefeedback" in args.sample:
    suffix += "_extremefeedback"
elif "nofeedback" in args.sample:
    suffix += "_nofeedback"
else:
    suffix += ""

# suffix += "_vary_gamma_high"
# suffix += "_seleff"
# suffix += "_nofeedback"
# suffix += "_DEScosmology"

if args.eta_prior != "No":
    suffix += "_{}prior".format(args.eta_prior)
if args.host_z_evol == "Yes":
    suffix += "_MDevolution"
os.environ["SUFFIX"] = suffix

if "BCEmu" in args.Pk_model:
    mcmc_file = '{}/emcee_{}{}{}.pkl'.format(args.dir_path, args.Pk_model[:5], args.N_params, suffix)
else:
    mcmc_file = '{}/emcee_{}{}.pkl'.format(args.dir_path, args.Pk_model[:6], suffix)
print(mcmc_file)

if args.Pk_model == "HMcode":
    map_estimate = np.array([7.8, 5, 0.5])

elif "BCEmu1" in args.Pk_model:
    map_estimate = np.array([14, 5, 0.5])

elif args.Pk_model == "BCEmu7_high_z" and args.N_params == 7:
    map_estimate = np.array([14, 3.5, 0.2, 1, 2.5, 8, 0.2, 5, 0.5])

elif args.Pk_model == "BCEmu7_high_z" and args.N_params == 5:
    # log_Mc, theta_ej, mu_beta, delta, eta
    map_estimate = np.array([14, 3.5, 1, 8, 0.2, 5, 0.5])

elif args.Pk_model == "BCEmu7_high_z" and args.N_params == 4:
    # log_Mc, theta_ej, mu_beta, delta
    map_estimate = np.array([14, 3.5, 1, 7, 5, 0.5])

elif args.Pk_model == "BCEmu8_high_z" and args.N_params == 6:
    # log_Mc, theta_ej, mu_beta, delta, eta, A, mu_host, sigma_host
    map_estimate = np.array([14, 3.5, 1, 7, 0.2, 0.055/2, 5, 0.5])

else:
    print("Unknown Pk_model provided.")

if __name__ == '__main__':
    ndim = len(map_estimate)
    nwalkers = 4*ndim
    burn = 500
    steps = 2000

    p0_x = np.random.uniform(low=mins, high=maxs, size=(nwalkers, ndim))

    # First stage: burn-in with StretchMove only
    moves_burn = [(emcee.moves.StretchMove(a=2.0), 1.0)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
            args=(dm_obs_list, z_list, args.Pk_model, mins, maxs),
            moves=moves_burn, vectorize=True)
    state = sampler.run_mcmc(p0_x, burn, progress=True)
    sampler.reset()
    
    # Second stage: production run with Stretch+KDE
    moves_prod = [(emcee.moves.StretchMove(a=2.0), 0.6),
            (emcee.moves.KDEMove(), 0.4)]
    sampler.moves = moves_prod
    sampler.run_mcmc(state, steps, progress=True)

    results = {
        "chain": sampler.get_chain(),
        "log_prob": sampler.get_log_prob(),
        "acceptance_fraction": sampler.acceptance_fraction,
    }

    with open(mcmc_file, "wb") as f:
        pickle.dump(results, f)

    samples = results["chain"]
    samples = samples.reshape(-1, samples.shape[-1])

    Ob0_arr = np.array([float(os.environ["Ob0"]) for _ in range(len(samples))])
    Om0_arr = np.array([float(os.environ["Om0"]) for _ in range(len(samples))])
    H0_arr = np.array([float(os.environ["H0"]) for _ in range(len(samples))])*(1e5/Mpc)
    s8_arr = np.array([float(os.environ["s8"]) for _ in range(len(samples))])

    if args.Pk_model == "HMcode":
        SPk_HMcode(H0_arr, Ob0_arr, Om0_arr, s8_arr, samples[:, 0])

    elif "BCEmu" in args.Pk_model:
        feedback_params_list = [{
            "log_Mc": float(os.environ["log_Mc"]), 
            "theta_ej": float(os.environ["theta_ej"]),
            "eta_delta": float(os.environ["eta_delta"]),
            "delta": float(os.environ["delta"]),
            "mu_beta": float(os.environ["mu_beta"]),
            "gamma": float(os.environ["gamma"]),
            "eta": float(os.environ["eta"]),
            "A": float(os.environ["A"]),
            "tau": float(os.environ["tau"])
            } for _ in range(len(samples))]
        if args.N_params==1:
            for i in range(len(samples)):
                feedback_params_list[i]["log_Mc"] = samples[i, 0]

        elif args.N_params==4:
            for i in range(len(samples)):
                # log_Mc, theta_ej, mu_beta, delta
                feedback_params_list[i]["log_Mc"] = samples[i, 0]
                feedback_params_list[i]["theta_ej"] = samples[i, 1]
                feedback_params_list[i]["mu_beta"] = samples[i, 2]
                feedback_params_list[i]["delta"] = samples[i, 3]

        elif args.N_params==5:
            for i in range(len(samples)):
                feedback_params_list[i]["log_Mc"] = samples[i, 0]
                feedback_params_list[i]["theta_ej"] = samples[i, 1]
                # feedback_params_list[i]["gamma"] = samples[i, 2]
                feedback_params_list[i]["mu_beta"] = samples[i, 2]
                feedback_params_list[i]["delta"] = samples[i, 3]
                feedback_params_list[i]["eta"] = samples[i, 4]

        elif (args.N_params==6) and args.Pk_model == "BCEmu8_high_z":
            for i in range(len(samples)):
                # log_Mc, theta_ej, mu_beta, delta, eta, A, mu_host, sigma_host
                feedback_params_list[i]["log_Mc"] = samples[i, 0]
                feedback_params_list[i]["theta_ej"] = samples[i, 1]
                feedback_params_list[i]["mu_beta"] = samples[i, 2]
                feedback_params_list[i]["delta"] = samples[i, 3]
                feedback_params_list[i]["eta"] = samples[i, 4]
                feedback_params_list[i]["A"] = samples[i, 5]
        

        elif args.N_params==7:
            for i in range(len(samples)):
                feedback_params_list[i]["log_Mc"] = samples[i, 0]
                feedback_params_list[i]["theta_ej"] = samples[i, 1]
                feedback_params_list[i]["eta_delta"] = samples[i, 2]
                feedback_params_list[i]["mu_beta"] = samples[i, 3]
                feedback_params_list[i]["gamma"] = samples[i, 4]
                feedback_params_list[i]["delta"] = samples[i, 5]
                feedback_params_list[i]["eta"] = samples[i, 6]
        
        else:
            print("Unknown number of parameters!")

        SPk_BCEmu(H0_arr, Ob0_arr, Om0_arr, s8_arr, feedback_params_list)

