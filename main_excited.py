#!/usr/bin/env python3
"""
CLI to process a single molecule.
"""
import copy
import os
import sys
import numpy as np
from jax.config import config as jax_config
from typing import Any, Tuple, Dict
from deeperwin.configuration import Configuration
import haiku as hk

def _setup_environment(raw_config: Dict, config: Configuration) -> None:
    # Set environment variable to control jax behaviour before importing jax
    if config.computation.disable_tensor_cores:
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    if config.computation.force_device_count and config.computation.n_local_devices:
        os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={config.computation.n_local_devices}'
    
    # update jax config
    jax_config.update("jax_enable_x64", config.computation.float_precision == "float64")

    import chex
    if config.computation.disable_jit:
        chex.fake_pmap_and_jit().start()

    from deeperwin.utils.multihost import configure_hardware, disable_slave_loggers

    """ Configure hardware usage """
    configure_hardware(config)

    from deeperwin.loggers import build_dpe_root_logger

    root_logger = build_dpe_root_logger(config.logging.basic)
    disable_slave_loggers(root_logger)

    return root_logger, config


def create_phsical(raw_config: Dict, config: Configuration) -> None:
    from deeperwin.checkpoints import load_data_for_reuse
    from deeperwin.utils.utils import replicate_across_devices
    """ Set random seed """
    if config.computation.rng_seed is None:
        rng_seed = np.random.randint(2**31, size=())
        config.computation.rng_seed = int(replicate_across_devices(rng_seed)[0])
    rng_seed = config.computation.rng_seed
    np.random.seed(rng_seed)

    """ Reusing/restarting old run: merge configs and load data"""
    if config.reuse.mode != 'no':
        config, params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_params = load_data_for_reuse(config, raw_config)
    else:
        params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_params = None, None, None, None, None, None

    return rng_seed, config, params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_params

def init_wavefunction_mcmc(raw_config: Dict, config: Configuration) -> None:
    rng_seed, config, params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_params_to_reuse = create_phsical(raw_config, config)

    if config.model.orbitals.transferable_atomic_orbitals and config.model.orbitals.transferable_atomic_orbitals.phisnet_model:
        N_ions_max = max([len(config.physical.Z)])
        all_elements = config.model.orbitals.transferable_atomic_orbitals.atom_types or config.physical.Z
        all_atomic_orbitals = _get_all_basis_functions(all_elements, config.model.orbitals.transferable_atomic_orbitals.basis_set)
        nb_orbitals_per_Z, irreps = _get_orbital_mapping(all_atomic_orbitals, all_elements)

        use_phisnet_hessian = False
        if config.optimization.shared_optimization is not None:
            use_phisnet_hessian = config.optimization.shared_optimization.distortion is not None

        phisnet_model = build_phisnet_model(phisnet_params_to_reuse,
                                            config.model.orbitals.transferable_atomic_orbitals.phisnet_model,
                                            irreps,
                                            N_ions_max,
                                            rng_seed,
                                            with_hessian=use_phisnet_hessian)
    else:
        phisnet_model, N_ions_max = None, None
        nb_orbitals_per_Z = get_n_basis_per_Z(config.pre_training.baseline.basis_set,
                                                tuple(config.physical.Z))

    """ Build wavefunction / initialize model parameters """
    log_psi_squared, psi, orbital_func, cache_func, params, fixed_params = build_log_psi_squared(config.model, config.physical, fixed_params, rng_seed, phisnet_model, N_ions_max, nb_orbitals_per_Z)
    if params_to_reuse:
        params = merge_params(params, params_to_reuse, config.reuse.check_param_count)

    return rng_seed, log_psi_squared, psi, orbital_func, cache_func, params, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_model

from deeperwin.model import build_log_psi_squared
from deeperwin.optimization import optimize_wavefunction, pretrain_orbitals, evaluate_wavefunction, optimize_multi_wavefunction
from deeperwin.utils.utils import merge_params
from deeperwin.utils.setup_utils import initialize_training_loggers, finalize_experiment_run
from deeperwin.loggers import LoggerCollection
from deeperwin.orbitals import _get_all_basis_functions, _get_orbital_mapping
from deeperwin.model.ml_orbitals.ml_orbitals import build_phisnet_model
from deeperwin.orbitals import get_n_basis_per_Z
import datetime

"""
Function that takes in a config file & runs the experiment using a single molecule/geometry
:param config_file: path to config file

Note that imports involving jax (so basically most of our code) can only be imported after the 
jax config flags have been set (this is considered best practice).
"""

config_file = 'sample_configs/config_minimal.yml'
raw_config, config = Configuration.load_configuration_file(config_file)
root_path = 'experiment_data/Be_2_2_fermient'
config.logging.save_path = root_path
N_energy = 16

if not os.path.exists(config.logging.save_path):
    os.makedirs(config.logging.save_path)

for i_energy in range(N_energy):
    # 查看工作目录下是否有已经训练好的模型，如果有则加载
    config.logging.save_path = root_path
    files = os.listdir(config.logging.save_path)
    ckpt_restore_filenames = []
    for file in files:
        ckpts = os.listdir(config.logging.save_path+'/'+file)
        ckpts.sort()
        ckpts_t = []
        for ckpt in ckpts:
            if 'zip' in ckpt:
                ckpts_t.append(ckpt)
        ckpt_restore_filenames.append(config.logging.save_path+'/'+file+'/'+ckpts_t[-1])
    # 创建当前模型的保存路径
    path_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    config.logging.save_path = os.path.join(config.logging.save_path, path_name)
    if not os.path.exists(config.logging.save_path):
        os.makedirs(config.logging.save_path)
    config.save(os.path.join(config.logging.save_path, "full_config.yml"))
    root_logger, config = _setup_environment(raw_config, config)

    log_psi_squareds = []
    orbital_funcs = []
    paramses = []
    fixed_paramses = []
    mcmc_states = []
    clipping_states = []
    cache_funcs = []
    psis = []

    # Initialize wavefunction and MCMC state
    # init_new_wavefunctions
    for i in range(len(ckpt_restore_filenames)+1):
        if len(ckpt_restore_filenames) == 0:
            config.reuse.mode = 'no'
            config.reuse.path = None
        else:
            config.reuse.mode = 'reuse'
            config.reuse.path = ckpt_restore_filenames[i-1]

        rng_seed, log_psi_squared, psi, orbital_func, cache_func, params, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_model = init_wavefunction_mcmc(raw_config, config)
        log_psi_squareds.append(log_psi_squared)
        psis.append(psi)
        orbital_funcs.append(orbital_func)
        cache_funcs.append(cache_func)
        paramses.append(params)
        fixed_paramses.append(fixed_params)
        mcmc_states.append(mcmc_state)
        clipping_states.append(clipping_state)

    """ Initialize training loggers """
    use_wandb_group = False
    exp_idx_in_group = None
    training_loggers: LoggerCollection = initialize_training_loggers(config, params, fixed_params, use_wandb_group, exp_idx_in_group, config.logging.save_path)

    """ Unsupervised variational wavefunction optimization  """
    if (config.optimization.n_epochs > 0)  or (0 in config.optimization.intermediate_eval.opt_epochs):
        root_logger.info("Starting optimization...")
        mcmc_state, params, opt_state, clipping_state = optimize_multi_wavefunction(
            log_psi_squareds,
            psis,
            cache_funcs,
            paramses,
            fixed_paramses,
            mcmc_states,
            config.optimization,
            config.physical,
            rng_seed,
            training_loggers,
            opt_state,
            clipping_states,
        )

    """ Finalize run"""
    finalize_experiment_run(config, training_loggers, params, fixed_params, mcmc_state, opt_state, clipping_state)