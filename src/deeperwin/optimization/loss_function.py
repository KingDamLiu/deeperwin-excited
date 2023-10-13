import logging
import jax
import jax.numpy as jnp
import kfac_jax
from deeperwin.configuration import ClippingConfig
from deeperwin.hamiltonian import get_local_energy
from deeperwin.utils.utils import pmean, without_cache
import functools

LOGGER = logging.getLogger("dpe")

def init_clipping_state():
    return jnp.array([0.0]).squeeze(), jnp.array([1e5]).squeeze()

def _update_clipping_state(E, clipping_state, clipping_config: ClippingConfig):
    del clipping_state
    center = dict(mean=jnp.nanmean,
                  median=jnp.nanmedian,
                  )[clipping_config.center](E)
    center = pmean(center)
    if clipping_config.width_metric == 'mae':
        width = jnp.nanmean(jnp.abs(E-center))
        width = pmean(width)
    elif clipping_config.width_metric == 'std':
        width = jnp.nanmean((E-center)**2)
        width = jnp.sqrt(pmean(width))
    else:
        raise NotImplementedError(f"Unknown clipping metric: {clipping_config.width_metric}")
    return center, width * clipping_config.clip_by

def clip_mean(x):
    a = jnp.nanpercentile(x, jnp.array([2, 98]))
    if 1:
        return jnp.nanmean(jnp.clip(x, a[0], a[1]))
    else:
        return jnp.nanmean(x[(x>a[0])&(x<a[1])])

def _clip_energies(E, clipping_state, clipping_config: ClippingConfig):
    center, width = clipping_state
    if (not clipping_config.from_previous_step) or (center is None):
        center, width = _update_clipping_state(E, clipping_state, clipping_config)

    if clipping_config.name == "hard":
        clipped_energies = jnp.clip(E, center - width, center + width)
    elif clipping_config.name == "tanh":
        clipped_energies = center + jnp.tanh((E - center) / width) * width
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.name: {clipping_config.name}")
    new_clipping_state = _update_clipping_state(clipped_energies, clipping_state, clipping_config)
    return clipped_energies, new_clipping_state

def build_value_and_grad_func(log_psi_sqr_funcs, psis, clipping_config: ClippingConfig, phys_config):
    """
    Returns a callable that computes the gradient of the mean local energy for a given set of MCMC walkers with respect to the model defined by `log_psi_func`.

    Args:
        log_psi_sqr_func (callable): A function representing the wavefunction model
        clipping_config (ClippingConfig): Clipping hyperparameters
        use_fwd_fwd_hessian (bool): If true, the second partial derivatives required for computing the local energy are obtained with a forward-forward scheme.

    """

    # Build custom total energy jvp. Based on https://github.com/deepmind/ferminet/blob/jax/ferminet/train.py
    spin_state = (phys_config.n_up, phys_config.n_dn)
    # @functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
    @jax.custom_jvp
    def total_energy(paramses, states, batchs):
        # TODO: why is spin state no integer anymore here now??
        params, state, batch = paramses[0], states[0], batchs[0]
        clipping_state = state 
        E_kin, E_pot = get_local_energy(log_psi_sqr_funcs[0], params, spin_state, *batch)
        E_loc = E_kin + E_pot
        E_mean = pmean(jnp.nanmean(E_loc))
        E_var = pmean(jnp.nanmean((E_loc - E_mean) ** 2))

        E_loc_clipped, states[0] = _clip_energies(E_loc, clipping_state, clipping_config)
        E_mean_clipped = pmean(jnp.nanmean(E_loc_clipped))
        E_var_clipped = pmean(jnp.nanmean((E_loc_clipped - E_mean_clipped) ** 2))

        psi_datas = []
        for i in range(len(psis)):
            psi_data = []
            for j in range(len(psis)):
                psi_data.append(psis[i](paramses[i], *spin_state, *batchs[j]))
            psi_datas.append(psi_data)

        # 计算波函数的重叠损失
        S = []
        for i in range(1):
            for j in range(1, len(psis)):
                psi_ij = pmean(clip_mean(psi_datas[i][j]/psi_datas[j][j]))
                psi_ji = pmean(clip_mean(psi_datas[j][i]/psi_datas[i][i]))
                S.append(jnp.abs(psi_ij*psi_ji)**0.5)

        aux = dict(E_mean=E_mean,
                   E_var=E_var,
                   E_mean_clipped=E_mean_clipped,
                   E_var_clipped=E_var_clipped,
                   E_kin = pmean(jnp.nanmean(E_kin)),
                   E_pot = pmean(jnp.nanmean(E_pot)),
                   psis = psi_datas,
                   S = S,
                   E_loc_clipped=E_loc_clipped)
        loss = E_mean_clipped
        return loss, (states, aux)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        paramses, states, batchs = primals
        params, state, batch = paramses[0], states[0], batchs[0]
        r, R, Z, fixed_params = batch
        batch_size = batch[0].shape[0]

        loss, (states, stats) = total_energy(paramses, states, batchs)
        psi_datas = stats["psis"]
        diff = stats["E_loc_clipped"] - stats["E_mean_clipped"]

        def func(params):
            return log_psi_sqr_funcs[0](params, *spin_state, r, R, Z, without_cache(fixed_params))

        log_psi_sqr, tangents_log_psi_sqr = jax.jvp(func, (primals[0][0],), (tangents[0][0],))
        kfac_jax.register_normal_predictive_distribution(log_psi_sqr[:, None])  # Register loss for kfac optimizer

        tangents_outs = jnp.dot(tangents_log_psi_sqr, diff) / batch_size
        # 波函数重叠的导数
        for i in range(len(log_psi_sqr_funcs)):
            for j in range(len(log_psi_sqr_funcs)):
                if i!=j:
                    psi_diff = psi_datas[j][i]/psi_datas[i][i]-pmean(clip_mean(psi_datas[j][i]/psi_datas[i][i]))
                    # S = jnp.abs(constants.pmean(clip_mean(psi_datas[i][j]/psi_datas[j][j]))*constants.pmean(clip_mean(psi_datas[j][i]/psi_datas[i][i])))**0.5-1e-4
                    tangents_outs += jnp.dot(tangents_log_psi_sqr, psi_diff) / batch_size * pmean(clip_mean(psi_datas[i][j]/psi_datas[j][j]))# *(1/(1-S)**2)

        primals_out = loss, (states, stats)
        tangents_out = tangents_outs, (states, stats)
        return primals_out, tangents_out

    return jax.value_and_grad(total_energy, has_aux=True)