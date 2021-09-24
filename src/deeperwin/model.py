from jax import numpy as jnp
import numpy as np
from deeperwin.orbitals import build_el_el_cusp_correction, evaluate_molecular_orbitals, get_baseline_solution
from deeperwin.utils import get_el_ion_distance_matrix, get_distance_matrix
from deeperwin.configuration import DeepErwinModelConfig, SimpleSchnetConfig, PhysicalConfig, CASSCFConfig
import logging
try:
    from register_curvature import register_repeated_dense
    from deeperwin.kfac_ferminet_alpha.layers_and_loss_tags import register_scale_and_shift
except ImportError:
    def register_repeated_dense(y, X, W, b):
        return y
    def register_scale_and_shift(y, inputs, has_scale, has_shift):
        return y


def get_number_of_params(nested_params):
    if hasattr(nested_params, 'shape'):
        return int(np.prod(nested_params.shape))
    elif isinstance(nested_params, dict):
        return sum([get_number_of_params(p) for p in nested_params.values()])
    else:
        return sum([get_number_of_params(p) for p in nested_params])

def scale(X, params, register):
    y = jnp.exp(params.squeeze())*X
    if not register:
        return y
    else:
        return register_scale_and_shift(y, [X, params], has_scale=True, has_shift=False)

def dense_layer(X, W, b, register):
    y = jnp.dot(X, W) + b
    if not register:
        return y
    else:
        return register_repeated_dense(y, X, W, b)

def ffwd_net(params, X, linear_output=True, register=True):
    for p in params[:-1]:
        X = dense_layer(X, *p, register)
        X = jnp.tanh(X)
    X = dense_layer(X, *params[-1], register)
    if linear_output:
        return X
    else:
        return jnp.tanh(X)

def init_ffwd_net(n_neurons, input_dim, n_parallel=None):
    if n_parallel is None:
        parallel_dims = []
    else:
        parallel_dims = [n_parallel]
    params = []
    for i, output_dim in enumerate(n_neurons):
        if i > 0:
            input_dim = n_neurons[i-1]
        scale = np.sqrt(6 / (input_dim + output_dim)) # glorot initializer
        params.append([
            jnp.array(np.random.uniform(-1, 1, parallel_dims + [input_dim, output_dim]) * scale),
            jnp.array(np.zeros(parallel_dims + [output_dim]))])
    return params


def get_rbf_features(dist, n_features, sigma_pauli):
    """
    Maps distances to a set of gaussians, which can then be used as a kind of "one-hot-encoding" of the distance
    Args:
        r (tf.Tensor): Tensor of shape [batch-dimensions x 1] representing pairwise distances
    Returns:
        (tf.Tensor): Tensor of shape [batch-dimensions x n_rbf_features]. The number of features returned is set in the config
    """
    r_rbf_max = 5.0
    q = jnp.linspace(0, 1.0, n_features)
    mu = q**2 * r_rbf_max

    if sigma_pauli:
        sigma = (1 / 7) * (1 + r_rbf_max * q)
    else:
        sigma = r_rbf_max / (n_features-1) * (2*q+1/(n_features-1))

    dist = dist[..., jnp.newaxis] # add dimension for features
    return dist ** 2 * jnp.exp(- dist - ((dist - mu) / sigma) ** 2)

def get_pairwise_features(dist, model_config: DeepErwinModelConfig, dist_feat=False):
    features = get_rbf_features(dist, model_config.n_rbf_features, model_config.sigma_pauli)
    eps = model_config.eps_dist_feat
    if len(model_config.distance_feature_powers) > 0 and dist_feat:
        f_r = jnp.stack([dist ** n if n>0 else 1/(dist**(-n) + eps) for n in model_config.distance_feature_powers], axis=-1)
        features = jnp.concatenate([f_r, features], axis=-1)
    return features

def build_dummy_embedding(config, name='dummy_embed'):
    n_el = config['n_electrons']
    emb_dim = config['embedding_dim']

    def _call_dummy_embed(dist_el_el, dist_el_ion, Z, params):
        return dist_el_el @ params['dummy_net']

    dummy_params = {}
    dummy_params['dummy_net'] = jnp.array(np.random.randn(n_el-1, emb_dim))
    return name, _call_dummy_embed, dummy_params

def build_simple_schnet(config: SimpleSchnetConfig, n_el, n_up, input_dim, name = "embed"):
    n_dn = n_el - n_up
    indices_u_u = np.array([[j for j in range(n_up) if j != i] for i in range(n_up)], dtype=int)
    indices_d_d = np.array([[j + n_up for j in range(n_dn) if j != i] for i in range(n_dn)], dtype=int)
    emb_dim = config.embedding_dim

    def _call_embed(features_el_el, features_el_ion, Z, params):
        f_pairs_u_u = features_el_el[..., :n_up, :n_up - 1, :]
        f_pairs_d_d = features_el_el[..., n_up:n_el, n_up:n_el - 1, :]
        f_pairs_u_d = features_el_el[..., :n_up, n_up - 1:n_el - 1, :]
        f_pairs_d_u = features_el_el[..., n_up:n_el, :n_up, :]

        f_pairs_u_u = jnp.reshape(f_pairs_u_u, f_pairs_u_u.shape[:-3]+ (n_up*(n_up-1), input_dim))
        f_pairs_d_d = jnp.reshape(f_pairs_d_d, f_pairs_d_d.shape[:-3] + (n_dn * (n_dn-1), input_dim))
        f_pairs_u_d = jnp.reshape(f_pairs_u_d, f_pairs_u_d.shape[:-3] + (n_up * n_dn, input_dim))
        f_pairs_d_u = jnp.reshape(f_pairs_d_u, f_pairs_d_u.shape[:-3] + (n_up * n_dn, input_dim))
        f_pairs_same = jnp.concatenate([f_pairs_u_u, f_pairs_d_d], axis=-2)
        f_pairs_diff = jnp.concatenate([f_pairs_u_d, f_pairs_d_u], axis=-2)

        Z = jnp.transpose(jnp.tile(Z[:, jnp.newaxis], features_el_el.shape[:-3]))
        Z = jnp.reshape(Z, features_el_el.shape[:-3] + (Z.shape[-1], 1))
        ion_embeddings = ffwd_net(params['ion_emb'], Z, linear_output=False, register=True)
        ion_embeddings = jnp.reshape(ion_embeddings[..., jnp.newaxis, :], Z.shape[:-2] + (1, Z.shape[-2], emb_dim))
        x = jnp.ones(features_el_el.shape[:-2] + (emb_dim,))
        for n in range(config.n_iterations):
            h_same = ffwd_net(params['h_same'][n], x, linear_output=False)
            h_diff = ffwd_net(params['h_diff'][n], x, linear_output=False)

            h_u_u = h_same[..., indices_u_u, :]
            h_d_d = h_same[..., indices_d_d, :]
            h_u_d = h_diff[..., jnp.newaxis, n_up:, :]
            h_d_u = h_diff[..., jnp.newaxis, :n_up, :]

            w_same = ffwd_net(params['w_same'][n], f_pairs_same)
            w_diff = ffwd_net(params['w_diff'][n], f_pairs_diff)
            w_u_u = jnp.reshape(w_same[..., :(n_up*(n_up-1)), :], w_same.shape[:-2] + (n_up, n_up-1, emb_dim))
            w_d_d = jnp.reshape(w_same[..., (n_up * (n_up - 1)):, :], w_same.shape[:-2] + (n_dn, n_dn - 1, emb_dim))
            w_u_d = jnp.reshape(w_diff[..., :(n_up * n_dn), :], w_diff.shape[:-2] + (n_up, n_dn, emb_dim))
            w_d_u = jnp.reshape(w_diff[..., (n_up * n_dn):, :], w_diff.shape[:-2] + (n_dn, n_up, emb_dim))
            w_el_ions = ffwd_net(params['w_el_ions'][n], features_el_ion)

            embeddings_el_el = jnp.concatenate([
                jnp.concatenate([w_u_u * h_u_u, w_u_d * h_u_d], axis=-2),
                jnp.concatenate([w_d_d * h_d_d, w_d_u * h_d_u], axis=-2)],
                axis=-3)
            embeddings_el_ions = w_el_ions * ion_embeddings

            x = jnp.sum(embeddings_el_el, axis=-2) + jnp.sum(embeddings_el_ions, axis=-2)
            if config.use_res_net:
                x = x + ffwd_net(params['g_func'][n], x, linear_output=False)
            else:
                x = ffwd_net(params['g_func'][n], x, linear_output=False)

        return x, embeddings_el_el, embeddings_el_ions

    embed_params = {}
    shape_w = config.n_hidden_w + [emb_dim]
    shape_h = config.n_hidden_h + [emb_dim]
    shape_g = config.n_hidden_g + [emb_dim]
    embed_params['w_same'] = [init_ffwd_net(shape_w, input_dim) for _ in range(config.n_iterations)]
    embed_params['w_diff'] = [init_ffwd_net(shape_w, input_dim) for _ in range(config.n_iterations)]
    embed_params['w_el_ions'] = [init_ffwd_net(shape_w, input_dim) for _ in range(config.n_iterations)]
    embed_params['h_same'] = [init_ffwd_net(shape_h, emb_dim) for _ in range(config.n_iterations)]
    embed_params['h_diff'] = [init_ffwd_net(shape_h, emb_dim) for _ in range(config.n_iterations)]
    embed_params['ion_emb'] = init_ffwd_net(shape_h, 1)
    embed_params['g_func'] = [init_ffwd_net(shape_g, emb_dim) for _ in range(config.n_iterations)]
    return name, _call_embed, embed_params


def calculate_shift_decay(d_el_ion, Z, decaying_parameter):
    scale = decaying_parameter / Z
    scaling = jnp.prod(jnp.tanh((d_el_ion/scale)**2), axis=-1)
    return scaling

def build_backflow_shift(config: DeepErwinModelConfig, n_el, name = "bf_shift"):
    input_dim = config.embedding.embedding_dim

    def _calc_shift(x, pair_embedding, nn_params, diff, dist):
        n_particles = diff.shape[-2]
        x_tiled = jnp.tile(jnp.expand_dims(x, axis=-2), (n_particles,1))
        features = jnp.concatenate([x_tiled, pair_embedding], axis=-1)
        shift = ffwd_net(nn_params, features)
        shift_weights = shift / (1 + dist[..., jnp.newaxis]**3)
        return jnp.sum(shift_weights * diff, axis=-2)

    def _call_bf_shift(x, diff_el_el, dist_el_el, embeddings_el_el, diff_el_ion, dist_el_ion, embeddings_el_ions, Z, params):
        shift_towards_electrons = _calc_shift(x, embeddings_el_el, params['w_el'], diff_el_el, dist_el_el)
        shift_towards_ions = _calc_shift(x, embeddings_el_ions, params['w_ion'], diff_el_ion, dist_el_ion)
        if config.use_trainable_shift_decay_radius:
            shift_decay = calculate_shift_decay(dist_el_ion, Z, params['scale_decay'])
        else:
            shift_decay = calculate_shift_decay(dist_el_ion, Z, 1.0)
        shift = (shift_towards_electrons+ shift_towards_ions)*shift_decay[..., jnp.newaxis]
        shift = scale(shift, params['scale_el'], register=config.register_scale) #jnp.exp(params['scale_el'])*shift_decay*(shift_ion + shift_el)
        return shift

    bf_params = {}
    bf_params['w_el'] = init_ffwd_net(config.n_hidden_bf_shift + [config.output_shift], 2*input_dim)
    bf_params['w_ion'] = init_ffwd_net(config.n_hidden_bf_shift + [config.output_shift], 2*input_dim)

    bf_params['scale_el'] = jnp.array([-3.5])
    if config.use_trainable_shift_decay_radius:
        bf_params['scale_decay'] = jnp.array([0.5])
    return name, _call_bf_shift, bf_params


def build_backflow_factor(config: DeepErwinModelConfig, n_electrons, n_up, name="bf_fac"):
    n_dn = n_electrons - n_up
    n_dets = config.baseline.n_determinants
    input_dim = config.embedding.embedding_dim

    def _call_bf_factor(embeddings, params):
        #n_dn = embeddings.shape[-2] - n_up
        bf_up = ffwd_net(params['up'], embeddings[..., :n_up, :], linear_output=True)
        bf_dn = ffwd_net(params['dn'], embeddings[..., n_up:, :], linear_output=True)

        bf_up = jnp.reshape(bf_up, bf_up.shape[:-2] + (n_dets * n_up * n_up,))
        bf_dn = jnp.reshape(bf_dn, bf_dn.shape[:-2] + (n_dets * n_dn * n_dn,))

        bf = jnp.concatenate([bf_up, bf_dn], axis=-1)
        bf = 1.0 + scale(bf, params['scale'], register=config.register_scale)

        bf_up = jnp.reshape(bf[..., :n_dets*n_up*n_up], bf_up.shape[:-1] + (n_up, n_dets*n_up)) # output-shape: [batch x n_up x n_dets * n_up_orb]
        bf_dn = jnp.reshape(bf[..., n_dets*n_up*n_up:], bf_dn.shape[:-1] + (n_dn, n_dets*n_dn))

        bf_up = jnp.reshape(bf_up, bf_up.shape[:-1] + (n_dets, n_up))  # output-shape: [batch x n_up x n_dets x n_up_orb]
        bf_dn = jnp.reshape(bf_dn, bf_dn.shape[:-1] + (n_dets, n_dn))

        bf_up = jnp.swapaxes(bf_up, -3, -2) # output-shape: [batch x n_dets x n_up x n_up_orb]
        bf_dn = jnp.swapaxes(bf_dn, -3, -2)

        return bf_up, bf_dn

    bf_params = {}
    bf_params['up'] = init_ffwd_net(config.n_hidden_bf_factor + [n_dets * n_up], input_dim)
    bf_params['dn'] = init_ffwd_net(config.n_hidden_bf_factor + [n_dets * n_dn], input_dim)
    bf_params['scale'] = jnp.array([-2.0])
    return name, _call_bf_factor, bf_params

def build_jastrow_factor(config: DeepErwinModelConfig, n_up, name="jastrow"):
    def _call_jastrow(embeddings, params):
        jastrow_up = ffwd_net(params['up'], embeddings[..., :n_up, :], linear_output=True)
        jastrow_dn = ffwd_net(params['dn'], embeddings[..., n_up:, :], linear_output=True)
        jastrow = scale(jnp.sum(jastrow_up, axis=(-2,-1)) + jnp.sum(jastrow_dn, axis=(-2,-1)), params['scale'], register=config.register_scale) #jnp.exp(params['scale']) * (jnp.sum(jastrow_up, axis=(-2,-1)) + jnp.sum(jastrow_dn, axis=(-2,-1)))
        return jastrow

    jas_params = {}
    jas_params['up'] = init_ffwd_net(config.n_hidden_jastrow + [1], config.embedding.embedding_dim)
    jas_params['dn'] = init_ffwd_net(config.n_hidden_jastrow + [1], config.embedding.embedding_dim)
    jas_params['scale'] = jnp.array([0.0])
    return name, _call_jastrow, jas_params

def _evaluate_sum_of_determinants(mo_matrix_up, mo_matrix_dn, ci_weights):
    LOG_EPSILON = 1e-8
    sign_up, log_up = jnp.linalg.slogdet(mo_matrix_up)
    sign_dn, log_dn = jnp.linalg.slogdet(mo_matrix_dn)
    log_total = log_up + log_dn
    sign_total = sign_up * sign_dn
    log_shift = jnp.max(log_total, axis=-1, keepdims=True)
    psi = jnp.exp(log_total - log_shift) * sign_total
    psi = jnp.sum(psi * ci_weights, axis=-1)  # sum over determinants
    log_psi_sqr = 2 * (jnp.log(jnp.abs(psi) + LOG_EPSILON) + jnp.squeeze(log_shift, -1))
    return log_psi_sqr

def _build_baseline_slater_determinants(el_ion_diff, el_ion_dist, fixed_params, n_up, cusp_type):
    atomic_orbitals, cusp_params, mo_coeff, ind_orb, ci_weights = fixed_params['baseline']
    if cusp_type == "mo":
        mo_matrix_up = evaluate_molecular_orbitals(el_ion_diff[..., :n_up, :, :], el_ion_dist[..., :n_up, :], atomic_orbitals, mo_coeff[0],
                                                   cusp_params[0], cusp_type)
        mo_matrix_dn = evaluate_molecular_orbitals(el_ion_diff[..., n_up:, :, :], el_ion_dist[..., n_up:, :], atomic_orbitals, mo_coeff[1],
                                                   cusp_params[1], cusp_type)
    else:
        mo_matrix_up = evaluate_molecular_orbitals(el_ion_diff[..., :n_up, :, :], el_ion_dist[..., :n_up, :], atomic_orbitals, mo_coeff[0],
                                                   cusp_params, cusp_type)
        mo_matrix_dn = evaluate_molecular_orbitals(el_ion_diff[..., n_up:, :, :], el_ion_dist[..., n_up:, :], atomic_orbitals, mo_coeff[1],
                                                   cusp_params, cusp_type)
    mo_matrix_up = mo_matrix_up[..., ind_orb[0]]
    mo_matrix_dn = mo_matrix_dn[..., ind_orb[1]]
    if len(mo_matrix_up.shape) == 4:  # including batch dimension
        mo_matrix_up = jnp.transpose(mo_matrix_up, [0, 2, 1, 3])
        mo_matrix_dn = jnp.transpose(mo_matrix_dn, [0, 2, 1, 3])
    else:
        mo_matrix_up = jnp.transpose(mo_matrix_up, [1, 0, 2])
        mo_matrix_dn = jnp.transpose(mo_matrix_dn, [1, 0, 2])
    return mo_matrix_up, mo_matrix_dn, ci_weights


def init_log_psi_squared_fixed_params(casscf_config: CASSCFConfig, physical_config: PhysicalConfig):
    logging.debug("Calculating baseline solution...")
    baseline_solution, (E_hf, E_cas) = get_baseline_solution(physical_config, casscf_config)
    initial_fixed_params = dict(baseline=baseline_solution, E_hf=E_hf, E_casscf=E_cas)
    logging.debug(f"Finished baseline calculation: E_casscf={E_cas:.6f}")
    return initial_fixed_params

def build_log_psi_squared(config: DeepErwinModelConfig, physical_config: PhysicalConfig, init_fixed_params = True, name = "log_psi_squared"):
    n_electrons, n_up = physical_config.n_electrons, physical_config.n_up
    initial_trainable_params = {}

    initial_fixed_params = init_log_psi_squared_fixed_params(config.baseline, physical_config) if init_fixed_params else None

    if config.embedding.name == "dummy":
        name_embed, embed_call, embed_params = build_dummy_embedding(config)
    elif config.embedding.name == "simple_schnet":
        name_embed, embed_call, embed_params = build_simple_schnet(config.embedding, n_electrons, n_up,
                                                                   config.n_pairwise_features)
    initial_trainable_params[name_embed] = embed_params
    if config.use_bf_factor:
        name_bf_factor, bf_factor_call, bf_factor_params = build_backflow_factor(config, n_electrons, n_up)
        initial_trainable_params[name_bf_factor] = bf_factor_params
    if config.use_bf_shift:
        name_bf_shift, bf_shift_call, bf_shift_params = build_backflow_shift(config, n_electrons)
        initial_trainable_params[name_bf_shift] = bf_shift_params
    if config.use_jastrow:
        name_jastrow, jastrow_call, jastrow_params = build_jastrow_factor(config, n_up)
        initial_trainable_params[name_jastrow] = jastrow_params

    el_el_cusp_call = build_el_el_cusp_correction(n_electrons, n_up, config.baseline.cusps)

    def _call(r, R, Z, params, fixed_params):
        # Calculate basic features from input coordinates
        diff_el_el, dist_el_el = get_distance_matrix(r)
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
        features_el_el = get_pairwise_features(dist_el_el, config, dist_feat=True)
        features_el_ion = get_pairwise_features(dist_el_ion, config, dist_feat=True)

        # Calculate embedding of electron coordinates
        embeddings, pairwise_feat, pairwise_feat_ions = embed_call(features_el_el, features_el_ion, Z, params[name_embed])
        if config.use_bf_shift:
            backflow_shift = bf_shift_call(embeddings, diff_el_el, dist_el_el, pairwise_feat, diff_el_ion, dist_el_ion, pairwise_feat_ions, Z, params[name_bf_shift])
            r = r + backflow_shift
            diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)

        # Evaluate atomic and molecular orbitals for every determinant
        mo_matrix_up, mo_matrix_dn, ci_weights = _build_baseline_slater_determinants(diff_el_ion, dist_el_ion, fixed_params, n_up, config.baseline.cusps.cusp_type)

        # Modify molecular orbitals using backflow factor
        if config.use_bf_factor:
            backflow_factor_up, backflow_factor_dn = bf_factor_call(embeddings, params[name_bf_factor])
            mo_matrix_up *= backflow_factor_up
            mo_matrix_dn *= backflow_factor_dn

        # Calculate slater-determinants
        log_psi_sqr = _evaluate_sum_of_determinants(mo_matrix_up, mo_matrix_dn, ci_weights)

        # Apply a jastrow factor to the total wavefunction
        if config.use_jastrow:
            jastrow = jastrow_call(embeddings, params[name_jastrow])
            log_psi_sqr += jastrow
        # Apply electron-electron-cusps
        if config.baseline.cusps.use:
            log_psi_sqr += el_el_cusp_call(dist_el_el)
        return log_psi_sqr

    logging.debug(f"Number of parameters: {get_number_of_params(initial_trainable_params)}")

    return name, _call, initial_trainable_params, initial_fixed_params


def build_log_psi_squared_baseline_model(baseline_config: CASSCFConfig, physical_config: PhysicalConfig, init_fixed_params = True, name = "log_psi_squared"):
    n_electrons, n_up = physical_config.n_electrons, physical_config.n_up
    initial_fixed_params = init_log_psi_squared_fixed_params(baseline_config, physical_config) if init_fixed_params else None

    initial_trainable_params = {}
    el_el_cusp_call = build_el_el_cusp_correction(n_electrons, n_up, baseline_config.cusps)

    def _call(r, R, Z, params, fixed_params):
        # Calculate basic features from input coordinates
        _, dist_el_el = get_distance_matrix(r)
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)

        # Evaluate atomic and molecular orbitals for every determinant
        mo_matrix_up, mo_matrix_dn, ci_weights = _build_baseline_slater_determinants(diff_el_ion, dist_el_ion, fixed_params, n_up, baseline_config.cusps.cusp_type)

        # Calculate slater-determinants
        log_psi_sqr = _evaluate_sum_of_determinants(mo_matrix_up, mo_matrix_dn, ci_weights)

        # Apply electron-electron-cusps
        if baseline_config.cusps.use:
            log_psi_sqr += el_el_cusp_call(dist_el_el)
        return log_psi_sqr

    return name, _call, initial_trainable_params, initial_fixed_params

if __name__ == '__main__':
    pass