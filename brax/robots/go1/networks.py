from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.envs.base import RlwamEnv
from brax.training.agents.rlwam import networks as rlwam_networks
from brax.contact_training.network import make_contact_model

from omegaconf import DictConfig
from flax import linen
from pathlib import Path
import dill
import functools as ft
import jax
import jax.numpy as jp

_activations = {
    'swish': linen.swish,
    'tanh': linen.tanh
}


def ppo_network_factory(cfg: DictConfig, saved_policies_dir: Path):
    activation = _activations[cfg.actor_network.activation]

    starting_params = None
    if cfg.starting_policy_for_ppo is not None:
        # load starting policy for ppo
        path = saved_policies_dir / cfg.starting_policy_for_ppo
        with open(path, 'rb') as f:
            starting_params = dill.load(f)

    # network factory
    network_factory = ft.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                                   * cfg.actor_network.hidden_layers),
        value_hidden_layer_sizes=((cfg.critic_network.hidden_size,)
                                  * cfg.critic_network.hidden_layers),
        activation=activation
    )

    return starting_params, network_factory


def make_ppo_networks(cfg: DictConfig, saved_policies_dir: Path,
                      env: RlwamEnv, ppo_params_path: Path = None):
    if ppo_params_path is not None:
        path = ppo_params_path
    else:
        path = saved_policies_dir / 'go1_ppo_policy.pkl'
    with open(path, 'rb') as f:
        params = dill.load(f)

    # create the policy network
    normalize = lambda x, y: x
    if cfg.common.normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size*cfg.contact_generate.obs_history_length,
        env.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                                   * cfg.actor_network.hidden_layers)
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    return params, make_policy


def make_rlwam_networks(cfg: DictConfig, saved_policies_dir: Path,
                        env: RlwamEnv, ppo_params_path: Path = None,
                        rlwam_params_path: Path = None):
    activation = _activations[cfg.actor_network.activation]

    if cfg.start_with_ppo:
        # load policy from ppo
        if ppo_params_path is not None:
            path = ppo_params_path
        else:
            path = saved_policies_dir / 'go1_ppo_policy.pkl'
        with open(path, 'rb') as f:
            params = dill.load(f)

        # create the policy and value networks
        normalize = lambda x, y: x
        if cfg.common.normalize_observations:
            normalize = running_statistics.normalize
        ppo_network = ppo_networks.make_ppo_networks(
            env.observation_size*cfg.rlwam.obs_history_length,
            env.action_size,
            preprocess_observations_fn=normalize,
            policy_hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                                       * cfg.actor_network.hidden_layers),
            value_hidden_layer_sizes=((cfg.critic_network.hidden_size,)
                                      * cfg.critic_network.hidden_layers),
            activation=activation
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)
        make_value = ppo_networks.make_value_inference_fn(ppo_network)
        network_factory = None

        if cfg.reset_ppo_value_fn:
            key_value = jax.random.PRNGKey(cfg.reset_ppo_value_fn_seed)
            value_params = ppo_network.value_network.init(key_value)
            norm_params, policy_params, _ = params
            params = (norm_params, policy_params, value_params)

        if cfg.use_delta_rlwam_policy:
            rlwam_network = rlwam_networks.make_rlwam_networks(
                env.observation_size*cfg.rlwam.obs_history_length,
                env.action_size,
                preprocess_observations_fn=normalize,
                policy_hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                                           * cfg.actor_network.hidden_layers),
                value_hidden_layer_sizes=((cfg.critic_network.hidden_size,)
                                          * cfg.critic_network.hidden_layers),
                activation=activation
            )
            (ppo_norm_params, ppo_policy_params, ppo_value_params) = params
            ppo_policy = make_policy((ppo_norm_params, ppo_policy_params),
                                        deterministic=cfg.rlwam.deterministic_policy)
            make_policy = rlwam_networks.make_inference_fn(rlwam_network,
                                                            ppo_policy)
            make_value = rlwam_networks.make_value_inference_fn(rlwam_network)
            if rlwam_params_path is not None:
                with open(rlwam_params_path, 'rb') as f:
                    params = dill.load(f)
            else:
                key = jax.random.PRNGKey(cfg.rlwam.seed)
                rlwam_policy_params = rlwam_network.policy_network.init(key)
                if cfg.zero_final_layer_of_delta_rlwam_policy:
                    last_key = list(rlwam_policy_params['params'].keys())[-1]
                    rlwam_policy_params['params'][last_key]['kernel'] = jp.zeros_like(rlwam_policy_params['params'][last_key]['kernel'])
                params = (ppo_norm_params, rlwam_policy_params, ppo_value_params)

    else:
        # make a fresh network
        make_policy = None
        make_value = None
        params = None
        network_factory = ft.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=((cfg.actor_network.hidden_size,)
                                      * cfg.actor_network.hidden_layers),
            value_hidden_layer_sizes=((cfg.critic_network.hidden_size,)
                                      * cfg.critic_network.hidden_layers),
            activation=activation
        )

    return make_policy, make_value, params, network_factory


def make_force_model(cfg: DictConfig, saved_policies_dir: Path,
                     env: RlwamEnv):
    if cfg.contact_train.forces_in_q_coords:
        q = "_q_coords_"
        output_dim = 18
    else:
        q = "_"
        output_dim = 16
    if cfg.use_force_model:
        r = "jacreg_" + str(cfg.common.contact_jac_reg)
        path = (saved_policies_dir
                / ('go1_contact_model' + q + r + '.pkl'))
        with open(path, 'rb') as f:
            contact_params = dill.load(f)
        raw_force_model = make_contact_model(
            params=contact_params[0],
            batch_stats=contact_params[1],
            obs_size=env.observation_size,
            obs_history_length=cfg.rlwam.obs_history_length,
            controls_size=env.controls_size,
            hidden_ratio_1=cfg.contact_train.hidden_ratio_1,
            hidden_ratio_2=cfg.contact_train.hidden_ratio_2,
            activation=cfg.contact_train.activation,
            batch_norm=cfg.contact_train.batch_norm,
            output_dim=output_dim
        )
        if cfg.force_model_deadzone:
            threshold = cfg.force_model_deadzone_threshold

            def force_model(obs, u):
                return jp.where(raw_force_model(obs, u) < threshold,
                                jp.zeros((output_dim,)),
                                raw_force_model(obs, u))
        else:
            force_model = raw_force_model
    else:
        force_model = lambda obs, u: jp.zeros((output_dim,))

    return force_model
