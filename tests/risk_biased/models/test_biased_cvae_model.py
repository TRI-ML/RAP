import os
import pytest

import torch
from mmcv import Config

from risk_biased.models.biased_cvae_model import cvae_factory
from risk_biased.models.cvae_params import CVAEParams
from risk_biased.utils.config_argparse import config_argparse
from risk_biased.utils.cost import TTCCostTorch, TTCCostParams
from risk_biased.utils.risk import CVaREstimator
from risk_biased.models.latent_distributions import (
    GaussianLatentDistribution,
    ClassifiedLatentDistribution,
)


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(0)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "learning_config.py"
    )
    waymo_config_path = os.path.join(
        working_dir, "..", "..", "..", "risk_biased", "config", "waymo_config.py"
    )
    paths = [config_path, waymo_config_path]
    if isinstance(paths, str):
        cfg = Config.fromfile(paths)
    else:
        cfg = Config.fromfile(paths[0])
        for path in paths[1:]:
            c = Config.fromfile(path)
            cfg.update(c)
    cfg.batch_size = 4
    cfg.state_dim = 5
    cfg.map_state_dim = 2
    cfg.num_steps = 3
    cfg.num_steps_future = 4
    cfg.latent_dim = 2
    cfg.hidden_dim = 64
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.device = "cpu"
    return cfg


class TestBiasedCVAE:
    @pytest.fixture(autouse=True)
    def setup(self, params):
        cvae_params = CVAEParams.from_config(params)
        cost_function = TTCCostTorch(TTCCostParams.from_config(params))
        risk_estimator = CVaREstimator()
        self.model = cvae_factory(cvae_params, cost_function, risk_estimator)

    def _get_forward_input(self, params, num_agents, num_map_objects):
        x = torch.rand(
            params.batch_size, num_agents, params.num_steps, params.state_dim
        )
        mask_x = torch.rand(params.batch_size, num_agents, params.num_steps) > 0.1
        map = torch.rand(
            params.batch_size,
            num_map_objects,
            params.max_size_lane,
            params.map_state_dim,
        )
        mask_map = (
            torch.rand(params.batch_size, num_map_objects, params.max_size_lane) > 0.1
        )

        offset = torch.rand(params.batch_size, num_agents, params.dynamic_state_dim)
        risk_level = torch.rand(params.batch_size, num_agents)

        x_ego = torch.rand(params.batch_size, 1, params.num_steps, params.state_dim)
        y_ego = torch.rand(
            params.batch_size, 1, params.num_steps_future, params.state_dim
        )

        return dict(
            x=x,
            mask_x=mask_x,
            map=map,
            mask_map=mask_map,
            offset=offset,
            x_ego=x_ego,
            y_ego=y_ego,
            risk_level=risk_level,
        )

    def _get_predict_input(self, params, num_agents, num_map_objects, n_samples):
        predict_input = self._get_forward_input(params, num_agents, num_map_objects)
        predict_input["n_samples"] = n_samples
        return predict_input

    def _get_loss_input(
        self,
        params,
        num_agents,
        num_map_objects,
        map_seq_len,
        n_samples_risk,
        n_samples_biased,
    ):
        forward_input = self._get_forward_input(params, num_agents, num_map_objects)
        y = torch.rand(
            params.batch_size, num_agents, params.num_steps_future, params.state_dim
        )
        mask_y = (
            torch.rand(params.batch_size, num_agents, params.num_steps_future) > 0.1
        )

        mask_loss = torch.logical_and(
            mask_y,
            torch.rand(params.batch_size, num_agents, params.num_steps_future) > 0.1,
        )
        unnormalizer = lambda x, y: x
        risk_level = torch.rand(params.batch_size, num_agents)
        kl_weight = params.kl_weight
        kl_threshold = params.kl_threshold
        risk_weight = params.risk_weight
        dt = params.dt

        loss_input = forward_input
        loss_input.update(
            dict(
                y=y,
                mask_y=mask_y,
                mask_loss=mask_loss,
                unnormalizer=unnormalizer,
                risk_level=risk_level,
                kl_weight=kl_weight,
                kl_threshold=kl_threshold,
                risk_weight=risk_weight,
                n_samples_risk=n_samples_risk,
                n_samples_biased=n_samples_biased,
                dt=dt,
            )
        )
        return loss_input

    @pytest.mark.parametrize(
        "num_agents, num_map_objects, map_seq_len", [(4, 5, 3), (2, 4, 10)]
    )
    def test_forward(
        self, params, num_agents: int, num_map_objects: int, map_seq_len: int
    ):
        inputs = self._get_forward_input(params, num_agents, num_map_objects)

        future_sample_1, weights, distribution = self.model(**inputs)
        if isinstance(distribution, GaussianLatentDistribution):
            assert distribution.mu.shape == (
                params.batch_size,
                num_agents,
                params.latent_dim,
            )
            assert distribution.logvar.shape == (
                params.batch_size,
                num_agents,
                params.latent_dim,
            )
            assert self.model.prior_distribution.mu.requires_grad == False
            assert self.model.prior_distribution.logvar.requires_grad == False
            assert torch.all(
                self.model.prior_distribution.mu.data == torch.zeros(params.latent_dim)
            )
            assert torch.all(
                self.model.prior_distribution.logvar.data
                == torch.zeros(params.latent_dim)
            )
        elif isinstance(distribution, ClassifiedLatentDistribution):
            assert distribution.logits.shape == (
                params.batch_size,
                num_agents,
                params.num_vq,
            )
            assert distribution.codebook.shape == (params.num_vq, params.latent_dim)
            assert self.model.prior_distribution.logits.requires_grad == False
            assert torch.all(
                self.model.prior_distribution.logits.data == torch.zeros(params.num_vq)
            )
            assert torch.all(
                self.model.prior_distribution.codebook == distribution.codebook
            )

        # check shape
        assert future_sample_1.shape == (
            params.batch_size,
            num_agents,
            params.num_steps_future,
            params.dynamic_state_dim,
        )

        # make sure prior is correct

        future_sample_2, _, _ = self.model(**inputs)
        # make sure sampling is non-deterministic
        assert not torch.allclose(future_sample_1, future_sample_2)

    @pytest.mark.parametrize(
        "num_agents, num_map_objects, map_seq_len, n_samples_risk, n_samples_biased",
        [(4, 5, 3, 1, 2), (2, 4, 10, 3, 1)],
    )
    def test_get_loss(
        self,
        params,
        num_agents: int,
        num_map_objects: int,
        map_seq_len: int,
        n_samples_risk: int,
        n_samples_biased: int,
    ):
        inputs = self._get_loss_input(
            params,
            num_agents,
            num_map_objects,
            map_seq_len,
            n_samples_risk,
            n_samples_biased,
        )
        loss, _ = self.model.get_loss(**inputs)
        # make sure loss is scalar and not NaN
        assert not torch.isnan(loss).item()

    @pytest.mark.parametrize(
        "num_agents, num_map_objects, n_samples",
        [(7, 1, 0), (1, 4, 6), (1, 3, 2)],
    )
    def test_predict(
        self,
        params,
        num_agents: int,
        num_map_objects: int,
        n_samples: int,
    ):
        squeeze_sample_dim = n_samples <= 0
        input = self._get_predict_input(params, num_agents, num_map_objects, n_samples)
        future_sample_1, _, _ = self.model(**input)
        # check shape
        if squeeze_sample_dim:
            assert future_sample_1.shape == (
                params.batch_size,
                num_agents,
                params.num_steps_future,
                params.dynamic_state_dim,
            )
        else:
            assert future_sample_1.shape == (
                params.batch_size,
                num_agents,
                n_samples,
                params.num_steps_future,
                params.dynamic_state_dim,
            )

        future_sample_2, _, _ = self.model(**input)
        # make sure sampling is non-deterministic
        assert not torch.allclose(future_sample_1, future_sample_2)
