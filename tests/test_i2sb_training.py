import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage

from dataset.splits import rainfall_split_ids
from i2sb.depth_metrics import DepthMetrics, depth_metres
from i2sb.diffusion import Diffusion
from i2sb.network import Image256Net
from i2sb.runner import Runner, normalize_and_clip_gradients
from i2sb.training_contract import check_checkpoint_contract, training_contract
from i2sb.util import BaseWriter, custom_collate_fn


class ToyDataset(Dataset):
    depth_mean, depth_std = 0.987, 0.0343

    def __len__(self):
        return 5

    def __getitem__(self, index):
        target = torch.full((1, 4, 4), 0.1 * index)
        return target, torch.ones_like(target), None, np.zeros(24, dtype=np.int64), str(index), target + 0.5, 4.0, 1


class ToyRain(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(24, 1)

    def forward(self, rainfall):
        return self.projection(rainfall.float())


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Conv2d(2, 1, 1)
        self.seen_indices = []

    def forward(self, x, indices, rainfall, cond=None):
        self.seen_indices.extend(indices.tolist())
        return self.adapter(torch.cat((x, cond), 1)) + rainfall.view(-1, 1, 1, 1) + indices.view(-1, 1, 1, 1) * 0.001


def options(**overrides):
    opt = SimpleNamespace(
        device='cpu', interval=12, beta_max=0.1, ot_ode=False, ca4d=True, add_x1_noise=False,
        use_single_dem=True, single_max_depth=4.0, train_rain_ids=[1, 2], val_rain_ids=[3],
        test_dem_list=None, test_rain_list=None, val_nfe=3, validation_seed=12345,
        global_size=1, global_rank=0, microbatch=2, batch_size=4, num_workers=0,
        lr=5e-5, l2_norm=0.0, lr_gamma=0.99, lr_step=1, load=None,
        num_itr=2, val_every=1, grad_clip=0.5, log_writer='none',
        log_dir=Path('.log'), name='test-not-written', ckpt_path=Path('results/test-not-written'),
    )
    vars(opt).update(overrides)
    return opt


def toy_runner(opt):
    runner = Runner.__new__(Runner)
    runner.net, runner.rainfall_emb = ToyNet(), ToyRain()
    runner.diffusion = Diffusion(np.full(opt.interval, 0.01), 'cpu')
    runner.ema = ExponentialMovingAverage(list(runner.net.parameters()) + list(runner.rainfall_emb.parameters()), decay=0.999)
    runner.start_step, runner.best_rmse, runner.log = 0, float('inf'), Mock()
    runner.writer = BaseWriter(opt)
    return runner


class I2SBRegressionTests(unittest.TestCase):
    def test_discrete_label_is_inverse_of_x0_prediction(self):
        opt = options()
        runner = toy_runner(opt)
        x0, x1 = torch.randn(3, 1, 4, 4), torch.randn(3, 1, 4, 4)
        indices = torch.tensor([0, 5, 11])
        for ode in (False, True):
            xt = runner.diffusion.q_sample(indices, x0, x1, ot_ode=ode)
            label = runner.compute_label(indices, x0, xt, x1)
            torch.testing.assert_close(runner.compute_pred_x0(indices, xt, x1, label), x0)

    def test_network_preserves_indices_and_rejects_fractional_time(self):
        net = Image256Net.__new__(Image256Net)
        nn.Module.__init__(net)
        net.cond, net.noise_levels = False, torch.arange(1000)
        capture = Mock(side_effect=lambda x, t, rain: x)
        net.diffusion_model = capture
        x = torch.zeros(2, 1, 4, 4)
        net(x, torch.tensor([123, 999]), None)
        torch.testing.assert_close(capture.call_args.args[1], torch.tensor([123, 999]))
        for invalid in (torch.tensor([0.2, 0.8]), torch.tensor([-1, 0]), torch.tensor([1, 1000])):
            with self.assertRaises(ValueError):
                net(x, invalid, None)

    def test_posterior_coefficients(self):
        diffusion = Diffusion(np.full(12, 0.01), 'cpu')
        result = diffusion.p_posterior(2, 7, torch.ones(1), torch.zeros(1), ot_ode=True)
        torch.testing.assert_close(result, torch.tensor([3 / 8]))

    def test_sampler_uses_requested_nfe_and_restores_modes_and_ema(self):
        opt = options()
        runner = toy_runner(opt)
        runner.net.train()
        runner.rainfall_emb.train()
        original = [p.clone() for p in runner.net.parameters()]
        xs, _ = runner.ddpm_sampling(opt, torch.zeros(2, 1, 4, 4), torch.zeros(2, 24, dtype=torch.long),
                                     cond=torch.ones(2, 1, 4, 4), nfe=3, verbose=False)
        self.assertEqual(xs.shape, (2, 3, 1, 4, 4))
        self.assertEqual(len(runner.net.seen_indices), 2 * 3)
        self.assertGreater(max(runner.net.seen_indices), 1)
        self.assertTrue(runner.net.training and runner.rainfall_emb.training)
        for expected, actual in zip(original, runner.net.parameters()):
            torch.testing.assert_close(expected, actual)
        with self.assertRaises(ValueError):
            runner.ddpm_sampling(opt, torch.zeros(1), torch.zeros(1), ode_method='euler-maruyama')

    def test_event_split_is_disjoint_reproducible_and_excludes_test(self):
        opt = SimpleNamespace(val_fraction=0.1, split_seed=0, val_rain_list=None)
        training, validation = rainfall_split_ids(range(1, 183), opt)
        self.assertEqual(len(validation), 18)
        self.assertFalse(training & validation)
        self.assertEqual(training | validation, set(range(1, 183)))
        self.assertEqual((training, validation), rainfall_split_ids(reversed(range(1, 183)), opt))
        opt.val_rain_list = [9, 20]
        with self.assertRaises(ValueError):
            rainfall_split_ids([1, 2, 3], opt)

    def test_metres_and_pixel_weighted_aggregation(self):
        mean, std = 0.987, 0.0343
        normalized = (torch.tensor([1.0, 0.5, 0.0]).view(3, 1, 1, 1) - mean) / std
        decoded = depth_metres(normalized, mean, std, torch.tensor([4., 4., 3.]))
        torch.testing.assert_close(decoded.flatten(), torch.tensor([0., 2., 3.]))
        metric = DepthMetrics()
        metric.update(torch.tensor([1., 1.]), torch.zeros(2))
        metric.update(torch.tensor([4.]), torch.zeros(1))
        self.assertAlmostEqual(metric.compute()['mae_m'], 2.)
        self.assertAlmostEqual(metric.compute()['rmse_m'], 6 ** 0.5)
        with self.assertRaises(ValueError):
            metric.update(torch.tensor([float('nan')]), torch.zeros(1))

    def test_accumulation_matches_full_batch_including_partial_valid_count(self):
        torch.manual_seed(9)
        inputs, targets = torch.randn(7, 3), torch.randn(7, 1)
        reference = nn.Linear(3, 1)
        nn.functional.mse_loss(reference(inputs), targets).backward()
        for microbatch in (1, 2, 4, 7):
            model = copy.deepcopy(reference)
            model.zero_grad()
            for x, y in zip(inputs.split(microbatch), targets.split(microbatch)):
                (nn.functional.mse_loss(model(x), y) * len(x)).backward()
            normalize_and_clip_gradients(model.parameters(), 7, 1000.)
            for expected, actual in zip(reference.parameters(), model.parameters()):
                torch.testing.assert_close(expected.grad, actual.grad)

    def test_joint_clipping_includes_embedder_and_rejects_bad_gradient(self):
        net_parameter, rain_parameter = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        net_parameter.grad, rain_parameter.grad = torch.tensor([6.]), torch.tensor([8.])
        normalize_and_clip_gradients([net_parameter, rain_parameter], 2, 0.5)
        self.assertAlmostEqual(net_parameter.grad.item(), 0.3, places=5)
        self.assertAlmostEqual(rain_parameter.grad.item(), 0.4, places=5)
        rain_parameter.grad = torch.tensor([float('inf')])
        with self.assertRaises(RuntimeError):
            normalize_and_clip_gradients([net_parameter, rain_parameter], 2, 0.5)

    def test_validation_consumes_last_batch_and_preserves_rng(self):
        opt, dataset = options(), ToyDataset()
        runner = toy_runner(opt)
        sizes = []

        def sampling(opt, x1, y, **kwargs):
            sizes.append(len(x1))
            torch.randn(7)  # Sampling must not affect the next training draw.
            return x1[:, None], x1[:, None]

        runner.ddpm_sampling = sampling
        loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)
        state = torch.get_rng_state().clone()
        metrics = runner.evaluation(opt, 1, loader, None)
        self.assertEqual(sizes, [2, 2, 1])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertAlmostEqual(metrics['mae_m'], 0.5 * dataset.depth_std * 4, places=6)
        self.assertAlmostEqual(metrics['rmse_m'], metrics['ca4d_rmse_m'])

    def test_checkpoint_selection_final_save_and_step_metadata(self):
        opt = options(num_itr=3, val_every=2)
        runner = toy_runner(opt)
        runner.evaluation = Mock(side_effect=[{'rmse_m': 0.3}, {'rmse_m': 0.2}])
        saved = []

        def save(state, path):
            saved.append((str(path.name), copy.deepcopy(state)))

        with patch('i2sb.runner.torch.save', side_effect=save), patch('i2sb.runner.save_training_error_plot'):
            runner.train(opt, ToyDataset(), ToyDataset(), None)
        best = [state for name, state in saved if name == 'best.pt']
        self.assertEqual([state['best_rmse_m'] for state in best], [0.3, 0.2])
        self.assertEqual(saved[-1][0], 'latest.pt')
        self.assertEqual(saved[-1][1]['step'], 3)
        self.assertEqual(saved[-1][1]['sched']['last_epoch'], 3)
        self.assertEqual(saved[-1][1]['training_contract'], training_contract(opt))

    def test_legacy_or_changed_contract_is_rejected(self):
        opt = options()
        with self.assertRaises(ValueError):
            check_checkpoint_contract({}, opt)
        checkpoint = {'training_contract': training_contract(opt)}
        check_checkpoint_contract(checkpoint, opt)
        opt.val_nfe = 4
        with self.assertRaises(ValueError):
            check_checkpoint_contract(checkpoint, opt)

    def test_nonfinite_backward_skips_optimizer_ema_scheduler_and_saving(self):
        opt = options(num_itr=1)
        runner = toy_runner(opt)
        initial = [p.clone() for p in runner.net.parameters()]
        parameter = next(runner.rainfall_emb.parameters())
        hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
        with patch('i2sb.runner.torch.save') as save, patch('i2sb.runner.save_training_error_plot'):
            runner.train(opt, ToyDataset(), ToyDataset(), None)
            save.assert_not_called()
        hook.remove()
        self.assertEqual(runner.ema.num_updates, 0)
        for expected, actual in zip(initial, runner.net.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_skipped_microbatch_uses_actual_valid_sample_count(self):
        opt = options(num_itr=1)
        runner = toy_runner(opt)
        batch = custom_collate_fn([ToyDataset()[0], ToyDataset()[1]])
        good = runner.sample_batch(opt, iter([batch]), None)
        bad = list(good)
        bad[0] = torch.full_like(good[0], float('nan'))
        runner.sample_batch = Mock(side_effect=[good, tuple(bad)])
        runner.evaluation = Mock(return_value={'rmse_m': 0.2})
        with patch('i2sb.runner.normalize_and_clip_gradients', wraps=normalize_and_clip_gradients) as clip, \
                patch('i2sb.runner.torch.save'), patch('i2sb.runner.save_training_error_plot'):
            runner.train(opt, ToyDataset(), ToyDataset(), None)
        self.assertEqual(clip.call_args.args[1], 2)
        self.assertEqual(runner.ema.num_updates, 1)

    def test_missing_pair_is_not_silently_zero_filled(self):
        dataset = ToyDataset()
        broken = list(dataset[1])
        broken[5] = None
        with self.assertRaises(ValueError):
            custom_collate_fn([dataset[0], tuple(broken)])


if __name__ == '__main__':
    unittest.main()
