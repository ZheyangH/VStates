"""Run on CPU without SlowFast, Fairseq, or dataset downloads."""
import ast
from abc import ABC
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from vidsitu_code.state_transition import StateTransitionRepresentation


class TransitionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def model(self, mode='mlp'):
        return StateTransitionRepresentation(6, hidden_dim=8, mode=mode, heads=2)

    def test_shapes_gradients_and_order(self):
        for mode in ('mlp', 'attention'):
            model = self.model(mode)
            states = torch.randn(2, 5, 6, requires_grad=True)
            valid = torch.ones(2, 5, dtype=torch.bool)
            baseline = states.mean(1)
            result = model(states, valid, baseline)
            self.assertEqual(result.shape, (2, 6))
            self.assertTrue(torch.isfinite(result).all())
            result.square().sum().backward()
            self.assertGreater(model.project.weight.grad.abs().sum().item(), 0)
            before = model.transition_summary(states, valid)
            after = model.transition_summary(states.flip(1), valid)
            self.assertFalse(torch.allclose(before, after))

    def test_missing_frames_do_not_create_transitions(self):
        for mode in ('mlp', 'attention'):
            model = self.model(mode)
            states = torch.randn(1, 5, 6)
            valid = torch.tensor([[True, True, False, True, True]])
            baseline = states.mean(1)
            self.assertTrue(torch.equal(model(states, valid, baseline), baseline))
            valid = torch.tensor([[True, True, True, False, False]])
            changed = states.clone()
            changed[:, 3:] = float('nan')
            self.assertTrue(torch.allclose(model.transition_summary(states, valid),
                                           model.transition_summary(changed, valid)))

    def test_short_sequence_and_singleton(self):
        model = self.model()
        states = torch.randn(1, 2, 6)
        baseline = states.mean(1)
        self.assertTrue(torch.equal(model(states, torch.ones(1, 2, dtype=torch.bool), baseline), baseline))

    def test_stride_and_bad_configuration(self):
        model = StateTransitionRepresentation(6, window_size=2, stride=2, hidden_dim=8)
        result = model.transition_summary(torch.randn(1, 5, 6), torch.ones(1, 5, dtype=torch.bool))
        self.assertEqual(result.shape, (1, 6))
        for kwargs in ({'window_size': 1}, {'stride': 0}, {'mode': 'bad'},
                       {'mode': 'attention', 'hidden_dim': 7, 'heads': 2}):
            with self.assertRaises(ValueError):
                StateTransitionRepresentation(6, **kwargs)

    def test_real_aggregation_integration(self):
        # Load the actual aggregation classes without importing optional backbones.
        path = Path(__file__).parents[1] / 'vidsitu_code/mdl_sf_base.py'
        tree = ast.parse(path.read_text())
        names = {'PPState', 'StateAgg', 'AvgAgg', 'TransitionAgg'}
        selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[])
        namespace = dict(torch=torch, nn=nn, ABC=ABC,
                         StateTransitionRepresentation=StateTransitionRepresentation)
        exec(compile(selected, str(path), 'exec'), namespace)
        cfg = SimpleNamespace(window_size=3, stride=1, hidden_dim=8,
                              mode='mlp', heads=2, dropout=0.0)
        frames = torch.randn(1, 1, 4, 4, 2, 2, requires_grad=True)
        boxes = torch.tensor([[[[[0, 0, 2, 2]] * 4, [[0, 0, 0, 0]] * 4]]])
        agg = namespace['TransitionAgg']((1, 4, 4, 2, 2), 2, cfg)
        result = agg(frames, boxes)
        self.assertEqual(result.shape, (1, 2, 6))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[:, 1], torch.zeros_like(result[:, 1])))
        result.sum().backward()
        self.assertIsNotNone(agg.transition.project.weight.grad)
        self.assertIsNotNone(frames.grad)


if __name__ == '__main__':
    unittest.main()
