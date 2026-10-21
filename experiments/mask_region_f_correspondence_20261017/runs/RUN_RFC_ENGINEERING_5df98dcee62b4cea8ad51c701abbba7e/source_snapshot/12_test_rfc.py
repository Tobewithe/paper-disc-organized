"""CPU synthetic behavior checks. No COCO, base forward, optimizer step, or cache export."""
from __future__ import annotations
import copy
from dataclasses import replace
import unittest

import torch

from rfc_paths import ROOT, STUDY
from rfc_inputs import (make_sidecar, prepare_inputs, raw_location, sample_region_features,
                        spatial_indices, validate_replay, validate_native)
from rfc_model import RegionFAdapter, parameter_receipt
from rfc_controls import (native_area_gate, gated_delta, mix_native_rows,
                          proto_roi, functional_rms, match_functional_amplitude)
from rfc_contract import (read_json, check_source_references, check_training_design,
                          check_native_args, check_native_runtime, execution_readiness)
from adapter_model import CoefficientAdapter, loss_grids
from adapter_primitives import sample_spatial_roi
from soft_iou_loss import soft_iou_grouped_loss

CACHE_SHA = 'a'*64
PRED_SHA = 'b'*64


def fixture(count=4, same_level=False):
    generator = torch.Generator().manual_seed(431)
    shapes = [[256, 8, 12], [512, 4, 6], [512, 2, 3]]
    dense = [torch.randn(shape, generator=generator) for shape in shapes]
    ids = [0 if same_level else (0, 97, 120, 0)[i % 4] for i in range(count)]
    locations = [raw_location(i, shapes) for i in ids]
    cache = {
        'image_id': 43, 'proto': torch.randn(32, 16, 24, generator=generator)*.1,
        'c0': torch.randn(count, 32, generator=generator)*.1,
        'boxes_input': torch.tensor([[16., 16., 64., 48.]]*count).reshape(count, 4),
        'scores': torch.full((count,), .6), 'classes': torch.arange(count).remainder(80).float(),
        'input_hw': [64, 96], 'native_decode_hw': [64, 96], 'original_hw': [48, 80],
        'ratio_pad': ((1., 1.), (8., 8.)), 'batch_index': 7, 'batch_position': 2,
        'image_sha256': 'c'*64, 'input_tensor_sha256': 'd'*64,
        'head_feature_shapes': shapes, 'raw_ids': ids, 'raw_locations': locations,
        'output_ranks': list(range(count)), 'prediction_indices': list(range(100, 100+count)),
        'feature_sources': {'head_module': 'model.23', 'head_layer_index': 23,
                            'source_layer_indices': [16, 19, 22],
                            'feature_definition': 'one2one.feats at actual head input; before prediction subheads'},
        'device': 'cpu',
    }
    cache['features'] = [dense[loc['level']][:, loc['row'], loc['column']].clone() for loc in locations]
    return cache, dense


def inputs_for(cache, regions, phase='eval:0', gate=None):
    training = phase.startswith('train:')
    sidecar = make_sidecar(cache, regions, CACHE_SHA, PRED_SHA,
                           'train_all_rows' if training else 'eval_gated_rows')
    if not training and gate is None:
        gate = torch.ones(len(cache['c0']), dtype=torch.bool)
    return prepare_inputs(cache, sidecar, CACHE_SHA, PRED_SHA, phase, gate)


class RFCBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        spec = read_json(STUDY/'DESIGN_SPEC.json')
        cls.spec = spec
        cls.initial = torch.load(ROOT/spec['training']['initialization_reference'],
                                 map_location='cpu', weights_only=True)['state_dict']
        cls.g_state = torch.load(ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/'
                                'RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/last.pt',
                                map_location='cpu', weights_only=True)['state_dict']

    def setUp(self):
        self.cache, self.dense = fixture()
        self.regions = sample_region_features(self.cache, self.dense)

    def model(self, arm='A', initial=False):
        model = RegionFAdapter(arm)
        model.load_state_dict(self.initial if initial else self.g_state, strict=True)
        return model

    def test_parameter_and_state_compatibility(self):
        a, q, g = self.model('A'), self.model('Q'), CoefficientAdapter('B')
        self.assertEqual(parameter_receipt(a), parameter_receipt(q))
        self.assertEqual(parameter_receipt(a), parameter_receipt(g))
        self.assertEqual(parameter_receipt(a)['parameters'], 167872)
        self.assertEqual(parameter_receipt(a)['parameter_tensors'], 17)
        self.assertTrue(all(torch.equal(a.state_dict()[k], q.state_dict()[k]) for k in a.state_dict()))

    def test_actual_initialization_zero_for_both_layouts(self):
        inputs = inputs_for(self.cache, self.regions)
        for arm in ('A', 'Q'):
            self.assertTrue(torch.equal(self.model(arm, initial=True)(inputs, 'eval:0'), torch.zeros(4, 32)))

    def test_broadcast_F_matches_nonzero_G_exactly(self):
        grids = [f[:, None, None].expand(-1, 8, 8).clone() for f in self.cache['features']]
        inputs = inputs_for(self.cache, grids)
        g = CoefficientAdapter('B')
        g.load_state_dict(self.g_state)
        reference = g(self.cache, 'eval:0')
        self.assertGreater(float(reference.detach().abs().max()), 0.)
        for arm in ('A', 'Q'):
            self.assertTrue(torch.equal(self.model(arm)(inputs, 'eval:0'), reference))

    def test_Q_permutation_is_joint_bijection_not_global_inverse(self):
        order = spatial_indices(43, 0)
        self.assertEqual(sorted(order), list(range(64)))
        self.assertNotEqual(order, spatial_indices(43, 1))
        self.assertNotEqual(order, spatial_indices(44, 0))
        self.assertEqual(spatial_indices(43, 0, True), tuple(range(64)))
        original = torch.arange(3*64).reshape(3, 64)
        permuted = original[:, order]
        self.assertEqual(sorted(map(tuple, original.T.tolist())), sorted(map(tuple, permuted.T.tolist())))
        self.assertTrue(torch.equal(permuted[2]-permuted[0], torch.full((64,), 128)))

    def test_layout_affects_nonzero_forward_and_J_equals_Q(self):
        a, q = self.model('A'), self.model('Q')
        inputs = inputs_for(self.cache, self.regions)
        aligned, shuffled = a(inputs, 'eval:0'), q(inputs, 'eval:0')
        self.assertGreater(float((aligned-shuffled).detach().abs().max()), 1e-6)
        self.assertTrue(torch.equal(a(inputs, 'eval:0', intervention='J'), shuffled))

    def test_Q_permutation_constant_across_phases(self):
        q = self.model('Q')
        train = inputs_for(self.cache, self.regions, phase='train:0')
        evaluation = inputs_for(self.cache, self.regions)
        self.assertTrue(torch.equal(q(train, 'train:0'), q(train, 'train:2')))
        self.assertTrue(torch.equal(q(train, 'train:0'), q(evaluation, 'eval:0')))

    def test_intervention_and_scope_cannot_sneak_into_training(self):
        a, q = self.model('A'), self.model('Q')
        train = inputs_for(self.cache, self.regions, 'train:0')
        evaluation = inputs_for(self.cache, self.regions)
        with self.assertRaises(ValueError): a(train, 'train:0', intervention='J')
        with self.assertRaises(ValueError): q(evaluation, 'eval:0', intervention='J')
        with self.assertRaises(ValueError): a(evaluation, 'train:0')

    def test_spatial_sampler_agrees_with_analytic_field_across_scales_and_P(self):
        cache = copy.deepcopy(self.cache)
        maps = []
        for channels, height, width in cache['head_feature_shapes']:
            x = (torch.arange(width)+.5)*(96/width)
            y = (torch.arange(height)+.5)*(64/height)
            maps.append((x[None, :]+2*y[:, None])[None].expand(channels, -1, -1).clone())
        cache['features'] = [maps[l['level']][:, l['row'], l['column']].clone() for l in cache['raw_locations']]
        actual = sample_region_features(cache, maps)
        x = 16+(torch.arange(8)+.5)*6
        y = 16+(torch.arange(8)+.5)*4
        expected = x[None, :]+2*y[:, None]
        for grid in actual:
            self.assertTrue(torch.allclose(grid[0], expected, atol=2e-5, rtol=0))
        x_p = (torch.arange(24)+.5)*4
        y_p = (torch.arange(16)+.5)*4
        p = (x_p[None, :]+2*y_p[:, None])[None].expand(32, -1, -1).clone()
        sampled_p = sample_spatial_roi(p, cache['c0'], cache['boxes_input'], cache['input_hw'])
        self.assertTrue(torch.allclose(sampled_p[:, 0], torch.stack([v[0] for v in actual]), atol=2e-5, rtol=0))

    def test_sampling_chunk_boundary_preserves_duplicate_raw_rows(self):
        cache, dense = fixture(20, same_level=True)
        regions = sample_region_features(cache, dense)
        self.assertEqual(len(regions), 20)
        self.assertTrue(all(torch.equal(x, regions[0]) for x in regions))
        self.assertEqual(len(set(cache['raw_ids'])), 1)
        self.assertEqual(validate_native(cache), 20)

    def test_dense_raw_point_mismatch_is_rejected(self):
        self.dense[0][0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'original F'):
            sample_region_features(self.cache, self.dense)

    def test_sidecar_identity_includes_prediction_SHA_not_just_raw(self):
        sidecar = make_sidecar(self.cache, self.regions, CACHE_SHA, PRED_SHA, 'train_all_rows')
        self.assertNotEqual(sidecar['output_detection_ids'][0], sidecar['output_detection_ids'][3])
        bad = copy.deepcopy(sidecar)
        bad['output_detection_ids'][0] = bad['output_detection_ids'][3]
        with self.assertRaises(ValueError): prepare_inputs(self.cache, bad, CACHE_SHA, PRED_SHA, 'train:0')
        with self.assertRaises(ValueError): prepare_inputs(self.cache, sidecar, 'e'*64, PRED_SHA, 'train:0')
        with self.assertRaises(ValueError): prepare_inputs(self.cache, sidecar, CACHE_SHA, 'e'*64, 'train:0')

    def test_sidecar_rejects_extra_GT_or_wrong_geometry(self):
        sidecar = make_sidecar(self.cache, self.regions, CACHE_SHA, PRED_SHA, 'train_all_rows')
        for field, value in [('targets', [1]), ('oracle_coefficients', 'forbidden')]:
            bad = {**sidecar, field: value}
            with self.assertRaises(ValueError): prepare_inputs(self.cache, bad, CACHE_SHA, PRED_SHA, 'train:0')
        bad = copy.deepcopy(sidecar)
        bad['identity']['batch_position'] += 1
        with self.assertRaises(ValueError): prepare_inputs(self.cache, bad, CACHE_SHA, PRED_SHA, 'train:0')

    def test_exact_replay_detects_tensor_and_batch_drift(self):
        validate_replay(self.cache, copy.deepcopy(self.cache))
        for field in ('proto', 'c0', 'boxes_input', 'scores'):
            bad = copy.deepcopy(self.cache)
            bad[field].view(-1)[0] += .001
            with self.assertRaises(ValueError): validate_replay(self.cache, bad)
        bad = copy.deepcopy(self.cache)
        bad['batch_index'] += 1
        with self.assertRaises(ValueError): validate_replay(self.cache, bad)

    def test_training_cannot_omit_or_gate_any_F(self):
        regions = list(self.regions)
        regions[0] = None
        with self.assertRaises(ValueError): inputs_for(self.cache, regions, 'train:0')
        with self.assertRaises(ValueError): inputs_for(self.cache, self.regions, 'train:0', torch.ones(4, dtype=torch.bool))

    def test_eval_omits_only_gate_off_F_and_retains_all_rows(self):
        gate = torch.tensor([True, False, True, False])
        regions = [r if active else None for r, active in zip(self.regions, gate.tolist())]
        inputs = inputs_for(self.cache, regions, gate=gate)
        for rank in (1, 3):
            self.assertTrue(torch.equal(inputs.regions[rank], self.cache['features'][rank][:, None, None].expand(-1, 8, 8)))
        self.assertEqual(self.model()(inputs, 'eval:0').shape, (4, 32))
        with self.assertRaises(ValueError): inputs_for(self.cache, self.regions, gate=gate)

    def test_loss_backward_reaches_all_17_parameters_not_frozen_inputs(self):
        inputs = inputs_for(self.cache, self.regions, 'train:0')
        inputs = replace(inputs, proto=inputs.proto.clone().requires_grad_(),
                         c0=inputs.c0.clone().requires_grad_(),
                         features=tuple(f.clone().requires_grad_() for f in inputs.features),
                         regions=tuple(f.clone().requires_grad_() for f in inputs.regions))
        model = self.model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        delta = model(inputs, 'train:0')
        z, baseline, roi = loss_grids(self.cache, self.cache['c0']+delta)
        labels = {i: torch.zeros_like(z[i]) for i in range(4)}
        labels[1][:, :12] = 1
        loss, _ = soft_iou_grouped_loss(z, baseline, roi, labels)
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        self.assertEqual(len(grads), 17)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
        self.assertTrue(all(t.grad is None for t in (inputs.proto, inputs.c0, *inputs.features, *inputs.regions)))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_zero_output_image_is_supported(self):
        cache, dense = fixture(0)
        self.assertEqual(sample_region_features(cache, dense), [])
        inputs = inputs_for(cache, [], 'train:0')
        delta = self.model(initial=True)(inputs, 'train:0')
        self.assertEqual(delta.shape, (0, 32))
        delta.sum().backward()

    def test_original_proto_ROI_semantics_preserved(self):
        self.cache['boxes_input'][0] = torch.tensor([-2.3, 1.2, 20.5, 29.9])
        self.cache['boxes_input'][1] = torch.tensor([96., 64., 104., 72.])
        _, _, reference = loss_grids(self.cache, self.cache['c0'])
        self.assertTrue(torch.equal(proto_roi(self.cache, 'cpu'), reference))

    def test_functional_amplitude_uses_P_response_not_coefficient_L2(self):
        self.cache['proto'].zero_()
        self.cache['proto'][0] = 1
        self.cache['proto'][1] = 10
        first, second = torch.zeros(4, 32), torch.zeros(4, 32)
        first[:, 0], second[:, 1] = 4, 1
        a, b, receipt = match_functional_amplitude(self.cache, first, second, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.equal(a, first))
        self.assertTrue(torch.allclose(b[:, 1], torch.full((4,), .4)))
        self.assertTrue(torch.allclose(functional_rms(self.cache, a)[0], functional_rms(self.cache, b)[0], atol=1e-7, rtol=1e-5))
        self.assertEqual(receipt['reason'], ['matched']*4)

    def test_amplitude_zero_empty_ROI_and_gate_keep_denominator(self):
        first, second = torch.ones(4, 32), torch.ones(4, 32)*2
        self.cache['boxes_input'][1] = torch.tensor([96., 64., 104., 72.])
        second[2] = 0
        gate = torch.tensor([False, True, True, True])
        a, b, receipt = match_functional_amplitude(self.cache, first, second, gate)
        self.assertEqual(receipt['reason'], ['gate_off', 'empty_roi', 'near_zero_direction', 'matched'])
        self.assertTrue(torch.equal(a[:3], torch.zeros(3, 32)))
        self.assertTrue(torch.equal(b[:3], torch.zeros(3, 32)))
        self.assertTrue(receipt['all_rows_retained'])
        second[0, 0] = float('nan')
        with self.assertRaises(ValueError): match_functional_amplitude(self.cache, first, second, gate)

    def test_gate_boundary_and_exact_native_row_mixing(self):
        gate = native_area_gate(torch.tensor([0, 9215, 9216, 10000]))
        self.assertEqual(gate.tolist(), [False, False, True, True])
        rows = [dict(image_id=43, bbox=[1, 2, 3, 4], score=.5, category_id=i,
                     segmentation={'size': [48, 80], 'counts': f'native{i}'}, custom_identity=i) for i in range(4)]
        masks = [{'size': [48, 80], 'counts': f'changed{i}'} for i in range(4)]
        result = mix_native_rows(rows, masks, gate)
        self.assertEqual(result[:2], rows[:2])
        for i, output in enumerate(result):
            self.assertEqual({k:v for k,v in output.items() if k != 'segmentation'},
                             {k:v for k,v in rows[i].items() if k != 'segmentation'})
        self.assertEqual(rows[2]['segmentation']['counts'], 'native2')
        self.assertTrue(torch.equal(gated_delta(torch.ones(4, 32), gate)[:2], torch.zeros(2, 32)))

    def test_real_source_hashes_training_and_data_contracts(self):
        self.assertEqual(check_source_references()['files'], 33)
        self.assertEqual(check_training_design()['steps_per_arm'], 1500)

    def test_native_config_unknown_missing_and_value_drift_rejected(self):
        expected = {'batch': 16, 'rect': True, 'mask_ratio': 4, 'model': 'original.pt'}
        check_native_args(expected, {**expected, 'model': 'different-location.pt'})
        variants = [{**expected, 'batch': 8}, {**expected, 'mask_ratio': 1},
                    {**expected, 'unknown_model_parameter': True}, {k:v for k,v in expected.items() if k != 'rect'}]
        for observed in variants:
            with self.assertRaises(ValueError): check_native_args(expected, observed)
        runtime = self.spec['training']['numeric_environment_exact_reference']
        drift = copy.deepcopy(runtime)
        drift['cudnn_allow_tf32'] = False
        with self.assertRaises(ValueError): check_native_runtime(runtime, drift)

    def test_training_budget_drift_and_real_execution_not_ready(self):
        bad = copy.deepcopy(self.spec)
        bad['training']['formal_steps_per_arm'] = 1501
        with self.assertRaises(ValueError): check_training_design(bad)
        self.assertFalse(execution_readiness({'new_cache_export_authorized':False,'training_authorized':False},receipts={})['real_execution_ready'])

    def test_receipt_presence_does_not_mean_real_smoke_passed(self):
        authorized = {'new_cache_export_authorized': True, 'training_authorized': True}
        failed = {'EXECUTION_FREEZE.json': {'status': 'frozen', 'configuration_validation': 'passed'},
                  'REAL_INPUT_REPLAY.json': {'status': 'failed', 'tensor_identity_exact': False,
                                            'zero_delta_RLE_exact': True, 'source_fingerprints_match': True},
                  'REAL_SMOKE_VERIFICATION.json': {'status': 'passed', 'native_environment_parity': False,
                                                  'parameter_and_gradient_checks': 'passed'}}
        result = execution_readiness(authorized, receipts=failed)
        self.assertFalse(result['real_execution_ready'])
        self.assertIn('REAL_SMOKE_VERIFICATION.json: native_environment_parity not satisfied', result['outstanding'])
        self.assertIn('REAL_INPUT_REPLAY.json: tensor_identity_exact not satisfied', result['outstanding'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
