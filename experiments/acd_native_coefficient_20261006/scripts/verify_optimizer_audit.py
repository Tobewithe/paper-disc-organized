"""Exercise real GradScaler finite/overflow steps through the ACD audit.

A tiny CPU model supplies both native coefficient parameter prefixes. The
audited optimizer step is compared against the unchanged official trainer
step, with actual parameters and EMA tensors checked after every attempt.
This is an instrumentation test, not an experiment hyperparameter change.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.vendor:
        sys.path.insert(0, str(args.vendor.resolve()))

    import torch
    import train_acd as acd
    from ultralytics.utils.torch_utils import ModelEMA

    torch.set_num_threads(1)
    assert torch.__version__.split("+")[0] == "2.5.1", "Verification targets the actual PyTorch 2.5.1 runtime"
    torch.manual_seed(0)

    class TinyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cv4 = torch.nn.Linear(2, 1, bias=False)
            self.one2one_cv4 = torch.nn.Linear(2, 1, bias=False)
            self.bn = torch.nn.BatchNorm1d(2)

        def forward(self, value):
            value = self.bn(value)
            return torch.cat((self.cv4(value), self.one2one_cv4(value)), dim=1)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2, bias=False)]
                + [torch.nn.Identity() for _ in range(22)] + [TinyHead()])
            # A nontrivial frozen float verifies that EMA roundoff is synced.
            self.register_buffer("frozen_constant", torch.tensor([3.1415927]))
            with torch.no_grad():
                self.model[0].weight.copy_(torch.eye(2))
                self.model[23].cv4.weight.copy_(torch.tensor([[0.2, 0.3]]))
                self.model[23].one2one_cv4.weight.copy_(torch.tensor([[0.4, 0.5]]))

        def forward(self, value):
            return self.model[23](self.model[0](value))

    def state_copy(model):
        return {name: value.detach().clone() for name, value in model.state_dict().items()}

    def assert_state_equal(actual, expected, names, reason):
        for name in names:
            assert torch.equal(actual[name], expected[name]), f"{reason}: {name}"

    def new_scaler():
        return torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=100, enabled=True)

    def new_optimizer(model):
        return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)

    def new_stats():
        stats = defaultdict(int)
        for name in ("all_finite", "observed_all_gradients_finite",
                     "observed_all_scaled_gradients_finite", "all_observed_nonfinite_steps_skipped",
                     "all_applied_gradients_finite"):
            stats[name] = True
        stats["maximum_absolute_gradient"] = 0.0
        stats["unclassified_skipped_steps"] = 0
        return stats

    model = TinyModel().train()
    trainable = acd._freeze_for_coefficients(model)
    coefficient_names = sorted(trainable)
    frozen_names = set(model.state_dict()) - set(coefficient_names)
    reference_model = copy.deepcopy(model)
    initial_state = state_copy(model)

    trainer = object.__new__(acd.ACDTrainer)
    trainer.model = model
    trainer.optimizer = new_optimizer(model)
    trainer.scaler = new_scaler()
    trainer.ema = ModelEMA(model)
    trainer.amp = True
    trainer.accumulate = 1
    trainer.epoch = 0
    trainer._batch_counter = 0
    trainer._frozen_names = frozen_names
    trainer._gradient_stats = {branch: new_stats() for branch in ("one2many", "one2one")}
    trainer._optimizer_step_stats = new_stats()
    reference = SimpleNamespace(model=reference_model, optimizer=new_optimizer(reference_model),
                                scaler=new_scaler(), ema=ModelEMA(reference_model))
    assert trainer.scaler.is_enabled() and trainer.amp
    assert all(not module.training for module in model.modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))

    original_root = acd.RUN_ROOT
    log_records = []
    effects = []
    try:
        with tempfile.TemporaryDirectory(prefix="acd_optimizer_verification_") as directory:
            acd.RUN_ROOT = Path(directory)
            for attempt, phase in enumerate(("finite", "injected_inf", "finite_after_overflow"), 1):
                before = state_copy(model)
                ema_before = state_copy(trainer.ema.ema)
                scale_before = float(trainer.scaler.get_scale())
                updates_before = trainer.ema.updates
                sample = torch.tensor([[1.0, 2.0]])
                for subject in (trainer, reference):
                    loss = subject.model(sample).square().sum()
                    assert bool(torch.isfinite(loss)), "The synthetic forward loss must remain finite"
                    subject.scaler.scale(loss).backward()
                    if phase == "injected_inf":
                        # A single branch overflow must skip the entire shared
                        # optimizer, including the other branch's finite grads.
                        subject.model.model[23].cv4.weight.grad[0, 0] = float("inf")
                trainer._batch_counter = attempt
                trainer.optimizer_step()
                acd.SegmentationTrainer.optimizer_step(reference)

                after = state_copy(model)
                ema_after = state_copy(trainer.ema.ema)
                scale_after = float(trainer.scaler.get_scale())
                assert_state_equal(after, reference.model.state_dict(), after,
                                   "Audit changed official optimizer parameter behavior")
                assert_state_equal(ema_after, reference.ema.ema.state_dict(), coefficient_names,
                                   "Audit changed official coefficient EMA behavior")
                assert trainer.ema.updates == reference.ema.updates == updates_before + 1
                assert scale_after == float(reference.scaler.get_scale())
                assert_state_equal(after, initial_state, frozen_names, "Live frozen state changed")
                assert_state_equal(ema_after, initial_state, frozen_names, "EMA frozen state changed")
                assert all(bool(torch.isfinite(value).all()) for value in after.values()
                           if value.is_floating_point())
                assert all(bool(torch.isfinite(value).all()) for value in ema_after.values()
                           if value.is_floating_point())

                changed = [name for name in coefficient_names if not torch.equal(before[name], after[name])]
                if phase == "injected_inf":
                    assert not changed, "GradScaler overflow did not preserve coefficients exactly"
                    assert scale_after == scale_before * trainer.scaler.get_backoff_factor()
                    assert any(not torch.equal(ema_before[name], ema_after[name]) for name in coefficient_names), \
                        "Stock EMA must still move toward live coefficients on the skipped attempt"
                else:
                    assert len(changed) == len(coefficient_names), "A finite step failed to update a branch"
                    assert scale_after == scale_before

                records = [json.loads(line) for line in (acd.RUN_ROOT / "optimizer_step_audit.jsonl")
                           .read_text(encoding="utf-8").splitlines()]
                assert len(records) == attempt
                record = records[-1]
                assert record["optimizer_applied"] == (phase != "injected_inf")
                assert record["skipped_overflow_step"] == (phase == "injected_inf")
                assert record["optimizer_invoked"] == (phase != "injected_inf")
                assert record["scale_before"] == scale_before and record["scale_after"] == scale_after
                assert record["ema_updates_before"] == updates_before
                assert record["ema_updates_after"] == updates_before + 1
                assert record["learning_rates"] == [0.05] and record["accumulate"] == 1
                assert record["exception"] is None and not record["unsafe_step_prevented"]
                if phase == "injected_inf":
                    assert not record["scaled"]["one2many"]["all_gradients_finite"]
                    assert not record["unscaled"]["one2many"]["all_gradients_finite"]
                    assert record["unscaled"]["one2one"]["all_gradients_finite"]
                    assert record["at_optimizer_call"] is None
                log_records.append(record)
                effects.append({"attempt": attempt, "phase": phase,
                                "changed_coefficient_parameters": changed,
                                "live_coefficient_sha256": {name: acd._tensor_hash(after[name])
                                                             for name in coefficient_names},
                                "scale_before": scale_before, "scale_after": scale_after,
                                "ema_updates_before": updates_before,
                                "ema_updates_after": trainer.ema.updates,
                                "frozen_live_and_ema_exact": True,
                                "official_parameter_and_coefficient_ema_exact": True})
    finally:
        acd.RUN_ROOT = original_root

    summary = dict(trainer._optimizer_step_stats)
    assert summary["attempted_steps"] == 3 and summary["applied_steps"] == 2
    assert summary["skipped_overflow_steps"] == 1 and summary["ema_updates"] == 3
    assert not summary["observed_all_gradients_finite"]
    assert summary["all_applied_gradients_finite"] and summary["all_observed_nonfinite_steps_skipped"]
    assert summary["unsafe_step_prevented_count"] == summary["unclassified_skipped_steps"] == 0
    for branch, stats in trainer._gradient_stats.items():
        assert stats["attempted_steps"] == 3 and stats["applied_steps"] == 2
        assert stats["skipped_overflow_steps"] == 1 and stats["nonzero_steps"] == 2
        assert stats["all_finite"] and stats["all_applied_gradients_finite"]
        assert stats["observed_nonfinite_gradient_steps"] == (1 if branch == "one2many" else 0)
        assert stats["observed_nonfinite_scaled_gradient_steps"] == (1 if branch == "one2many" else 0)
    assert not trainer._gradient_stats["one2many"]["observed_all_gradients_finite"]
    assert not trainer._gradient_stats["one2many"]["observed_all_scaled_gradients_finite"]

    source = Path(acd.__file__).resolve()
    receipt = {"status": "PASS", "verified_at": datetime.now(timezone.utc).isoformat(),
               "torch": torch.__version__, "ultralytics": acd.ultralytics.__version__,
               "ultralytics_path": str(Path(acd.ultralytics.__file__).resolve()),
               "training_script": str(source),
               "training_script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "device": "cpu", "amp_flag": True, "real_grad_scaler_enabled": True,
               "synthetic_optimizer": "SGD", "synthetic_learning_rate": 0.05,
               "synthetic_init_scale": 8.0, "effects": effects,
               "optimizer_counters": summary,
               "branch_counters": {key: dict(value) for key, value in trainer._gradient_stats.items()},
               "optimizer_step_records": log_records,
               "scope": "Tiny CPU instrumentation test with real GradScaler; explicit Inf injection. No experiment optimizer, loss, AMP setting, data or hyperparameter was changed."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print("OPTIMIZER_AUDIT_VERIFICATION", json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
