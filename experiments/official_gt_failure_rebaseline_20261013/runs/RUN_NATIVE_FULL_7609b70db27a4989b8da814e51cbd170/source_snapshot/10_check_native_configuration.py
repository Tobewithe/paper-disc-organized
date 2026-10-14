"""Preflight the complete pinned configuration without inference or CUDA allocation."""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, require
from native_output_bridge import apply_runtime_policy, native_environment


def main():
    profile = read_json(ROOT / "paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json")
    for package, version in profile["packages"].items():
        require(importlib.metadata.version(package) == version, "version mismatch: " + package)
    require(platform.python_version() == profile["executor"]["python"], "wrong Python version")
    apply_runtime_policy(profile["runtime_policy"])
    env = native_environment()
    require(env["cuda_version"] == profile["executor"]["cuda_build"], "wrong CUDA build")
    for k, value in profile["runtime_policy"].items():
        require(env[k] == value, "runtime policy mismatch: " + k)
    from ultralytics.cfg import get_cfg
    import ultralytics
    config = get_cfg(overrides=profile["native_args"])
    for k, value in profile["native_args"].items():
        require(getattr(config, k) == value, "argument normalization drift: " + k)
    installed = Path(ultralytics.__file__).parent
    vendor = ROOT / "shared/vendor/ultralytics_8_4_100/ultralytics"
    count = 0
    for source in vendor.rglob("*"):
        if source.is_file() and source.suffix in {".py", ".yaml", ".yml"}:
            require(source.read_text(encoding="utf8") == (installed / source.relative_to(vendor)).read_text(encoding="utf8"),
                    "source mismatch: " + str(source.relative_to(vendor)))
            count += 1
    require(count >= 342, "incomplete source inventory")
    print(json.dumps({"status": "passed", "source_files": count, "environment": env}, indent=2))


if __name__ == "__main__":
    main()
