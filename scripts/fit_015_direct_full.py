"""015 direct-only experimental full fit (manual, CPU only).

Complements the post-hoc direct-only ablation (5 fold models) with one
full-data LR fit on identical frozen inputs/params/weights, per handoff
step 1. Does not modify any 015/ablation originals; writes only
artifacts/015-analog/direct-only-ablation/models/lr_direct_only_experimental_full.joblib
+ full_fit.json. Not deployed, no OOF claim for this model.

Manual run: .venv-015/bin/python scripts/fit_015_direct_full.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_015_candidate_ranker import CONFIG, fit_model, load_inputs, sha

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "artifacts/015-analog/features"
OUT_DIR = ROOT / "artifacts/015-analog/direct-only-ablation"
DIRECT_FEATURE = "direct_quality"


def main():
    manifest, arrays, ordered, weights, _, row_fold, _ = load_inputs(IN_DIR)
    names = manifest["feature_names"]
    base = manifest["baseline_feature_names"]
    assert DIRECT_FEATURE in manifest["analog_feature_names"], "direct_quality missing"
    feat_names = list(base) + [DIRECT_FEATURE]
    spec = {"name": "lr_direct_only_experimental_full", "family": "lr",
            "feature_names": feat_names,
            "feature_indices": [names.index(x) for x in feat_names],
            "array": "X"}
    if CONFIG["logistic"] != {"C": 1, "max_iter": 1000, "solver": "lbfgs"}:
        raise ValueError("015 fixed logistic parameters changed")
    rows = np.arange(len(arrays["X"]), dtype=np.int64)
    t0 = time.perf_counter()
    bundle, info = fit_model(spec, arrays, rows, weights)
    wall = time.perf_counter() - t0
    out_model = OUT_DIR / "models/lr_direct_only_experimental_full.joblib"
    joblib.dump(bundle, out_model, compress=3)
    record = {"spec": spec, "logistic_params": CONFIG["logistic"],
              "rows": info["rows"], "positives": info["positives"],
              "weight_sum": info["weight_sum"],
              "positive_weight_sum": info["positive_weight_sum"],
              "fit_wall_seconds": wall, "warnings": info["warnings"],
              "coefficient": info["coefficient"], "intercept": info["intercept"],
              "n_iter": info["n_iter"],
              "model_sha256": sha(out_model),
              "note": "Experimental full-data fit for later new-panel prediction only; "
                      "training rows are the 015 research panel, no OOF performance. Not deployed."}
    (OUT_DIR / "full_fit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"model": out_model.name, "rows": info["rows"],
                      "warnings": info["warnings"],
                      "sha256": record["model_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
