# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure FOCUS ROI recall against ground-truth boxes (e.g. V*Bench) with a local HF model.

Input is a JSONL file, one sample per line:
    {"image": "path.jpg", "question": "...", "bbox": [x, y, w, h], "targets": ["optional", "gt targets"]}
``bbox`` is the ground-truth object box in original-image pixels (x, y, width, height); a list of boxes is accepted.

Reported: ``hit`` (a crop contains the GT box center), ``iou50`` (a crop has IoU >= 0.5 with a GT box),
``coverage`` (fraction of the GT box area inside the union-free best crop), and the average number of crops.

    python scripts/eval_focus_roi.py --model Qwen/Qwen3-VL-2B-Instruct --data vstar.jsonl --out out_dir
"""

import argparse
import json
import os

from PIL import Image


def _boxes(sample):
    b = sample["bbox"]
    b = [b] if isinstance(b[0], (int, float)) else b
    return [(x, y, x + w, y + h) for x, y, w, h in b]


def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a):
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None, help="save crops and per-sample results here")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--use-gt-targets", action="store_true", help="skip target extraction (step I)")
    p.add_argument("--config", default="{}", help="JSON overrides for FocusConfig")
    args = p.parse_args()

    from verl.experimental.focus.focus_worker import FocusWorker

    worker = FocusWorker(args.model, json.loads(args.config), dtype=args.dtype, device_map=args.device_map)
    root = os.path.dirname(os.path.abspath(args.data))
    samples = [json.loads(line) for line in open(args.data) if line.strip()]
    samples = samples[: args.limit]
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    stats = {"hit": 0, "iou50": 0, "coverage": 0.0, "crops": 0, "empty": 0}
    rows = []
    for i, s in enumerate(samples):
        path = s["image"] if os.path.isabs(s["image"]) else os.path.join(root, s["image"])
        image = Image.open(path).convert("RGB")
        targets = s.get("targets") if args.use_gt_targets else None
        res = worker.focus(image, s["question"], targets)
        gts = _boxes(s)
        crops = [c["box"] for c in res["crops"]]
        hit = any(c[0] <= (g[0] + g[2]) / 2 < c[2] and c[1] <= (g[1] + g[3]) / 2 < c[3] for c in crops for g in gts)
        iou50 = any(_inter(c, g) / (_area(c) + _area(g) - _inter(c, g) + 1e-9) >= 0.5 for c in crops for g in gts)
        cov = max((_inter(c, g) / (_area(g) + 1e-9) for c in crops for g in gts), default=0.0)
        stats["hit"] += hit
        stats["iou50"] += iou50
        stats["coverage"] += cov
        stats["crops"] += len(crops)
        stats["empty"] += not crops
        rows.append({"idx": i, "targets": res["targets"], "crops": crops, "hit": hit, "iou50": iou50, "coverage": cov})
        print(
            f"[{i + 1}/{len(samples)}] targets={res['targets']} crops={len(crops)} hit={hit} iou50={iou50}", flush=True
        )
        if args.out:
            for j, c in enumerate(res["crops"]):
                image.crop(tuple(c["box"])).save(os.path.join(args.out, f"{i:04d}_{j}.jpg"))

    n = max(len(samples), 1)
    summary = {
        "n": len(samples),
        "hit": stats["hit"] / n,
        "iou50": stats["iou50"] / n,
        "mean_coverage": stats["coverage"] / n,
        "mean_crops": stats["crops"] / n,
        "no_crop_rate": stats["empty"] / n,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump({"summary": summary, "samples": rows}, f, indent=2)


if __name__ == "__main__":
    main()
