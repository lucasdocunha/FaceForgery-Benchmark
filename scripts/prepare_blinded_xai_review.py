"""Create a blinded human-review queue only when explicitly invoked.

Keep the private linkage file away from reviewers. No demographic or nuisance
attribute is inferred automatically. This command reads existing images but does
not load or execute any detector. Dataset permissions still apply to sharing.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explicability.contracts import contained, freeze, sha256
from src.explicability.prepare import read_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-linkage", type=Path, required=True,
                        help="Operator-only mapping, outside the review output directory")
    args = parser.parse_args()
    plan = read_plan(args.plan)
    out = args.output.resolve(); linkage_path = args.private_linkage.resolve()
    if linkage_path.is_relative_to(out):
        parser.error("Private linkage must be outside the reviewer output directory")
    if out.exists() or linkage_path.exists():
        parser.error("Use new destinations; do not overwrite annotations or a blinding key")
    samples = []
    for cohort in ("task3", "task3_controls"):
        samples.extend({**row, "private_cohort": cohort} for row in plan["cohorts"][cohort]["samples"])
    if len({row["id"] for row in samples}) != len(samples):
        raise ValueError("Hard-sample and control identities overlap")
    random.Random(plan["config"]["seed"]).shuffle(samples)
    # Validate every source before creating output, without substituting an image.
    from PIL import Image
    for row in samples:
        path = contained(args.images_root, row["img_name"])
        with Image.open(path) as image:
            image.verify()
    out.mkdir(parents=True); (out / "images").mkdir()
    queue, linkage = [], []
    for index, row in enumerate(samples):
        token = hashlib.sha256(f"{plan['plan_id']}:blinded:{row['id']}".encode()).hexdigest()[:20]
        path = contained(args.images_root, row["img_name"])
        with Image.open(path) as image:
            image.convert("RGB").save(out / "images" / f"{token}.png")
        queue.append({"review_order": index + 1, "blind_id": token, "image": f"images/{token}.png",
            "rater_id": "", "illumination": "", "occlusion": "", "visible_compression": "",
            "blur": "", "assessability": "", "confidence": "", "notes": ""})
        linkage.append({"blind_id": token, "id": row["id"], "img_name": row["img_name"],
            "y_true": row["y_true"], "cohort": row["private_cohort"], "source_sha256": sha256(path)})
    fields = ["review_order", "blind_id", "image", "rater_id", "illumination", "occlusion",
              "visible_compression", "blur", "assessability", "confidence", "notes"]
    with (out / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(queue)
    (out / "README.txt").write_text(
        "Review original images without detector outputs or attribution maps.\n"
        "illumination: typical / dim / bright / mixed / unassessable\n"
        "occlusion: none / partial / severe / unassessable\n"
        "visible_compression and blur: none / mild / severe / unassessable\n"
        "assessability: assessable / uncertain / unassessable\n"
        "confidence: low / medium / high\n"
        "These are subjective visible-condition annotations, not measurements of cause.\n"
        "Do not infer race, ethnicity, or other demographic attributes.\n"
        "Use independent copies for at least two raters; retain disagreements.\n"
        "Share only as allowed by the dataset's access and redistribution terms.\n", encoding="utf-8")
    freeze(linkage_path, {"plan_id": plan["plan_id"], "warning": "OPERATOR ONLY; never send to blinded reviewers",
                          "records": linkage})
    print(json.dumps({"review_images": len(queue), "output": str(out), "private_linkage": str(linkage_path)}, indent=2))


if __name__ == "__main__":
    main()
