#!/usr/bin/env python3
"""
Download all Roboflow datasets using Private API key.
"""

import os
import sys
import shutil
from pathlib import Path
from roboflow import Roboflow

API_KEY = "Vppn97B7Xrb5jkYfU1gN"
BASE_DIR = Path("/home/nozima/table-cleanliness-detection/datasets")

DATASETS = {
    "tier1": [
        ("varun-qlsfy", "table-detection-gomjy", "1711 imgs - dirty/clean/occupied tables"),
        ("tablevision", "table-model", "131 imgs - food-court occupied/empty tables"),
        ("visionspace-adnsg", "ai-smart-restaurant-surveillance-veh3i", "3000 imgs - CCTV restaurant surveillance"),
        ("insper-erh3j", "restaurant-tables-ymr9k", "1544 imgs - tables, people, plates, waiters"),
    ],
    "tier2": [
        ("rohith-vr9nb", "clean-table", "857 imgs - occupied/clean/dirty"),
        ("parth-more", "table-cleanliness-detection", "1749 imgs - table cleanliness"),
    ],
    "tier3": [
        ("georges-workspace", "food-tray-detection", "214 imgs - food tray detection"),
        ("food-rlxlo", "dish-detection", "199 imgs - dish detection"),
        ("anton-althoff", "dirty-kitchen", "110 imgs - dirty kitchen"),
        ("data-rgpls", "leftover-food-detection-wpjyv", "100 imgs - leftover food"),
    ],
    "tier4": [
        ("project-eyfif", "proj3-food-waste-detection", "175 imgs - canteen food waste"),
    ],
}


def main():
    print("=" * 60)
    print("ROBOFLOW DATASET DOWNLOADER (Private API Key)")
    print("=" * 60)

    rf = Roboflow(api_key=API_KEY)
    stats = {"ok": 0, "fail": 0, "skip": 0}

    for tier, ds_list in DATASETS.items():
        tier_dir = BASE_DIR / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*40}")
        print(f"  {tier.upper()}")
        print(f"{'='*40}")

        for workspace, project_name, desc in ds_list:
            # Use project name as folder name (strip workspace suffix if present)
            folder_name = project_name.split("/")[-1]
            target = tier_dir / folder_name

            # Skip if already has images
            if target.exists():
                img_count = len(list(target.rglob("*.jpg"))) + len(list(target.rglob("*.png")))
                if img_count > 0:
                    print(f"\n  [SKIP] {folder_name} — already has {img_count} images")
                    stats["skip"] += 1
                    continue
                # Remove empty/broken folder
                shutil.rmtree(target, ignore_errors=True)

            print(f"\n  [DL] {folder_name}: {desc}")
            try:
                project = rf.workspace(workspace).project(project_name)
                versions = project.versions()

                if not versions:
                    print(f"  [WARN] No versions found for {folder_name}")
                    stats["fail"] += 1
                    continue

                # Get latest version
                latest = versions[0]
                print(f"    Version {latest.version} — downloading...")

                latest.download("yolov8", location=str(target))

                # Count downloaded images
                img_count = len(list(target.rglob("*.jpg"))) + len(list(target.rglob("*.png")))
                print(f"  [OK] {folder_name} — {img_count} images")
                stats["ok"] += 1

            except Exception as e:
                print(f"  [ERR] {folder_name}: {e}")
                stats["fail"] += 1

    # Summary
    print("\n" + "=" * 60)
    print("ROBOFLOW DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  OK:      {stats['ok']}")
    print(f"  Skipped: {stats['skip']}")
    print(f"  Failed:  {stats['fail']}")

    # Show folder sizes
    print("\nDataset sizes:")
    for tier in ["tier1", "tier2", "tier3", "tier4"]:
        tier_path = BASE_DIR / tier
        if tier_path.exists():
            for d in sorted(tier_path.iterdir()):
                if d.is_dir():
                    imgs = len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.png")))
                    print(f"  {tier}/{d.name}: {imgs} images")


if __name__ == "__main__":
    main()
