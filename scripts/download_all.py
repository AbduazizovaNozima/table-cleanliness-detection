#!/usr/bin/env python3
"""
Download all datasets for table cleanliness detection project.
Works without Roboflow API key — uses public download URLs where possible.
"""

import os
import sys
import json
import subprocess
import zipfile
import shutil
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE_DIR = Path(__file__).resolve().parent.parent / "datasets"

# ============================================================
# ROBOFLOW DATASETS — grouped by tier
# ============================================================
ROBOFLOW_DATASETS = {
    "tier1": [
        {
            "name": "table-detection-gomjy",
            "workspace": "varun-qlsfy",
            "project": "table-detection-gomjy",
            "desc": "1711 imgs - dirty/clean/occupied tables",
        },
        {
            "name": "table-model",
            "workspace": "tablevision",
            "project": "table-model",
            "desc": "131 imgs - food-court occupied/empty tables",
        },
        {
            "name": "ai-smart-restaurant-surveillance",
            "workspace": "visionspace-adnsg",
            "project": "ai-smart-restaurant-surveillance-veh3i",
            "desc": "3000 imgs - CCTV restaurant surveillance",
        },
        {
            "name": "restaurant-tables",
            "workspace": "insper-erh3j",
            "project": "restaurant-tables-ymr9k",
            "desc": "1544 imgs - tables, people, plates, waiters",
        },
    ],
    "tier2": [
        {
            "name": "clean-table",
            "workspace": "rohith-vr9nb",
            "project": "clean-table",
            "desc": "857 imgs - occupied/clean/dirty",
        },
        {
            "name": "table-cleanliness-detection",
            "workspace": "parth-more",
            "project": "table-cleanliness-detection",
            "desc": "1749 imgs - table cleanliness",
        },
    ],
    "tier3": [
        {
            "name": "food-tray-detection",
            "workspace": "georges-workspace",
            "project": "food-tray-detection",
            "desc": "214 imgs - food tray detection",
        },
        {
            "name": "dish-detection",
            "workspace": "food-rlxlo",
            "project": "dish-detection",
            "desc": "199 imgs - dish detection",
        },
        {
            "name": "dirty-kitchen",
            "workspace": "anton-althoff",
            "project": "dirty-kitchen",
            "desc": "110 imgs - dirty kitchen",
        },
        {
            "name": "leftover-food-detection",
            "workspace": "data-rgpls",
            "project": "leftover-food-detection-wpjyv",
            "desc": "100 imgs - leftover food",
        },
    ],
    "tier4": [
        {
            "name": "food-waste-detection",
            "workspace": "project-eyfif",
            "project": "proj3-food-waste-detection",
            "desc": "175 imgs - canteen food waste",
        },
    ],
}

# HuggingFace datasets
HF_DATASETS = [
    {
        "name": "FoodLogAthl-218",
        "repo": "FoodLog/FoodLogAthl-218",
        "desc": "6925 imgs - messy multi-dish food photos",
        "tier": "tier3",
    },
    {
        "name": "food-waste-dataset",
        "repo": "Voxel51/food-waste-dataset",
        "desc": "375 imgs - food before/after waste",
        "tier": "tier3",
    },
]


def try_roboflow_public_api(workspace, project):
    """Try to get dataset info from Roboflow public API."""
    url = f"https://universe.roboflow.com/api/{workspace}/{project}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def try_roboflow_download(workspace, project, version=1, fmt="yolov8"):
    """Try multiple methods to download a Roboflow dataset."""
    # Method 1: Public universe API
    urls_to_try = [
        f"https://universe.roboflow.com/ds/{workspace}/{project}/{version}/{fmt}",
        f"https://app.roboflow.com/{workspace}/{project}/{version}/{fmt}",
        f"https://universe.roboflow.com/{workspace}/{project}/dataset/{version}/download/{fmt}",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }

    for url in urls_to_try:
        try:
            r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith(("application/zip", "application/octet")):
                return r.content
        except:
            continue

    return None


def download_roboflow_via_cli(ds_info, target_dir):
    """Download Roboflow dataset using CLI curl approach."""
    name = ds_info["name"]
    workspace = ds_info["workspace"]
    project = ds_info["project"]
    target = target_dir / name

    if target.exists() and any(target.iterdir()):
        print(f"  [SKIP] {name} — already exists")
        return "skip"

    print(f"  [TRY] {name}: {ds_info['desc']}")

    # Try to get project info
    info = try_roboflow_public_api(workspace, project)
    if info:
        print(f"    Found project info: {json.dumps({k: info[k] for k in ['name', 'id'] if k in info}, indent=2)[:200]}")

    # Try to download
    data = try_roboflow_download(workspace, project)
    if data:
        target.mkdir(parents=True, exist_ok=True)
        zip_path = target / "dataset.zip"
        zip_path.write_bytes(data)
        try:
            with zipfile.ZipFile(str(zip_path), 'r') as z:
                z.extractall(str(target))
            zip_path.unlink()
            print(f"  [OK] {name} downloaded and extracted")
            return "ok"
        except:
            print(f"  [ERR] {name} — zip extraction failed")
            return "fail"

    # If download failed, save URL for manual download
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://universe.roboflow.com/{workspace}/{project}"
    with open(target / "DOWNLOAD_MANUALLY.txt", "w") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Description: {ds_info['desc']}\n")
        f.write(f"URL: {url}\n")
        f.write(f"\nTo download:\n")
        f.write(f"1. Go to {url}\n")
        f.write(f"2. Click 'Download Dataset'\n")
        f.write(f"3. Select YOLOv8 format\n")
        f.write(f"4. Download and extract to this folder\n")
        f.write(f"\nOR use Roboflow Python:\n")
        f.write(f"  from roboflow import Roboflow\n")
        f.write(f"  rf = Roboflow(api_key='YOUR_KEY')\n")
        f.write(f"  project = rf.workspace('{workspace}').project('{project}')\n")
        f.write(f"  version = project.versions()[0]\n")
        f.write(f"  version.download('yolov8', location='{target}')\n")

    print(f"  [MANUAL] {name} — saved download instructions to {target / 'DOWNLOAD_MANUALLY.txt'}")
    return "manual"


def download_huggingface(ds_info):
    """Download a HuggingFace dataset."""
    name = ds_info["name"]
    tier = ds_info["tier"]
    target = BASE_DIR / tier / name

    if target.exists() and any(target.iterdir()):
        print(f"  [SKIP] {name} — already exists")
        return "skip"

    print(f"  [DL] {name}: {ds_info['desc']}")
    target.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        dataset = load_dataset(ds_info["repo"], trust_remote_code=True)

        split_name = list(dataset.keys())[0]
        ds = dataset[split_name]
        total = len(ds)
        print(f"    {total} samples in split '{split_name}', columns: {ds.column_names}")

        # Save images if there's an image column
        img_dir = target / "images"
        img_dir.mkdir(exist_ok=True)

        img_col = None
        for col in ds.column_names:
            feat_str = str(ds.features[col])
            if "Image" in feat_str:
                img_col = col
                break
        if img_col is None:
            for candidate in ["image", "img", "photo"]:
                if candidate in ds.column_names:
                    img_col = candidate
                    break

        saved = 0
        if img_col:
            for i, sample in enumerate(ds):
                if i % 1000 == 0 and i > 0:
                    print(f"    Saved {i}/{total} images...")
                try:
                    img = sample[img_col]
                    if img is not None:
                        img.save(str(img_dir / f"{i:06d}.jpg"))
                        saved += 1
                except Exception:
                    continue
            print(f"    Saved {saved}/{total} images")
        else:
            # No image column — save raw data
            raw_dir = target / "raw"
            raw_dir.mkdir(exist_ok=True)
            ds.save_to_disk(str(raw_dir))
            print(f"    Saved raw dataset to {raw_dir}")

        # Save metadata/labels
        meta = {"columns": ds.column_names, "features": {col: str(ds.features[col]) for col in ds.column_names}, "num_rows": total}
        label_cols = [c for c in ds.column_names if c != img_col]
        if label_cols:
            labels = []
            for i, sample in enumerate(ds):
                row = {}
                for c in label_cols:
                    val = sample[c]
                    if isinstance(val, (str, int, float, bool)):
                        row[c] = val
                    else:
                        row[c] = str(val)
                row["_image_file"] = f"{i:06d}.jpg"
                labels.append(row)
            with open(target / "labels.json", "w") as f:
                json.dump(labels, f)
            print(f"    Saved labels for {len(labels)} samples")

        with open(target / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [OK] {name} downloaded")
        return "ok"
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
        import traceback
        traceback.print_exc()
        return "fail"


def download_github_dtgen():
    """Clone DTGen dirty tableware dataset."""
    name = "DTGen"
    target = BASE_DIR / "tier3" / name

    if target.exists() and any(target.iterdir()):
        print(f"  [SKIP] {name} — already exists")
        return "skip"

    print(f"  [DL] {name}: dirty tableware (clean/lightly/heavily dirty)")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/cyicz123/DTGen.git", str(target)],
            check=True, capture_output=True, text=True, timeout=300
        )
        print(f"  [OK] {name} cloned")
        return "ok"
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
        return "fail"


def main():
    print("=" * 60)
    print("TABLE CLEANLINESS DETECTION — DATASET DOWNLOADER")
    print("=" * 60)

    stats = {"ok": 0, "fail": 0, "skip": 0, "manual": 0}

    # --- ROBOFLOW DATASETS ---
    for tier, datasets in ROBOFLOW_DATASETS.items():
        tier_dir = BASE_DIR / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        tier_label = {"tier1": "TIER 1 — Real restaurant tables",
                      "tier2": "TIER 2 — Clean/dirty schema",
                      "tier3": "TIER 3 — Dirty signal",
                      "tier4": "TIER 4 — Academic"}[tier]
        print(f"\n▸ {tier_label} (Roboflow)")
        for ds in datasets:
            result = download_roboflow_via_cli(ds, tier_dir)
            stats[result] += 1

    # --- HUGGINGFACE DATASETS ---
    print(f"\n▸ TIER 3 — HuggingFace datasets")
    for ds in HF_DATASETS:
        result = download_huggingface(ds)
        stats[result] += 1

    # --- GITHUB ---
    print(f"\n▸ TIER 3 — GitHub DTGen")
    result = download_github_dtgen()
    stats[result] += 1

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Downloaded:  {stats['ok']}")
    print(f"  Skipped:     {stats['skip']}")
    print(f"  Manual:      {stats['manual']}")
    print(f"  Failed:      {stats['fail']}")

    print("\nFolder structure:")
    for tier in ["tier1", "tier2", "tier3", "tier4"]:
        tier_path = BASE_DIR / tier
        if tier_path.exists():
            subdirs = sorted([d.name for d in tier_path.iterdir() if d.is_dir()])
            print(f"  {tier}: {subdirs}")

    # Count total images
    total_images = 0
    for img_path in BASE_DIR.rglob("*.jpg"):
        total_images += 1
    for img_path in BASE_DIR.rglob("*.png"):
        total_images += 1
    print(f"\nTotal images found: {total_images}")

    if stats["manual"] > 0:
        print(f"\n⚠ {stats['manual']} dataset(s) need manual download.")
        print("  Check DOWNLOAD_MANUALLY.txt files in each folder.")
        print("  You need a free Roboflow account + API key for these.")


if __name__ == "__main__":
    main()
