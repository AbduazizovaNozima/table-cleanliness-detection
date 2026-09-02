# Table Cleanliness Detection

A computer vision system for automatically detecting restaurant table cleanliness status — **clean**, **dirty**, or **occupied** — using YOLO and Vision-Language Models.

## Overview

This project collects and processes multi-source datasets from Roboflow, HuggingFace, and GitHub, then trains a detection model to classify restaurant tables in real time. The approach combines object detection (YOLOv8) with vision-language model evaluation (Qwen-VL) for high-accuracy scene understanding.

**Key stats across collected datasets:**
- 11 Roboflow datasets (7,000+ labeled images)
- 2 HuggingFace datasets
- 4 tiers of dataset difficulty/context

## Project Structure

```
table-cleanliness-detection/
├── scripts/
│   ├── download_roboflow.py   # Download via Roboflow SDK (requires API key)
│   └── download_all.py        # Download via public URLs + HuggingFace
├── datasets/                  # Downloaded data (gitignored — too large)
│   ├── tier1/                 # Real restaurant table photos
│   ├── tier2/                 # Clean/dirty schema datasets
│   ├── tier3/                 # Supporting dirty-signal data
│   └── tier4/                 # Academic food waste datasets
├── labeled/                   # Custom labeled data (gitignored)
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/table-cleanliness-detection
cd table-cleanliness-detection

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download Datasets

**Option 1 — Roboflow SDK** (requires API key):
```bash
cp .env.example .env
# Edit .env and add your ROBOFLOW_API_KEY
source .env
python scripts/download_roboflow.py
```

**Option 2 — Public URLs** (no API key needed):
```bash
python scripts/download_all.py
```

Get a free Roboflow API key at [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api).

## Dataset Tiers

| Tier | Focus | Images |
|------|-------|--------|
| Tier 1 | Real restaurant tables (dirty/clean/occupied) | ~6,400 |
| Tier 2 | Clean vs. dirty schema | ~2,600 |
| Tier 3 | Supporting signals (food trays, dirty kitchen) | ~1,600 |
| Tier 4 | Academic canteen food waste | ~175 |

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **VLM Evaluation:** Qwen-VL (via HuggingFace Transformers)
- **Dataset Management:** Roboflow SDK, HuggingFace Datasets
- **Training:** PyTorch, Accelerate

## License

MIT
