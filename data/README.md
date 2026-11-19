# Dataset Directory

Place your crowd-counting dataset here before running `train.py`.

CrowdSense supports two publicly available datasets out of the box.

---

## Option 1 — ShanghaiTech

**Paper:** Single-Image Crowd Counting via Multi-Column Convolutional Neural Network (CVPR 2016)  
**Download:** https://github.com/desenzhou/ShanghaiTechDataset

After downloading, extract so the layout looks like:

```
data/
└── ShanghaiTech/
    ├── part_A/
    │   ├── train_data/
    │   │   ├── images/          ← *.jpg crowd images
    │   │   └── ground-truth/    ← GT_*.mat annotation files
    │   └── test_data/
    │       ├── images/
    │       └── ground-truth/
    └── part_B/
        ├── train_data/
        │   ├── images/
        │   └── ground-truth/
        └── test_data/
            ├── images/
            └── ground-truth/
```

Each `.mat` file contains an `image_info` struct with a `.location` field
of shape `(N, 2)` holding the (row, col) coordinates of N annotated heads.

---

## Option 2 — UCF_CC_50

**Paper:** Multi-Source Multi-Scale Counting in Extremely Dense Crowd Images (CVPR 2013)  
**Download:** https://www.crcv.ucf.edu/data/crowd_counting.php

After downloading, extract so the layout looks like:

```
data/
└── UCF_CC_50/
    ├── *.jpg          ← 50 crowd images
    └── *_ann.mat      ← annotation files (field: annPoints, shape (N, 2))
```

---

## Running Training

```bash
# ShanghaiTech Part A
python train.py --dataset shanghaitech --part A --data-root data/ShanghaiTech

# ShanghaiTech Part B
python train.py --dataset shanghaitech --part B --data-root data/ShanghaiTech

# UCF_CC_50
python train.py --dataset ucf --data-root data/UCF_CC_50
```

The best model weights are saved to `model/weights.pth`.

---

> **DO NOT commit dataset files to the git repository.**
> Large binary files should be excluded via `.gitignore`.

