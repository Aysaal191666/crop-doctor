# Experiments log

## Setup (all runs)
- Dataset: nizorogbezuode/rice-leaf-images (3355 images, 4 classes, 2637 near-duplicate groups)
- Split: group-aware, seed 42. Train 2395 / val 480 / test 480
- Test set is used for final reporting only

## Run 1: Baseline
- Model: MobileNetV3-Large (ImageNet pretrained), 224 px
- Training: flips only (no random crop), 10 epochs, AdamW lr 1e-4, batch 32, class-weighted loss
- Best validation accuracy: 0.752

| Metric (test) | Value |
|---|---|
| Accuracy | 0.746 |
| Macro F1 | 0.718 |
| Diseased called Healthy | 45 / 267 (16.9%) |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Brown Spot | 0.663 | 0.813 | 0.731 | 75 |
| Healthy | 0.796 | 0.826 | 0.811 | 213 |
| Hispa | 0.595 | 0.588 | 0.591 | 80 |
| Leaf Blast | 0.841 | 0.661 | 0.740 | 112 |

Confusion matrix (rows = true, columns = predicted: Brown Spot, Healthy, Hispa, Leaf Blast):

    [[ 61  10   1   3]
     [ 12 176  20   5]
     [  0  27  47   6]
     [ 19   8  11  74]]

Observations:
- Hispa is the weakest class; it is mostly confused with Healthy (27 Hispa called Healthy, 20 Healthy called Hispa)
- Leaf Blast is often called Brown Spot (19 of 112)
- Train accuracy 0.90 vs validation 0.75: overfitting after about epoch 5
- Diseased-called-Healthy rate (16.9%) is far above the 5% target
- Test set has 480 images, so differences of a few points are within noise