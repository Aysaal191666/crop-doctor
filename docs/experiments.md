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



## Run 2: EfficientNet-B0 (model swap only)
- Same settings as Run 1 except the model: EfficientNet-B0, 224 px, flips only, 10 epochs
- Best validation accuracy: 0.777

| Metric (test) | Value |
|---|---|
| Accuracy | 0.758 |
| Macro F1 | 0.731 |
| Diseased called Healthy | 44 / 267 (16.5%) |

Confusion matrix (rows = true, columns = predicted: Brown Spot, Healthy, Hispa, Leaf Blast):

    [[ 60   8   3   4]
     [ 14 176  17   6]
     [  0  28  44   8]
     [ 12   8   8  84]]

Observations:
- Swapping the model gave only a small gain (macro F1 +0.013), within the noise of a 480-image test set
- Leaf Blast improved (F1 0.74 to 0.79); Hispa did not (F1 0.58)

## Run 3: EfficientNet-B0 + rotation and colour augmentation
- Same as Run 2 plus RandomRotation(20) and ColorJitter(0.2), 10 epochs
- Best validation accuracy: 0.771

| Metric (test) | Value |
|---|---|
| Accuracy | 0.773 |
| Macro F1 | 0.752 |
| Diseased called Healthy | 31 / 267 (11.6%) |

Confusion matrix (rows = true, columns = predicted: Brown Spot, Healthy, Hispa, Leaf Blast):

    [[ 64   7   1   3]
     [ 12 171  19  11]
     [  1  18  49  12]
     [ 12   6   7  87]]

Observations:
- Macro F1 +0.021 and diseased-called-Healthy 44 to 31 versus Run 2, but validation accuracy did not improve (0.777 vs 0.771), so the gain is not confirmed
- Hispa called Healthy dropped from 28 to 18; Healthy called Leaf Blast rose from 6 to 11 (trade-off)
- Train/validation gap is smaller (0.77 vs 0.73), so more epochs may help

## Summary (test set, 480 images)

| Run | Model | Augmentation | Accuracy | Macro F1 | Diseased called Healthy |
|---|---|---|---|---|---|
| 1 | MobileNetV3 | flips | 0.746 | 0.718 | 45/267 |
| 2 | EfficientNet-B0 | flips | 0.758 | 0.731 | 44/267 |
| 3 | EfficientNet-B0 | flips + rotation + colour | 0.773 | 0.752 | 31/267 |

Note: runs were compared on the test set here. From now on, compare on validation and use the test set only for the final result.



## Error analysis and confidence thresholds (Run 3 model)
- Model: Run 3 (EfficientNet-B0 + rotation and colour augmentation)
- Rule: the app answers only if confidence passes a threshold; a higher bar for "Healthy" than for a disease name. Otherwise it asks for a retake.
- Thresholds were chosen on the validation set (grid search, minimum 70% of photos answered, then fewest diseased-called-Healthy). The test set was checked once.
- Chosen: disease >= 0.4, healthy >= 0.6

| | Validation | Test |
|---|---|---|
| Photos answered | 83% | 82% |
| Accuracy on answered photos | 0.781 | 0.794 |
| Diseased called Healthy | 20/267 (7.5%) | 11/267 (4.1%) |

Test without thresholds: 100% answered, accuracy 0.773, diseased called Healthy 31/267 (11.6%).

Findings:
- The diseased-called-Healthy count depends only on the Healthy threshold; the disease threshold only trades coverage against accuracy
- The test result (4.1%) is better than validation (7.5%) for the same rule; with 11 cases the 95% range is roughly 2% to 7%, so the 5% target is not confirmed
- The 70% coverage minimum was a product choice: (disease 0.4, healthy 0.7) answered 69% on validation with 8/267 missed (3.0%), and was excluded by 1 point
- In 12 sampled mistakes, 6 were diseased called Healthy; 4 had confidence under 60% (would be sent to retake), 2 (64% and 73%) would pass the threshold
- All thresholds were tuned on lab-style photos; field-photo behaviour is not yet measured (cross-source evaluation is next)
