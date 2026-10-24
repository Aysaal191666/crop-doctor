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


## Cross-source evaluation (Run 3 model)
- External data: RiceyLeafDisease (Mendeley Data, Bangladesh, 2023), original images only (1,701; the 5,188 augmented copies were excluded)
- 944 photos in the 4 classes the model knows; 757 photos of 4 diseases it never saw
- The photos are field photos (grass, mud, water, blur, several leaves); training photos were single leaves on plain paper

| | Internal test | External (field photos) |
|---|---|---|
| Accuracy | 0.773 | 0.375 |
| Macro F1 | 0.752 | 0.211 |

Confusion matrix on external photos (rows = true, columns = predicted: Brown Spot, Healthy, Hispa, Leaf Blast):

    [[ 52   0   0 215]
     [ 49   3   7  98]
     [ 77   1   1 136]
     [  7   0   0 298]]

Findings:
- The model collapsed to "Leaf Blast": 747 of 944 photos (79%) were called Leaf Blast
- 154 of 157 healthy leaves were called diseased. "Diseased called Healthy" is only 1/787 because the model almost never says Healthy (4 times in 944), so that safety metric is misleading here
- The confidence thresholds did not help: 98% of in-scope photos and 100% of photos of unseen diseases were still answered
- Likely cause (not proven): the model learned cues specific to the lab-style photos rather than the disease itself
- Possible confounds not yet separated: photos were resized to a square (distorting non-square photos); external labels were not verified; some external photos are near-duplicates
- Conclusion: the current model must not be presented as field-ready



## Run A: lab-style + field photos (4 classes)
- Same settings as Run 3 (EfficientNet-B0, 224 px, flips + rotation + colour jitter, 10 epochs, lr 1e-4, class-weighted loss), plus field photos in training
- Train 3,069 (2,395 lab-style + 674 field), validation 614 (480 + 134); best validation accuracy 0.831
- Field photos: RiceyLeafDisease originals, group-aware split (1,458 groups from 1,701 photos); only the four known classes used here

| Metric | Run 3 | Run A |
|---|---|---|
| Lab-style test: accuracy | 0.773 | 0.806 |
| Lab-style test: macro F1 | 0.752 | 0.773 |
| Lab-style test: diseased called Healthy | 31/267 (11.6%) | 47/267 (17.6%) |
| Field test (136 photos): accuracy | 0.375 (model never saw field photos) | 0.882 |
| Field test: macro F1 | 0.211 | 0.884 |
| Field test: diseased called Healthy | 1/787 | 1/112 |

Confusion matrix, lab-style test (rows = true, columns = predicted: Brown Spot, Healthy, Hispa, Leaf Blast):

    [[ 61   6   0   8]
     [  2 197  10   4]
     [  0  33  41   6]
     [ 10   8   6  88]]

Confusion matrix, field test:

    [[33  0  2  3]
     [ 2 20  2  0]
     [ 0  1 29  1]
     [ 5  0  0 38]]

Findings:
- Training with field photos removed the collapse seen in the cross-source test
- Lab-style accuracy and macro F1 did not drop (the small rise is within noise); diseased-called-Healthy got worse (31 to 47), mostly Hispa called Healthy (33 of 80); cause not investigated
- Hispa recall: 0.51 on lab-style photos vs 0.94 on field photos
- The field test has only 136 photos (Healthy 24), so numbers are rough (about +/-5.5 points)
- Likely optimistic: the field photos come from a few locations and photos of the same plant may sit on both sides of the split; not comparable with the 0.375 result because that model never saw field photos
- Run A has no way to say "other disease"; 103 field test photos of unseen diseases were not evaluated
- Confidence thresholds from the earlier ticket do not apply to this model and must be redone



## Run B: lab-style + field photos + "Other" class
- Same as Run A plus a fifth class, Other, made of the four field diseases the model had not been trained on (Bacterial Leaf Blight, Leaf scald, Narrow Brown Leaf Spot, Sheath Blight)
- Train 3,615 (2,395 lab-style + 674 field known + 546 field Other), validation 722 (480 + 134 + 108); best validation accuracy 0.831
- Test: lab-style 480 photos; field 239 photos (136 known classes + 103 Other)

| Metric | Run A | Run B |
|---|---|---|
| Lab-style test: accuracy | 0.806 | 0.775 |
| Lab-style test: macro F1 | 0.773 | 0.763 |
| Lab-style test: diseased called Healthy | 47/267 (17.6%) | 25/267 (9.4%) |
| Field test: accuracy | 0.882 (136 photos) | 0.870 (239 photos) |
| Field test: diseased called Healthy | 1/112 | 0/215 |
| Same 136 known-class field photos: correct | 120 | 115 |
| Other: recall / precision | n/a | 0.903 / 0.930 |

Confusion matrix, lab-style test (rows = true, columns = predicted: Brown Spot, Healthy, Hispa, Leaf Blast, Other):

    [[ 64   4   1   6   0]
     [  5 164  35   9   0]
     [  1  17  57   5   0]
     [ 13   4   8  87   0]
     [  0   0   0   0   0]]

Confusion matrix, field test:

    [[34  0  0  1  3]
     [ 2 22  0  0  0]
     [ 1  0 28  0  2]
     [10  0  0 31  2]
     [ 5  0  1  4 93]]

Findings:
- 93 of 103 photos of the four unseen diseases were labelled Other; no diseased field photo was called Healthy (0/215; 95% upper bound about 1.4%)
- On the same 136 known-class field photos, Run B got 5 fewer right than Run A (115 vs 120), within noise; Leaf Blast recall fell from 0.88 to 0.72 (10 called Brown Spot, only 43 photos)
- On lab-style photos, Hispa called Healthy dropped from 33 to 17 while Healthy called Hispa rose from 10 to 35, so the Hispa/Healthy boundary moves between runs; these are single runs, so part of the change may be chance
- Other was trained on these same four diseases, so it does not show that a new disease would be caught; all Other photos are field-style, so lab-style photos of unseen diseases are untested
- Field result likely optimistic (small test set, one source, same-plant photos may span the split)
- Run B is carried forward because it can say "other disease", a product decision, not because it scored higher
- The confidence thresholds from the earlier ticket do not apply to this model and must be redone on validation data



## Confidence thresholds for the mixed-source model (Run B)
- Rule: the app answers only if confidence passes a threshold, with a higher bar for "Healthy"; otherwise it asks for a retake. "Other" counts as a disease.
- Chosen on the combined validation set (480 lab-style + 242 field photos) with the same rule as before: at least 70% of photos answered, then fewest diseased-called-Healthy, then highest coverage. Each test set was checked once.
- Chosen: disease >= 0.4, healthy >= 0.9. On validation: 75% answered (lab 63%, field 98%), diseased called Healthy 1/482.

| | Without thresholds | With thresholds |
|---|---|---|
| Lab-style test: photos answered | 100% | 69% |
| Lab-style test: accuracy on answered | 0.775 | 0.751 |
| Lab-style test: diseased called Healthy | 25/267 (9.4%) | 3/267 (1.1%) |
| Field test: photos answered | 100% | 98% |
| Field test: accuracy on answered | 0.870 | 0.868 |
| Field test: diseased called Healthy | 0/215 | 0/215 |

Findings:
- The diseased-called-Healthy count depends only on the Healthy threshold; the disease threshold trades coverage against accuracy (validation accuracy on answered rose from 0.84 to 0.92 while coverage fell)
- The model is far more confident on field photos (97 to 99% answered at most settings) than on lab-style photos (63% answered at healthy >= 0.9), so the thresholds change little for field photos
- On lab-style photos accuracy on answered photos fell (0.775 to 0.751): confidence did not separate right from wrong answers; the drop in diseased-called-Healthy comes from suppressing low-confidence "Healthy" answers
- Not measured: how many truly healthy leaves are sent to "retake photo" because of the 0.9 bar
- Small counts: 3/267 has a 95% range of about 0.2% to 3.3%; 0/215 means the true rate is probably below about 1.4%; the 5% target looks met but is not proven
- The lab-style test answered 69%, just under the 70% minimum that held on validation (75%)
- Alternative on validation: (disease 0.4, healthy 0.8) answers 83% and misses 3/482; not adopted because choosing after seeing test results would be peeking
- Validation also selected the training epoch, so validation numbers are slightly optimistic
- All results are on lab-style photos and one Bangladeshi field dataset; no photos from Andhra Pradesh yet.



## ONNX export (Run B model)
- Exported with the legacy TorchScript exporter (opset 17, dynamic batch size); files: model.onnx, model_meta.json (classes, input size, normalisation, thresholds), export_report.json
- Equivalence check against PyTorch on both test sets (719 photos): identical predictions on all of them; largest probability difference 0.000025 (lab-style) and 0.000007 (field)

| | PyTorch | ONNX Runtime |
|---|---|---|
| File size | 16.3 MB | 16.0 MB |
| Speed, 1 photo, median / 95th percentile (Colab server CPU) | 39.5 / 54.4 ms | 17.2 / 19.5 ms |

Findings:
- ONNX gives the same answers as PyTorch and is about 2.3x faster at the median; no quantisation was applied, so the file is not smaller
- Speed was measured on a Colab server CPU, for the model step only (no image decoding, resizing or network); it is not a phone measurement
- The confidence thresholds were not re-run on the ONNX model; probabilities differ by at most 0.000025
- The API must repeat the training preprocessing exactly: RGB, resize straight to 224 x 224 (no crop), scale to 0-1, normalise with the ImageNet mean/std
- TFLite export is not done (only needed for a native Android app)




## API serving check (Run B model, FastAPI + ONNX Runtime)
- The API (`api/main.py`) preprocesses the photo, runs the ONNX model, and applies the thresholds from `models/model_meta.json` (disease 0.4, healthy 0.9). Answers are "result", "other" or "retake".
- `api/eval_serving.py` runs the real API code over the 480 lab-style test photos and compares with the Colab run of the same model

| Lab-style test (480 photos) | Colab (PyTorch) | API pipeline |
|---|---|---|
| Accuracy, no thresholds | 0.775 | 0.775 |
| Diseased called Healthy, no thresholds | 25/267 | 25/267 |
| With thresholds: photos answered | 69% | 69% |
| With thresholds: accuracy on answered | 0.751 | 0.751 |
| With thresholds: diseased called Healthy | 3/267 | 3/267 |

- The confusion matrix is identical to the Colab run, so the serving pipeline reproduces the trained model on square lab-style photos
- Not yet verified: non-square photos (field photos, phone photos). The test `test_preprocess_matches_training` covers this but needs PyTorch and was skipped in Codespaces
- The API rotates sideways phone photos upright using the EXIF tag; training did not do this
- Two manual checks through the `/docs` page: a healthy lab-style leaf was answered "Leaf Blast" (confidence 0.54; Healthy 0.23), an ordinary error of the kind seen in the test set (about 4% of healthy test photos are called Leaf Blast); a photo of a wooden table was answered "Other disease" (confidence 0.53; Brown Spot 0.30). The model never saw non-leaf photos, so this answer is luck, not a capability. One photo each proves nothing about rates
- Open limits: no "not a leaf" check; no login or rate limit; library versions not yet pinned; the database file may not survive a restart on a hosting service