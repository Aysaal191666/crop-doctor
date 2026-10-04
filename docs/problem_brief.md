# Crop Doctor: Problem Brief (v0.1)

## Problem
Small farmers in Andhra Pradesh often identify crop diseases late or incorrectly,
because expert advice is not always available. Wrong or late treatment reduces yield.

## Users
- Small and marginal farmers
- Agriculture extension officers
- Pesticide shop owners who advise farmers

## Solution (v1 scope)
A mobile-friendly app: the user photographs a rice leaf and gets the likely
disease, a confidence level, and simple advice. If the model is unsure,
the app asks for a retake instead of guessing.

## Scope v1
- Crop: rice only
- Classes: Healthy, Brown Spot, Hispa, Leaf Blast
- Languages: English first, Telugu later

## Out of scope (for now)
- Other crops, pest detection, soil analysis, payments

## Success metrics (targets, to be revised after real measurements)
- Macro F1 >= 0.80 on a held-out test set
- Diseased leaf called "Healthy" < 5% of diseased leaves
- Prediction latency < 2 seconds per image
- Cross-source (field photo) performance measured and reported honestly

## Risks
- Dataset has plain-background photos; field performance may be lower
- About 21% near-duplicate images found; splits must be group-aware
- Possible label noise in subtle classes (Hispa vs Healthy)
- The app is a decision aid, not a final diagnosis