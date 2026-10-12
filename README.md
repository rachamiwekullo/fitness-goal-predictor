# Fitness Goal Predictor

A scikit-learn Random Forest exercise that classifies whether a synthetic
fitness-log day reached 10,000 steps, using sleep, water, and bench-press
measurements as features. It reports a reproducible hold-out accuracy and
example predictions.

This is a learning demo, not a validated fitness or medical model. Its dataset
is small, invented, and unsuitable for real-world decisions; the displayed
probability is not a calibrated confidence score.

## Run

Requires Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
python predictor.py
python -m unittest -v
```

## Project source

Based on the classification, prediction, and model-evaluation exercises in
Week 7, including the mini-project in Day 5.
