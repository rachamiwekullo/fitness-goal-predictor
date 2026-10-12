# Fitness Goal Predictor

A Random Forest classifier that predicts whether a sample activity-log day
reaches 10,000 steps from sleep, water, and bench-press measurements. The
program reports a hold-out score and predictions for example days.

## What It Does

- Trains a scikit-learn Random Forest on a small synthetic dataset
- Evaluates the model on a reproducible hold-out split
- Predicts whether example days meet the step target
- Reports the model's predicted probability for the goal-hit class
- Rejects negative prediction inputs

## Setup

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the example predictions:

```bash
python predictor.py
```

Use the prediction helper from Python:

```python
from predictor import train_model, predict_day

model = train_model()
result = predict_day(model, sleep_hours=8, water_glasses=8, bench_press_kg=84)
print(result)
```

Run tests with `python -m unittest -v`.

## Sample Output

```text
Hold-out accuracy: 100%
sleep=8.0h water=8 glasses bench=84.0kg -> 10k+ steps (model probability: 52%)
sleep=6.0h water=5 glasses bench=78.0kg -> below 10k steps (model probability: 0%)
```

## Stack

Python, NumPy, scikit-learn

This is a learning demo, not a validated fitness or medical model. The
training data is small and synthetic, and the predicted probability is not a
calibrated confidence score. Hold-out accuracy on this sample does not imply
real-world performance.
