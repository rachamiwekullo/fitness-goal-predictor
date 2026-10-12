"""Train and evaluate a small classifier on synthetic fitness examples."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURE_NAMES = ["sleep_hours", "water_glasses", "bench_press_kg"]
FEATURES = np.array([
    [6.5, 6, 80], [7.2, 8, 85], [5.8, 5, 75], [8.0, 10, 90],
    [7.5, 9, 88], [6.0, 6, 78], [7.8, 8, 86], [8.2, 10, 92],
    [5.5, 4, 70], [7.0, 7, 82], [6.8, 8, 84], [8.5, 11, 95],
    [7.1, 7, 83], [6.2, 5, 77], [8.1, 9, 89], [7.6, 8, 87],
    [5.9, 6, 76], [8.3, 10, 91], [6.7, 7, 81], [7.9, 9, 88],
    [6.4, 6, 79], [8.4, 11, 93], [7.3, 8, 85], [5.7, 5, 74],
    [7.7, 9, 86], [6.1, 6, 78], [8.6, 10, 94], [7.4, 8, 84],
], dtype=float)
STEPS = np.array([
    8_400, 10_200, 7_200, 11_300, 10_600, 7_900, 10_400, 11_500,
    6_500, 9_600, 9_800, 12_000, 9_700, 7_000, 10_900, 10_500,
    7_100, 11_600, 8_900, 10_700, 7_800, 11_800, 10_100, 6_900,
    10_300, 7_600, 11_700, 9_900,
], dtype=int)
GOAL_LABELS = (STEPS >= 10_000).astype(int)


def train_model() -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    return model.fit(FEATURES, GOAL_LABELS)


def evaluate_model() -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        FEATURES,
        GOAL_LABELS,
        test_size=0.25,
        random_state=42,
        stratify=GOAL_LABELS,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    return float(model.score(x_test, y_test))


def predict_day(model: RandomForestClassifier, sleep_hours: float, water_glasses: int,
                bench_press_kg: float) -> dict[str, float | bool]:
    if sleep_hours < 0 or water_glasses < 0 or bench_press_kg < 0:
        raise ValueError("Prediction inputs must be non-negative.")
    features = np.array([[sleep_hours, water_glasses, bench_press_kg]], dtype=float)
    probabilities = model.predict_proba(features)[0]
    hit_index = list(model.classes_).index(1)
    hit_probability = float(probabilities[hit_index])
    return {
        "hit_goal": bool(model.predict(features)[0]),
        "probability": round(hit_probability, 3),
    }


def main() -> None:
    print(f"Hold-out accuracy: {evaluate_model():.0%}")
    model = train_model()
    for inputs in [(8.0, 8, 84.0), (6.0, 5, 78.0), (7.5, 9, 88.0)]:
        prediction = predict_day(model, *inputs)
        label = "10k+ steps" if prediction["hit_goal"] else "below 10k steps"
        print(f"sleep={inputs[0]}h water={inputs[1]} glasses bench={inputs[2]}kg"
              f" -> {label} (model probability: {prediction['probability']:.0%})")


if __name__ == "__main__":
    main()
