import unittest

from predictor import evaluate_model, predict_day, train_model


class FitnessGoalPredictorTests(unittest.TestCase):
    def test_evaluation_is_reproducible_and_bounded(self):
        accuracy = evaluate_model()
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_prediction_has_expected_shape(self):
        result = predict_day(train_model(), 8.0, 8, 84.0)
        self.assertIsInstance(result["hit_goal"], bool)
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)

    def test_negative_input_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_day(train_model(), -1, 8, 84.0)


if __name__ == "__main__":
    unittest.main()
