import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from api.services import app
from api.services import get_risk

class TestSoma(unittest.TestCase):
    def test_soma_positiva(self):
        self.assertEqual(app.soma(2, 3), 5)

    def test_soma_negativa(self):
        self.assertEqual(app.soma(-1, -1), -2)
    
    def test_is_number(self):
        self.assertIsInstance(app.soma(-1, -1), (int, float))


class TestGetRisk(unittest.TestCase):
    @patch('api.infra.mlflow_server')
    @patch('api.utils.utils')
    def test_get_risk_returns_expected_prediction(self, mock_get_params, mock_load_model_and_scaler):
        mock_get_params.return_value = [5000, 700, 1000, 2]

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = [[0.5, 0.7, 0.4, 0.1]]

        mock_model = MagicMock()
        mock_model.predict.return_value = [2]

        mock_load_model_and_scaler.return_value = (mock_model, mock_scaler)

        input_data = {
            "monthly_income": 5000,
            "credit_score": 700,
            "current_debt": 1000,
            "late_payments": 2
        }

        result = get_risk(input_data)

        mock_get_params.assert_called_once_with(input_data)
        mock_scaler.transform.assert_called_once_with([[mock_get_params.return_value]])
        mock_model.predict.assert_called_once_with([[0.5, 0.7, 0.4, 0.1]])
        self.assertEqual(result, [2])


if __name__ == "__main__":
    unittest.main()
