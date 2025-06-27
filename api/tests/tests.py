import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from api.utils import utils
from api.services import app ,services

class TestSoma(unittest.TestCase):
    def test_soma_positiva(self):
        self.assertEqual(app.soma(2, 3), 5)

    def test_soma_negativa(self):
        self.assertEqual(app.soma(-1, -1), -2)
    
    def test_is_number(self):
        self.assertIsInstance(app.soma(-1, -1), (int, float))



class TestGetRisk(unittest.TestCase):
    def setUp(self):
        self.valid_run_id = "12345"
        self.valid_data = {"some_key": "some_value"}
        self.valid_params = [1000.0, 700.0, 5000.0, 2.0]  # monthly_income, credit_score, current_debt, late_payments
        self.valid_X = np.array([self.valid_params])
        self.scaled_X = np.array([[0.0, 0.0, 0.0, 0.0]])  # Mock de saída normalizada
        self.prediction = np.array([1])  # Mock de previsão

    @patch("mlflow.sklearn.load_model")
    @patch("api.utils.utils.get_params_by_prediction")
    def test_valid_input(self, mock_get_params, mock_load_model):
        mock_get_params.return_value = self.valid_params
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = self.scaled_X
        mock_mlp = MagicMock()
        mock_mlp.predict.return_value = self.prediction
        mock_load_model.side_effect = [mock_scaler, mock_mlp]

        result = services.get_risk(self.valid_data, self.valid_run_id)

        self.assertEqual(result, self.prediction[0])
        mock_get_params.assert_called_once_with(self.valid_data)
        mock_load_model.assert_any_call(f"runs:/{self.valid_run_id}/scaler_model")
        mock_load_model.assert_any_call(f"runs:/{self.valid_run_id}/model")
        mock_scaler.transform.assert_called_once()
        mock_mlp.predict.assert_called_once_with(self.scaled_X)

    def test_missing_run_id(self):
        with self.assertRaises(ValueError) as cm:
            services.get_risk(self.valid_data, run_id=None)
        self.assertEqual(str(cm.exception), "The run_id must be provided.")

    @patch("mlflow.sklearn.load_model")
    def test_model_load_failure(self, mock_load_model):
        mock_load_model.side_effect = Exception("MLflow connection error")
        with self.assertRaises(RuntimeError) as cm:
            services.get_risk(self.valid_data, self.valid_run_id)
        self.assertTrue("Error loading models from MLflow" in str(cm.exception))

    @patch("api.utils.utils.get_params_by_prediction")
    def test_invalid_params(self, mock_get_params):
        mock_get_params.side_effect = Exception("Invalid data format")
        with self.assertRaises(ValueError) as cm:
           services. get_risk(self.valid_data, self.valid_run_id)
        self.assertTrue("Error extracting parameters" in str(cm.exception))

    @patch("mlflow.sklearn.load_model")
    @patch("api.utils.utils.get_params_by_prediction")
    def test_missing_values(self, mock_get_params, mock_load_model):
        mock_get_params.return_value = [1000.0, None, 5000.0, 2.0]
        mock_load_model.side_effect = [MagicMock(), MagicMock()]
        with self.assertRaises(ValueError) as cm:
            services.get_risk(self.valid_data, self.valid_run_id)
        self.assertEqual(str(cm.exception), "Input data contains invalid or missing values.")

   # @patch("mlflow.sklearn.load_model")
   # @patch("api.utils.utils.get_params_by_prediction")
   # def test_non_numeric_values(self, mock_get_params, mock_load_model):
    ##    mock_get_params.return_value = [1000.0, "invalid", 5000.0, 2.0]
      #  mock_load_model.side_effect = [MagicMock(), MagicMock()]
      #  with self.assertRaises(ValueError) as cm:
       #     services.get_risk(self.valid_data, self.valid_run_id)
       # self.assertEqual(str(cm.exception), "Input data contains invalid or missing values.")

if __name__ == "__main__":
    unittest.main()
