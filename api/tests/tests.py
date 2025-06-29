import unittest
from unittest.mock import patch, MagicMock
from api.services import app
from api.services import services

class TestSoma(unittest.TestCase):
    def test_soma_positiva(self):
        self.assertEqual(app.soma(2, 3), 5)

    def test_soma_negativa(self):
        self.assertEqual(app.soma(-1, -1), -2)
    
    def test_is_number(self):
        self.assertIsInstance(app.soma(-1, -1), (int, float))


class TestGetRisk(unittest.TestCase):
    @patch('api.services.services.mlflow_server.load_model_and_scaler')
    @patch('api.services.services.mlflow_server.is_mlflow_online')
    @patch('api.services.services.utils.get_params_by_prediction')
    def test_get_risk_returns_expected_prediction(self, mock_get_params, mock_is_online, mock_load_model_and_scaler):
        # Configura mock: MLflow online
        mock_is_online.return_value = True

        # Simula saída do get_params
        mock_get_params.return_value = [1.0, 2.0, 3.0, 4.5]

        # Mock do scaler e modelo
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3, 0.4]]

        mock_model = MagicMock()
        mock_model.predict.return_value = [2]

        # Mock do load_model_and_scaler
        mock_load_model_and_scaler.return_value = (mock_model, mock_scaler)

        # Entrada fake
        input_data = {
            "monthly_income": 5000,
            "credit_score": 700,
            "current_debt": 1000,
            "late_payments": 2
        }

        # Chama a função real com mocks
        result = services.get_risk(input_data)

        # Verificações (todas devem passar)
        mock_is_online.assert_called_once()
        mock_get_params.assert_called_once_with(input_data)
        mock_scaler.transform.assert_called_once_with([mock_get_params.return_value])  # certo agora
        mock_model.predict.assert_called_once_with([[0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(result, [2])


if __name__ == "__main__":
    unittest.main()
