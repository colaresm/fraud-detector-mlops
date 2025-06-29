import unittest
from api.services import app
from api.services import services
import unittest
from unittest.mock import patch, MagicMock

class TestSoma(unittest.TestCase):
    def test_soma_positiva(self):
        self.assertEqual(app.soma(2, 3), 5)

    def test_soma_negativa(self):
        self.assertEqual(app.soma(-1, -1), -2)
    
    def test_is_number(self):
        self.assertIsInstance(app.soma(-1, -1), (int, float))


class TestGetRisk(unittest.TestCase):

    @patch('api.infra.mlflow_server.load_model_and_scaler')
    @patch('api.utils.utils.get_params_by_prediction')
    def test_get_risk_returns_expected_prediction(self, mock_get_params, mock_load_model_and_scaler):
        mock_model = MagicMock()
        mock_scaler = MagicMock()

        mock_get_params.return_value = [5.1, 3.5, 1.4, 0.2]
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_model.predict.return_value = ['setosas']

        mock_load_model_and_scaler.return_value = (mock_model, mock_scaler)

        input_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        result = services.get_risk(input_data)

        self.assertEqual(result, ['setosa'])
        mock_get_params.assert_called_once_with(input_data)
        mock_scaler.transform.assert_called_once()
        mock_model.predict.assert_called_once()

if __name__ == '__main__':
    unittest.main()
