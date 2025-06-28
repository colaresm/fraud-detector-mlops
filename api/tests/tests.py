import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from api.services import services,app

class TestSoma(unittest.TestCase):
    def test_soma_positiva(self):
        self.assertEqual(app.soma(2, 3), 5)

    def test_soma_negativa(self):
        self.assertEqual(app.soma(-1, -1), -2)
    
    def test_is_number(self):
        self.assertIsInstance(app.soma(-1, -1), (int, float))




if __name__ == "__main__":
    unittest.main()
