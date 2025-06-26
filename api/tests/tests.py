import unittest
from api.services import app

class TestSoma(unittest.TestCase):
    def test_soma_positiva(self):
        self.assertEqual(app.soma(2, 3), 5)

    def test_soma_negativa(self):
        self.assertEqual(app.soma(-1, -1), -2)

if __name__ == '__main__':
    unittest.main()
