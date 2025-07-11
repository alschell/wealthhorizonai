import unittest
from core.coordinator import CoordinatorAgent
from core.state import SharedState

class TestSystem(unittest.TestCase):
    def setUp(self):
        self.state = SharedState()
        self.coord = CoordinatorAgent(self.state)

    def test_performance(self):
        perf = self.coord.process_query("performance")
        self.assertIsInstance(perf, dict)

    def test_graphic(self):
        graphic = self.coord.process_query("graphic compare portfolio p1 to sp500")
        self.assertIn("data:image/png;base64", graphic)

if __name__ == '__main__':
    unittest.main()
