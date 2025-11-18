import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import unittest
import json
from src.inference.app import app

class APITestCase(unittest.TestCase):
    """
    Unit test for the phishing URL prediction API.
    """

    def setUp(self):
        """Creating a Flask test client before each test."""
        self.client = app.test_client()

    def test_predict(self):
        """Ensuring prediction endpoint returns a phishing score""" 
        response = self.client.post(
            "/predict",
            json = {"url" : "www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php"}
        )       

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("phishing", data)
        self.assertIsInstance(data["phishing"], float)

    def test_missing_url(self):
        """Ensuring API returns an error when 'url' is missing"""
        response = self.client.post("/predict", json = {})
        self.assertNotEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()