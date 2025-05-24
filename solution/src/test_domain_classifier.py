import unittest

from src.domain_classifier import classify_domain
from src.summarizer import IndicSummarizer

class TestDomainClassifier(unittest.TestCase):
    def test_legal(self):
        text = "This agreement is made between the parties."
        self.assertEqual(classify_domain(text), "legal")

    def test_medical(self):
        text = "The patient exhibited symptoms of hypertension and required treatment."
        self.assertEqual(classify_domain(text), "medical")

    def test_technical(self):
        text = "The software engineer optimized the code and fixed a bug."
        self.assertEqual(classify_domain(text), "technical")

    def test_unknown(self):
        # Empty or whitespace text should return unknown
        self.assertEqual(classify_domain("   "), "unknown")

class TestSummarizerIntegration(unittest.TestCase):
    def test_summarizer_succeeds_on_legal(self):
        text = "This agreement is made between the parties."
        summarizer = IndicSummarizer(method="neural", lang_code="te")
        summary = summarizer.summarize(text)
        # Summary should be a non-empty string
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

if __name__ == "__main__":
    unittest.main()
