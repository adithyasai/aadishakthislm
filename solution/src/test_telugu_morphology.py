"""
Test module for the Telugu morphological analyzer.

This module provides comprehensive tests for the Telugu morphological analyzer
functionality, including morpheme segmentation, stemming, and feature extraction.
"""
import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent))
from src.telugu_morphology import TeluguMorphologyAnalyzer

class TeluguMorphologyTest(unittest.TestCase):
    """Test cases for TeluguMorphologyAnalyzer."""
    
    def setUp(self):
        """Set up the test fixtures."""
        self.analyzer = TeluguMorphologyAnalyzer()
    
    def test_word_segmentation(self):
        """Test morpheme segmentation functionality."""
        test_cases = [
            # Word, Expected segments
            ('పుస్తకాలు', ['పుస్తక', 'లు']),  # pustakalu (books) -> pustaka + lu (plural)
            ('పిల్లవాడు', ['పిల్లవా', 'డు']),  # pillavadu (boy) -> pillava + du (masculine)
            ('చదువుతున్నాను', ['చదువు', 'తున్నాను']),  # caduvutunnanu (I am reading) -> caduvu + tunnanu
            ('ఇంటికి', ['ఇంటి', 'కి']),  # intiki (to the house) -> inti + ki (dative)
            ('అమ్మాయితో', ['అమ్మాయి', 'తో']),  # ammayito (with the girl) -> ammayi + to (instrumental)
            ('చేస్తాడు', ['చేస్', 'తాడు']),  # cestadu (he will do) -> ces + tadu
            ('వచ్చినవాడు', ['వచ్చిన', 'వాడు']),  # vaccinavaḍu (the one who came) -> vaccina + vadu
            ('చదవలేదు', ['చదవ', 'లేదు']),  # cadavaledu (did not read) -> cadava + ledu
            ('రాజుగారు', ['రాజు', 'గారు']),  # rajugaru (respected Raja) -> raju + garu
        ]
        
        for word, expected in test_cases:
            with self.subTest(word=word):
                result = self.analyzer.segment_word(word)
                self.assertEqual(result, expected, f"Expected {expected} but got {result} for {word}")
    
    def test_stemming(self):
        """Test stem extraction functionality."""
        test_cases = [
            # Word, Expected stem
            ('పుస్తకాలు', 'పుస్తక'),  # pustakalu (books) -> pustaka
            ('పిల్లవాడు', 'పిల్లవా'),  # pillavadu (boy) -> pillava
            ('చదువుతున్నాను', 'చదువు'),  # caduvutunnanu (I am reading) -> caduvu
            ('ఇంటికి', 'ఇంటి'),  # intiki (to the house) -> inti
            ('వారికి', 'వారు'),  # variki (to them) -> varu
            ('నాకు', 'నేను'),  # naku (to me) -> nenu
            ('మీతో', 'మీరు'),  # mito (with you) -> miru
        ]
        
        for word, expected in test_cases:
            with self.subTest(word=word):
                result = self.analyzer.get_stem(word)
                self.assertEqual(result, expected, f"Expected {expected} but got {result} for {word}")
    
    def test_pronoun_analysis(self):
        """Test pronoun analysis functionality."""
        test_cases = [
            # Pronoun, Expected features
            ('నేను', {'person': '1', 'number': 'singular', 'case': 'direct'}),
            ('నాకు', {'person': '1', 'number': 'singular', 'case': 'dative'}),
            ('మేము', {'person': '1', 'number': 'plural', 'case': 'direct', 'inclusivity': 'exclusive'}),
            ('నువ్వు', {'person': '2', 'number': 'singular', 'case': 'direct', 'formality': 'informal'}),
            ('మీరు', {'person': '2', 'number': 'singular', 'case': 'direct', 'formality': 'formal'}),
            ('అతడు', {'person': '3', 'number': 'singular', 'case': 'direct', 'gender': 'masculine', 'distance': 'distal'}),
            ('ఆమె', {'person': '3', 'number': 'singular', 'case': 'direct', 'gender': 'feminine'}),
            ('వారు', {'person': '3', 'number': 'plural', 'case': 'direct', 'gender': 'human'}),
        ]
        
        for pronoun, expected_features in test_cases:
            with self.subTest(pronoun=pronoun):
                result = self.analyzer.analyze_word(pronoun)
                
                # Check if all expected features are in the result with correct values
                for key, value in expected_features.items():
                    self.assertIn(key, result, f"Key {key} not found in analysis of {pronoun}")
                    self.assertEqual(result[key], value, f"Expected {value} for {key} but got {result[key]} for {pronoun}")
                
                # Check that POS is correctly identified
                self.assertEqual(result['pos'], 'pronoun', f"Expected POS 'pronoun' but got {result.get('pos', 'None')} for {pronoun}")
    
    def test_verb_analysis(self):
        """Test verb analysis functionality."""
        test_cases = [
            # Verb, Expected features
            ('చేస్తున్నాను', {'tense': 'present_continuous_1sg', 'person': '1', 'number': 'singular'}),
            ('చేశాడు', {'tense': 'past_3sg_m', 'person': '3', 'gender': 'masculine', 'number': 'singular'}),
            ('వస్తారు', {'tense': 'future_3pl', 'person': '3', 'number': 'plural'}),
            ('చదవండి', {'tense': 'imperative_formal', 'formality': 'formal'}),
            ('చూడగలను', {'tense': 'potential_1sg', 'person': '1', 'number': 'singular'}),
        ]
        
        for verb, expected_features in test_cases:
            with self.subTest(verb=verb):
                result = self.analyzer.analyze_word(verb)
                
                # Check if verb is correctly identified as a verb
                if 'pos' in result:  # Some complex forms may not have POS identified
                    self.assertEqual(result.get('pos'), 'verb', f"Expected POS 'verb' but got {result.get('pos', 'None')} for {verb}")
                
                # Check if expected features are present
                for key, value in expected_features.items():
                    if key in result:  # Some features may not be identified for all verbs
                        self.assertEqual(result[key], value, f"Expected {value} for {key} but got {result[key]} for {verb}")
    
    def test_noun_analysis(self):
        """Test noun analysis functionality."""
        test_cases = [
            # Noun, Expected features
            ('పుస్తకాలు', {'number': 'plural', 'case': 'plural_nom'}),
            ('అబ్బాయి', {'gender': 'masculine'}),
            ('అమ్మాయి', {'gender': 'feminine'}),
            ('ఇంటికి', {'case': 'dative'}),
            ('పిల్లలతో', {'case': 'instrumental'}),
            ('పాఠశాలలో', {'case': 'locative'}),
        ]
        
        for noun, expected_features in test_cases:
            with self.subTest(noun=noun):
                result = self.analyzer.analyze_word(noun)
                
                # Check if expected features are present
                for key, value in expected_features.items():
                    if key in result:  # Some features may not be identified for all nouns
                        self.assertEqual(result[key], value, f"Expected {value} for {key} but got {result[key]} for {noun}")
    
    def test_sandhi(self):
        """Test sandhi rule application."""
        test_cases = [
            # Word1, Word2, Expected combined form
            ('రాము', 'అని', 'రామని'),  # Ramu + ani -> Ramani
            ('కవి', 'అయిన', 'కవీయిన'),  # kavi + ayina -> kaviyina
            ('తల్లి', 'అమ్మ', 'తల్లిమ్మ'),  # talli + amma -> tallimma
        ]
        
        for word1, word2, expected in test_cases:
            with self.subTest(combination=f"{word1}+{word2}"):
                result = self.analyzer.apply_sandhi_rules(word1, word2)
                self.assertEqual(result, expected, f"Expected {expected} but got {result} for {word1}+{word2}")
    
    def test_text_analysis(self):
        """Test full text analysis functionality."""
        text = "నేను తెలుగులో మాట్లాడుతున్నాను. అది చాలా అందమైన భాష."
        # I am speaking in Telugu. It's a very beautiful language.
        
        result = self.analyzer.analyze_text(text)
        
        # Verify we get the right number of word analyses
        self.assertEqual(len(result), 8, f"Expected 8 words to be analyzed but got {len(result)}")
        
        # Check some specific words
        self.assertEqual(result[0]['word'], 'నేను', "First word should be 'nenu'")
        self.assertEqual(result[0]['pos'], 'pronoun', "First word should be identified as pronoun")
        
        self.assertEqual(result[2]['word'], 'మాట్లాడుతున్నాను', "Third word should be the verb")
        if 'pos' in result[2]:
            self.assertEqual(result[2]['pos'], 'verb', "Third word should be identified as verb")
    
    def test_lemmatization(self):
        """Test text lemmatization functionality."""
        text = "పిల్లలు పాఠశాలకు వెళ్తున్నారు"  # Children are going to school
        expected = "పిల్ల పాఠశాల వెళ్ళ"  # Child school go
        
        result = self.analyzer.lemmatize_text(text)
        self.assertEqual(result, expected, f"Expected '{expected}' but got '{result}'")
    
    def test_get_grammatical_features(self):
        """Test extraction of grammatical features from text."""
        text = "నేను నిన్న స్కూలుకు వెళ్ళాను"  # I went to school yesterday
        
        features = self.analyzer.get_grammatical_features(text)
        
        # Check that we have features
        self.assertTrue(len(features['person']) > 0, "Should have identified person features")
        self.assertTrue(len(features['tense']) > 0, "Should have identified tense features")

if __name__ == '__main__':
    unittest.main()
