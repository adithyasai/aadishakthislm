import torch
from pathlib import Path
import os
import argparse
import logging
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import string

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GrammaticalSummarizer:
    """Improved summarization that preserves grammatical correctness"""
    
    def __init__(self, preserve_grammar=True, compression_level="moderate"):
        self.preserve_grammar = preserve_grammar
        # Compression level: "light", "moderate", "aggressive"
        self.compression_level = compression_level
        
    def preprocess_text(self, text, lang="te"):
        """Preprocess text based on language with better sentence boundary detection"""
        # Handle different sentence splitting for different languages
        if lang == "te":
            # Telugu sentence boundaries typically end with . or !
            # Use lookahead to avoid splitting on decimal points or abbreviations
            sentences = re.split(r'(?<=[.!?])\s+', text)
        elif lang == "hi":
            # Hindi sentence boundaries often use Devanagari danda (।)
            sentences = re.split(r'(?<=[.!?।])\s+', text)
        else:
            # Fallback to NLTK for other languages
            sentences = sent_tokenize(text)
            
        # Clean up sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def tokenize(self, text, lang="te"):
        """Tokenize text into words based on language while preserving structure"""
        if lang == "te" or lang == "hi":
            # For Indian languages, use a more careful approach that preserves structure
            # First, normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # For Telugu/Hindi, we'll split by spaces but keep significant punctuation
            # This helps preserve the grammatical structure
            tokens = []
            words = text.split()
            
            for word in words:
                # Check if word ends with punctuation
                if word and word[-1] in '.!?,;:':
                    if len(word) > 1:
                        tokens.append(word[:-1])  # Add word without punctuation
                    tokens.append(word[-1])  # Add punctuation as separate token
                else:
                    tokens.append(word)
            
            return tokens
        else:
            # For other languages, use NLTK's word_tokenize
            return nltk.word_tokenize(text)
    
    def get_word_frequencies(self, sentences, lang="te"):
        """Calculate word frequencies and identify important words"""
        word_frequencies = Counter()
        
        for sentence in sentences:
            words = self.tokenize(sentence, lang)
            for word in words:
                if len(word) > 1 and word not in string.punctuation:  # Skip single character words and punctuation
                    word_frequencies[word] += 1
                
        # Normalize frequencies
        max_freq = max(word_frequencies.values()) if word_frequencies else 1
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_freq
            
        return word_frequencies
    
    def identify_verbs_and_nouns(self, sentence, lang="te"):
        """Identify potential verbs and nouns in the sentence to maintain grammar"""
        # This is a simplified approach - a proper POS tagger for Telugu/Hindi would be better
        words = self.tokenize(sentence, lang)
        
        # Potential verb endings for Telugu
        te_verb_endings = ['ాయి', 'ాలి', 'డం', 'ారు', 'ాను', 'ావు', 'ాము']
        # Potential verb endings for Hindi
        hi_verb_endings = ['ना', 'ता', 'ते', 'ती', 'या', 'गा', 'गी', 'गे']
        
        verbs = []
        nouns = []
        
        for word in words:
            # Check for verbs based on endings
            if lang == "te":
                if any(word.endswith(end) for end in te_verb_endings):
                    verbs.append(word)
                else:
                    nouns.append(word)
            elif lang == "hi":
                if any(word.endswith(end) for end in hi_verb_endings):
                    verbs.append(word)
                else:
                    nouns.append(word)
            else:
                # For other languages, we'd need a POS tagger
                nouns.append(word)
                
        return verbs, nouns
    
    def get_key_phrases(self, sentences, word_frequencies, lang="te", top_n=5):
        """Extract key phrases from the text based on word importance"""
        phrases = []
        phrase_scores = {}
        
        for sentence in sentences:
            words = self.tokenize(sentence, lang)
            if len(words) < 3:
                continue
                
            # Create bigrams and trigrams
            for i in range(len(words)-1):
                # Only create phrases without punctuation
                if words[i] not in string.punctuation and words[i+1] not in string.punctuation:
                    # Bigrams
                    bigram = " ".join([words[i], words[i+1]])
                    if len(bigram) > 4:  # Skip very short bigrams
                        phrases.append(bigram)
                        # Score is the average of word frequencies
                        score = sum(word_frequencies.get(word, 0) for word in [words[i], words[i+1]]) / 2
                        phrase_scores[bigram] = score
                
                # Trigrams
                if i < len(words)-2 and words[i+2] not in string.punctuation:
                    trigram = " ".join([words[i], words[i+1], words[i+2]])
                    if len(trigram) > 6:  # Skip very short trigrams
                        phrases.append(trigram)
                        # Score is the average of word frequencies
                        score = sum(word_frequencies.get(word, 0) for word in [words[i], words[i+1], words[i+2]]) / 3
                        phrase_scores[trigram] = score
        
        # Sort phrases by score and return top N
        return sorted(phrases, key=lambda p: phrase_scores.get(p, 0), reverse=True)[:top_n]
    
    def score_sentences(self, sentences, word_frequencies, lang="te"):
        """Score sentences based on word importance and topic relevance"""
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            words = self.tokenize(sentence, lang)
            word_count = len(words)
            
            # Avoid division by zero
            if word_count == 0:
                continue
                
            # Score based on important words
            score = 0
            for word in words:
                if word in word_frequencies:
                    score += word_frequencies[word]
            
            # Normalize by sentence length with preference for medium length sentences
            if 5 <= word_count <= 25:
                # Give full weight to sentences of good length
                sentence_scores[i] = score / max(1, word_count)
            else:
                # Penalize very short or long sentences
                length_factor = min(1.0, 5/word_count if word_count < 5 else 25/word_count)
                sentence_scores[i] = (score / max(1, word_count)) * length_factor
                
            # Boost score for sentences appearing earlier in the text
            position_factor = 1.0 - (i / max(1, len(sentences)))
            sentence_scores[i] *= (1 + position_factor * 0.5)
                
        return sentence_scores
    
    def calculate_sentence_similarity(self, sentences, lang="te"):
        """Calculate similarity between sentences to reduce redundancy"""
        try:
            # Convert sentences to TF-IDF vectors
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(sentences)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(vectors)
            return similarity_matrix
        except:
            # Fallback if TF-IDF fails
            n = len(sentences)
            return np.identity(n)
    
    def compress_sentence_grammatically(self, sentence, word_frequencies, lang="te"):
        """Compress a sentence while preserving grammatical structure"""
        words = self.tokenize(sentence, lang)
        if len(words) <= 5:
            return sentence  # Don't compress very short sentences
            
        # Set compression rate based on level
        if self.compression_level == "light":
            compression_rate = 0.8
        elif self.compression_level == "aggressive":
            compression_rate = 0.5
        else:  # "moderate" (default)
            compression_rate = 0.65
            
        # Identify potential verbs and nouns to retain
        verbs, nouns = self.identify_verbs_and_nouns(sentence, lang)
        
        # Always keep verbs to maintain grammatical structure
        must_keep = set(verbs)
        
        # Add top nouns to keep
        noun_scores = {noun: word_frequencies.get(noun, 0) for noun in nouns}
        sorted_nouns = sorted(nouns, key=lambda n: noun_scores.get(n, 0), reverse=True)
        top_nouns_count = max(1, int(len(nouns) * compression_rate))
        must_keep.update(sorted_nouns[:top_nouns_count])
        
        # Score other words
        word_scores = {}
        for i, word in enumerate(words):
            # Prioritize keeping words that are near verbs
            verb_proximity = min([abs(i - words.index(verb)) for verb in verbs], default=len(words))
            
            # Important words get high scores
            if word in must_keep:
                word_scores[i] = 1000  # Very high score for must-keep words
            elif word in string.punctuation:
                word_scores[i] = 900   # High score for punctuation
            else:
                # Score based on word frequency and verb proximity
                freq_score = word_frequencies.get(word, 0) * 100
                proximity_score = max(0, (len(words) - verb_proximity) / len(words) * 50)
                word_scores[i] = freq_score + proximity_score
        
        # Calculate how many words to keep based on compression level
        words_to_keep = max(len(must_keep) + 3, int(len(words) * compression_rate))
        
        # Select the top-scoring word indices to keep
        indices_to_keep = sorted(
            range(len(words)), 
            key=lambda i: word_scores.get(i, 0), 
            reverse=True
        )[:words_to_keep]
        
        # Sort indices to maintain original word order
        indices_to_keep.sort()
        
        # Reconstruct the compressed sentence
        compressed_words = [words[i] for i in indices_to_keep]
        
        # Join the words back together
        if lang == "te" or lang == "hi":
            result = " ".join(compressed_words)
            
            # Clean up spacing around punctuation
            for punct in ".!?,;:":
                result = result.replace(f" {punct}", punct)
                
            return result
        else:
            return " ".join(compressed_words)
    
    def get_summary(self, text, lang="te", min_sentences=1, max_sentences=3):
        """Generate a concise summary from the text that maintains grammar"""
        if not text or text.isspace():
            return "Empty text provided."
            
        # Preprocess text
        sentences = self.preprocess_text(text, lang)
        
        if not sentences:
            return "Could not extract proper sentences from the text."
            
        # Handle very short texts
        if len(sentences) <= min_sentences:
            return text
            
        # Calculate word frequencies
        word_frequencies = self.get_word_frequencies(sentences, lang)
        
        # Get key phrases
        key_phrases = self.get_key_phrases(sentences, word_frequencies, lang)
        
        # Score sentences
        sentence_scores = self.score_sentences(sentences, word_frequencies, lang)
        
        # Calculate sentence similarity to avoid redundancy
        similarity_matrix = self.calculate_sentence_similarity(sentences, lang)
        
        # Select top sentences using MMR (Maximal Marginal Relevance)
        selected_indices = []
        candidate_indices = list(sentence_scores.keys())
        
        while len(selected_indices) < max_sentences and candidate_indices:
            # Calculate MMR scores
            mmr_scores = {}
            for i in candidate_indices:
                if not selected_indices:
                    # First sentence selection is based on pure sentence score
                    mmr_scores[i] = sentence_scores[i]
                else:
                    # For subsequent sentences, consider both relevance and redundancy
                    redundancy = max(similarity_matrix[i][j] for j in selected_indices)
                    mmr_scores[i] = sentence_scores[i] * (1 - 0.7 * redundancy)  # 0.7 is redundancy penalty
            
            # Select best sentence by MMR
            if mmr_scores:
                best_idx = max(mmr_scores, key=mmr_scores.get)
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            else:
                break
        
        # Ensure we have the minimum number of sentences
        if len(selected_indices) < min_sentences and len(sentences) >= min_sentences:
            # Add more sentences if needed
            remaining = [i for i in range(len(sentences)) if i not in selected_indices]
            sorted_remaining = sorted(remaining, key=lambda i: sentence_scores.get(i, 0), reverse=True)
            selected_indices.extend(sorted_remaining[:min_sentences - len(selected_indices)])
        
        # Sort indices to preserve original order
        selected_indices.sort()
        
        # Compress sentences if needed while preserving grammar
        final_sentences = []
        for idx in selected_indices:
            sentence = sentences[idx]
            if self.preserve_grammar:
                sentence = self.compress_sentence_grammatically(sentence, word_frequencies, lang)
            final_sentences.append(sentence)
        
        # Create final summary
        summary = ". ".join(final_sentences)
        
        # Add proper ending punctuation if missing
        if summary and not summary.endswith(('.', '!', '?', '।')):
            summary += '.'
        
        # If key phrases were found, add them as "Key topics: ..."
        if key_phrases:
            key_topics = ", ".join(key_phrases[:3])  # Limit to top 3 phrases
            summary += f"\n\nKey topics: {key_topics}"
        
        return summary

def generate_grammatical_summary(text: str, lang_code: str = 'te', **kwargs) -> str:
    """
    Generate a basic grammatical summary (stub implementation).
    """
    # Stub: return first sentence or full text
    sentences = text.split('.')
    return sentences[0] + ('.' if sentences[0] else '')

def main():
    parser = argparse.ArgumentParser(description='Generate grammatically correct summaries for Telugu or Hindi text')
    parser.add_argument('--text', type=str, help='Input text to summarize')
    parser.add_argument('--text_file', type=str, help='Path to a file containing the text to summarize')
    parser.add_argument('--lang', type=str, choices=['te', 'hi'], required=True, help='Language code')
    parser.add_argument('--output_file', type=str, help='Path to save the summary')
    parser.add_argument('--compression', type=str, choices=['light', 'moderate', 'aggressive'], 
                        default='moderate', help='Level of compression to apply')
    
    args = parser.parse_args()
    
    # Get text from file if provided
    if args.text_file and not args.text:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                args.text = f.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return
    
    # Use sample text if not provided
    if not args.text:
        if args.lang == 'te':
            args.text = """పర్యావరణ పరిరక్షణ మనందరిది బాధ్యత. రోజు రోజుకు వాతావరణంలో మార్పులు, కాలుష్యం పెరుగుతున్నాయి. వృక్షాలను నరికడం, ప్లాస్టిక్ వినియోగం, మరియు పరిశ్రమల వల్ల వాయు కాలుష్యం పెరుగుతున్నాయి. మన భవిష్యత్తు తరాలకు ఆరోగ్యవంతమైన జీవితం కల్పించాలంటే ఇప్పుడు నుంచే చర్యలు తీసుకోవాలి. మొక్కలు నాటడం, పునర్వినియోగ వనరులను వినియోగించడం వంటి చర్యలు చాలా ముఖ్యం. ప్రతి ఒక్కరూ ఈ విషయంలో చైతన్యం కలిగి ఉండాలి."""
        else:  # Hindi
            args.text = """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत गणराज्य की आधिकारिक भाषाओं में से एक है। हिंदी की लिपि देवनागरी है। वर्तमान हिंदी का विकास खड़ी बोली से हुआ है, जो दिल्ली के आसपास के क्षेत्रों में बोली जाती थी। हिंदी शब्द फारसी मूल का है, जिसका अर्थ है 'सिंधु नदी का'। मुगल काल में सिंधु नदी के पूर्व में रहने वाले लोगों और उनकी भाषा को 'हिंदी' कहा जाता था।"""
    
    # Create a summarizer with the specified compression level
    summarizer = GrammaticalSummarizer(preserve_grammar=True, compression_level=args.compression)
    
    print(f"\nOriginal {args.lang} text:")
    print("-" * 80)
    print(args.text)
    
    print("\nGenerated Summary:")
    print("-" * 80)
    try:
        summary = "Summary: " + summarizer.get_summary(args.text, args.lang)
        print(summary)
    except Exception as e:
        logger.error(f"Error: {e}")
        print("Error generating summary. Please check the logs for details.")
    
    # Save to file if requested
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Original {args.lang} text:\n{args.text}\n\nGenerated Summary:\n{summary}")
            print(f"\nSummary saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving to file: {e}")

if __name__ == "__main__":
    main()

import os
import torch
from pathlib import Path
import logging
import sys

# Add parent directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import custom model classes
from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"
MODEL_DIR = BASE_DIR / "models" / "final_model"

def simple_summary():
    # Sample texts
    te_text = """తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష."""
    hi_text = """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है।"""
    
    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer from {CONFIG_PATH}...")
        tokenizer = IndicTokenizer(str(CONFIG_PATH))
        logger.info("Tokenizer loaded successfully")
        
        # Load model configuration
        logger.info(f"Loading model configuration from {MODEL_DIR}...")
        config = IndicSLMConfig.from_pretrained(MODEL_DIR)
        logger.info(f"Model configuration loaded successfully")
        
        # Load model
        logger.info(f"Loading model from {MODEL_DIR}...")
        model = IndicSLM.from_pretrained(MODEL_DIR, config=config)
        logger.info("Model loaded successfully")
        
        # Put model in evaluation mode
        model.eval()
        
        print("\nAttempting to generate text with the model:")
        print("-" * 80)
        
        # Try with Telugu text
        print("\nTelugu input:")
        print(te_text)
        
        # Tokenize input
        lang = "te"
        input_text = f"<{lang}> {te_text}"
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate text with low memory settings
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=50,
                    num_beams=2,
                    do_sample=True,
                    early_stopping=True
                )
                
                # Decode output
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print("\nGenerated Telugu output:")
                print(output_text)
                
            except Exception as e:
                logger.error(f"Error generating Telugu text: {e}")
        
        # Try with Hindi text
        print("\nHindi input:")
        print(hi_text)
        
        # Tokenize input
        lang = "hi"
        input_text = f"<{lang}> {hi_text}"
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate text with low memory settings
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=50,
                    num_beams=2,
                    do_sample=True,
                    early_stopping=True
                )
                
                # Decode output
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print("\nGenerated Hindi output:")
                print(output_text)
                
            except Exception as e:
                logger.error(f"Error generating Hindi text: {e}")
        
    except Exception as e:
        logger.error(f"Error in simple summary: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    simple_summary()