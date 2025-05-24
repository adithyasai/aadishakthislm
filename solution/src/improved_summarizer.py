import os
import sys
import logging
import re
import string
import math
import nltk
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available - using basic similarity methods")

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK punkt is available for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ImprovedIndicSummarizer:
    """
    A comprehensive summarization system for Indian languages that combines
    extractive and abstractive elements to create coherent summaries
    """
    
    def __init__(self, lang_code: str = 'te'):
        """
        Initialize the summarizer
        
        Args:
            lang_code: Language code ('te' for Telugu, 'hi' for Hindi)
        """
        self.lang_code = lang_code
        
        # Initialize resources
        self.stopwords = self._load_stopwords()
        self.domain_terms = self._load_domain_terms()
        self.coherence_markers = self._load_coherence_markers()
        
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for the language"""
        # Telugu stopwords
        te_stopwords = {
            'మరియు', 'కూడా', 'ఒక', 'అది', 'ఈ', 'ఆ', 'అయితే', 'కానీ', 'లేదా', 
            'నుండి', 'వారు', 'మేము', 'నేను', 'మీరు', 'అతను', 'ఆమె', 'వాటిని', 
            'దాని', 'గా', 'తో', 'లో', 'పై', 'కి', 'ను', 'అందువలన', 'కోసం', 'గురించి',
            'వంటి', 'అన్ని', 'కొన్ని', 'ఏదైనా', 'ఎక్కడ', 'ఎప్పుడు', 'ఎందుకు', 'ఎలా',
            'అందరూ', 'చాలా', 'కొంత', 'మరి', 'ఇలా', 'అలా', 'తర్వాత', 'ముందు', 'అక్కడ',
            'ఇక్కడ', 'ఉంది', 'ఉంటుంది', 'ఉన్నారు', 'ఉన్నాయి', 'వచ్చింది', 'వెళ్ళింది'
        }
        
        # Hindi stopwords
        hi_stopwords = {
            'और', 'का', 'एक', 'में', 'की', 'है', 'यह', 'तथा', 'को', 'इस', 'पर', 'से', 
            'हैं', 'लिए', 'गया', 'किया', 'अपने', 'न', 'हो', 'कि', 'वह', 'वे', 'हम', 
            'था', 'होता', 'करना', 'किए', 'कुछ', 'भी', 'थे', 'द्वारा', 'जा', 'रहा', 'हुआ',
            'जैसे', 'सभी', 'कई', 'कोई', 'कहाँ', 'कब', 'क्यों', 'कैसे', 'अब', 'सब', 'उन',
            'वहाँ', 'यहाँ', 'तब', 'जब', 'होगा', 'करेगा', 'थी', 'थीं', 'उनके', 'उनका',
            'इनका', 'जिस', 'जिसे', 'जिसका', 'वाले', 'वाला', 'वाली', 'होने', 'बहुत', 'कम'
        }
        
        if self.lang_code == 'te':
            return te_stopwords
        elif self.lang_code == 'hi':
            return hi_stopwords
        else:
            return set()
    
    def _load_domain_terms(self) -> Dict[str, float]:
        """Load domain-specific terms with their importance weights"""
        # Telugu domain terms
        te_domain_terms = {
            # Language and literature domain
            'భాష': 3.0, 'సాహిత్యం': 3.0, 'కవిత': 2.5, 'ద్రావిడ': 2.5, 'తెలుగు': 2.0, 
            'లిపి': 2.5, 'వ్యాకరణం': 2.5, 'కావ్యం': 2.0, 'నిఘంటువు': 2.0, 'ప్రాచీన': 2.0,
            # Educational domain
            'విద్య': 3.0, 'బోధన': 2.5, 'విద్యార్థి': 2.0, 'పాఠశాల': 2.0, 'కళాశాల': 2.0,
            'విశ్వవిద్యాలయం': 2.5, 'పరీక్ష': 2.0, 'ఉపాధ్యాయుడు': 2.0, 'పాఠ్యాంశాలు': 2.0,
            # Science and technology
            'శాస్త్రం': 3.0, 'సాంకేతిక': 2.5, 'అభివృద్ధి': 2.5, 'పరిశోధన': 2.5,
            'విజ్ఞానం': 2.5, 'ఆవిష్కరణ': 2.0, 'నమూనా': 2.0, 'ప్రయోగం': 2.0,
            # Essential entities (proper nouns, places, dates)
            'భారతదేశం': 2.0, 'ఆంధ్రప్రదేశ్': 2.0, 'తెలంగాణ': 2.0, 'ప్రపంచ': 2.0, 
            'అధికారిక': 2.0, 'శతాబ్దం': 2.0, 'జనాభా': 2.0
        }
        
        # Hindi domain terms
        hi_domain_terms = {
            # Language and literature domain
            'भाषा': 3.0, 'साहित्य': 3.0, 'लिपि': 2.5, 'व्याकरण': 2.5, 'काव्य': 2.5,
            'देवनागरी': 2.5, 'हिंदी': 2.0, 'संस्कृत': 2.0, 'शब्द': 2.0, 'कविता': 2.0,
            # Educational domain
            'शिक्षा': 3.0, 'विद्यालय': 2.5, 'विद्यार्थी': 2.0, 'अध्ययन': 2.5, 
            'विश्वविद्यालय': 2.5, 'परीक्षा': 2.0, 'अध्यापक': 2.0, 'पाठ्यक्रम': 2.0,
            # Science and technology
            'विज्ञान': 3.0, 'तकनीकी': 2.5, 'विकास': 2.5, 'अनुसंधान': 2.5,
            'आविष्कार': 2.5, 'प्रयोग': 2.0, 'नमूना': 2.0,
            # Essential entities (proper nouns, places, dates)
            'भारत': 2.0, 'दिल्ली': 2.0, 'हिंदुस्तान': 2.0, 'विश्व': 2.0,
            'आधिकारिक': 2.0, 'शताब्दी': 2.0, 'जनसंख्या': 2.0, 'मुगल': 2.0
        }
        
        if self.lang_code == 'te':
            return te_domain_terms
        elif self.lang_code == 'hi':
            return hi_domain_terms
        else:
            return {}
    
    def _load_coherence_markers(self) -> Dict[str, List[str]]:
        """Load coherence markers by type for improved transitions"""
        # Telugu coherence markers
        te_markers = {
            'causal': ['కాబట్టి', 'అందువలన', 'దీనివలన', 'ఎందుకంటే'],
            'additive': ['మరియు', 'కూడా', 'అలాగే', 'అంతేకాకుండా', 'ఇంకా', 'పైగా'],
            'contrastive': ['అయితే', 'కానీ', 'అయినప్పటికీ'],
            'sequential': ['మొదటగా', 'తరువాత', 'చివరగా', 'ముందుగా', 'ఆపై', 'అటుపిమ్మట'],
            'illustrative': ['ఉదాహరణకు', 'అంటే', 'అనగా'],
            'summative': ['సారాంశంలో', 'చివరకు', 'మొత్తంమీద']
        }
        
        # Hindi coherence markers
        hi_markers = {
            'causal': ['इसलिए', 'अतः', 'क्योंकि', 'इसके कारण'],
            'additive': ['और', 'भी', 'तथा', 'इसके अलावा', 'साथ ही'],
            'contrastive': ['लेकिन', 'फिर भी', 'परंतु', 'इसके बावजूद', 'किंतु'],
            'sequential': ['पहले', 'फिर', 'अंत में', 'इसके बाद', 'आगे'],
            'illustrative': ['उदाहरण के लिए', 'जैसे', 'यानी', 'अर्थात्'],
            'summative': ['संक्षेप में', 'निष्कर्षतः', 'अंततः', 'कुल मिलाकर']
        }
        
        if self.lang_code == 'te':
            return te_markers
        elif self.lang_code == 'hi':
            return hi_markers
        else:
            return {}
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Improved sentence boundary detection for Telugu and Hindi with better handling
        of special characters and punctuation
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Normalize newlines and whitespace
        text = re.sub(r'[\n\r]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Handle different sentence endings based on language
        if self.lang_code == 'te':
            # Telugu uses both Western punctuation and Devanagari danda
            # First clean up any irregular spacing around sentence boundaries
            text = re.sub(r'([.!?।॥])\s*', r'\1 ', text)
            
            # Split on sentence boundaries
            raw_sentences = re.split(r'(?<=[.!?।॥])\s+', text)
            
            # Handle cases where sentence ends without a period
            if raw_sentences and not raw_sentences[-1].strip().endswith(('.', '!', '?', '।', '॥')):
                # Check if it's a complete sentence or just a fragment
                last_part = raw_sentences[-1].strip()
                if len(last_part.split()) > 3:  # Consider it a sentence if it has more than 3 words
                    raw_sentences[-1] = last_part + '.'
                else:
                    # If it's just a fragment, attach it to the previous sentence
                    if len(raw_sentences) > 1:
                        raw_sentences[-2] = raw_sentences[-2] + ' ' + last_part
                        raw_sentences.pop()
            
        elif self.lang_code == 'hi':
            # Hindi primarily uses Devanagari danda but also adopts Western punctuation
            text = re.sub(r'([.!?।॥])\s*', r'\1 ', text)
            
            # Split on sentence boundaries
            raw_sentences = re.split(r'(?<=[.!?।॥])\s+', text)
            
            # Handle cases where sentence ends without a period
            if raw_sentences and not raw_sentences[-1].strip().endswith(('.', '!', '?', '।', '॥')):
                last_part = raw_sentences[-1].strip()
                if len(last_part.split()) > 3:
                    raw_sentences[-1] = last_part + '।'
                else:
                    if len(raw_sentences) > 1:
                        raw_sentences[-2] = raw_sentences[-2] + ' ' + last_part
                        raw_sentences.pop()
        else:
            # For other languages, use NLTK's sentence tokenizer
            raw_sentences = nltk.sent_tokenize(text)
        
        # Post-process sentences
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if s and len(s.split()) >= 2:  # Require at least 2 words for a valid sentence
                sentences.append(s)
        
        return sentences
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words with proper handling of punctuation and special characters
        """
        # Handle language-specific tokenization
        if self.lang_code == 'te':
            # For Telugu, we need to handle special characters
            for punct in string.punctuation + '।॥':
                text = text.replace(punct, f' {punct} ')
            
            tokens = text.split()
            return tokens
            
        elif self.lang_code == 'hi':
            # For Hindi, handle Devanagari-specific punctuation
            for punct in string.punctuation + '।॥':
                text = text.replace(punct, f' {punct} ')
            
            tokens = text.split()
            return tokens
            
        else:
            # For other languages, use NLTK's word tokenizer
            return nltk.word_tokenize(text)
    
    def calculate_word_importance(self, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate word importance scores for weighting in summarization
        """
        # Create a list of all tokens
        all_tokens = []
        for sentence in sentences:
            tokens = self.tokenize_text(sentence)
            all_tokens.extend([t.lower() for t in tokens if t not in string.punctuation])
        
        # Count token frequencies
        token_counter = Counter(all_tokens)
        
        # Remove stopwords and calculate importance
        word_scores = {}
        max_freq = max(token_counter.values()) if token_counter else 1
        
        for token, count in token_counter.items():
            if token in self.stopwords or len(token) <= 1:
                continue
            
            # Base score from frequency
            score = count / max_freq
            
            # Boost domain-specific terms
            if token in self.domain_terms:
                score *= self.domain_terms[token]
            
            # Length factor for complex words (often more meaningful in Indian languages)
            length_factor = min(1.0, math.log(len(token) + 1, 10) / math.log(10, 10))
            score *= (1 + 0.2 * length_factor)
            
            word_scores[token] = score
        
        return word_scores
    
    def score_sentences(self, sentences: List[str], word_scores: Dict[str, float]) -> Dict[int, float]:
        """
        Score sentences based on multiple factors for better sentence selection
        """
        sentence_scores = {}
        
        # Document position weights
        position_weights = {}
        n_sentences = len(sentences)
        
        for i in range(n_sentences):
            # First and last sentences are typically more important
            if i == 0:
                position_weights[i] = 1.2  # First sentence bonus
            elif i == n_sentences - 1:
                position_weights[i] = 1.1  # Last sentence bonus
            elif i < n_sentences // 3:
                position_weights[i] = 1.0  # First third
            elif i >= 2 * n_sentences // 3:
                position_weights[i] = 0.9  # Last third
            else:
                position_weights[i] = 0.8  # Middle sentences
        
        # Score each sentence
        for i, sentence in enumerate(sentences):
            tokens = self.tokenize_text(sentence)
            content_tokens = [t.lower() for t in tokens if t.lower() not in self.stopwords and t not in string.punctuation]
            
            if not content_tokens:
                sentence_scores[i] = 0
                continue
            
            # Content importance score
            content_score = sum(word_scores.get(token, 0) for token in content_tokens)
            
            # Normalize by effective sentence length with logarithmic scaling
            # This avoids over-penalizing long sentences
            length_norm = math.log(len(content_tokens) + 1) / math.log(10)
            normalized_content_score = content_score / length_norm
            
            # Apply position weight
            position_score = position_weights.get(i, 0.8)
            
            # Check for presence of coherence markers that could indicate important sentences
            has_marker = False
            for marker_type, markers in self.coherence_markers.items():
                if any(marker in sentence.lower() for marker in markers):
                    has_marker = True
                    break
            
            # Give a small bonus to sentences with coherence markers
            coherence_bonus = 1.1 if has_marker else 1.0
            
            # Combine scores: content (70%), position (20%), coherence (10%)
            final_score = (0.7 * normalized_content_score) + (0.2 * position_score) + (0.1 * coherence_bonus)
            
            sentence_scores[i] = final_score
        
        return sentence_scores
    
    def calculate_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Calculate similarity between sentences for redundancy reduction
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        # Check if we have access to scikit-learn for better vectorization
        if SKLEARN_AVAILABLE:
            try:
                # Create a custom tokenizer that removes stopwords
                def custom_tokenize(text):
                    tokens = self.tokenize_text(text)
                    return [t.lower() for t in tokens if t.lower() not in self.stopwords and t not in string.punctuation]
                
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    tokenizer=custom_tokenize,
                    use_idf=True,
                    min_df=1
                )
                
                # Create sentence vectors
                tfidf_matrix = vectorizer.fit_transform(sentences)
                
                # Calculate cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Zero out diagonal (self-similarity)
                np.fill_diagonal(similarity_matrix, 0)
                
                return similarity_matrix
                
            except Exception as e:
                logger.warning(f"Error calculating TF-IDF similarity: {e}")
                # Fall through to token overlap method
        
        # Token overlap method (fallback)
        for i in range(n):
            tokens_i = set(self.tokenize_text(sentences[i].lower()))
            tokens_i = {t for t in tokens_i if t not in self.stopwords and t not in string.punctuation}
            
            for j in range(n):
                if i == j:
                    continue
                    
                tokens_j = set(self.tokenize_text(sentences[j].lower()))
                tokens_j = {t for t in tokens_j if t not in self.stopwords and t not in string.punctuation}
                
                # Calculate Jaccard similarity
                if tokens_i and tokens_j:
                    intersection = len(tokens_i.intersection(tokens_j))
                    union = len(tokens_i.union(tokens_j))
                    similarity_matrix[i, j] = intersection / union if union > 0 else 0
        
        return similarity_matrix
    
    def select_sentences_mmr(self, sentences: List[str], scores: Dict[int, float], 
                            similarity_matrix: np.ndarray, num_sentences: int, 
                            lambda_param: float = 0.7) -> List[int]:
        """
        Select sentences using Maximal Marginal Relevance (MMR) to balance
        relevance and redundancy
        """
        # Validate inputs
        if not sentences or not scores:
            return []
        
        # Initialize with empty selected set
        selected_indices = []
        candidate_indices = list(range(len(sentences)))
        
        while len(selected_indices) < num_sentences and candidate_indices:
            mmr_scores = {}
            
            for i in candidate_indices:
                if not selected_indices:
                    # For first sentence, just use the original score
                    mmr_scores[i] = scores[i]
                else:
                    # Calculate redundancy as max similarity to any selected sentence
                    redundancy = max(similarity_matrix[i][j] for j in selected_indices)
                    
                    # MMR formula: relevance - redundancy trade-off
                    mmr_scores[i] = lambda_param * scores[i] - (1 - lambda_param) * redundancy
            
            # Select the sentence with highest MMR score
            if mmr_scores:
                best_idx = max(mmr_scores.keys(), key=lambda i: mmr_scores[i])
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            else:
                break
        
        # Return indices in original order
        return sorted(selected_indices)
    
    def ensure_coherence(self, selected_sentences: List[str]) -> List[str]:
        """
        Improve coherence by adding connector words between sentences
        """
        if not selected_sentences or len(selected_sentences) <= 1:
            return selected_sentences
        
        # Keep first sentence as is
        coherent_sentences = [selected_sentences[0]]
        
        # Process subsequent sentences
        for i in range(1, len(selected_sentences)):
            current_sentence = selected_sentences[i]
            
            # Check if the sentence already starts with a connector
            has_connector = False
            for marker_type, markers in self.coherence_markers.items():
                if any(current_sentence.lower().startswith(marker.lower()) for marker in markers):
                    has_connector = True
                    break
            
            if not has_connector:
                # Determine appropriate connector based on position and context
                if i == 1:
                    # Second sentence connectors
                    if self.lang_code == 'te':
                        connector = "అలాగే, "
                    else:  # Hindi
                        connector = "इसके अलावा, "
                        
                elif i == len(selected_sentences) - 1:
                    # Last sentence connectors
                    if self.lang_code == 'te':
                        connector = "చివరగా, "
                    else:  # Hindi
                        connector = "अंत में, "
                        
                else:
                    # Middle sentence connectors - alternate between types
                    connector_types = ['additive', 'sequential', 'illustrative'] 
                    current_type = connector_types[(i-1) % len(connector_types)]
                    
                    # Get markers of current type
                    type_markers = self.coherence_markers.get(current_type, [])
                    if type_markers:
                        # Choose a marker based on position
                        marker_idx = (i - 1) % len(type_markers)
                        connector = type_markers[marker_idx] + ", "
                    else:
                        # Fallback connectors
                        if self.lang_code == 'te':
                            connector = "ఇంకా, "
                        else:  # Hindi
                            connector = "इसके अतिरिक्त, "
                
                # Add the connector to the sentence
                if current_sentence[0].isupper():
                    # Preserve capitalization if the sentence begins with uppercase
                    coherent_sentences.append(connector + current_sentence)
                else:
                    # Start with connector
                    coherent_sentences.append(connector + current_sentence)
            else:
                # Sentence already has a connector, use as is
                coherent_sentences.append(current_sentence)
        
        return coherent_sentences
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        Extract important keywords from the text
        """
        # Split into sentences and tokenize
        sentences = self.segment_sentences(text)
        all_tokens = []
        
        for sentence in sentences:
            tokens = self.tokenize_text(sentence)
            # Filter tokens
            filtered_tokens = [t.lower() for t in tokens 
                              if t.lower() not in self.stopwords 
                              and t not in string.punctuation
                              and len(t) > 1]
            all_tokens.extend(filtered_tokens)
        
        if not all_tokens:
            return []
        
        # Get word frequencies
        token_counter = Counter(all_tokens)
        
        # Apply domain-specific weights
        weighted_scores = {}
        for token, count in token_counter.items():
            # Base score is the frequency
            score = count
            
            # Apply domain weight if available
            if token in self.domain_terms:
                score *= self.domain_terms[token]
            
            weighted_scores[token] = score
        
        # Get top keywords
        top_keywords = [token for token, _ in 
                       sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:num_keywords]]
        
        return top_keywords
    
    def clean_punctuation(self, text: str) -> str:
        """
        Clean up any irregularities in punctuation spacing
        """
        # Fix spaces before punctuation
        text = re.sub(r'\s+([,.!?:;।॥])', r'\1', text)
        
        # Ensure space after punctuation
        text = re.sub(r'([,.!?:;।॥])([^\s])', r'\1 \2', text)
        
        # Fix multiple punctuation
        text = re.sub(r'([.!?।॥])[.!?।॥]+', r'\1', text)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def summarize(self, text: str, compression_ratio: float = 0.3, 
                 max_sentences: int = 5) -> str:
        """
        Generate a comprehensive summary with improved coherence
        
        Args:
            text: Text to summarize
            compression_ratio: Target ratio of summary size to original size
            max_sentences: Maximum number of sentences in the summary
        
        Returns:
            A coherent summary of the input text
        """
        if not text or len(text.strip()) == 0:
            return "Empty text provided."
        
        # Split into sentences
        sentences = self.segment_sentences(text)
        
        if not sentences:
            return "Could not extract proper sentences from the text."
        
        # Return original text if too short
        if len(sentences) <= 2:
            return text
        
        # Calculate word importance scores
        word_scores = self.calculate_word_importance(sentences)
        
        # Score sentences
        sentence_scores = self.score_sentences(sentences, word_scores)
        
        # Calculate similarity matrix for redundancy detection
        similarity_matrix = self.calculate_similarity_matrix(sentences)
        
        # Calculate target number of sentences based on compression ratio
        target_sentences = min(
            max_sentences,
            max(1, int(len(sentences) * compression_ratio))
        )
        
        # Select sentences using MMR
        selected_indices = self.select_sentences_mmr(
            sentences,
            sentence_scores,
            similarity_matrix,
            target_sentences,
            lambda_param=0.7  # Balance between relevance (0.7) and diversity (0.3)
        )
        
        # Get selected sentences
        selected_sentences = [sentences[i] for i in selected_indices]
        
        # Improve coherence with connector words
        coherent_sentences = self.ensure_coherence(selected_sentences)
        
        # Join sentences into a summary
        summary = ' '.join(coherent_sentences)
        
        # Clean up any punctuation issues
        summary = self.clean_punctuation(summary)
        
        # Extract keywords
        keywords = self.extract_keywords(text, num_keywords=5)
        
        # Add keywords section
        if keywords:
            if self.lang_code == 'te':
                keyword_heading = "ముఖ్యమైన అంశాలు"
            elif self.lang_code == 'hi':
                keyword_heading = "महत्वपूर्ण बिंदु"
            else:
                keyword_heading = "Key Points"
                
            keywords_text = ", ".join(keywords)
            summary = f"{summary}\n\n{keyword_heading}: {keywords_text}"
        
        return summary

# Create a convenient function to generate summaries
def generate_improved_summary(text: str, lang: str = 'te', 
                             compression_ratio: float = 0.3,
                             max_sentences: int = 5) -> str:
    """
    Generate an improved summary for Telugu or Hindi text
    
    Args:
        text: The text to summarize
        lang: Language code ('te' for Telugu, 'hi' for Hindi)
        compression_ratio: Target ratio of summary length to original text
        max_sentences: Maximum number of sentences in the summary
    
    Returns:
        A coherent, well-formed summary
    """
    summarizer = ImprovedIndicSummarizer(lang_code=lang)
    return summarizer.summarize(text, compression_ratio, max_sentences)

if __name__ == "__main__":
    # Sample usage
    te_text = """తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష. ఇది భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో నాలుగవది మరియు ద్రావిడ భాషా కుటుంబంలో రెండవ అతిపెద్ద భాష. 2011 జనాభా లెక్కల ప్రకారం, భారతదేశంలో 81.1 మిలియన్లకు పైగా ప్రజలు తెలుగును మాతృభాషగా మాట్లాడతారు, సింగపూర్‌లో అధికారిక భాషగా గుర్తింపు పొందింది. తెలుగు భాష చరిత్ర క్రీ.పూ. 400 నాటిది, ఇది భారతదేశ ప్రాచీన భాషలలో ఒకటి. 10వ శతాబ్దం నాటి నన్నయ్య రచించిన ఆంధ్రమహాభారతాన్ని తెలుగు తొలి కావ్యంగా భావిస్తారు. తెలుగు సాహిత్యం మూడు యుగాలుగా విభజించబడింది: కవిత్రయం, శతక యుగం మరియు ఆధునిక యుగం. ప్రపంచవ్యాప్తంగా 85 మిలియన్లకు పైగా జనాభా తెలుగు మాట్లాడతారు."""
    
    hi_text = """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत गणराज्य की आधिकारिक भाषाओं में से एक है। हिंदी की लिपि देवनागरी है। वर्तमान हिंदी का विकास खड़ी बोली से हुआ है, जो दिल्ली के आसपास के क्षेत्रों में बोली जाती थी। हिंदी शब्द फारसी मूल का है, जिसका अर्थ है 'सिंधु नदी का'। मुगल काल में सिंधु नदी के पूर्व में रहने वाले लोगों और उनकी भाषा को 'हिंदी' कहा जाता था। हिंदी भाषा के विकास को भाषावैज्ञानिक आधार पर चार कालों में बांटा गया है: आदिकाल (12वीं से 14वीं शताब्दी), भक्तिकाल (14वीं से 16वीं शताब्दी), रीतिकाल (17वीं से 19वीं शताब्दी) और आधुनिक काल (19वीं शताब्दी से अब तक)।"""
    
    print("Telugu Summary:")
    print("-" * 80)
    te_summary = generate_improved_summary(te_text, 'te', 0.5)
    print(te_summary)
    
    print("\n\nHindi Summary:")
    print("-" * 80)
    hi_summary = generate_improved_summary(hi_text, 'hi', 0.5)
    print(hi_summary)

import re
import logging
import math
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelniveau)s - %(message)s')
logger = logging.getLogger(__name__)

# Advanced summarization using a hybrid approach
def generate_improved_summary(text: str, lang: str = 'te', max_sentences: int = 5, 
                             compression_ratio: float = 0.3) -> str:
    """
    Generate an improved summary using a hybrid extractive-abstractive approach
    
    Args:
        text: Text to summarize
        lang: Language code ('te' for Telugu, 'hi' for Hindi)
        max_sentences: Maximum number of sentences to include
        compression_ratio: Target ratio of summary length to original text
        
    Returns:
        Generated summary text
    """
    if not text or len(text.strip()) == 0:
        return "Empty text provided."
    
    try:
        # Step 1: Segment text into sentences
        sentences = _segment_sentences(text, lang)
        
        # For very short texts, return as is
        if len(sentences) <= 2:
            return text
        
        # Step 2: Analyze and score sentences
        sentence_scores = _score_sentences_advanced(sentences, lang)
        
        # Step 3: Calculate target summary length
        target_count = max(1, min(max_sentences, 
                                  int(len(sentences) * compression_ratio + 0.5)))
        
        # Step 4: Select sentences for summary
        selected_indices = _select_diverse_sentences(sentences, sentence_scores, target_count)
        selected_sentences = [sentences[i] for i in selected_indices]
        
        # Step 5: Apply post-processing for coherence and flow
        enhanced_summary = _enhance_summary_coherence(selected_sentences, lang)
        
        return enhanced_summary
        
    except Exception as e:
        logger.error(f"Error generating improved summary: {e}")
        # Fallback to basic summary
        sentences = _segment_sentences(text, lang)
        return sentences[0] if sentences else "Summarization failed."

def _segment_sentences(text: str, lang: str) -> List[str]:
    """Segment text into sentences with improved handling for Indian languages"""
    # Clean and normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Language-specific sentence splitting
    if lang == 'te':
        # Telugu uses both Western and Devanagari punctuation
        sentences = re.split(r'(?<=[.!?।॥])\s+', text)
    elif lang == 'hi':
        # Hindi primarily uses Devanagari punctuation
        sentences = re.split(r'(?<=[.!?।॥])\s+', text)
    else:
        # Default fallback
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Post-process sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Require at least 2 words for a valid sentence
        if sentence and len(sentence.split()) >= 2:
            # Fix sentences lacking ending punctuation
            if not sentence[-1] in '.!?।॥':
                sentence += '.'
            clean_sentences.append(sentence)
    
    return clean_sentences

def _score_sentences_advanced(sentences: List[str], lang: str) -> Dict[int, float]:
    """Score sentences with multiple factors for importance"""
    # Extract all words for frequency analysis
    all_words = []
    for sentence in sentences:
        words = [w.lower() for w in sentence.split()]
        all_words.extend(words)
        
    # Calculate word frequencies
    word_freq = Counter(all_words)
    max_freq = max(word_freq.values()) if word_freq else 1
    
    # Stopwords for each language
    stopwords = _get_stopwords(lang)
    
    # Important domain terms by language
    domain_terms = {
        'te': {'భాష': 2.0, 'భారతదేశం': 1.5, 'తెలుగు': 1.5, 'ప్రభుత్వం': 1.5, 
              'సాహిత్యం': 1.5, 'చరిత్ర': 1.5, 'సంస్కృతి': 1.5, 'విద్య': 1.5},
        'hi': {'भाषा': 2.0, 'भारत': 1.5, 'हिंदी': 1.5, 'सरकार': 1.5, 
              'साहित्य': 1.5, 'इतिहास': 1.5, 'संस्कृति': 1.5, 'शिक्षा': 1.5}
    }
    
    term_weights = domain_terms.get(lang, {})
    
    # Score calculation for each sentence
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        words = [w.lower() for w in sentence.split()]
        
        if not words:
            sentence_scores[i] = 0
            continue
        
        # Filter out stopwords for scoring
        content_words = [w for w in words if w not in stopwords]
        
        if not content_words:
            content_words = words  # Fallback if all words are stopwords
            
        # Base score from word frequency
        freq_score = sum(word_freq[w] / max_freq for w in content_words) / len(content_words)
        
        # Term importance boost
        term_score = 0
        for word in content_words:
            if word in term_weights:
                term_score += term_weights[word]
        term_score = term_score / len(content_words) if content_words else 0
        
        # Position bias
        position_score = 0
        if i == 0:  # First sentence
            position_score = 1.5
        elif i == len(sentences) - 1:  # Last sentence
            position_score = 1.2
        elif i < len(sentences) / 3:  # First third
            position_score = 1.0
        else:
            position_score = 0.8
        
        # Sentence length factor (prefer medium-length sentences)
        length = len(words)
        if length < 5:
            length_factor = 0.7  # Too short
        elif length > 25:
            length_factor = 0.8  # Too long
        else:
            length_factor = 1.0  # Just right
        
        # Combine all factors
        final_score = (0.4 * freq_score + 
                      0.25 * term_score + 
                      0.25 * position_score + 
                      0.1 * length_factor)
        
        sentence_scores[i] = final_score
    
    return sentence_scores

def _get_sentence_similarity(sentences: List[str], lang: str) -> np.ndarray:
    """Calculate similarity between sentences to avoid redundancy"""
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    
    # Stopwords for each language
    stopwords = _get_stopwords(lang)
    
    # Calculate pairwise similarity
    for i in range(n):
        # Tokenize and filter first sentence
        words_i = [w.lower() for w in sentences[i].split() 
                 if w.lower() not in stopwords and len(w) > 1]
        words_i_set = set(words_i)
        
        for j in range(i+1, n):
            # Tokenize and filter second sentence
            words_j = [w.lower() for w in sentences[j].split() 
                     if w.lower() not in stopwords and len(w) > 1]
            words_j_set = set(words_j)
            
            # Skip if either sentence has no content words
            if not words_i_set or not words_j_set:
                continue
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(words_i_set.intersection(words_j_set))
            union = len(words_i_set.union(words_j_set))
            
            similarity = intersection / union if union > 0 else 0
            
            # Store in symmetric matrix
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix

def _select_diverse_sentences(sentences: List[str], scores: Dict[int, float], 
                             target_count: int) -> List[int]:
    """Select diverse sentences using the Maximal Marginal Relevance approach"""
    if not sentences or not scores:
        return []
    
    # Calculate similarity matrix
    similarity = _get_sentence_similarity(sentences, 'te')  # Default to Telugu
    
    # Initialize variables
    selected = []
    candidates = list(range(len(sentences)))
    
    # Always select the highest scoring sentence first
    if candidates:
        best_idx = max(candidates, key=lambda i: scores[i])
        selected.append(best_idx)
        candidates.remove(best_idx)
    
    # MMR algorithm
    lambda_param = 0.7  # Balance between relevance and diversity
    
    while len(selected) < target_count and candidates:
        # Find the candidate with the best MMR score
        best_mmr = -float('inf')
        best_idx = -1
        
        for i in candidates:
            # Relevance component
            relevance = scores[i]
            
            # Redundancy component (max similarity to any selected sentence)
            redundancy = max([similarity[i, j] for j in selected]) if selected else 0
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        
        if best_idx != -1:
            selected.append(best_idx)
            candidates.remove(best_idx)
        else:
            break
    
    # Sort indices to maintain original order
    return sorted(selected)

def _enhance_summary_coherence(sentences: List[str], lang: str) -> str:
    """Improve coherence between sentences"""
    if not sentences:
        return ""
    
    # For single sentences, no enhancement needed
    if len(sentences) <= 1:
        return sentences[0]
    
    # Connector phrases by language and type
    connectors = {
        'te': {
            'sequence': ['మరియు', 'అలాగే', 'అదేవిధంగా'],
            'contrast': ['అయితే', 'కానీ', 'అయినప్పటికీ'],
            'cause': ['అందువలన', 'దీని ఫలితంగా', 'కాబట్టి'],
            'conclusion': ['చివరగా', 'ముగింపులో', 'సారాంశంలో']
        },
        'hi': {
            'sequence': ['और', 'साथ ही', 'इसी प्रकार'],
            'contrast': ['लेकिन', 'परंतु', 'फिर भी'],
            'cause': ['इसलिए', 'इसके कारण', 'अतः'],
            'conclusion': ['अंत में', 'निष्कर्ष में', 'संक्षेप में']
        }
    }
    
    lang_connectors = connectors.get(lang, connectors['te'])
    
    # Apply connectors based on sentence position and content
    enhanced_sentences = [sentences[0]]  # First sentence remains unchanged
    
    for i in range(1, len(sentences)):
        curr_sent = sentences[i]
        
        # Determine appropriate connector type based on position
        if i == len(sentences) - 1:
            connector_type = 'conclusion'
        elif i % 3 == 1:
            connector_type = 'sequence'
        elif i % 3 == 2:
            connector_type = 'contrast'
        else:
            connector_type = 'cause'
        
        # Get available connectors of this type
        available_connectors = lang_connectors.get(connector_type, lang_connectors['sequence'])
        
        # Choose a connector (cycle through options)
        connector = available_connectors[i % len(available_connectors)]
        
        # Add connector at the beginning of the sentence
        enhanced_sentence = f"{connector}, {curr_sent}"
        enhanced_sentences.append(enhanced_sentence)
    
    # Join sentences with proper spacing
    return ' '.join(enhanced_sentences)

def _get_stopwords(lang: str) -> Set[str]:
    """Get stopwords for the specified language"""
    if lang == 'te':
        # Telugu stopwords
        return {
            'మరియు', 'ఉంది', 'ఉన్నాయి', 'మీద', 'ద్వారా', 'తో', 'కూడా', 'కు', 'లో', 
            'వద్ద', 'నుండి', 'చేత', 'ఈ', 'ఆ', 'అది', 'వారు', 'తాము', 'మేము', 'నేను', 
            'మీరు', 'అతను', 'ఆమె', 'అయితే', 'కానీ', 'లేదా', 'మరొక', 'కొన్ని', 'కొంత',
            'అన్ని', 'ఏదైనా', 'కాదు', 'లేదు', 'వేరే', 'కాబట్టి', 'అందుకే', 'అప్పుడు'
        }
    elif lang == 'hi':
        # Hindi stopwords
        return {
            'और', 'का', 'एक', 'में', 'की', 'है', 'यह', 'तथा', 'को', 'इस', 'पर', 'से', 
            'हैं', 'लिए', 'गया', 'किया', 'अपने', 'न', 'हो', 'कि', 'वह', 'वे', 'हम', 
            'था', 'होता', 'करना', 'किए', 'कुछ', 'भी', 'थे', 'द्वारा', 'ने', 'बहुत',
            'कर', 'इसके', 'या', 'हुआ', 'तो', 'ही', 'हुई', 'जो', 'अब', 'सब', 'वहाँ', 'उस'
        }
    else:
        return set()


if __name__ == "__main__":
    # Example usage with Telugu text
    te_text = """తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష. ఇది భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో నాలుగవది మరియు ద్రావిడ భాషా కుటుంబంలో రెండవ అతిపెద్ద భాష. 2011 జనాభా లెక్కల ప్రకారం, భారతదేశంలో 81.1 మిలియన్లకు పైగా ప్రజలు తెలుగును మాతృభాషగా మాట్లాడతారు, సింగపూర్‌లో అధికారిక భాషగా గుర్తింపు పొందింది. తెలుగు భాష చరిత్ర క్రీ.పూ. 400 నాటిది, ఇది భారతదేశ ప్రాచీన భాషలలో ఒకటి. 10వ శతాబ్దం నాటి నన్నయ్య రచించిన ఆంధ్రమహాభారతాన్ని తెలుగు తొలి కావ్యంగా భావిస్తారు. తెలుగు సాహిత్యం మూడు యుగాలుగా విభజించబడింది: కవిత్రయం, శతక యుగం మరియు ఆధునిక యుగం. ప్రపంచవ్యాప్తంగా 85 మిలియన్లకు పైగా జనాభా తెలుగు మాట్లాడతారు."""
    
    # Generate improved summary
    summary = generate_improved_summary(te_text, 'te')
    
    print("Original text:")
    print("-" * 60)
    print(te_text)
    print("\nImproved summary:")
    print("-" * 60)
    print(summary)

import os
import torch
import logging
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to Python path if needed
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import custom modules
from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer
from src.neural_summarizer import NeuralSummarizer

class ImprovedSummarizer:
    """
    Enhanced summarizer that leverages relative position embeddings for 
    better understanding of Indian languages like Telugu and Hindi that have 
    relatively flexible word order.
    
    This summarizer extends the NeuralSummarizer with explicit support for
    relative position embeddings and improved attention-based sentence selection.
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 config_path: str = None, 
                 lang_code: str = 'te',
                 attention_type: str = 'relative'):
        """
        Initialize the improved summarizer with a pretrained model with 
        relative position embeddings
        
        Args:
            model_path: Path to the model directory
            config_path: Path to the model config file
            lang_code: Language code ('te' for Telugu, 'hi' for Hindi)
            attention_type: Type of attention mechanism to use ('relative' is recommended)
        """
        self.lang_code = lang_code
        self.model_loaded = False
        self.attention_type = attention_type
        
        # Set default paths if not provided
        if not model_path or not config_path:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = model_path or str(base_dir / "models" / "final_model")
            config_path = config_path or str(base_dir / "configs" / "model_config.json")
        
        try:
            # Load tokenizer
            self.tokenizer = IndicTokenizer(config_path)
            
            # Load or create model config with relative position embeddings
            if os.path.exists(os.path.join(model_path, "config.json")):
                self.config = IndicSLMConfig.from_pretrained(model_path)
                # Update attention type to ensure we're using relative position embeddings
                self.config.attention_type = attention_type
            else:
                # Create new config with relative position embeddings
                logger.info(f"Creating new model config with {attention_type} position embeddings")
                self.config = IndicSLMConfig(
                    vocab_size=self.tokenizer.get_vocab_size(),
                    attention_type=attention_type,
                    max_relative_position=32  # Good default for Indian languages
                )
            
            # Load or initialize model
            if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                self.model = IndicSLM.from_pretrained(model_path, config=self.config)
            else:
                logger.info("Initializing new model")
                self.model = IndicSLM(self.config)
            
            self.model.eval()  # Set to evaluation mode
            self.model_loaded = True
            logger.info(f"Model with {attention_type} position embeddings loaded successfully")
            
            # Initialize the base neural summarizer for text preprocessing and structure
            self.neural_summarizer = NeuralSummarizer(
                model_path=model_path,
                config_path=config_path,
                lang_code=lang_code
            )
            
        except Exception as e:
            logger.error(f"Error initializing improved summarizer: {e}")
            logger.error("Falling back to base neural summarizer")
            self.model = None
            self.neural_summarizer = NeuralSummarizer(
                model_path=model_path,
                config_path=config_path,
                lang_code=lang_code
            )
    
    def _compute_sentence_embeddings(self, sentences: List[str]) -> List[torch.Tensor]:
        """
        Compute sentence embeddings using the model with relative position embeddings
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            List of sentence embeddings
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot compute embeddings")
            return None
        
        embeddings = []
        with torch.no_grad():
            for sentence in sentences:
                # Tokenize the sentence
                tokens = self.tokenizer.encode(sentence)
                input_ids = torch.tensor([tokens], dtype=torch.long)
                
                # Get model outputs
                outputs = self.model(input_ids=input_ids)
                
                # Use the last layer representation as sentence embedding
                # We use mean pooling over tokens to get a fixed-size representation
                sentence_embedding = outputs['logits'].mean(dim=1).squeeze().cpu()
                embeddings.append(sentence_embedding)
        
        return embeddings
    
    def _compute_sentence_similarities(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute sentence similarities based on embeddings
        
        Args:
            embeddings: List of sentence embeddings
            
        Returns:
            Similarity matrix between sentences
        """
        if not embeddings:
            return None
        
        # Stack embeddings into a tensor
        embeddings_tensor = torch.stack(embeddings)
        
        # Normalize embeddings
        norm = embeddings_tensor.norm(dim=1, keepdim=True)
        normalized_embeddings = embeddings_tensor / norm
        
        # Compute cosine similarity
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        return similarity_matrix
    
    def _rank_sentences_by_centrality(self, similarity_matrix: torch.Tensor) -> List[int]:
        """
        Rank sentences by their centrality in the document based on the similarity matrix
        
        Args:
            similarity_matrix: Similarity matrix between sentences
            
        Returns:
            List of sentence indices sorted by importance
        """
        if similarity_matrix is None:
            return []
        
        # Compute centrality scores (sum of similarities)
        centrality_scores = similarity_matrix.sum(dim=1).cpu().numpy()
        
        # Create (index, score) pairs and sort by score in descending order
        indexed_scores = [(i, score) for i, score in enumerate(centrality_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted indices
        return [idx for idx, _ in indexed_scores]
    
    def _enhance_summary_with_relative_attention(self, 
                                                text: str, 
                                                compression_ratio: float = 0.5,
                                                max_length: int = 200) -> str:
        """
        Generate an enhanced summary using relative position embeddings
        
        Args:
            text: Text to summarize
            compression_ratio: Target ratio of summary length to original text
            max_length: Maximum summary length in words
            
        Returns:
            Enhanced summary
        """
        if not text or len(text.strip()) == 0:
            return "Empty text provided."
        
        if not self.model_loaded:
            # Fall back to neural summarizer if model not loaded
            return self.neural_summarizer.summarize(text, compression_ratio)
        
        try:
            # Segment text into sentences
            sentences = self.neural_summarizer._segment_text(text)
            
            if len(sentences) <= 2:
                return text  # Return original for very short texts
            
            # Calculate target number of sentences for summary
            target_sentences = max(1, min(5, int(len(sentences) * compression_ratio + 0.5)))
            
            # Compute sentence embeddings using the model with relative position embeddings
            sentence_embeddings = self._compute_sentence_embeddings(sentences)
            
            if not sentence_embeddings:
                # Fall back to neural summarizer if embeddings failed
                return self.neural_summarizer.summarize(text, compression_ratio)
            
            # Compute sentence similarities
            similarity_matrix = self._compute_sentence_similarities(sentence_embeddings)
            
            # Rank sentences by centrality
            ranked_indices = self._rank_sentences_by_centrality(similarity_matrix)
            
            # Select top sentences while maintaining original order
            selected_indices = sorted(ranked_indices[:target_sentences])
            selected_sentences = [sentences[i] for i in selected_indices]
            
            # Apply neural summarizer's compression and flow enhancement
            compressed_sentences = []
            for sentence in selected_sentences:
                # Apply more aggressive compression to longer sentences
                if len(sentence.split()) > 15:
                    compressed = self.neural_summarizer._compress_sentence(sentence, aggressive=True)
                elif len(sentence.split()) > 10:
                    compressed = self.neural_summarizer._compress_sentence(sentence, aggressive=False)
                else:
                    compressed = sentence  # No compression for short sentences
                
                compressed_sentences.append(compressed)
            
            # Enhance flow with appropriate connectors
            enhanced_sentences = self.neural_summarizer._enhance_flow(compressed_sentences)
            
            # Join sentences into summary
            summary = ' '.join(enhanced_sentences)
            
            # Apply final enhancements
            summary = self.neural_summarizer._enhance_final_summary(summary)
            
            # Truncate if necessary
            words = summary.split()
            if len(words) > max_length:
                summary = ' '.join(words[:max_length])
                # Ensure the summary ends with a proper ending
                if self.lang_code == 'te':
                    summary += '.'
                elif self.lang_code == 'hi':
                    summary += '।'
                else:
                    summary += '.'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in enhanced summarization: {e}")
            # Fall back to neural summarizer
            return self.neural_summarizer.summarize(text, compression_ratio)
    
    def summarize(self, 
                 text: str, 
                 compression_ratio: float = 0.5,
                 max_length: int = 200) -> str:
        """
        Generate a summary of the input text using relative position embeddings
        
        Args:
            text: Text to summarize
            compression_ratio: Target ratio of summary size to original text
            max_length: Maximum summary length in words
        
        Returns:
            A coherent, well-formed summary
        """
        return self._enhance_summary_with_relative_attention(
            text, compression_ratio, max_length
        )

def generate_improved_summary(text: str, 
                             lang: str = 'te', 
                             model_dir: str = None, 
                             config_path: str = None,
                             compression_ratio: float = 0.5,
                             max_length: int = 200) -> str:
    """
    Generate an improved summary leveraging relative position embeddings
    
    This function is the main entry point for generating summaries with
    the improved summarizer.
    
    Args:
        text: Text to summarize
        lang: Language code ('te' for Telugu, 'hi' for Hindi)
        model_dir: Path to the model directory (optional, will use default if None)
        config_path: Path to the model config (optional, will use default if None)
        compression_ratio: Target ratio of summary to original text (0.0-1.0)
        max_length: Maximum summary length in words
    
    Returns:
        A coherent summary of the text
    """
    try:
        # Initialize the improved summarizer
        summarizer = ImprovedSummarizer(
            model_path=model_dir,
            config_path=config_path,
            lang_code=lang,
            attention_type='relative'  # Explicitly using relative position embeddings
        )
        
        # Generate the summary
        summary = summarizer.summarize(text, compression_ratio, max_length)
        return summary
        
    except Exception as e:
        logger.error(f"Error in improved summarization: {e}")
        logger.error("Falling back to basic summarization")
        
        # Simple fallback - return the first 1-2 sentences
        sentences = re.split(r'(?<=[.!?।॥])\s+', text)
        if not sentences:
            return "Summarization failed."
        
        # Return first sentence for very short texts, or first two for longer ones
        if len(sentences) <= 3:
            return sentences[0]
        else:
            return ' '.join(sentences[:2])


if __name__ == "__main__":
    # Example usage with Telugu text
    te_text = """ప్రస్తుతం భారత్‌-పాకిస్తాన్ మధ్య పరిస్థితి మరింత ఉద్రిక్తంగా మారుతున్నది. కశ్మీర్ సరిహద్దుల్లో తరచూ కాల్పులు జరగడం, ఉగ్రవాద చర్యలు కొనసాగడం ఈ ఉద్రిక్తతకు ప్రధాన కారణాలు. ఇటీవలి కాలంలో పాక్ మద్దతు ఉన్న ఉగ్రవాదుల చొరబాటు ప్రయత్నాలు పెరిగినట్లు భారత సైన్యం పేర్కొంది. అదే సమయంలో, రాజకీయ నేతల మధ్య మాటల యుద్ధం కూడా తీవ్రంగా సాగుతోంది. ఇరు దేశాల ప్రజలు శాంతిని కోరుతున్నా, సరిహద్దుల్లో పరిస్థితి ఇంకా గందరగోళంగా ఉంది. ఈ పరిస్థితిని చర్చల ద్వారా పరిష్కరించాలనే సూచనలు అంతర్జాతీయంగా వెల్లువెత్తుతున్నాయి."""
    
    # Generate and print the summary
    te_summary = generate_improved_summary(te_text, 'te')
    print("Telugu text:")
    print(te_text)
    print("\nGenerated summary with relative position embeddings:")
    print(te_summary)
    
    # Example Hindi text
    hi_text = """भारत और पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर लगातार गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना के अनुसार, हाल के दिनों में पाकिस्तान समर्थित आतंकवादियों की घुसपैठ की कोशिशों में वृद्धि हुई है। इसी समय, राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। दोनों देशों के लोग शांति चाहते हैं, लेकिन सीमा पर स्थिति अभी भी अस्थिर है। इस स्थिति को वार्ता के माध्यम से सुलझाने के सुझाव अंतरराष्ट्रीय स्तर पर दिए जा रहे हैं।"""
    
    # Generate and print the Hindi summary
    hi_summary = generate_improved_summary(hi_text, 'hi')
    print("\n\nHindi text:")
    print(hi_text)
    print("\nGenerated summary with relative position embeddings:")
    print(hi_summary)