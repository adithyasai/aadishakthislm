import re
import logging
from typing import List, Dict, Tuple, Set, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class HindiMorphologyAnalyzer:
    """
    Morphological analyzer for Hindi language.
    
    Handles morpheme segmentation, stemming, and identification of grammatical features
    for Hindi text processing.
    """
    
    def __init__(self, resources_path: Optional[str] = None):
        """
        Initialize the Hindi morphology analyzer.
        
        Args:
            resources_path: Path to morphological resources (e.g., suffix lists, rules)
        """
        # Initialize the resource path
        if resources_path is None:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            resources_path = str(base_dir / "data" / "resources" / "hindi")
        
        self.resources_path = resources_path
        
        # Load resources
        self._load_resources()
        
        logger.info("Hindi morphology analyzer initialized")
    
    def _load_resources(self):
        """Load morphological resources for Hindi"""
        # Gender markers
        self.gender_markers = {
            'ा': 'masculine',  # Masculine marker (e.g. लड़का)
            'ी': 'feminine',   # Feminine marker (e.g. लड़की)
        }
        
        # Number markers
        self.number_markers = {
            'ा': 'singular',   # Singular (e.g. लड़का)
            'े': 'plural',     # Plural for masculine (e.g. लड़के)
            'ियां': 'plural',   # Plural for feminine (e.g. लड़कियां)
            'ियाँ': 'plural',   # Alternate plural for feminine
            'यां': 'plural',    # Another plural form
            'याँ': 'plural',    # Another plural form
            'एं': 'plural',     # Another plural form
            'एँ': 'plural',     # Another plural form
        }
        
        # Case markers (postpositions)
        self.case_markers = {
            'ने': 'ergative',    # Ergative case (subject of transitive verb)
            'को': 'accusative',  # Accusative/dative case
            'से': 'instrumental', # Instrumental/ablative case
            'का': 'genitive_m_sg', # Genitive masculine singular
            'के': 'genitive_m_pl', # Genitive masculine plural
            'की': 'genitive_f',    # Genitive feminine
            'में': 'locative',     # Locative (in)
            'पर': 'locative',     # Locative (on)
            'तक': 'terminative',  # Terminative (until)
        }
        
        # Tense/aspect markers for verbs
        self.tense_markers = {
            'ता': 'habitual_m_sg',  # Habitual aspect, masculine singular
            'ते': 'habitual_m_pl',  # Habitual aspect, masculine plural
            'ती': 'habitual_f',     # Habitual aspect, feminine
            'रहा': 'progressive_m_sg', # Progressive aspect, masculine singular
            'रहे': 'progressive_m_pl', # Progressive aspect, masculine plural
            'रही': 'progressive_f',    # Progressive aspect, feminine
            'चुका': 'completive_m_sg', # Completive aspect, masculine singular
            'चुके': 'completive_m_pl', # Completive aspect, masculine plural
            'चुकी': 'completive_f',    # Completive aspect, feminine
            'गा': 'future_m_sg',    # Future tense, masculine singular
            'गे': 'future_m_pl',    # Future tense, masculine plural
            'गी': 'future_f',       # Future tense, feminine
            'या': 'perfective_m_sg', # Perfective aspect, masculine singular
            'ये': 'perfective_m_pl', # Perfective aspect, masculine plural
            'यी': 'perfective_f',    # Perfective aspect, feminine
            'ना': 'infinitive',      # Infinitive form
        }
        
        # List of common verb roots
        self.verb_roots = {
            'कर': 'करना',    # to do
            'कह': 'कहना',    # to say
            'जा': 'जाना',     # to go
            'आ': 'आना',      # to come
            'दे': 'देना',     # to give
            'ले': 'लेना',     # to take
            'पा': 'पाना',     # to get/find
            'रख': 'रखना',    # to keep
            'सक': 'सकना',    # can/to be able to
            'हो': 'होना',     # to be/happen
            'लिख': 'लिखना',  # to write
            'पढ़': 'पढ़ना',    # to read
            'देख': 'देखना',   # to see
            'सुन': 'सुनना',   # to hear
            'बोल': 'बोलना',   # to speak
            'खा': 'खाना',     # to eat
            'पी': 'पीना',      # to drink
            'सोच': 'सोचना',   # to think
            'समझ': 'समझना',  # to understand
            'चल': 'चलना',     # to walk
        }
        
        # Common prefixes
        self.prefixes = {
            'अ': 'negative',      # negation (e.g. अशुद्ध - impure)
            'अन': 'negative',     # negation (e.g. अनजान - unknown)
            'अप': 'negative',     # negative/bad (e.g. अपमान - insult)
            'अधि': 'superior',    # superior/over (e.g. अधिकार - authority)
            'अनु': 'following',   # following/after (e.g. अनुवाद - translation)
            'अभि': 'towards',     # towards/intensive (e.g. अभिमान - pride)
            'प्र': 'forward',      # forward/forth (e.g. प्रगति - progress)
            'परा': 'away',        # away/beyond (e.g. परावर्तन - reflection)
            'परि': 'around',      # around/complete (e.g. परिवार - family)
            'प्रति': 'counter',    # counter/against (e.g. प्रतिक्रिया - reaction)
            'उप': 'sub',          # sub/near (e.g. उपवन - garden)
            'सम': 'together',     # together/complete (e.g. समझ - understanding)
            'सु': 'good',         # good/well (e.g. सुंदर - beautiful)
            'दुर': 'bad',         # bad/difficult (e.g. दुर्घटना - accident)
            'दुस': 'bad',         # bad/difficult (e.g. दुस्वप्न - nightmare)
            'नि': 'down',         # down/without (e.g. निवास - residence)
            'विन': 'without',     # without (e.g. विनम्र - humble)
            'वि': 'apart',        # apart/special (e.g. विशेष - special)
        }
        
        # Common derivational suffixes
        self.derivation_suffixes = {
            'वाला': 'doer',       # doer/possessor (e.g. दूधवाला - milkman)
            'दार': 'holder',      # holder/possessor (e.g. जमींदार - landlord)
            'गर': 'doer',         # doer (e.g. जादूगर - magician)
            'कार': 'maker',       # maker/doer (e.g. कलाकार - artist)
            'आई': 'abstract_noun', # abstract noun (e.g. लंबाई - length)
            'पन': 'abstract_noun', # abstract quality (e.g. बचपन - childhood)
            'हट': 'abstract_noun', # abstract quality (e.g. रिझहट - annoyance)
            'ता': 'abstract_noun', # abstract quality (e.g. सुंदरता - beauty)
            'त्व': 'abstract_noun', # abstract quality (e.g. महत्व - importance)
            'इत': 'past_participle', # past participle (e.g. लिखित - written)
            'आऊ': 'tendency',      # having tendency (e.g. बिकाऊ - saleable)
            'इया': 'diminutive',   # diminutive (e.g. डिबिया - small box)
        }
        
        # Honorific suffixes
        self.honorific_suffixes = {
            'जी': 'honorific',    # Respectful suffix (e.g. रामजी - respected Ram)
            'साहब': 'honorific',  # Respectful title (e.g. मोहन साहब)
            'महोदय': 'honorific', # Sir/Mr. (formal)
            'महोदया': 'honorific', # Madam/Mrs. (formal)
            'श्री': 'honorific',   # Mr./Respected (formal prefix)
            'श्रीमती': 'honorific', # Mrs. (formal prefix)
            'श्रीमान': 'honorific', # Mr. (formal prefix)
        }
        
        # Oblique stems - irregular forms
        self.oblique_stems = {
            'मैं': 'मुझ',     # I -> me
            'तू': 'तुझ',      # you (intimate) -> you (oblique)
            'हम': 'हम',       # we -> us
            'तुम': 'तुम',     # you -> you (oblique)
            'आप': 'आप',      # you (formal) -> you (formal oblique)
            'यह': 'इस',       # this -> this (oblique)
            'वह': 'उस',       # that/he/she -> that/him/her (oblique)
            'ये': 'इन',       # these -> these (oblique)
            'वे': 'उन',       # those/they -> those/them (oblique)
            'कौन': 'किस',     # who -> whom (oblique)
        }
        
        # Pronouns
        self.pronouns = {
            'मैं': {'person': '1', 'number': 'singular', 'case': 'direct'},
            'मुझे': {'person': '1', 'number': 'singular', 'case': 'accusative'},
            'मुझको': {'person': '1', 'number': 'singular', 'case': 'accusative'},
            'मेरा': {'person': '1', 'number': 'singular', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'singular'},
            'मेरे': {'person': '1', 'number': 'singular', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'plural'},
            'मेरी': {'person': '1', 'number': 'singular', 'case': 'genitive', 'gender': 'feminine'},
            'हम': {'person': '1', 'number': 'plural', 'case': 'direct'},
            'हमें': {'person': '1', 'number': 'plural', 'case': 'accusative'},
            'हमको': {'person': '1', 'number': 'plural', 'case': 'accusative'},
            'हमारा': {'person': '1', 'number': 'plural', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'singular'},
            'हमारे': {'person': '1', 'number': 'plural', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'plural'},
            'हमारी': {'person': '1', 'number': 'plural', 'case': 'genitive', 'gender': 'feminine'},
            'तू': {'person': '2', 'number': 'singular', 'case': 'direct', 'formality': 'intimate'},
            'तुझे': {'person': '2', 'number': 'singular', 'case': 'accusative', 'formality': 'intimate'},
            'तुम': {'person': '2', 'number': 'singular', 'case': 'direct', 'formality': 'familiar'},
            'तुम्हें': {'person': '2', 'number': 'singular', 'case': 'accusative', 'formality': 'familiar'},
            'आप': {'person': '2', 'number': 'singular', 'case': 'direct', 'formality': 'formal'},
            'आपको': {'person': '2', 'number': 'singular', 'case': 'accusative', 'formality': 'formal'},
            'वह': {'person': '3', 'number': 'singular', 'case': 'direct'},
            'उसे': {'person': '3', 'number': 'singular', 'case': 'accusative'},
            'उसका': {'person': '3', 'number': 'singular', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'singular'},
            'उसके': {'person': '3', 'number': 'singular', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'plural'},
            'उसकी': {'person': '3', 'number': 'singular', 'case': 'genitive', 'gender': 'feminine'},
            'वे': {'person': '3', 'number': 'plural', 'case': 'direct'},
            'उन्हें': {'person': '3', 'number': 'plural', 'case': 'accusative'},
            'उनका': {'person': '3', 'number': 'plural', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'singular'},
            'उनके': {'person': '3', 'number': 'plural', 'case': 'genitive', 'gender': 'masculine', 'possessed_number': 'plural'},
            'उनकी': {'person': '3', 'number': 'plural', 'case': 'genitive', 'gender': 'feminine'},
        }
    
    def segment_word(self, word: str) -> List[str]:
        """
        Segment a Hindi word into morphemes (prefixes, stem, suffixes).
        
        Args:
            word: Hindi word to segment
            
        Returns:
            List of morphemes
        """
        if not word:
            return []
        
        morphemes = []
        remaining = word
        
        # Check for prefixes
        for prefix, prefix_type in sorted(self.prefixes.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.startswith(prefix) and len(remaining) > len(prefix) + 1:
                morphemes.append(prefix)
                remaining = remaining[len(prefix):]
                break
        
        # Check for pronouns (which don't follow regular morphology)
        if word in self.pronouns:
            # If it's a pronoun, just return the word as a single morpheme
            return [word]
        
        # Check if it's a declined postposition
        for case_marker, case_type in sorted(self.case_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if word == case_marker:
                return [case_marker]
        
        # Check for verb inflections
        for tense_marker, tense_type in sorted(self.tense_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(tense_marker) and len(remaining) > len(tense_marker):
                # Possible verb root
                root_part = remaining[:-len(tense_marker)]
                
                # Check if it's a known verb root
                for verb_root in self.verb_roots:
                    if root_part == verb_root or root_part.endswith(verb_root):
                        if root_part != verb_root:
                            # There might be a prefix
                            prefix_part = root_part[:-len(verb_root)]
                            if prefix_part:
                                morphemes.append(prefix_part)
                            morphemes.append(verb_root)
                        else:
                            morphemes.append(root_part)
                        
                        morphemes.append(tense_marker)
                        return morphemes
        
        # Check for noun/adjective inflections
        # First look for case markers
        for case_marker, case_type in sorted(self.case_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(" " + case_marker):
                # Separate postposition
                stem_part = remaining[:-len(case_marker)-1]  # Remove the space too
                morphemes.append(stem_part)
                morphemes.append(case_marker)
                return morphemes
        
        # Check for gender and number markers
        for gender_marker, gender_type in sorted(self.gender_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(gender_marker) and len(remaining) > len(gender_marker):
                stem_part = remaining[:-len(gender_marker)]
                morphemes.append(stem_part)
                morphemes.append(gender_marker)
                return morphemes
        
        # Check for derivational suffixes
        for suffix, suffix_type in sorted(self.derivation_suffixes.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                stem_part = remaining[:-len(suffix)]
                morphemes.append(stem_part)
                morphemes.append(suffix)
                return morphemes
        
        # If no segmentation found, treat the whole word as one morpheme
        morphemes.append(remaining)
        return morphemes
    
    def get_stem(self, word: str) -> str:
        """
        Get the stem of a Hindi word by removing inflectional affixes.
        
        Args:
            word: Word to get stem for
            
        Returns:
            Stem of the word
        """
        # For pronouns, return the direct case form
        for pronoun, properties in self.pronouns.items():
            if word == pronoun:
                # Find the direct case form for this pronoun
                for p, props in self.pronouns.items():
                    if (props.get('person') == properties.get('person') and 
                        props.get('number') == properties.get('number') and 
                        props.get('case') == 'direct'):
                        return p
        
        # For verbs, get the root form
        segments = self.segment_word(word)
        if len(segments) > 1:
            # Check if we segmented a verb
            for segment in segments:
                if segment in self.verb_roots:
                    return segment
            
            # For nouns and adjectives, typically the first segment is the stem
            return segments[0]
        
        # If no segmentation was possible or only one segment found
        return word
    
    def analyze_word(self, word: str) -> Dict[str, any]:
        """
        Perform morphological analysis of a Hindi word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary with morphological features
        """
        # Initialize with basic information
        result = {
            'word': word,
            'stem': self.get_stem(word),
            'segments': self.segment_word(word)
        }
        
        # Check if it's a pronoun
        if word in self.pronouns:
            result.update(self.pronouns[word])
            result['pos'] = 'pronoun'
            return result
        
        # Check for prefixes
        for prefix, prefix_type in self.prefixes.items():
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                result['prefix'] = prefix
                result['prefix_type'] = prefix_type
                break
        
        # Check for postpositions
        for case_marker, case_type in self.case_markers.items():
            if word == case_marker:
                result['pos'] = 'postposition'
                result['case_type'] = case_type
                return result
            if word.endswith(" " + case_marker):
                result['case'] = case_type
                break
        
        # Check for gender and number markers
        for gender_marker, gender_type in self.gender_markers.items():
            if word.endswith(gender_marker):
                result['gender'] = gender_type
                break
        
        for number_marker, number_type in self.number_markers.items():
            if word.endswith(number_marker):
                result['number'] = number_type
                break
        
        # Check for verb markers
        for tense_marker, tense_type in self.tense_markers.items():
            if word.endswith(tense_marker):
                result['tense'] = tense_type
                
                # Extract gender, number information from the tense marker
                if '_m_' in tense_type:
                    result['gender'] = 'masculine'
                    if tense_type.endswith('_sg'):
                        result['number'] = 'singular'
                    elif tense_type.endswith('_pl'):
                        result['number'] = 'plural'
                elif '_f' in tense_type:
                    result['gender'] = 'feminine'
                
                # Identify if it's a verb
                stem = result.get('stem', '')
                if stem in self.verb_roots:
                    result['pos'] = 'verb'
                    result['verb_root'] = stem
                    result['verb_form'] = self.verb_roots[stem]
                
                break
        
        # Check for derivation suffixes to determine part of speech
        if 'pos' not in result:
            for suffix, suffix_type in self.derivation_suffixes.items():
                if word.endswith(suffix):
                    result['derivation_suffix'] = suffix
                    result['derivation_type'] = suffix_type
                    
                    # Determine part of speech from derivation type
                    if suffix_type in ['abstract_noun', 'doer', 'holder', 'maker']:
                        result['pos'] = 'noun'
                    
                    break
        
        # If still no POS determined, make an educated guess
        if 'pos' not in result:
            # If it has gender or number, it's likely a noun or adjective
            if 'gender' in result or 'number' in result:
                # Look at other features to determine if it's a noun or adjective
                if word.endswith('ता') or word.endswith('त्व'):
                    result['pos'] = 'noun'
                else:
                    # Default to noun if we can't determine clearly
                    result['pos'] = 'noun'
        
        return result
    
    def analyze_text(self, text: str) -> List[Dict[str, any]]:
        """
        Analyze all words in a Hindi text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of analysis dictionaries, one per word
        """
        # Simple word segmentation (could be improved with better tokenization)
        words = re.findall(r'[\u0900-\u097F]+', text)
        return [self.analyze_word(word) for word in words]
    
    def get_root_forms(self, text: str) -> List[str]:
        """
        Extract root forms from Hindi text.
        
        Args:
            text: Hindi text to analyze
            
        Returns:
            List of root forms
        """
        analyses = self.analyze_text(text)
        return [analysis['stem'] for analysis in analyses]