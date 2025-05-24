import re
import logging
from typing import List, Dict, Tuple, Set, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TeluguMorphologyAnalyzer:
    """
    Morphological analyzer for Telugu language.
    
    Handles morpheme segmentation, stemming, and identification of grammatical features
    for Telugu text processing. Supports root form extraction, grammatical feature analysis,
    and morphological decomposition of Telugu words.
    """
    
    def __init__(self, resources_path: Optional[str] = None):
        """
        Initialize the Telugu morphology analyzer.
        
        Args:
            resources_path: Path to morphological resources (e.g., suffix lists, rules)
        """
        # Initialize the resource path
        if resources_path is None:
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            resources_path = str(base_dir / "data" / "resources" / "telugu")
        
        self.resources_path = resources_path
        
        # Load resources
        self._load_resources()
        
        logger.info("Telugu morphology analyzer initialized")
    
    def _load_resources(self):
        """Load morphological resources for Telugu"""
        # Gender markers
        self.gender_markers = {
            'డు': 'masculine',  # Masculine singular (e.g., వాడు - he)
            'రాలు': 'feminine',  # Feminine (e.g., కుమారురాలు - daughter)
            'ఆయన': 'masculine_hon',  # Masculine honorific
            'అమ్మ': 'feminine',  # Feminine (e.g., అమ్మ - mother)
            'రాలి': 'feminine',  # Another feminine marker
        }
        
        # Number markers
        self.number_markers = {
            'లు': 'plural',     # General plural marker (e.g., పుస్తకాలు - books)
            'ములు': 'plural',   # Another plural marker
            'ఇద్దరు': 'dual_human',  # Dual (two people)
            'ముగ్గురు': 'three_human',  # Three people
            'నలుగురు': 'four_human',  # Four people
        }
        
        # Case suffixes (vibhakti)
        self.case_suffixes = {
            'డు': 'masculine_nom',  # Masculine nominative
            'ము': 'neuter_nom',     # Neuter nominative
            'లు': 'plural_nom',     # Plural nominative
            'ని': 'accusative',     # Accusative
            'కి': 'dative',         # Dative
            'తో': 'instrumental',   # Instrumental
            'లో': 'locative',       # Locative
            'న': 'locative',        # Locative (e.g., ఇంటిన - at home)
            'నుండి': 'ablative',    # Ablative (from)
            'నుంచి': 'ablative',    # Alternative ablative
            'పై': 'locative_on',    # Locative (on)
            'కోసం': 'benefactive',  # Benefactive (for the sake of)
            'కొరకు': 'benefactive', # Alternative benefactive
            'వద్ద': 'locative_near', # Locative (near)
            'వరకు': 'terminative',  # Terminative (until)
            'వలన': 'causative',    # Causative (because of)
            'వలె': 'comparative',   # Comparative (like)
            'లాగా': 'comparative',   # Alternative comparative
            'యొక్క': 'genitive',     # Genitive (possessive)
            'దగ్గర': 'locative_near', # Near
            'లోపల': 'locative_inside', # Inside
            'బయట': 'locative_outside', # Outside
            'లోని': 'locative_within', # Within
        }
        
        # Tense/Aspect markers - expanded
        self.tense_markers = {
            # Present continuous forms
            'తున్నాను': 'present_continuous_1sg',  # I am doing
            'తున్నాము': 'present_continuous_1pl',  # We are doing
            'తున్నావు': 'present_continuous_2sg',  # You are doing
            'తున్నారు': 'present_continuous_2pl',  # You all are doing
            'తున్నాడు': 'present_continuous_3sg_m', # He is doing
            'తున్నది': 'present_continuous_3sg_f',  # She is doing
            'తున్నారు': 'present_continuous_3pl',   # They are doing
            
            # Past forms
            'ాను': 'past_1sg',  # I did
            'ాము': 'past_1pl',  # We did
            'ావు': 'past_2sg',  # You did
            'ారు': 'past_2pl',  # You all did
            'ాడు': 'past_3sg_m', # He did
            'ింది': 'past_3sg_f', # She did
            'ారు': 'past_3pl',   # They did
            
            # Future forms
            'తాను': 'future_1sg',  # I will do
            'తాము': 'future_1pl',  # We will do
            'తావు': 'future_2sg',  # You will do
            'తారు': 'future_2pl',  # You all will do
            'తాడు': 'future_3sg_m', # He will do
            'తుంది': 'future_3sg_f', # She will do
            'తారు': 'future_3pl',   # They will do
            
            # Habitual forms
            'తాను': 'habitual_1sg',  # I usually do
            'తాము': 'habitual_1pl',  # We usually do
            'తావు': 'habitual_2sg',  # You usually do
            'తారు': 'habitual_2pl',  # You all usually do
            'తాడు': 'habitual_3sg_m', # He usually does
            'తుంది': 'habitual_3sg_f', # She usually does
            'తారు': 'habitual_3pl',   # They usually do
            
            # Perfect forms
            'ాను': 'perfect_1sg',  # I have done
            'ాము': 'perfect_1pl',  # We have done
            'ావు': 'perfect_2sg',  # You have done
            'ారు': 'perfect_2pl',  # You all have done
            'ాడు': 'perfect_3sg_m', # He has done
            'ింది': 'perfect_3sg_f', # She has done
            'ారు': 'perfect_3pl',   # They have done
            
            # Infinitive & other forms
            'డం': 'infinitive',  # To do
            'ాలి': 'obligative',  # Should do/have to do
            'వచ్చు': 'potential',  # Can do/may do
            'లేదు': 'negative',  # Not do/didn't do
            'వద్దు': 'prohibitive',  # Don't do
            'గలను': 'potential_1sg',  # I can do
            'కూడదు': 'prohibitive_emph',  # Must not do
            'ండి': 'imperative_formal',  # Please do (polite command)
            'వయ్యండి': 'imperative_formal_emph',  # Please do (more emphatic)
        }
        
        # Common verb roots - expanded
        self.verb_roots = {
            'చేయ': 'చేయడం',  # to do
            'చేస': 'చేయడం',  # to do (variant stem)
            'పోవ': 'పోవడం',  # to go
            'పో': 'పోవడం',  # to go (variant)
            'చూడ': 'చూడటం',  # to see
            'చూస': 'చూడటం',  # to see (variant)
            'వచ్చ': 'వచ్చుట',  # to come
            'వస్త': 'వచ్చుట',  # to come (variant)
            'వెళ్ళ': 'వెళ్ళడం',  # to go
            'వెళ్ళిపో': 'వెళ్ళిపోవడం',  # to go away
            'ఇవ్వ': 'ఇవ్వడం',  # to give
            'ఇస్త': 'ఇవ్వడం',  # to give (variant)
            'తీసుకొన': 'తీసుకొనడం',  # to take
            'తీసుకుంట': 'తీసుకొనడం',  # to take (variant)
            'మాట్లాడ': 'మాట్లాడటం',  # to speak
            'మాట్లాడుత': 'మాట్లాడటం',  # to speak (variant)
            'అర్థం చేసుకొన': 'అర్థం చేసుకొనడం',  # to understand
            'అర్థం చేసుకుంట': 'అర్థం చేసుకొనడం',  # to understand (variant)
            'ఆలోచించ': 'ఆలోచించడం',  # to think
            'ఆలోచిస్త': 'ఆలోచించడం',  # to think (variant)
            'వినిపించ': 'వినిపించడం',  # to be heard
            'విన': 'వినడం',  # to hear
            'వింట': 'వినడం',  # to hear (variant)
            'తిన': 'తినడం',  # to eat
            'తింట': 'తినడం',  # to eat (variant)
            'త్రాగ': 'త్రాగడం',  # to drink
            'త్రాగుత': 'త్రాగడం',  # to drink (variant)
            'నిలబడ': 'నిలబడటం',  # to stand
            'నిలుచుంట': 'నిలబడటం',  # to stand (variant)
            'కూర్చొన': 'కూర్చొనడం',  # to sit
            'కూర్చుంట': 'కూర్చొనడం',  # to sit (variant)
            'రాయ': 'రాయడం',  # to write
            'రాస్త': 'రాయడం',  # to write (variant)
            'చదవ': 'చదవడం',  # to read
            'చదువుత': 'చదవడం',  # to read (variant)
            'రా': 'రావడం',  # to come
            'వస్త': 'రావడం',  # to come (variant)
            'ఉండ': 'ఉండటం',  # to be/stay
            'ఉంట': 'ఉండటం',  # to be/stay (variant)
            'తెలుసుకొన': 'తెలుసుకొనడం',  # to learn/know
            'తెలుసుకుంట': 'తెలుసుకొనడం',  # to learn/know (variant)
            'కావాల': 'కావాలనుకొనుట',  # to want/need
            'అడగ': 'అడగడం',  # to ask
            'అడుగుత': 'అడగడం',  # to ask (variant)
            'పడుకొన': 'పడుకొనడం',  # to sleep/lie down
            'పడుకుంట': 'పడుకొనడం',  # to sleep/lie down (variant)
            'ఇష్టపడ': 'ఇష్టపడటం',  # to like
            'ఇష్టపడుత': 'ఇష్టపడటం',  # to like (variant)
            'నమ్మ': 'నమ్మడం',  # to believe/trust
            'నమ్ముత': 'నమ్మడం',  # to believe/trust (variant)
        }
        
        # Pronouns - expanded with more detailed information
        self.pronouns = {
            # First person
            'నేను': {'person': '1', 'number': 'singular', 'case': 'direct'},
            'నాకు': {'person': '1', 'number': 'singular', 'case': 'dative'},
            'నన్ను': {'person': '1', 'number': 'singular', 'case': 'accusative'},
            'నా': {'person': '1', 'number': 'singular', 'case': 'genitive'},
            'నాతో': {'person': '1', 'number': 'singular', 'case': 'instrumental'},
            'నావలన': {'person': '1', 'number': 'singular', 'case': 'causative'},
            
            'మేము': {'person': '1', 'number': 'plural', 'case': 'direct', 'inclusivity': 'exclusive'},
            'మాకు': {'person': '1', 'number': 'plural', 'case': 'dative', 'inclusivity': 'exclusive'},
            'మమ్ములను': {'person': '1', 'number': 'plural', 'case': 'accusative', 'inclusivity': 'exclusive'},
            'మా': {'person': '1', 'number': 'plural', 'case': 'genitive', 'inclusivity': 'exclusive'},
            'మాతో': {'person': '1', 'number': 'plural', 'case': 'instrumental', 'inclusivity': 'exclusive'},
            
            'మనము': {'person': '1', 'number': 'plural', 'case': 'direct', 'inclusivity': 'inclusive'},
            'మనకు': {'person': '1', 'number': 'plural', 'case': 'dative', 'inclusivity': 'inclusive'},
            'మనలను': {'person': '1', 'number': 'plural', 'case': 'accusative', 'inclusivity': 'inclusive'},
            'మన': {'person': '1', 'number': 'plural', 'case': 'genitive', 'inclusivity': 'inclusive'},
            'మనతో': {'person': '1', 'number': 'plural', 'case': 'instrumental', 'inclusivity': 'inclusive'},
            
            # Second person
            'నువ్వు': {'person': '2', 'number': 'singular', 'case': 'direct', 'formality': 'informal'},
            'నీకు': {'person': '2', 'number': 'singular', 'case': 'dative', 'formality': 'informal'},
            'నిన్ను': {'person': '2', 'number': 'singular', 'case': 'accusative', 'formality': 'informal'},
            'నీ': {'person': '2', 'number': 'singular', 'case': 'genitive', 'formality': 'informal'},
            'నీతో': {'person': '2', 'number': 'singular', 'case': 'instrumental', 'formality': 'informal'},
            
            'మీరు': {'person': '2', 'number': 'singular', 'case': 'direct', 'formality': 'formal'},
            'మీకు': {'person': '2', 'number': 'singular', 'case': 'dative', 'formality': 'formal'},
            'మిమ్ములను': {'person': '2', 'number': 'singular', 'case': 'accusative', 'formality': 'formal'},
            'మీ': {'person': '2', 'number': 'singular', 'case': 'genitive', 'formality': 'formal'},
            'మీతో': {'person': '2', 'number': 'singular', 'case': 'instrumental', 'formality': 'formal'},
            
            # Third person - masculine
            'వాడు': {'person': '3', 'number': 'singular', 'case': 'direct', 'gender': 'masculine', 'distance': 'proximate'},
            'వాడికి': {'person': '3', 'number': 'singular', 'case': 'dative', 'gender': 'masculine', 'distance': 'proximate'},
            'వాడిని': {'person': '3', 'number': 'singular', 'case': 'accusative', 'gender': 'masculine', 'distance': 'proximate'},
            'వాడి': {'person': '3', 'number': 'singular', 'case': 'genitive', 'gender': 'masculine', 'distance': 'proximate'},
            
            'అతడు': {'person': '3', 'number': 'singular', 'case': 'direct', 'gender': 'masculine', 'distance': 'distal'},
            'అతనికి': {'person': '3', 'number': 'singular', 'case': 'dative', 'gender': 'masculine', 'distance': 'distal'},
            'అతనిని': {'person': '3', 'number': 'singular', 'case': 'accusative', 'gender': 'masculine', 'distance': 'distal'},
            'అతని': {'person': '3', 'number': 'singular', 'case': 'genitive', 'gender': 'masculine', 'distance': 'distal'},
            
            # Third person - feminine
            'ఆమె': {'person': '3', 'number': 'singular', 'case': 'direct', 'gender': 'feminine'},
            'ఆమెకు': {'person': '3', 'number': 'singular', 'case': 'dative', 'gender': 'feminine'},
            'ఆమెను': {'person': '3', 'number': 'singular', 'case': 'accusative', 'gender': 'feminine'},
            'ఆమె': {'person': '3', 'number': 'singular', 'case': 'genitive', 'gender': 'feminine'},
            
            # Third person - neuter
            'అది': {'person': '3', 'number': 'singular', 'case': 'direct', 'gender': 'neuter'},
            'దానికి': {'person': '3', 'number': 'singular', 'case': 'dative', 'gender': 'neuter'},
            'దానిని': {'person': '3', 'number': 'singular', 'case': 'accusative', 'gender': 'neuter'},
            'దాని': {'person': '3', 'number': 'singular', 'case': 'genitive', 'gender': 'neuter'},
            
            # Third person - plural human
            'వారు': {'person': '3', 'number': 'plural', 'case': 'direct', 'gender': 'human'},
            'వారికి': {'person': '3', 'number': 'plural', 'case': 'dative', 'gender': 'human'},
            'వారిని': {'person': '3', 'number': 'plural', 'case': 'accusative', 'gender': 'human'},
            'వారి': {'person': '3', 'number': 'plural', 'case': 'genitive', 'gender': 'human'},
            
            # Third person - plural non-human
            'అవి': {'person': '3', 'number': 'plural', 'case': 'direct', 'gender': 'non-human'},
            'వాటికి': {'person': '3', 'number': 'plural', 'case': 'dative', 'gender': 'non-human'},
            'వాటిని': {'person': '3', 'number': 'plural', 'case': 'accusative', 'gender': 'non-human'},
            'వాటి': {'person': '3', 'number': 'plural', 'case': 'genitive', 'gender': 'non-human'},
        }
        
        # Common prefixes in Telugu
        self.prefixes = {
            'ని': 'negative',  # Negative prefix (e.g., నిరుపయోగం - useless)
            'దుర్': 'negative',  # Negative/bad prefix (e.g., దుర్వాసన - bad smell)
            'సు': 'good',  # Good prefix (e.g., సుగంధం - good smell)
            'పరి': 'around',  # Around/complete (e.g., పరిపాలన - administration)
            'ప్ర': 'forward',  # Forward/intense (e.g., ప్రశాంతత - tranquility)
            'ఉప': 'sub',  # Sub/near (e.g., ఉపనది - tributary)
            'అతి': 'excessive',  # Over/excessive (e.g., అతివృష్టి - excessive rain)
            'అధి': 'superior',  # Over/above (e.g., అధికారం - authority)
            'పై': 'upper',  # Upper/above (e.g., పైఅంతస్తు - upper floor)
            'కింద': 'lower',  # Below/under (e.g., కిందిఅంతస్తు - lower floor)
            'అన': 'negative',  # Negative (e.g., అనుచితం - improper)
        }
        
        # Derivational suffixes
        self.derivation_suffixes = {
            'కారుడు': 'doer_masc',  # Doer masculine (e.g., రచయితకారుడు - writer)
            'కారురాలు': 'doer_fem',  # Doer feminine (e.g., రచయితకారురాలు - female writer)
            'దారుడు': 'holder_masc',  # Holder masculine (e.g., అధికారదారుడు - official)
            'దారురాలు': 'holder_fem',  # Holder feminine (e.g., అధికారదారురాలు - female official)
            'తనము': 'abstract_quality',  # Abstract quality (e.g., మంచితనము - goodness)
            'త్వము': 'abstract_quality',  # Abstract quality (e.g., ఏకత్వము - oneness)
            'దలు': 'collective',  # Collective (e.g., ఆవుదలు - herd of cows)
            'బడి': 'place',  # Place (e.g., చదువుబడి - school)
            'గా': 'adverbial',  # Adverbial (e.g., బాగా - well)
            'గాడు': 'occupational_masc',  # Occupational masculine (e.g., పనిగాడు - worker)
            'గత్తె': 'occupational_fem',  # Occupational feminine (e.g., పనిగత్తె - female worker)
            'వంతుడు': 'possessor_masc',  # Possessor masculine (e.g., గుణవంతుడు - virtuous man)
            'వంతురాలు': 'possessor_fem',  # Possessor feminine (e.g., గుణవంతురాలు - virtuous woman)
            'కర్త': 'doer',  # Doer (e.g., నియంతకర్త - controller)
            'జనకుడు': 'creator_masc',  # Creator masculine (e.g., సృష్టిజనకుడు - creator)
            'జనకురాలు': 'creator_fem',  # Creator feminine (e.g., సృష్టిజనకురాలు - female creator)
        }
        
        # Sandhi rules (simplified for implementation)
        self.sandhi_rules = [
            # Vowel sandhi
            ('ా', 'అ', 'ా'),
            ('ి', 'అ', 'ి'),
            ('ు', 'అ', 'ు'),
            ('ె', 'అ', 'ె'),
            ('ే', 'అ', 'ే'),
            ('ొ', 'అ', 'ొ'),
            ('ో', 'అ', 'ో'),
            # Consonant sandhi (simplified)
            ('్', 'అ', ''),
            ('ి', 'ఇ', 'ీ'),
            ('ు', 'ఉ', 'ూ'),
            ('ె', 'ఎ', 'ే'),
            ('ొ', 'ఒ', 'ో'),
        ]
        
        # Honorific forms
        self.honorific_forms = {
            'గారు': 'honorific',  # Respectful suffix (e.g., రాజుగారు - respected Raja)
            'మహాశయ': 'formal_honorific',  # Formal honorific (e.g., రాజుమహాశయ - honorable Raja)
            'అయ్య': 'respectful_masc',  # Respectful for male (e.g., రామయ్య - respected Ram)
            'అమ్మ': 'respectful_fem',  # Respectful for female (e.g., లక్ష్మిఅమ్మ - respected Lakshmi)
            'దేవి': 'goddess_honorific',  # Goddess/Female honorific (e.g., సీతాదేవి - goddess Sita)
            'శ్రీ': 'auspicious_masc',  # Auspicious prefix for male (e.g., శ్రీరాము - auspicious Ram)
            'శ్రీమతి': 'auspicious_fem',  # Auspicious prefix for female (e.g., శ్రీమతిసీత - auspicious Sita)
        }
        
    def segment_word(self, word: str) -> List[str]:
        """
        Segment a Telugu word into morphemes (roots, suffixes).
        
        Args:
            word: Telugu word to segment
            
        Returns:
            List of morphemes
        """
        if not word:
            return []
        
        morphemes = []
        remaining = word
        
        # Check if it's a pronoun (which don't follow regular morphology)
        if word in self.pronouns:
            return [word]
        
        # Check for prefixes
        for prefix, prefix_type in sorted(self.prefixes.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.startswith(prefix) and len(remaining) > len(prefix) + 1:
                morphemes.append(prefix)
                remaining = remaining[len(prefix):]
                break
                
        # Check for case suffixes
        for suffix, case_type in sorted(self.case_suffixes.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                root_part = remaining[:-len(suffix)]
                morphemes.append(root_part)
                morphemes.append(suffix)
                return morphemes
        
        # Check for tense markers
        for marker, tense in sorted(self.tense_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(marker) and len(remaining) > len(marker):
                root_part = remaining[:-len(marker)]
                
                # Check if it's a recognizable verb root
                for verb_root in sorted(self.verb_roots.keys(), key=len, reverse=True):
                    if root_part == verb_root or root_part.startswith(verb_root):
                        if root_part != verb_root:
                            # There might be a prefix or additional part
                            prefix_part = root_part[:len(root_part)-len(verb_root)]
                            if prefix_part:
                                morphemes.append(prefix_part)
                            morphemes.append(verb_root)
                        else:
                            morphemes.append(root_part)
                        
                        morphemes.append(marker)
                        return morphemes
        
        # Check for noun/adjective inflections - gender markers
        for marker, gender_type in sorted(self.gender_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(marker) and len(remaining) > len(marker):
                stem_part = remaining[:-len(marker)]
                morphemes.append(stem_part)
                morphemes.append(marker)
                return morphemes
                
        # Check for number markers
        for marker, number_type in sorted(self.number_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(marker) and len(remaining) > len(marker):
                stem_part = remaining[:-len(marker)]
                morphemes.append(stem_part)
                morphemes.append(marker)
                return morphemes
        
        # Check for derivational suffixes
        for suffix, suffix_type in sorted(self.derivation_suffixes.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix):
                stem_part = remaining[:-len(suffix)]
                morphemes.append(stem_part)
                morphemes.append(suffix)
                return morphemes
        
        # Check for honorific forms
        for form, honorific_type in sorted(self.honorific_forms.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(form) and len(remaining) > len(form):
                stem_part = remaining[:-len(form)]
                morphemes.append(stem_part)
                morphemes.append(form)
                return morphemes
        
        # If no segmentation found, treat the whole word as one morpheme
        morphemes.append(remaining)
        return morphemes
    
    def get_stem(self, word: str) -> str:
        """
        Get the stem of a Telugu word by removing inflectional suffixes.
        
        Args:
            word: Word to get stem for
            
        Returns:
            Stem of the word
        """
        # For pronouns, return the direct case form
        if word in self.pronouns:
            # Find the corresponding base/direct form if it's an oblique form
            pronoun_info = self.pronouns[word]
            person = pronoun_info.get('person')
            number = pronoun_info.get('number')
            gender = pronoun_info.get('gender', '')
            
            # Look for direct case form
            for p, props in self.pronouns.items():
                if (props.get('person') == person and 
                    props.get('number') == number and
                    props.get('gender', '') == gender and
                    props.get('case') == 'direct'):
                    return p
        
        # General segmentation approach
        segments = self.segment_word(word)
        
        # If multiple segments, typically the first is the stem
        if len(segments) > 1:
            # Check if we segmented a verb with a tense marker
            for segment in segments:
                if segment in self.verb_roots:
                    return segment
            
            # For other words, first segment is typically the stem
            return segments[0]
        
        # If no segmentation was possible or only one segment
        return word
    
    def apply_sandhi_rules(self, word1: str, word2: str) -> str:
        """
        Apply Telugu sandhi rules to combine two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Combined word with sandhi rules applied
        """
        if not word1 or not word2:
            return word1 + word2
        
        # Get the last character of first word and first character of second word
        last_char = word1[-1]
        first_char = word2[0]
        
        # Apply sandhi rules if a matching rule exists
        for c1, c2, result in self.sandhi_rules:
            if last_char == c1 and first_char == c2:
                return word1[:-1] + result + word2[1:]
        
        # No rule matched, simply concatenate
        return word1 + word2
    
    def analyze_word(self, word: str) -> Dict[str, any]:
        """
        Perform morphological analysis of a Telugu word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary with morphological features
        """
        # Initialize with basic information
        result = {
            'word': word,
            'stem': self.get_stem(word),
            'segments': self.segment_word(word),
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
        
        # Check for case markers
        for suffix, case_type in self.case_suffixes.items():
            if word.endswith(suffix) and word != suffix:
                result['case'] = case_type
                
                # Extract gender information from case marker
                if '_m' in case_type or 'masculine' in case_type:
                    result['gender'] = 'masculine'
                elif '_f' in case_type or 'feminine' in case_type:
                    result['gender'] = 'feminine'
                elif '_n' in case_type or 'neuter' in case_type:
                    result['gender'] = 'neuter'
                
                # Extract number information
                if 'plural' in case_type:
                    result['number'] = 'plural'
                else:
                    result['number'] = 'singular'
                
                break
        
        # Check for gender markers
        if 'gender' not in result:
            for marker, gender_type in self.gender_markers.items():
                if word.endswith(marker):
                    result['gender'] = gender_type
                    break
        
        # Check for number markers
        if 'number' not in result:
            for marker, number_type in self.number_markers.items():
                if word.endswith(marker):
                    result['number'] = number_type
                    break
        
        # Check for tense/aspect
        for marker, tense in sorted(self.tense_markers.items(), key=lambda x: len(x[0]), reverse=True):
            if word.endswith(marker):
                result['tense'] = tense
                
                # Extract person, gender, and number information from tense marker
                if '_1' in tense:
                    result['person'] = '1'
                elif '_2' in tense:
                    result['person'] = '2'
                elif '_3' in tense:
                    result['person'] = '3'
                
                if '_sg' in tense or tense.endswith('_m') or tense.endswith('_f'):
                    result['number'] = 'singular'
                elif '_pl' in tense:
                    result['number'] = 'plural'
                
                if '_m' in tense:
                    result['gender'] = 'masculine'
                elif '_f' in tense:
                    result['gender'] = 'feminine'
                
                # Check if it's a verb
                stem = result.get('stem', '')
                for verb_root in self.verb_roots:
                    if stem == verb_root or stem.startswith(verb_root):
                        result['pos'] = 'verb'
                        result['verb_root'] = verb_root
                        result['verb_form'] = self.verb_roots.get(verb_root, '')
                        break
                
                break
        
        # Check for derivational suffixes to determine part of speech
        if 'pos' not in result:
            for suffix, suffix_type in self.derivation_suffixes.items():
                if word.endswith(suffix):
                    result['derivation_suffix'] = suffix
                    result['derivation_type'] = suffix_type
                    
                    # Determine part of speech from derivation type
                    if 'doer' in suffix_type or 'holder' in suffix_type or 'creator' in suffix_type:
                        result['pos'] = 'noun'
                    elif 'quality' in suffix_type:
                        result['pos'] = 'abstract_noun'
                    elif 'place' in suffix_type or 'collective' in suffix_type:
                        result['pos'] = 'noun'
                    elif 'adverbial' in suffix_type:
                        result['pos'] = 'adverb'
                    
                    break
        
        # Check for honorific forms
        for form, honorific_type in self.honorific_forms.items():
            if word.endswith(form):
                result['honorific'] = form
                result['honorific_type'] = honorific_type
                
                # Extract gender if present in honorific type
                if '_masc' in honorific_type or 'masc' in honorific_type:
                    result['gender'] = 'masculine'
                elif '_fem' in honorific_type or 'fem' in honorific_type:
                    result['gender'] = 'feminine'
                
                break
        
        # If still no POS determined, make an educated guess
        if 'pos' not in result:
            # If it has gender or number, it's likely a noun or adjective
            if 'gender' in result or 'number' in result:
                # Try to distinguish between nouns and adjectives
                if word.endswith('మైన') or word.endswith('గల'):
                    result['pos'] = 'adjective'
                else:
                    # Default to noun if we can't determine clearly
                    result['pos'] = 'noun'
            elif word.endswith('గా'):
                result['pos'] = 'adverb'
        
        return result
    
    def analyze_text(self, text: str) -> List[Dict[str, any]]:
        """
        Analyze all words in a Telugu text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of analysis dictionaries, one per word
        """
        # Simple word segmentation using Telugu Unicode range
        words = re.findall(r'[\u0C00-\u0C7F]+', text)
        return [self.analyze_word(word) for word in words]
    
    def get_root_forms(self, text: str) -> List[str]:
        """
        Extract root forms from Telugu text.
        
        Args:
            text: Telugu text to analyze
            
        Returns:
            List of root forms
        """
        analyses = self.analyze_text(text)
        return [analysis['stem'] for analysis in analyses]
    
    def get_grammatical_features(self, text: str) -> Dict[str, List[str]]:
        """
        Extract grammatical features from Telugu text.
        
        Args:
            text: Telugu text to analyze
            
        Returns:
            Dictionary with grammatical features
        """
        analyses = self.analyze_text(text)
        
        # Initialize feature collections
        features = {
            'pos': [],
            'gender': [],
            'number': [],
            'person': [],
            'case': [],
            'tense': [],
            'honorific': []
        }
        
        # Collect features from analyses
        for analysis in analyses:
            for feature_name in features.keys():
                if feature_name in analysis and analysis[feature_name]:
                    features[feature_name].append(analysis[feature_name])
        
        return features
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize Telugu text by replacing each word with its stem form.
        
        Args:
            text: Telugu text to lemmatize
            
        Returns:
            Lemmatized text
        """
        words = re.findall(r'([\u0C00-\u0C7F]+)|(\s+)', text)
        result = ""
        
        for telugu_word, space in words:
            if telugu_word:
                # Get stem and replace word
                stem = self.get_stem(telugu_word)
                result += stem
            if space:
                result += space
                
        return result