#!/usr/bin/env python3
"""
Hybrid Transliteration + Translation Pipeline using Helsinki MarianMT
Builds a system that intelligently combines translation and transliteration for Hindi.
"""

import re
import sys
from typing import List, Dict, Tuple, Optional
from transformers import pipeline

class HybridTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-hi"):
        """Initialize the translator with MarianMT model."""
        print(f"Loading model: {model_name}")
        self.translator = pipeline("translation", model=model_name)
        print("Model loaded successfully!")

        # Known proper nouns and acronyms that should be preserved/transliterated
        self.known_entities = {
            'anits': 'एनआईटीएस',
            'anil neerukonda institute of technology and sciences': 'अनिल नीरुकोंडा इंस्टीट्यूट ऑफ टेक्नोलॉजी एंड साइंसेस',
            'iit': 'आईआईटी',
            'mit': 'एमआईटी',
            'nit': 'एनआईटी',
            'iisc': 'आईआईएससी',
            'google': 'गूगल',
            'microsoft': 'माइक्रोसॉफ्ट',
            'apple': 'ऐपल',
            'samsung': 'सैमसंग',
            'delhi': 'दिल्ली',
            'mumbai': 'मुंबई',
            'bharat': 'भारत',
            'mata': 'माता',
            'jai': 'जय',
            'raju': 'राजू',
            'john': 'जॉन'
        }

    def preprocess_text(self, text: str) -> Tuple[str, List[str], List[str]]:
        """
        Preprocess input text:
        - Detect and preserve acronyms
        - Detect proper nouns
        - Normalize phonetic vowels
        """
        # Find acronyms (all caps words)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)

        # Find proper nouns (capitalized words)
        proper_nouns = []
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                proper_nouns.append(word.lower())

        # Create a copy for processing
        processed_text = text.lower()

        # Normalize phonetic vowels
        processed_text = re.sub(r'aa+', 'a', processed_text)
        processed_text = re.sub(r'ee+', 'i', processed_text)
        processed_text = re.sub(r'oo+', 'u', processed_text)

        return processed_text, acronyms, proper_nouns

    def detect_sentence_type(self, text: str) -> str:
        """
        Determine if sentence contains English grammar or is phonetic.
        Returns: 'translate' or 'transliterate'
        """
        # Check for English grammar indicators
        english_indicators = [
            r'\b(is|am|are|was|were|be|being|been)\b',
            r'\b(the|a|an)\b',
            r'\b(and|or|but|so|because|although)\b',
            r'\b(i|you|he|she|it|we|they)\b',
            r'\b(in|on|at|to|from|with|by)\b'
        ]

        english_score = sum(1 for pattern in english_indicators if re.search(pattern, text.lower()))

        # If significant English grammar detected, allow translation
        if english_score >= 2:
            return 'translate'
        else:
            return 'transliterate'

    def transliterate_phonetic(self, text: str) -> str:
        """
        Simple phonetic transliteration for Hindi Devanagari.
        """
        # Split into words and transliterate each
        words = text.split()
        transliterated_words = []

        for word in words:
            # Direct mapping for known words
            if word in self.known_entities:
                transliterated_words.append(self.known_entities[word])
                continue

            # Character-by-character transliteration
            result = ''
            i = 0
            while i < len(word):
                # Try 2-character combinations first
                if i + 1 < len(word):
                    two_char = word[i:i+2]
                    if two_char in self.char_map:
                        result += self.char_map[two_char]
                        i += 2
                        continue

                # Single character
                char = word[i]
                if char in self.char_map:
                    result += self.char_map[char]
                else:
                    result += char
                i += 1

            transliterated_words.append(result)

        return ' '.join(transliterated_words)

    def _init_char_map(self):
        """Initialize character mapping for transliteration."""
        self.char_map = {
            # Vowels
            'a': 'अ', 'i': 'इ', 'u': 'उ', 'e': 'ए', 'o': 'ओ',
            # Consonants with a
            'ka': 'का', 'ki': 'की', 'ku': 'कु', 'ke': 'के', 'ko': 'को',
            'kha': 'खा', 'khi': 'खी', 'khu': 'खु', 'khe': 'खे', 'kho': 'खो',
            'ga': 'गा', 'gi': 'गी', 'gu': 'गु', 'ge': 'गे', 'go': 'गो',
            'gha': 'घा', 'ghi': 'घी', 'ghu': 'घु', 'ghe': 'घे', 'gho': 'घो',
            'cha': 'चा', 'chi': 'ची', 'chu': 'चु', 'che': 'चे', 'cho': 'चो',
            'ja': 'जा', 'ji': 'जी', 'ju': 'जु', 'je': 'जे', 'jo': 'जो',
            'jha': 'झा', 'jhi': 'झी', 'jhu': 'झु', 'jhe': 'झे', 'jho': 'झो',
            'ta': 'टा', 'ti': 'टी', 'tu': 'टु', 'te': 'टे', 'to': 'टो',
            'tha': 'ठा', 'thi': 'ठी', 'thu': 'ठु', 'the': 'ठे', 'tho': 'ठो',
            'da': 'डा', 'di': 'डी', 'du': 'डु', 'de': 'डे', 'do': 'डो',
            'dha': 'ढा', 'dhi': 'ढी', 'dhu': 'ढु', 'dhe': 'ढे', 'dho': 'ढो',
            'na': 'णा', 'ni': 'णी', 'nu': 'णु', 'ne': 'णे', 'no': 'णो',
            'pa': 'पा', 'pi': 'पी', 'pu': 'पु', 'pe': 'पे', 'po': 'पो',
            'pha': 'फा', 'phi': 'फी', 'phu': 'फु', 'phe': 'फे', 'pho': 'फो',
            'ba': 'बा', 'bi': 'बी', 'bu': 'बु', 'be': 'बे', 'bo': 'बो',
            'bha': 'भा', 'bhi': 'भी', 'bhu': 'भु', 'bhe': 'भे', 'bho': 'भो',
            'ma': 'मा', 'mi': 'मी', 'mu': 'मु', 'me': 'मे', 'mo': 'मो',
            'ya': 'या', 'yi': 'यी', 'yu': 'यु', 'ye': 'ये', 'yo': 'यो',
            'ra': 'रा', 'ri': 'री', 'ru': 'रु', 're': 'रे', 'ro': 'रो',
            'la': 'ला', 'li': 'ली', 'lu': 'लु', 'le': 'ले', 'lo': 'लो',
            'va': 'वा', 'vi': 'वी', 'vu': 'वु', 've': 'वे', 'vo': 'वो',
            'sha': 'शा', 'shi': 'शी', 'shu': 'शु', 'she': 'शे', 'sho': 'शो',
            'sa': 'सा', 'si': 'सी', 'su': 'सु', 'se': 'से', 'so': 'सो',
            'ha': 'हा', 'hi': 'ही', 'hu': 'हु', 'he': 'हे', 'ho': 'हो',
            # Single consonants (without a)
            'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ch': 'च', 'j': 'ज',
            'jh': 'झ', 't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
            'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म', 'y': 'य',
            'r': 'र', 'l': 'ल', 'v': 'व', 'sh': 'श', 's': 'स', 'h': 'ह'
        }

    def mask_entities(self, text: str, acronyms: List[str], proper_nouns: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Mask proper nouns and acronyms before translation.
        Returns masked text and mapping dictionary.
        """
        masked_text = text
        entity_map = {}

        # Mask acronyms and proper nouns
        entities_to_mask = acronyms + proper_nouns
        for i, entity in enumerate(entities_to_mask):
            placeholder = f"__ENTITY_{i}__"
            entity_map[placeholder] = entity
            # Replace in text (case insensitive)
            masked_text = re.sub(r'\b' + re.escape(entity) + r'\b', placeholder, masked_text, flags=re.IGNORECASE)

        return masked_text, entity_map

    def unmask_entities(self, text: str, entity_map: Dict[str, str]) -> str:
        """
        Restore masked entities with their transliterated versions.
        """
        result = text
        for placeholder, entity in entity_map.items():
            # Get transliterated version
            if entity.lower() in self.known_entities:
                transliterated = self.known_entities[entity.lower()]
            else:
                transliterated = self.transliterate_phonetic(entity)

            result = result.replace(placeholder, transliterated)

        return result

    def translate_single(self, text: str) -> str:
        """Translate a single sentence."""
        # Initialize character map
        self._init_char_map()

        # Preprocess
        processed_text, acronyms, proper_nouns = self.preprocess_text(text)

        # Detect sentence type
        sentence_type = self.detect_sentence_type(processed_text)

        if sentence_type == 'transliterate':
            # For phonetic sentences, transliterate directly
            result = self.transliterate_phonetic(processed_text)
            return self.validate_devanagari(result)

        # For translation sentences: mask entities, translate, then unmask
        masked_text, entity_map = self.mask_entities(processed_text, acronyms, proper_nouns)

        # Translate with MarianMT (direct translation, no prompts)
        translation_result = self.translator(
            masked_text,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

        translated_text = translation_result[0]['translation_text']

        # Unmask entities with transliteration
        final_text = self.unmask_entities(translated_text, entity_map)

        return self.validate_devanagari(final_text)

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple sentences."""
        return [self.translate_single(text) for text in texts]

    def validate_devanagari(self, text: str) -> str:
        """Validate and clean Devanagari Unicode output."""
        # Remove any non-Devanagari characters except common punctuation
        cleaned = ''.join(c for c in text if '\u0900' <= c <= '\u097F' or c in ' \n\t.,!?;:')

        # Ensure proper spacing
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

def main():
    """Main function with 20 comprehensive test examples."""
    translator = HybridTranslator()

    # 20 comprehensive test examples covering various scenarios
    test_sentences = [
        # Phonetic transliteration cases
        "mera naam raju hai",
        "bharat mata ki jai",
        "ram aur shyam",
        "sita aur geeta",

        # Mixed translation + transliteration cases
        "i am in anits",
        "i study in iit delhi",
        "my name is john",
        "i work at google",
        "she lives in mumbai",

        # Full translation cases
        "hello how are you",
        "what is your name",
        "i love eating pizza",
        "the weather is nice today",
        "please help me",

        # Complex sentences with proper nouns
        "i study in anil neerukonda institute of technology and sciences",
        "he works for microsoft in bangalore",
        "we visited taj hotel in delhi",
        "my friend rajesh studies at mit",
        "i bought a samsung phone from amazon",

        # Edge cases
        "this is a test sentence",
        "the quick brown fox jumps over the lazy dog"
    ]

    print("Hybrid Transliteration + Translation Pipeline")
    print("=" * 60)

    for sentence in test_sentences:
        print(f"Input:  {sentence}")
        result = translator.translate_single(sentence)
        print(f"Output: {result}")
        print("-" * 60)

    # Batch processing example
    print("\nBatch Processing Example:")
    batch_results = translator.translate_batch(test_sentences[:3])
    for i, result in enumerate(batch_results):
        print(f"{i+1}. {result}")

if __name__ == "__main__":
    main()
