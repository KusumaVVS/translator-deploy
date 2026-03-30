#!/usr/bin/env python3
"""
Context-Aware Hybrid Transliteration + Translation System
Maintains conversation context for pronoun and topic continuity using Helsinki MarianMT.
"""

import re
import json
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Deque
from transformers import pipeline

class ContextManager:
    """Maintains rolling memory of recent conversation history."""

    def __init__(self, max_history: int = 3):
        self.history: Deque[Dict] = deque(maxlen=max_history)

    def add_interaction(self, user_input: str, system_output: str) -> None:
        """Add a new interaction to the context history."""
        self.history.append({
            'input': user_input,
            'output': system_output,
            'timestamp': datetime.now().isoformat()
        })

    def get_recent_context(self) -> List[Dict]:
        """Get recent conversation context for prompt building."""
        return list(self.history)

    def clear_context(self) -> None:
        """Clear all conversation history."""
        self.history.clear()

class EntityMemory:
    """Stores and manages detected entities (proper nouns, acronyms)."""

    def __init__(self):
        self.entities: Dict[str, str] = {}  # lowercase -> exact spelling

    def add_entity(self, entity: str) -> None:
        """Add an entity to memory with its exact spelling."""
        if entity:
            self.entities[entity.lower()] = entity

    def get_entity(self, entity_key: str) -> Optional[str]:
        """Retrieve exact spelling of an entity."""
        return self.entities.get(entity_key.lower())

    def extract_entities(self, text: str, known_entities: set) -> List[str]:
        """Extract potential entities from text."""
        entities = []

        # Find acronyms (all caps)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.extend(acronyms)

        # Find capitalized words (potential proper nouns)
        words = text.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if (len(clean_word) > 1 and clean_word[0].isupper() and
                clean_word.lower() not in {'i', 'i\'m', 'i\'ll', 'i\'ve', 'i\'d'}):
                entities.append(clean_word)

        # Also check for known proper nouns (even if lowercase)
        words_lower = text.lower().split()
        for word in words_lower:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in known_entities:
                entities.append(clean_word)

        return list(set(entities))  # Remove duplicates

    def update_from_text(self, text: str, known_entities: set = None) -> None:
        """Update entity memory from input text."""
        entities = self.extract_entities(text, known_entities or set())
        for entity in entities:
            self.add_entity(entity)

class TopicMemory:
    """Manages active conversation topics for pronoun resolution."""

    def __init__(self, max_topics: int = 3):
        self.topics: List[Dict] = []
        self.max_topics = max_topics

    def extract_topics(self, text: str) -> List[str]:
        """Extract main noun topics from text."""
        topics = []

        # Common topic keywords (can be expanded)
        topic_keywords = {
            'college', 'university', 'school', 'institute', 'company', 'organization',
            'hostel', 'home', 'office', 'place', 'city', 'country', 'person', 'people',
            'placement', 'job', 'work', 'study', 'class', 'course', 'subject'
        }

        words = text.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in topic_keywords:
                topics.append(clean_word)

        return topics

    def update_topics(self, text: str) -> None:
        """Update active topics from input text."""
        new_topics = self.extract_topics(text)

        for topic in new_topics:
            # Remove if already exists (to update recency)
            self.topics = [t for t in self.topics if t['topic'] != topic]

            # Add to front (most recent)
            self.topics.insert(0, {
                'topic': topic,
                'last_mentioned': datetime.now().isoformat()
            })

        # Keep only max topics
        self.topics = self.topics[:self.max_topics]

    def get_active_topics(self) -> List[str]:
        """Get list of currently active topics."""
        return [t['topic'] for t in self.topics]

    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve pronoun to most likely topic."""
        pronoun_mappings = {
            'waha': ['college', 'university', 'place', 'office', 'home'],
            'udhar': ['college', 'university', 'place', 'office', 'home'],
            'vaha': ['college', 'university', 'place', 'office', 'home'],
            'usme': ['college', 'university', 'company', 'organization'],
            'unme': ['college', 'university', 'company', 'organization'],
            'iski': ['college', 'university', 'company', 'organization'],
            'unki': ['college', 'university', 'company', 'organization']
        }

        if pronoun.lower() in pronoun_mappings:
            relevant_topics = pronoun_mappings[pronoun.lower()]
            for topic in self.get_active_topics():
                if topic in relevant_topics:
                    return topic

        return None

class InputClassifier:
    """Classifies input text type for appropriate processing."""

    @staticmethod
    def classify_input(text: str) -> str:
        """Classify input as PHONETIC, ENGLISH, or MIXED."""
        text_lower = text.lower().strip()

        # Check for high punctuation density (indicates garbled or abbreviated text)
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        word_count = len(text.split())
        if word_count > 0 and punctuation_count / word_count > 0.5:
            return 'MIXED'  # Likely garbled English with punctuation

        # Check for very short words (abbreviations, typos)
        words = text.split()
        short_word_ratio = sum(1 for word in words if len(word) <= 3) / len(words) if words else 0
        if short_word_ratio > 0.6:
            return 'MIXED'  # Likely abbreviations or typos

        # Check for English grammar indicators
        english_indicators = [
            r'\b(is|am|are|was|were|be|being|been)\b',
            r'\b(the|a|an)\b',
            r'\b(and|or|but|so|because|although)\b',
            r'\b(i|you|he|she|it|we|they)\b',
            r'\b(in|on|at|to|from|with|by)\b'
        ]

        english_score = sum(1 for pattern in english_indicators if re.search(pattern, text_lower))

        # Check for Hindi phonetic patterns
        hindi_phonetic_indicators = [
            r'mera|teri|uski|unki|hamara',
            r'hai|hoon|ho|hain|tha|thi|the',
            r'mein|tum|yeh|woh|waha',
            r'kar|karna|karta|karti|karte'
        ]

        hindi_score = sum(1 for pattern in hindi_phonetic_indicators if re.search(pattern, text_lower))

        # Classification logic
        if hindi_score > english_score:
            return 'PHONETIC'
        elif english_score > hindi_score:
            return 'ENGLISH'
        else:
            return 'MIXED'

class PromptConstructor:
    """Builds dynamic prompts with conversation context."""

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def build_prompt(self, current_input: str, input_type: str) -> str:
        """Build context-aware prompt for the translation model."""
        context = self.context_manager.get_recent_context()

        # Keep prompt under 400 characters to avoid model limits
        max_prompt_length = 400
        prompt_parts = []

        if context:
            prompt_parts.append("CONTEXT:")
            # Only include most recent 2 interactions to keep under limit
            recent_context = context[-2:] if len(context) > 2 else context

            for interaction in recent_context:
                # Truncate long inputs/outputs
                input_text = interaction['input'][:50] + "..." if len(interaction['input']) > 50 else interaction['input']
                output_text = interaction['output'][:50] + "..." if len(interaction['output']) > 50 else interaction['output']

                prompt_parts.append(f"User: {input_text}")
                prompt_parts.append(f"Output: {output_text}")

        prompt_parts.append(f"Input: {current_input}")

        # Task instruction based on input type (keep short)
        if input_type == 'PHONETIC':
            task = "Transliterate to Devanagari, maintain context."
        elif input_type == 'ENGLISH':
            task = "Translate to Hindi, maintain context."
        else:  # MIXED
            task = "Hybrid: transliterate phonetic parts, translate English parts."

        prompt_parts.append(f"Task: {task}")

        full_prompt = "\n".join(prompt_parts)

        # Truncate if too long
        if len(full_prompt) > max_prompt_length:
            full_prompt = full_prompt[:max_prompt_length - 10] + "..."

        return full_prompt

class ContextAwareTranslator:
    """Main orchestrator for context-aware translation system."""

    def __init__(self, model_name='M2M100'):
        # Initialize components
        self.context_manager = ContextManager()
        self.entity_memory = EntityMemory()
        self.topic_memory = TopicMemory()
        self.input_classifier = InputClassifier()
        self.prompt_constructor = PromptConstructor(self.context_manager)

        # Known proper nouns/acronyms that should always be transliterated
        self.known_proper_nouns = {
            'anits', 'mit', 'iit', 'nit', 'iisc', 'iim', 'google', 'microsoft',
            'apple', 'samsung', 'tesla', 'amazon', 'harvard', 'delhi', 'mumbai',
            'bangalore', 'john', 'pizza', 'dominos', 'eiffel', 'wembley', 'taj',
            'python', 'programming', 'kusuma', 'gandhi', 'mahatma', 'nelson', 'mandela'
        }

        # Known Hindi names to English transliterations for hi-en translation
        self.hindi_to_english_names = {
            'कुसुम': 'Kusuma',
            'गांधी': 'Gandhi',
            'महात्मा': 'Mahatma',
            'नेल्सन': 'Nelson',
            'मंडेला': 'Mandela',
            'दिल्ली': 'Delhi',
            'मुंबई': 'Mumbai',
            'चेन्नई': 'Chennai',
            'कोलकाता': 'Kolkata',
            'बैंगलोर': 'Bangalore',
            'पुणे': 'Pune',
            'हैदराबाद': 'Hyderabad',
            'अहमदाबाद': 'Ahmedabad',
            'जयपुर': 'Jaipur',
            'लखनऊ': 'Lucknow'
        }

        # Initialize translation models
        self.model_name = model_name
        print(f"Loading {model_name} models...")

        if model_name == 'M2M100':
            self.translator_en_hi = pipeline('translation', model='facebook/m2m100_418M', src_lang='en', tgt_lang='hi')
            self.translator_hi_en = pipeline('translation', model='facebook/m2m100_418M', src_lang='hi', tgt_lang='en')
        elif model_name == 'Helsinki':
            self.translator_en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
            self.translator_hi_en = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'M2M100' or 'Helsinki'")

        print("Models loaded successfully!")

    def translate(self, text: str, direction: str = 'en-hi') -> str:
        """Main translation method with context awareness."""
        if not text or not text.strip():
            return ""

        # Clean input
        text = text.strip()

        if direction == 'en-hi':
            # Special cases for famous quotes
            text_lower = text.lower().strip()
            # Note: Removed hardcoded poor quality translation - let model handle it naturally
            # if "it always seems impossible until" in text_lower and "done" in text_lower:
            #     return "यह हमेशा असंभव लगता है जब तक कि यह नहीं हो जाता"

            # Classify input type
            input_type = self.input_classifier.classify_input(text)

            # Update memories with current input
            self.entity_memory.update_from_text(text, self.known_proper_nouns)
            self.topic_memory.update_topics(text)

            # Replace detected proper nouns with transliterated versions for better translation
            processed_text = text
            for entity in self.entity_memory.entities:
                from modules.translate_logic import transliterate_en_to_hi
                transliterated = transliterate_en_to_hi(entity.lower())
                processed_text = processed_text.replace(entity, transliterated)

            # Handle different input types appropriately
            if input_type == 'PHONETIC':
                # For phonetic Hindi input, use transliteration to Devanagari
                from modules.translate_logic import transliterate_en_to_hi
                translation = transliterate_en_to_hi(processed_text)
            else:  # ENGLISH or MIXED
                # Use translation model
                try:
                    result = self.translator_en_hi(
                        processed_text,
                        max_length=200,
                        num_beams=2,
                        early_stopping=True
                    )
                    if result and len(result) > 0 and 'translation_text' in result[0]:
                        translation = result[0]['translation_text']
                    else:
                        translation = self._fallback_translation(processed_text)
                except Exception as e:
                    print(f"Translation error: {e}")
                    translation = self._fallback_translation(processed_text)
        elif direction == 'hi-en':
            # Replace known Hindi names with English transliterations for better translation
            processed_text = text
            for hindi_name, english_name in self.hindi_to_english_names.items():
                processed_text = processed_text.replace(hindi_name, english_name)

            # For Hindi to English, use the hi-en model
            try:
                result = self.translator_hi_en(
                    processed_text,
                    max_length=200,
                    num_beams=2,
                    early_stopping=True
                )
                if result and len(result) > 0 and 'translation_text' in result[0]:
                    translation = result[0]['translation_text']
                else:
                    translation = "Translation failed."
            except Exception as e:
                print(f"Translation error: {e}")
                translation = "Translation failed."
        else:
            return "Unsupported direction."

        # Clean up the result
        translation = self._postprocess_translation(translation)

        # Update context with this interaction (only for en-hi for now)
        if direction == 'en-hi':
            self.context_manager.add_interaction(text, translation)

        return translation

    def _postprocess_translation(self, text: str) -> str:
        """Clean up translation output."""
        if not text:
            return text

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove any control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)

        return text

    def _fallback_translation(self, text: str) -> str:
        """Fallback to basic translation without context."""
        try:
            result = self.translator_en_hi(text, max_length=512, truncation=True)
            if result and len(result) > 0:
                return result[0]['translation_text']
        except Exception as e:
            print(f"Fallback translation error: {e}")

        return "Translation failed. Please try again."

    def get_context_status(self) -> Dict:
        """Get current status of context and memories."""
        return {
            'conversation_history': len(self.context_manager.history),
            'entities': list(self.entity_memory.entities.keys()),
            'active_topics': self.topic_memory.get_active_topics(),
            'recent_interactions': self.context_manager.get_recent_context()
        }

    def reset_context(self) -> None:
        """Reset all context and memories."""
        self.context_manager.clear_context()
        self.entity_memory.entities.clear()
        self.topic_memory.topics.clear()

# Example usage and testing
def main():
    """Demonstrate the context-aware translation system."""
    translator = ContextAwareTranslator()

    # Example conversation
    examples = [
        "mera college anits hai",
        "waha placement ache hote hai",
        "tum waha padhte ho",
        "he works at google",
        "iski salary kitni hai"
    ]

    print("Context-Aware Translation Demo")
    print("=" * 50)

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Input:  {example}")

        result = translator.translate(example)
        print(f"Output length: {len(result)} characters")

        # Show context status
        status = translator.get_context_status()
        print(f"Active topics: {status['active_topics']}")
        print(f"Entities: {status['entities']}")
        print(f"Conversation history: {len(status['recent_interactions'])} interactions")

        print("-" * 50)

if __name__ == "__main__":
    main()