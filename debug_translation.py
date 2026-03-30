#!/usr/bin/env python3
"""
Debug script to analyze the problematic translation.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.translate_logic import translate, normalize_text, preprocess_text

def debug_translation():
    """Debug the translation issue."""

    original = 'संघर्ष प्रगति का आमंत्रण है जो इसे स्वीकारता है उसका जीवन निखर जाता है वा 5000. (९90'
    print("Original Text:")
    print(original.encode('utf-8').decode('utf-8', errors='replace'))
    print()

    # Test normalization
    normalized = normalize_text(original)
    print("After normalization:")
    print(normalized.encode('utf-8').decode('utf-8', errors='replace'))
    print()

    # Test preprocessing
    preprocessed = preprocess_text(normalized, 'hi-en')
    print("After preprocessing:")
    print(preprocessed.encode('utf-8').decode('utf-8', errors='replace'))
    print()

    # Test translation
    print("Testing translation with different models:")
    models = ['Helsinki-NLP', 'M2M100']

    for model in models:
        try:
            result = translate(original, 'hi-en', model_name=model)
            print(f"{model}: {result.encode('utf-8').decode('utf-8', errors='replace')}")
        except Exception as e:
            print(f"{model}: ERROR - {e}")

    print()
    print("Expected translation should be something like:")
    print("'Struggle is an invitation to progress, the life of the one who accepts it becomes bright or 5000. (990)'")

if __name__ == "__main__":
    debug_translation()