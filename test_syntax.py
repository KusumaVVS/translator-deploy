#!/usr/bin/env python3
"""Quick syntax check script"""
import sys
import py_compile

# Check translate_logic.py
try:
    py_compile.compile('modules/translate_logic.py', doraise=True)
    print("modules/translate_logic.py: OK")
except py_compile.PyCompileError as e:
    print(f"modules/translate_logic.py: ERROR - {e}")
    sys.exit(1)

# Check hybrid_transliteration_translation.py
try:
    py_compile.compile('hybrid_transliteration_translation.py', doraise=True)
    print("hybrid_transliteration_translation.py: OK")
except py_compile.PyCompileError as e:
    print(f"hybrid_transliteration_translation.py: ERROR - {e}")
    sys.exit(1)

# Check app.py
try:
    py_compile.compile('app.py', doraise=True)
    print("app.py: OK")
except py_compile.PyCompileError as e:
    print(f"app.py: ERROR - {e}")
    sys.exit(1)

print("\nAll files passed syntax check!")
