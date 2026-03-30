# Context-Aware Hybrid Transliteration + Translation System

## System Overview

A context-aware translation system that maintains conversation continuity for pronouns and topics using HuggingFace M2M100 (facebook/m2m100_418M) with memory support.

## Architecture Components

### 1. ContextManager
- **Purpose**: Maintain rolling memory of conversation history
- **Implementation**: Deque with max length 3 for recent inputs/outputs
- **Structure**:
  ```python
  class ContextManager:
      def __init__(self, max_history=3):
          self.history = deque(maxlen=max_history)

      def add_interaction(self, user_input, system_output):
          self.history.append({
              'input': user_input,
              'output': system_output,
              'timestamp': datetime.now()
          })
  ```

### 2. EntityMemory
- **Purpose**: Store and reuse detected entities (proper nouns, acronyms)
- **Implementation**: Dictionary mapping for entity persistence
- **Features**:
  - Extract entities from text (ANITS, IIT, person names)
  - Maintain exact spelling for future reuse
  - Handle case variations

### 3. TopicMemory
- **Purpose**: Track active conversation topics
- **Implementation**: List with max 3 topics, prioritized by recency
- **Features**:
  - Extract main noun keywords
  - Maintain topic relevance scoring
  - Resolve pronoun references

### 4. PromptConstructor
- **Purpose**: Build dynamic prompts with context
- **Format**:
  ```
  CONTEXT:
  User: [previous_input_1]
  Output: [previous_output_1]

  User: [previous_input_2]
  Output: [previous_output_2]

  CURRENT INPUT:
  [current_user_input]

  TASK:
  Transliterate/translate maintaining pronoun and topic continuity.
  ```

### 5. PronounResolver
- **Purpose**: Handle pronoun continuity using topic memory
- **Mappings**:
  - he/she/it/they → resolved from topic memory
  - waha/udhar/vaha → location/topic references
  - usme/unme → topic references

### 6. InputClassifier
- **Purpose**: Classify input type for appropriate processing
- **Types**:
  - PHONETIC: Hindi-like pronunciation ("mera naam raju hai")
  - ENGLISH: Standard English grammar ("my name is raju")
  - MIXED: Combination of both

### 7. ContextAwareTranslator
- **Purpose**: Main orchestrator integrating all components
- **Workflow**:
  1. Classify input
  2. Extract entities and topics
  3. Build context-aware prompt
  4. Generate translation with M2M100 model
  5. Update memories
  6. Validate and return result

## Implementation Plan

### Phase 1: Core Components
1. Implement ContextManager
2. Implement EntityMemory
3. Implement TopicMemory

### Phase 2: Processing Components
4. Implement PromptConstructor
5. Implement PronounResolver
6. Implement InputClassifier

### Phase 3: Integration
7. Implement ContextAwareTranslator
8. Add validation and fallbacks
9. Create comprehensive tests

### Phase 4: Deployment
10. Integrate into existing app.py
11. Update UI for context display
12. Add context reset functionality

## Example Usage

```python
# Initialize system
translator = ContextAwareTranslator()

# First interaction
result1 = translator.translate("mera college anits hai")
# Output: "मेरा कॉलेज ANITS है"
# Memories updated with entity "ANITS" and topic "college"

# Second interaction
result2 = translator.translate("waha placement ache hote hai")
# Output: "वहाँ प्लेसमेंट अच्छे होते हैं"
# Context maintained, "waha" refers to "college"

# Third interaction
result3 = translator.translate("tum waha padhte ho")
# Output: "तुम वहाँ पढ़ते हो"
# Pronoun "waha" correctly resolved to college context
```

## Key Features

- **Pronoun Continuity**: Maintains consistent references across conversation
- **Topic Awareness**: Remembers active subjects (college, company, etc.)
- **Entity Consistency**: Reuses exact spellings of proper nouns
- **Dynamic Prompts**: Context-aware prompt engineering for better translations
- **Fallback Mechanisms**: Graceful degradation if context is insufficient
- **Memory Management**: Efficient rolling memory with configurable limits

## Validation Strategy

- Unit tests for each component
- Integration tests for conversation flows
- Edge case handling (no context, ambiguous pronouns, etc.)
- Performance benchmarking
- Accuracy validation against expected outputs