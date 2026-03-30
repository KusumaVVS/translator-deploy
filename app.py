import json
import os
import logging

# Suppress transformers verbose warnings before importing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Production-ready Tesseract path from environment variable
# Default to Linux path for PaaS platforms (Render/Heroku/Docker)
TESSERACT_PATH = os.environ.get('TESSERACT_PATH', '/usr/bin')

# Handle Windows vs Linux path separators
if os.name == 'nt':  # Windows
    os.environ["PATH"] += f";{TESSERACT_PATH}"
else:  # Linux/Mac
    os.environ["PATH"] = f"{TESSERACT_PATH}:{os.environ.get('PATH', '')}"

import pytesseract

# Set tesseract executable path
tesseract_exe = os.path.join(TESSERACT_PATH, 'tesseract' if os.name != 'nt' else 'tesseract.exe')
pytesseract.pytesseract.tesseract_cmd = tesseract_exe

print(f"Tesseract path set to: {tesseract_exe}")
print(f"Tesseract executable exists: {os.path.exists(tesseract_exe)}")
print("TESSERACT VERSION IN FLASK:", pytesseract.get_tesseract_version())

print("TESSERACT VERSION IN FLASK:", pytesseract.get_tesseract_version())

from datetime import datetime
from flask import Flask, render_template, request, jsonify

from modules.translate_logic import translate
from modules.context_aware_translator import ContextAwareTranslator
from modules.text_to_speech import text_to_speech
from modules.speech_to_text import speech_to_text
from modules.image_ocr import ocr_from_image_bytes

# Initialize context-aware translator with M2M100 model for better quality
context_translator = ContextAwareTranslator(model_name='M2M100')


app = Flask(__name__)

def load_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def save_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_settings():
    if os.path.exists('settings.json'):
        with open('settings.json', 'r') as f:
            return json.load(f)
    return {"default_direction": "en-hi", "audio_enabled": True, "theme_mode": "light"}

def save_settings(settings):
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

def load_context():
    if os.path.exists('context.json'):
        with open('context.json', 'r') as f:
            return json.load(f)
    return []

def save_context(context):
    with open('context.json', 'w') as f:
        json.dump(context, f, indent=4)

def load_global_context():
    if os.path.exists('global_context.json'):
        try:
            with open('global_context.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If corrupted, reset to empty list
            print("Warning: global_context.json is corrupted, resetting to empty.")
            return []
    return []

def save_global_context(context):
    # Keep only the last 20 translations to avoid too much context
    if len(context) > 20:
        context = context[-20:]
    with open('global_context.json', 'w') as f:
        json.dump(context, f, indent=4)

@app.route("/", methods=["GET", "POST"]) # FIX: Allow both GET and POST
def home():
    translated_text = None
    audio_path = None
    input_text = ""
    direction = "en-hi"

    if request.method == "POST":
        # Logic for POST request (form submission/translation)
        input_text = request.form.get("input_text")
        direction = request.form.get("direction")
        
        # Ensure the input isn't empty before attempting translation/TTS
        if input_text:
            translated_text = translate(input_text, direction, model_name='M2M100')
            audio_path = text_to_speech(translated_text, direction)

    return render_template(
        "index.html",
        input_text=input_text,
        translated_text=translated_text,
        audio_path=audio_path,
        direction=direction
    )

@app.route("/saved")
def saved():
    saved_data = load_data('saved.json')
    return render_template("saved.html", saved=saved_data)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    settings = load_settings()
    if request.method == "POST":
        default_direction = request.form.get("default_direction")
        audio_enabled = request.form.get("audio_enabled") == "on"
        settings["default_direction"] = default_direction
        settings["audio_enabled"] = audio_enabled
        save_settings(settings)
    return render_template("settings.html", settings=settings)



@app.route("/get_saved")
def get_saved():
    saved_data = load_data('saved.json')
    return jsonify(saved_data)

@app.route("/get_settings")
def get_settings():
    settings = load_settings()
    return jsonify(settings)

@app.route("/get_context")
def get_context():
    contexts = load_context()
    return jsonify(contexts)

@app.route("/create_context", methods=["POST"])
def create_context():
    data = request.get_json()
    name = data.get('name', 'New Conversation')
    contexts = load_context()
    context_id = str(len(contexts) + 1)
    new_context = {
        'id': context_id,
        'name': name,
        'messages': [],
        'created': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat()
    }
    contexts.append(new_context)
    save_context(contexts)
    return jsonify({'context_id': context_id, 'message': 'Context created successfully'})

@app.route("/reset_context", methods=["POST"])
def reset_context():
    """Reset the context-aware translator context."""
    try:
        context_translator.reset_context()
        return jsonify({'message': 'Context reset successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to reset context: {str(e)}'}), 500

@app.route("/get_context_status")
def get_context_status():
    """Get current context status."""
    try:
        status = context_translator.get_context_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': f'Failed to get context status: {str(e)}'}), 500

@app.route("/clear_context", methods=["POST"])
def clear_context():
    data = request.get_json()
    context_id = data.get('context_id')
    if not context_id:
        return jsonify({'error': 'Context ID required'}), 400

    contexts = load_context()
    for c in contexts:
        if c['id'] == context_id:
            c['messages'] = []
            c['last_updated'] = datetime.now().isoformat()
            save_context(contexts)
            return jsonify({'message': 'Context cleared successfully'})
    return jsonify({'error': 'Context not found'}), 404

@app.route("/clear_global_context", methods=["POST"])
def clear_global_context():
    # Clear the global context file
    with open('global_context.json', 'w') as f:
        json.dump([], f)
    return jsonify({'message': 'Global context cleared successfully'})

@app.route("/save_settings", methods=["POST"])
def save_settings_route():
    data = request.get_json()
    settings = {
        "default_direction": data.get("default_direction", "en-hi"),
        "audio_enabled": data.get("audio_enabled", True),
        "theme_mode": data.get("theme_mode", "light")
    }
    save_settings(settings)
    return jsonify({"message": "Settings saved successfully"})

@app.route("/save_translation", methods=["POST"])
def save_translation():
    data = request.get_json()
    input_text = data.get("input_text")
    translated_text = data.get("translated_text")
    direction = data.get("direction")
    timestamp = datetime.now().isoformat()

    if not input_text or not translated_text:
        return jsonify({"error": "Missing input or translated text"}), 400

    saved_data = load_data('saved.json')
    saved_data.append({
        "input_text": input_text,
        "translated_text": translated_text,
        "direction": direction,
        "timestamp": timestamp
    })
    save_data('saved.json', saved_data)
    return jsonify({"message": "Translation saved successfully"})

# Update home route to save history
@app.route("/translate", methods=["POST"])
def translate_route():
    input_text = request.form.get("input_text")
    direction = request.form.get("direction")
    context_id = request.form.get("context_id")
    if not input_text:
        return render_template("index.html", input_text="", translated_text=None, audio_path=None, direction=direction)

    # Translate without context (context feature removed for better translation quality)
    translated_text = translate(input_text, direction)
    audio_path = text_to_speech(translated_text, direction)

    # Update context if context_id provided
    if context_id:
        contexts = load_context()
        for c in contexts:
            if c['id'] == context_id:
                c['messages'].append(input_text)
                c['messages'].append(translated_text)
                c['last_updated'] = datetime.now().isoformat()
                break
        save_context(contexts)

    return render_template("index.html", input_text=input_text, translated_text=translated_text, audio_path=audio_path, direction=direction)

@app.route('/translate_api', methods=['POST'])
def translate_api():
    data = request.get_json()
    input_text = data.get('input_text', '').strip()
    direction = data.get('direction')
    if not direction:
        # Auto-detect direction based on input language
        if any('\u0900' <= c <= '\u097F' for c in input_text):
            direction = 'hi-en'
        elif any('\u0C00' <= c <= '\u0C7F' for c in input_text):
            direction = 'te-en'
        else:
            direction = 'en-hi'
    model_name = data.get('model_name', 'M2M100')  # Default to M2M100
    context_id = data.get('context_id')

    if not input_text:
        return jsonify({'error': 'Please enter text to translate.'}), 400

    try:
        # Use context-aware translator for Hindi translations (en-hi, hi-en)
        # Use regular translate() function for Telugu translations (en-te, te-en)
        if direction in ['en-hi', 'hi-en']:
            translated_text = context_translator.translate(input_text, direction)
        else:
            # For Telugu, use the translate function which supports it
            from modules.translate_logic import translate as translate_func
            translated_text = translate_func(input_text, direction, model_name='M2M100')
        
        audio_path = None
        if translated_text:
            audio_path = text_to_speech(translated_text, direction)

        # Get context status for debugging
        context_status = context_translator.get_context_status()

        return jsonify({
            'translated_text': translated_text,
            'audio_path': audio_path,
            'direction': direction,
            'model_used': context_translator.model_name,
            'context_info': {
                'active_topics': context_status['active_topics'],
                'conversation_history': len(context_status['recent_interactions']),
                'entities': context_status['entities']
            }
        })
    except Exception as e:
        print(f"Translation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Translation failed: {str(e)}"}), 500

def preprocess_image_for_ocr(image_path):
    """Convert image to PNG format for better OCR compatibility."""
    try:
        from PIL import Image
        print(f"Preprocessing image: {image_path}")

        # Open image
        img = Image.open(image_path)
        print(f"Original image mode: {img.mode}, format: {img.format}")

        # Convert to RGB if not already (handles WEBP, RGBA, etc.)
        if img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
            print("Converted to RGB mode")

        # Create new filename with PNG extension
        new_path = image_path.rsplit(".", 1)[0] + "_processed.png"

        # Save as PNG
        img.save(new_path, "PNG")
        print(f"Saved processed image as: {new_path}")

        # Clean up original if different
        if new_path != image_path and os.path.exists(image_path):
            os.remove(image_path)
            print("Cleaned up original image file")

        return new_path

    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return image_path  # Return original path if preprocessing fails

@app.route('/translate_image', methods=['POST'])
def translate_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    direction = request.form.get('direction', 'auto')  # Default to auto-detection
    model_name = request.form.get('model_name', 'M2M100')

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_filename = None
    try:
        print(f"Filename: {image_file.filename}")
        print("Saving uploaded file...")

        # Save with original extension first
        file_ext = image_file.filename.rsplit(".", 1)[-1].lower() if "." in image_file.filename else "png"
        temp_filename = f'temp_uploaded.{file_ext}'
        image_file.save(temp_filename)

        print(f"File saved as: {temp_filename}")
        print(f"File exists: {os.path.exists(temp_filename)}")
        print(f"File size: {os.path.getsize(temp_filename) if os.path.exists(temp_filename) else 0} bytes")

        # Preprocess image for better OCR compatibility
        processed_filename = preprocess_image_for_ocr(temp_filename)
        print(f"Processed image path: {processed_filename}")

        # Read processed image as bytes
        with open(processed_filename, 'rb') as f:
            image_bytes = f.read()

        # Clean up processed file
        if os.path.exists(processed_filename):
            os.remove(processed_filename)

        # OCR now returns both text and detected source language
        ocr_text, detected_source_language = ocr_from_image_bytes(image_bytes, direction)
        print(f"OCR detected source language: {detected_source_language}")

        # Auto-detect translation direction based on detected source language
        if direction == 'auto':
            # Use detected source language to determine translation direction
            if detected_source_language == 'hindi':
                direction = 'hi-en'
            elif detected_source_language == 'telugu':
                direction = 'te-en'
            elif detected_source_language == 'english':
                direction = 'en-hi'
            else:  # mixed
                # For mixed text, default to en-hi (more common use case)
                direction = 'en-hi'
            print(f"Auto-detected translation direction: {direction}")

        if not ocr_text or len(ocr_text.strip()) == 0:
            return jsonify({'success': False, 'error': 'No readable text found in image'})

        # Use the robust translate function which has better fallback handling
        translated_text = translate(ocr_text, direction, model_name='M2M100')

        if not translated_text:
            return jsonify({'error': 'Translation failed - no result from translation engine'}), 500

        # Check if translation result is meaningful (not just transliteration artifacts)
        # For OCR text with errors, we might get partial results, but that's still valid
        if len(translated_text.strip()) < 1:
            return jsonify({'error': 'Translation failed - empty result from translation engine'}), 500

        # Generate audio for the translated text
        audio_path = text_to_speech(translated_text, direction)

        return jsonify({
            'success': True,
            'extracted_text': ocr_text,
            'translated_text': translated_text
        })
    except Exception as e:
        print(f"Image translation error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a more user-friendly error message
        error_msg = "Image processing failed. Please ensure the image contains readable text and try again."
        if "OCR" in str(e).lower():
            error_msg = "Could not extract text from the image. Please try a clearer image with visible text."
        elif "translation" in str(e).lower():
            error_msg = "Translation failed. Please try again or use different text."
        return jsonify({'error': error_msg}), 500
    finally:
        # Clean up any remaining temp files
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                print(f"Cleaned up temp file: {temp_filename}")
            except:
                pass

@app.route("/speech_to_text", methods=["POST"])
def speech_to_text_route():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    direction = request.form.get('direction', 'en-hi')

    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded audio file temporarily
    audio_path = os.path.join('static', 'temp_audio.wav')
    audio_file.save(audio_path)

    # Perform speech-to-text
    transcribed_text = speech_to_text(audio_path, direction)

    # Clean up the temporary file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    if transcribed_text:
        return jsonify({"text": transcribed_text})
    else:
        return jsonify({"error": "Speech recognition failed"}), 500

if __name__ == "__main__":
    # Clean up old audio files on startup
    from modules.text_to_speech import cleanup_old_audio_files
    cleanup_old_audio_files()
    
    # Temporarily disable the reloader to prevent the double-run that
    # causes conflicts in model loading/cleanup.
    app.run(debug=True, use_reloader=False)
