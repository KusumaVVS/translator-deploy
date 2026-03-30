from gtts import gTTS
import os
import time

def text_to_speech(text, direction):
    """
    Convert text to speech using gTTS and save as an audio file.
    Reuses fixed filenames (output_en.mp3 and output_hi.mp3) to save space.
    Adds timestamp query parameter for cache-busting instead of filename timestamps.

    :param text: The text to convert to speech
    :param direction: Translation direction ('en-hi', 'en-te', 'hi-en', 'te-en') to determine output language
    :return: Path to the generated audio file with cache-busting query parameter
    """
    if not text or not text.strip():
        print("Error in text-to-speech: No text to send to TTS API")
        return None

    if direction == 'en-hi':
        lang = 'hi'
    elif direction == 'en-te':
        lang = 'te'
    elif direction == 'te-en':
        lang = 'en'
    else:
        lang = 'en'

    # Use fixed filenames to prevent accumulation of many files
    # Only 2 files: output_en.mp3 and output_hi.mp3
    audio_file = f"static/output_{lang}.mp3"
    os.makedirs('static', exist_ok=True)

    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(audio_file)
        
        # Add timestamp as query parameter for cache-busting (browser will reload)
        # This prevents browser caching while keeping only 2 files
        timestamp = int(time.time() * 1000)
        return f"{audio_file}?t={timestamp}"
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

def cleanup_old_audio_files():
    """
    Remove old timestamped audio files, keeping only output_en.mp3 and output_hi.mp3
    """
    static_dir = 'static'
    if not os.path.exists(static_dir):
        return
    
    # Keep only these files
    keep_files = {'output_en.mp3', 'output_hi.mp3', 'output.mp3', 'output_en.mp3', 'output_hi.mp3'}
    
    removed_count = 0
    for filename in os.listdir(static_dir):
        if filename.endswith('.mp3'):
            filepath = os.path.join(static_dir, filename)
            # Remove timestamped files (output_hi_1234567890.mp3, output_en_1234567890.mp3)
            # or old format (output_1234567890.mp3)
            if filename not in keep_files and (filename.startswith('output_') or filename.startswith('output')):
                try:
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} old audio files")
    
    return removed_count
