import speech_recognition as sr

def speech_to_text(audio_file_path, direction):
    """
    Convert speech from an audio file to text using speech_recognition.

    :param audio_file_path: Path to the audio file
    :param direction: Translation direction ('en-hi' or 'hi-en') to determine language
    :return: Transcribed text or None if recognition fails
    """
    recognizer = sr.Recognizer()

    # Determine language based on direction
    if direction == 'en-hi':
        language = 'en-US'  # English for input when translating to Hindi
    else:
        language = 'hi-IN'  # Hindi for input when translating to English

    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language)
            return text
    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return None
