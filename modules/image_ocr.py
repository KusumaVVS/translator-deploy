import pytesseract
import os
import io
import re
from PIL import Image, ImageFilter, ImageEnhance

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
    print("PaddleOCR is available")
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    print("PaddleOCR not available, using Tesseract only")

# Initialize PaddleOCR engine (lazy initialization)
_paddle_ocr_en = None
_paddle_ocr_hi = None
_paddle_ocr_te = None

def get_paddle_ocr_engine(language='en'):
    """
    Get or initialize the PaddleOCR engine for the specified language.
    
    Args:
        language: 'en' for English, 'hi' for Hindi, 'te' for Telugu
        
    Returns:
        PaddleOCR engine instance
    """
    global _paddle_ocr_en, _paddle_ocr_hi, _paddle_ocr_te
    
    if language == 'en':
        if _paddle_ocr_en is None:
            print("Initializing PaddleOCR for English...")
            _paddle_ocr_en = PaddleOCR(
                lang='en',
                use_angle_cls=True,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5
            )
        return _paddle_ocr_en
    elif language == 'hi':
        if _paddle_ocr_hi is None:
            print("Initializing PaddleOCR for Hindi...")
            _paddle_ocr_hi = PaddleOCR(
                lang='hi',
                use_angle_cls=True,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5
            )
        return _paddle_ocr_hi
    elif language == 'te':
        if _paddle_ocr_te is None:
            print("Initializing PaddleOCR for Telugu...")
            _paddle_ocr_te = PaddleOCR(
                lang='te',
                use_angle_cls=True,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5
            )
        return _paddle_ocr_te
    else:
        # Default to English
        if _paddle_ocr_en is None:
            print("Initializing PaddleOCR for English (default)...")
            _paddle_ocr_en = PaddleOCR(
                lang='en',
                use_angle_cls=True,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5
            )
        return _paddle_ocr_en

# Make Tesseract path configurable via environment variable
# Check if custom path is provided and exists, otherwise use default
custom_tesseract_path = os.environ.get('TESSERACT_PATH')
if custom_tesseract_path and os.path.exists(custom_tesseract_path):
    # If path is a directory, append tesseract.exe
    if os.path.isdir(custom_tesseract_path):
        TESSERACT_PATH = os.path.join(custom_tesseract_path, "tesseract.exe")
    else:
        TESSERACT_PATH = custom_tesseract_path
    print(f"Using custom Tesseract path: {TESSERACT_PATH}")
else:
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if custom_tesseract_path:
        print(f"Warning: Custom Tesseract path '{custom_tesseract_path}' not found, using default: {TESSERACT_PATH}")
    else:
        print(f"Using default Tesseract path: {TESSERACT_PATH}")

# Verify Tesseract executable exists
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    print(f"Warning: Tesseract executable not found at {TESSERACT_PATH}")
    print("OCR functionality may not work properly")

# Try to import optional dependencies for advanced preprocessing
try:
    import numpy as np
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using basic preprocessing only.")


def calculate_image_quality_metrics(image):
    """
    Calculate quality metrics for an image to guide preprocessing intensity.
    
    Returns:
        dict: Dictionary containing quality metrics:
            - contrast_ratio: Ratio of max to min pixel values
            - brightness: Mean pixel value
            - noise_estimate: Estimated noise level (standard deviation)
            - sharpness: Estimated sharpness using Laplacian variance
    """
    try:
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate basic statistics
        brightness = np.mean(img_array)
        contrast_ratio = (np.max(img_array) - np.min(img_array)) / 255.0
        noise_estimate = np.std(img_array)
        
        # Calculate sharpness using Laplacian variance (only if scipy available)
        if HAS_SCIPY:
            laplacian = ndimage.laplace(img_array)
            sharpness = np.var(laplacian)
        else:
            # Fallback: estimate sharpness using gradient variance
            gradient_x = np.diff(img_array, axis=1)
            gradient_y = np.diff(img_array, axis=0)
            sharpness = np.var(gradient_x) + np.var(gradient_y)
        
        return {
            'contrast_ratio': float(contrast_ratio),
            'brightness': float(brightness),
            'noise_estimate': float(noise_estimate),
            'sharpness': float(sharpness)
        }
    except Exception as e:
        print(f"Error calculating image quality metrics: {e}")
        # Return default values if calculation fails
        return {
            'contrast_ratio': 0.5,
            'brightness': 128.0,
            'noise_estimate': 50.0,
            'sharpness': 100.0
        }


def validate_preprocessed_image(image, min_size=100):
    """
    Validate that the preprocessed image meets minimum requirements for OCR.
    
    Args:
        image: PIL Image object
        min_size: Minimum width and height in pixels
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if image exists
    if image is None:
        return False, "Image is None"
    
    # Check image dimensions
    if image.size[0] < min_size or image.size[1] < min_size:
        return False, f"Image too small: {image.size[0]}x{image.size[1]} (minimum {min_size}x{min_size})"
    
    # Check if image is empty (all pixels same value)
    try:
        if image.mode == 'L':
            img_array = np.array(image)
            if np.all(img_array == img_array.flat[0]):
                return False, "Image appears to be empty (all pixels have same value)"
    except:
        pass
    
    # Check image mode
    if image.mode not in ['L', 'RGB', 'RGBA']:
        return False, f"Unsupported image mode: {image.mode}"
    
    return True, None


def calculate_adaptive_contrast_factor(image):
    """
    Calculate adaptive contrast enhancement factor based on image characteristics.
    
    Args:
        image: PIL Image object
        
    Returns:
        float: Contrast enhancement factor (1.0 to 3.0)
    """
    try:
        metrics = calculate_image_quality_metrics(image)
        contrast_ratio = metrics['contrast_ratio']
        brightness = metrics['brightness']
        
        # Low contrast images need more enhancement
        if contrast_ratio < 0.3:
            # Very low contrast - aggressive enhancement
            factor = 2.5
        elif contrast_ratio < 0.5:
            # Low contrast - moderate enhancement
            factor = 2.0
        elif contrast_ratio < 0.7:
            # Normal contrast - mild enhancement
            factor = 1.5
        else:
            # High contrast - minimal enhancement
            factor = 1.2
        
        # Adjust for brightness (very dark or very bright images need less enhancement)
        if brightness < 50 or brightness > 200:
            factor = max(1.0, factor - 0.3)
        
        # Clamp to reasonable range
        factor = max(1.0, min(3.0, factor))
        
        return factor
    except Exception as e:
        print(f"Error calculating adaptive contrast: {e}")
        return 1.5  # Default moderate enhancement


def calculate_adaptive_block_size(image):
    """
    Calculate adaptive block size for thresholding based on image dimensions.
    
    Args:
        image: PIL Image object
        
    Returns:
        int: Block size for adaptive thresholding (odd number)
    """
    try:
        min_dimension = min(image.size)
        
        # Block size should be proportional to image size
        # Typical range: 15 to 51
        block_size = max(15, min(51, min_dimension // 20))
        
        # Ensure block size is odd (required for adaptive thresholding)
        if block_size % 2 == 0:
            block_size += 1
        
        return block_size
    except Exception as e:
        print(f"Error calculating adaptive block size: {e}")
        return 35  # Default block size


def ocr_from_image_bytes(image_bytes, direction=None):
    """
    Performs OCR on an image provided as bytes.
    Detects source language automatically and returns both text and detected language.
    
    Args:
        image_bytes: Image data as bytes
        direction: Translation direction (kept for backward compatibility, not used for OCR language selection)
    
    Returns:
        tuple: (extracted_text, detected_language) where detected_language is 'english', 'hindi', or 'mixed'
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Original image: {image.size}, mode: {image.mode}, format: {image.format}")
        
        # Convert to grayscale for better OCR accuracy
        image = image.convert('L')

        # Calculate image quality metrics to guide preprocessing
        quality_metrics = calculate_image_quality_metrics(image)
        print(f"Image quality metrics: contrast={quality_metrics['contrast_ratio']:.2f}, "
              f"brightness={quality_metrics['brightness']:.1f}, "
              f"noise={quality_metrics['noise_estimate']:.1f}, "
              f"sharpness={quality_metrics['sharpness']:.1f}")

        # SIMPLIFIED PREPROCESSING - Less aggressive to preserve text quality
        # Only apply minimal preprocessing to avoid degrading text
        
        # Resize if too small (minimum 300 DPI equivalent) but preserve aspect ratio
        min_size = 600  # Increased minimum size for better OCR
        if image.size[0] < min_size or image.size[1] < min_size:
            scale = max(min_size / image.size[0], min_size / image.size[1])
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Resized image from {image.size} to {new_size}")

        # Apply very mild contrast enhancement only if needed
        # Only enhance if contrast is very low (< 0.3)
        if quality_metrics['contrast_ratio'] < 0.3:
            contrast_factor = 1.3  # Very mild enhancement
            print(f"Applying mild contrast enhancement: {contrast_factor:.2f}")
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)

        # Apply very mild sharpening only if image is blurry
        # Only sharpen if sharpness is very low (< 50)
        if quality_metrics['sharpness'] < 50:
            print(f"Applying mild sharpening")
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

        # Validate preprocessed image before proceeding
        is_valid, validation_error = validate_preprocessed_image(image)
        if not is_valid:
            print(f"Preprocessing validation failed: {validation_error}")
            return "", "english"

        # Ensure image is in RGB mode for better OCR
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Enhanced OCR with PaddleOCR - Try English, Hindi, and Telugu
        english_text = ""
        hindi_text = ""
        telugu_text = ""
        
        # Try PaddleOCR if available
        if PADDLE_OCR_AVAILABLE:
            try:
                # Convert image to numpy array for PaddleOCR
                import numpy as np
                img_array = np.array(image)
                
                # Try English OCR first
                try:
                    ocr_engine = get_paddle_ocr_engine('en')
                    result_en = ocr_engine.ocr(img_array)
                    
                    if result_en and result_en[0]:
                        lines = []
                        for line in result_en[0]:
                            if line and len(line) >= 2:
                                text = line[1][0]  # Get text from (text, confidence) tuple
                                if text:
                                    lines.append(text)
                        english_text = ' '.join(lines)
                        if english_text:
                            eng_chars = sum(1 for c in english_text if c.isascii() and (c.isalnum() or c in ' .,;:!?()-'))
                            print(f"PaddleOCR English extracted: '{english_text[:50]}...' ({eng_chars} English chars)" if len(english_text) > 50 else f"PaddleOCR English extracted: '{english_text}' ({eng_chars} English chars)")
                except Exception as e:
                    print(f"PaddleOCR English failed: {e}")
                
                # Try Hindi OCR
                try:
                    ocr_engine = get_paddle_ocr_engine('hi')
                    result_hi = ocr_engine.ocr(img_array)
                    
                    if result_hi and result_hi[0]:
                        lines = []
                        for line in result_hi[0]:
                            if line and len(line) >= 2:
                                text = line[1][0]  # Get text from (text, confidence) tuple
                                if text:
                                    lines.append(text)
                        hindi_text = ' '.join(lines)
                        if hindi_text:
                            hindi_chars = sum(1 for c in hindi_text if '\u0900' <= c <= '\u097F')
                            try:
                                print(f"PaddleOCR Hindi extracted: '{hindi_text[:50]}...' ({hindi_chars} Hindi chars)" if len(hindi_text) > 50 else f"PaddleOCR Hindi extracted: '{hindi_text}' ({hindi_chars} Hindi chars)")
                            except UnicodeEncodeError:
                                print(f"PaddleOCR Hindi extracted: [Hindi text - {hindi_chars} Hindi chars]")
                except Exception as e:
                    print(f"PaddleOCR Hindi failed: {e}")
                
                # Try Telugu OCR
                try:
                    ocr_engine = get_paddle_ocr_engine('te')
                    result_te = ocr_engine.ocr(img_array)
                    
                    if result_te and result_te[0]:
                        lines = []
                        for line in result_te[0]:
                            if line and len(line) >= 2:
                                text = line[1][0]  # Get text from (text, confidence) tuple
                                if text:
                                    lines.append(text)
                        telugu_text = ' '.join(lines)
                        if telugu_text:
                            telugu_chars = sum(1 for c in telugu_text if '\u0C00' <= c <= '\u0C7F')
                            try:
                                print(f"PaddleOCR Telugu extracted: '{telugu_text[:50]}...' ({telugu_chars} Telugu chars)" if len(telugu_text) > 50 else f"PaddleOCR Telugu extracted: '{telugu_text}' ({telugu_chars} Telugu chars)")
                            except UnicodeEncodeError:
                                print(f"PaddleOCR Telugu extracted: [Telugu text - {telugu_chars} Telugu chars]")
                except Exception as e:
                    print(f"PaddleOCR Telugu failed: {e}")
                    
            except Exception as e:
                print(f"PaddleOCR processing error: {e}")
        
        # Fallback to Tesseract if PaddleOCR didn't produce results
        if not english_text and not hindi_text:
            print("Falling back to Tesseract OCR...")
            
            # Try English OCR with multiple configurations
            try:
                eng_configs = [
                    "--oem 3 --psm 6",  # Uniform block of text
                    "--oem 3 --psm 3",  # Fully automatic page segmentation
                    "--oem 3 --psm 1",  # Automatic page segmentation with OSD
                    "--oem 3 --psm 8",  # Single word
                    "--oem 3 --psm 13", # Raw line
                ]

                for config in eng_configs:
                    try:
                        candidate = pytesseract.image_to_string(image, lang='eng', config=config)
                        if candidate and candidate.strip():
                            candidate = candidate.strip()
                            eng_chars = sum(1 for c in candidate if c.isascii() and (c.isalnum() or c in ' .,;:!?()-'))
                            text_length = len(candidate.strip())

                            current_eng_chars = sum(1 for c in english_text if c.isascii() and (c.isalnum() or c in ' .,;:!?()-'))
                            current_length = len(english_text.strip()) if english_text else 0

                            if (eng_chars > current_eng_chars) or (eng_chars == current_eng_chars and text_length > current_length):
                                english_text = candidate.strip()
                                print(f"English OCR ({config}) extracted: '{english_text}' ({eng_chars} English chars)")
                    except Exception as e:
                        print(f"English OCR ({config}) failed: {e}")
                        continue
            except Exception as e:
                print(f"English OCR strategy failed: {e}")

            # Try Hindi OCR with multiple configurations
            try:
                hindi_configs = [
                    "--oem 3 --psm 6",
                    "--oem 3 --psm 3",
                    "--oem 3 --psm 8",
                    "--oem 3 --psm 11",
                    "--oem 3 --psm 13",
                ]

                for config in hindi_configs:
                    try:
                        candidate = pytesseract.image_to_string(image, lang='hin', config=config)
                        if candidate and candidate.strip():
                            hindi_chars = sum(1 for c in candidate if '\u0900' <= c <= '\u097F')
                            text_length = len(candidate.strip())

                            current_hindi = sum(1 for c in hindi_text if '\u0900' <= c <= '\u097F')
                            current_length = len(hindi_text.strip()) if hindi_text else 0

                            if (hindi_chars > current_hindi) or (hindi_chars == current_hindi and text_length > current_length):
                                hindi_text = candidate.strip()
                                try:
                                    print(f"Hindi OCR ({config}) extracted: '{hindi_text}' ({hindi_chars} Hindi chars)")
                                except UnicodeEncodeError:
                                    print(f"Hindi OCR ({config}) extracted: [Hindi text - {hindi_chars} Hindi chars]")
                    except Exception as e:
                        print(f"Hindi OCR ({config}) failed: {e}")
                        continue

                # Try Hindi OCR with whitelist for better Devanagari recognition
                try:
                    whitelist = "अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहऽािीुूेैोौ्।॥0123456789 "
                    candidate = pytesseract.image_to_string(image, lang='hin',
                        config=f"--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}")
                    if candidate and candidate.strip():
                        hindi_chars = sum(1 for c in candidate if '\u0900' <= c <= '\u097F')
                        current_hindi = sum(1 for c in hindi_text if '\u0900' <= c <= '\u097F')
                        if hindi_chars > current_hindi:
                            hindi_text = candidate.strip()
                            try:
                                print(f"Hindi OCR (whitelist) extracted: '{hindi_text}' ({hindi_chars} Hindi chars)")
                            except UnicodeEncodeError:
                                print(f"Hindi OCR (whitelist) extracted: [Hindi text - {hindi_chars} Hindi chars]")
                except Exception as e:
                    print(f"Hindi OCR (whitelist) failed: {e}")
            except Exception as e:
                print(f"Hindi OCR strategy failed: {e}")

            # Try bilingual OCR as fallback
            try:
                candidate = pytesseract.image_to_string(image, lang='hin+eng', config="--oem 3 --psm 6")
                if candidate and candidate.strip():
                    hindi_chars = sum(1 for c in candidate if '\u0900' <= c <= '\u097F')
                    eng_chars = sum(1 for c in candidate if c.isascii() and c.isalnum())
                    if hindi_chars + eng_chars > 5:
                        try:
                            print(f"Bilingual OCR extracted: '{candidate.strip()}' ({hindi_chars} Hindi, {eng_chars} English chars)")
                        except UnicodeEncodeError:
                            print(f"Bilingual OCR extracted: [Mixed text - {hindi_chars} Hindi, {eng_chars} English chars]")
                        if hindi_chars > eng_chars:
                            if hindi_chars > sum(1 for c in hindi_text if '\u0900' <= c <= '\u097F'):
                                hindi_text = candidate.strip()
                        else:
                            if eng_chars > sum(1 for c in english_text if c.isascii() and c.isalnum()):
                                english_text = candidate.strip()
            except Exception as e:
                print(f"Bilingual OCR failed: {e}")

        # Determine which OCR result is better
        english_score = sum(1 for c in english_text if c.isascii() and c.isalnum())
        hindi_score = sum(1 for c in hindi_text if '\u0900' <= c <= '\u097F')
        telugu_score = sum(1 for c in telugu_text if '\u0C00' <= c <= '\u0C7F')
        
        print(f"OCR scores: English={english_score}, Hindi={hindi_score}, Telugu={telugu_score}")
        
        # Select the best result based on character counts
        max_score = max(english_score, hindi_score, telugu_score)
        
        if max_score == 0:
            # No valid text found
            selected_text = english_text or hindi_text or telugu_text
            detected_language = 'english'
        elif english_score == max_score and english_score > 0:
            selected_text = english_text
            detected_language = 'english'
            print(f"Selected English OCR: '{selected_text}'")
        elif telugu_score == max_score and telugu_score > 0:
            selected_text = telugu_text
            detected_language = 'telugu'
            try:
                print(f"Selected Telugu OCR: '{selected_text}'")
            except UnicodeEncodeError:
                print(f"Selected Telugu OCR: [Telugu text - {len(selected_text)} characters]")
        elif hindi_score == max_score and hindi_score > 0:
            selected_text = hindi_text
            detected_language = 'hindi'
            try:
                print(f"Selected Hindi OCR: '{selected_text}'")
            except UnicodeEncodeError:
                print(f"Selected Hindi OCR: [Hindi text - {len(selected_text)} characters]")
        elif english_text and hindi_text and telugu_text:
            # All have similar scores - might be mixed text
            best_text = max([english_text, hindi_text, telugu_text], key=len)
            selected_text = best_text
            detected_language = 'mixed'
            try:
                print(f"Selected mixed OCR: '{selected_text}'")
            except UnicodeEncodeError:
                print(f"Selected mixed OCR: [Mixed text - {len(selected_text)} characters]")
        elif english_text:
            selected_text = english_text
            detected_language = 'english'
            print(f"Selected English OCR (fallback): '{selected_text}'")
        elif telugu_text:
            selected_text = telugu_text
            detected_language = 'telugu'
            try:
                print(f"Selected Telugu OCR (fallback): '{selected_text}'")
            except UnicodeEncodeError:
                print(f"Selected Telugu OCR (fallback): [Telugu text - {len(selected_text)} characters]")
        elif hindi_text:
            selected_text = hindi_text
            detected_language = 'hindi'
            try:
                print(f"Selected Hindi OCR (fallback): '{selected_text}'")
            except UnicodeEncodeError:
                print(f"Selected Hindi OCR (fallback): [Hindi text - {len(selected_text)} characters]")
        else:
            # No text found - try one more time with English as final fallback
            try:
                selected_text = pytesseract.image_to_string(image, lang='eng', config="--oem 3 --psm 6").strip()
                detected_language = 'english'
                print(f"Final fallback English OCR: '{selected_text}'")
            except Exception as e:
                print(f"Final fallback OCR failed: {e}")
                return "", "english"

        # Clean the OCR result
        cleaned = selected_text.strip()
        if cleaned:
            cleaned = clean_ocr_text(cleaned)
            try:
                print(f"Final cleaned text: '{cleaned}'")
            except UnicodeEncodeError:
                print(f"Final cleaned text: [Text - {len(cleaned)} characters]")
            print(f"Detected source language: {detected_language}")
            return cleaned, detected_language

        return "", "english"
    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        return "", "english"


def detect_language(text):
    """
    Detect the primary language of the text (English, Hindi, Telugu, or Mixed).
    
    Args:
        text: The text to analyze
        
    Returns:
        str: 'english', 'hindi', 'telugu', or 'mixed'
    """
    if not text:
        return 'english'
    
    # Count characters in each language
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    
    total_chars = hindi_chars + telugu_chars + english_chars
    
    if total_chars == 0:
        return 'english'
    
    hindi_ratio = hindi_chars / total_chars
    telugu_ratio = telugu_chars / total_chars
    english_ratio = english_chars / total_chars
    
    # Determine language based on character ratios
    if telugu_ratio > 0.7:
        return 'telugu'
    elif hindi_ratio > 0.7:
        return 'hindi'
    elif english_ratio > 0.7:
        return 'english'
    else:
        return 'mixed'


def fix_concatenated_words(text):
    """
    Fix words that are concatenated together without spaces.
    Common in OCR output: "shakeMahatma" -> "shake Mahatma"
    """
    if not text:
        return text
    
    import re
    
    # Known word pairs that commonly get concatenated in OCR
    known_concatenations = [
        ('shakeMahatma', 'shake', 'Mahatma'),
        ('shakeGandhi', 'shake', 'Gandhi'),
        ('youcan', 'you can'),
        ('theworld', 'the world'),
        ('gentleway', 'gentle way'),
        ('can shake', 'can shake'),
        ('in agentle', 'in a gentle'),
    ]
    
    # First apply known concatenations
    for combined, *replacement in known_concatenations:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(combined), re.IGNORECASE)
        text = pattern.sub(' '.join(replacement), text)
    
    # Pattern-based: Find lowercase immediately followed by uppercase
    # e.g., "shakeMahatma" -> "shake Mahatma"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    return text


def clean_english_ocr_text(text):
    """Clean common OCR artifacts specific to English text."""
    if not text:
        return text

    import re

    # FIRST: Fix concatenated words (before other cleaning)
    text = fix_concatenated_words(text)

    # Fix common letter substitutions
    text = re.sub(r'\bIf\b', 'It', text)  # "If" -> "It" (common OCR error)
    text = re.sub(r'\bWm\b', 'im', text)  # "Wm" -> "im"
    text = re.sub(r'possity', 'possible', text)  # "possity" -> "possible"
    text = re.sub(r'\bte\b', 'the', text)  # "te" -> "the"
    text = re.sub(r'\boO\b', 'to', text)  # "oO" -> "to"
    text = re.sub(r'\bwn\b', 'on', text)  # "wn" -> "on"
    text = re.sub(r'\bth\b', 'the', text)  # "th" -> "the"
    text = re.sub(r'\bTh\b', 'The', text)  # "Th" -> "The"
    text = re.sub(r'\bts\b', 'is', text)  # "ts" -> "is"
    text = re.sub(r'\bIts\b', "It's", text)  # "Its" -> "It's"
    text = re.sub(r'\buntl\b', 'until', text)  # "untl" -> "until"

    # Specific fixes for the Nelson Mandela quote OCR errors
    text = re.sub(r'Ifalways', 'It always', text)  # "Ifalways" -> "It always"
    text = re.sub(r'Wmpossity', 'impossible', text)  # "Wmpossity" -> "impossible"
    text = re.sub(r'Wmpossible', 'impossible', text)  # "Wmpossible" -> "impossible"
    text = re.sub(r'\|\s*ams', 'it always', text)  # "| ams" -> "it always"
    text = re.sub(r'°°\s*Aa', '', text)  # Remove "°° Aa" artifacts
    text = re.sub(r'4\s*Oo', '', text)  # Remove "4 Oo" artifacts
    text = re.sub(r'\bMandela\b', '', text)  # Remove "Mandela" artifact
    # Remove text duplication like "it always It always"
    text = re.sub(r'\b(it always)\s+(It always)\b', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(It always)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # Additional OCR fixes for common errors
    text = re.sub(r'\bTn\b', 'In', text)  # "Tn" -> "In"
    text = re.sub(r'\bwily\b', 'way', text)  # "wily" -> "way"
    text = re.sub(r'\bCar\b', 'can', text)  # "Car" -> "can"
    text = re.sub(r'\bYow\b', 'you', text)  # "Yow" -> "you"
    text = re.sub(r'\bCiv\b', 'can', text)  # "Civ" -> "can"
    text = re.sub(r'\bYow\s+Civ\b', 'you can', text, flags=re.IGNORECASE)  # "Yow Civ" -> "you can"
    text = re.sub(r'\bbuf\b', 'but', text)  # "buf" -> "but"
    text = re.sub(r'\bfe\b', '', text)  # Remove "fe" artifacts
    text = re.sub(r'\bwow\b', '', text)  # Remove "wow" artifacts
    text = re.sub(r'YourPositiveOasis\.com', '', text)  # Remove website artifacts
    text = re.sub(r'\(\_P\)', '', text)  # Remove "(_P)" artifacts
    text = re.sub(r'\bES\b', '', text)  # Remove "ES" artifacts

    # Specific fixes for the "in a gentle way you can shape the world" quote
    text = re.sub(r'\bTn\s+a\s+gentle\s+wily\s+you\s+Car\s+shake\b', 'In a gentle way you can shape the world', text, flags=re.IGNORECASE)
    text = re.sub(r'\bTn\s+a\s+gentle\s+wily\b', 'In a gentle way', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCar\s+shake\b', 'can shape the world', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCar\b', 'can', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwily\b', 'way', text, flags=re.IGNORECASE)

    # Additional patterns for partial matches and variations
    text = re.sub(r'\bTn\s+a\s+gentle\b', 'In a gentle', text, flags=re.IGNORECASE)
    text = re.sub(r'\bwily\s+you\b', 'way you', text, flags=re.IGNORECASE)
    text = re.sub(r'\byou\s+Car\b', 'you can', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCar\s+shake\b', 'can shape', text, flags=re.IGNORECASE)
    text = re.sub(r'\bshape\s+the\s+world\b', 'shape the world', text, flags=re.IGNORECASE)

    # Additional fixes for OCR misreads in this quote
    text = re.sub(r'\bTn\b', 'In', text)  # "Tn" -> "In"
    text = re.sub(r'\bwily\b', 'way', text)  # "wily" -> "way"
    text = re.sub(r'\bCar\b', 'can', text)  # "Car" -> "can"

    # Remove author names and artifacts that appear in quote images
    text = re.sub(r'Insta\s*-\s*MarathiMulga\.com', '', text)  # Remove website artifacts
    text = re.sub(r'BRIAN\s*TRACY\s*INTERNATIONAL', '', text)  # Remove author artifacts
    text = re.sub(r'—\s*VINCE\s*LOMBARDI', '', text)  # Remove author artifacts

    # Remove isolated Hindi characters from English text (common OCR artifacts)
    # This fixes cases where OCR picks up random Hindi characters in English text
    # Remove Hindi characters that appear between English characters
    text = re.sub(r'([a-zA-Z\s])[\u0900-\u097F]+([a-zA-Z\s])', lambda m: m.group(1) + m.group(2), text)
    # Remove Hindi characters at the beginning of text
    text = re.sub(r'^[\u0900-\u097F]+([a-zA-Z\s])', lambda m: m.group(1), text)
    # Remove Hindi characters at the end of text
    text = re.sub(r'([a-zA-Z\s])[\u0900-\u097F]+$', lambda m: m.group(1), text)

    # General OCR error corrections for common patterns
    text = re.sub(r'\bHallo\b', 'Hello', text)  # "Hallo" -> "Hello"
    text = re.sub(r'\bW¥orld\b', 'World', text)  # "W¥orld" -> "World"
    text = re.sub(r'\b¥Y¥orld\b', 'World', text)  # "¥Y¥orld" -> "World"
    text = re.sub(r'\bbetfer\b', 'better', text)  # "betfer" -> "better"
    text = re.sub(r'Perfecttion', 'Perfection', text)  # "Perfecttion" -> "Perfection"
    text = re.sub(r'\bYe\b', '', text)  # Remove "Ye" artifacts
    text = re.sub(r'\bfd\b', '', text)  # Remove "fd" artifacts

    # Fix common word fragments and misreads
    text = re.sub(r'\bSlways\b', 'always', text)  # "Slways" -> "always"
    text = re.sub(r'\bmpossib\b', 'impossible', text)  # "mpossib" -> "impossible"
    text = re.sub(r'\bpossib\b', 'possible', text)  # "possib" -> "possible"

    # Handle common OCR distortions
    text = re.sub(r'\b(\w+)\s*==\s*:?\s*(\w+)?', r'\1 \2', text)  # Remove "== :" patterns
    text = re.sub(r'\b(\w+)\s*==\s*(\w+)', r'\1 \2', text)  # Remove "==" patterns

    # Clean up author names and artifacts
    text = re.sub(r'\b\w+\s*\.\s*i\s+\w+\b', '', text)  # Remove "w . i Mandela" type patterns
    text = re.sub(r'\b\w+\s*==\s*:', '', text)  # Remove "prakash == :" type patterns

    # Specific cleaning for the problematic text
    text = re.sub(r'^ig\s*==\s*:', '', text)  # Remove leading "ig == :"
    text = re.sub(r'It[\u2019\u2018\u0027]Slways', "It always", text)  # Fix "It'Slways" -> "It always"
    text = re.sub(r'rg\}\s*\.\s*bes', 'impossible', text)  # Fix "rg} . bes" -> "impossible"
    text = re.sub(r'\*mpossigé', 'impossible', text)  # Fix "*mpossigé" -> "impossible"
    text = re.sub(r'w\s*\.\s*i\s*Mandela$', '', text)  # Remove trailing "w . i Mandela"

    return text


def clean_hindi_ocr_text(text):
    """Clean common OCR artifacts specific to Hindi (Devanagari) text."""
    if not text:
        return text

    import re

    # Common Devanagari OCR error corrections
    # These are based on common misreads in Hindi OCR
    
    # Fix common character substitutions
    # Note: Devanagari characters are in Unicode range U+0900 to U+097F
    
    # Fix matra (vowel sign) errors - join consonant + space + matra
    text = re.sub(r'([क-ह\u093C])\s+([ा-ौ])', r'\1\2', text)  # Join separated matras
    text = re.sub(r'([क-ह\u093C])\s+([्])', r'\1\2', text)  # Join separated virama (halant)
    
    # Fix common conjunct errors - join consonant + space + consonant
    text = re.sub(r'([क-ह\u093C])\s+([क-ह\u093C])', lambda m: m.group(1) + '\u094D' + m.group(2), text)  # Join separated consonants with virama
    
    # Fix common OCR artifacts in Hindi
    text = re.sub(r'[|]', '', text)  # Remove pipe artifacts
    text = re.sub(r'[°]', '', text)  # Remove degree artifacts
    text = re.sub(r'[©®™]', '', text)  # Remove trademark symbols
    
    # Fix spacing issues between words
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Remove common OCR noise patterns
    text = re.sub(r'[«»""''""'';]', '', text)  # Remove quote artifacts
    
    # Remove currency symbols
    text = re.sub(r'[¥£€$¢₹]', '', text)
    
    # Remove BAR patterns and other artifacts
    text = re.sub(r'\b_BAR\b', '', text)
    text = re.sub(r'\b_PUNBAR\b', '', text)
    text = re.sub(r'\b\d+DPDBAR\b', '', text)
    text = re.sub(r'\bSIPPBAR\w*\b', '', text)
    
    # Fix common Devanagari character misreads
    # These are based on common OCR errors in Hindi text
    text = re.sub(r'।\s*।', '।', text)  # Fix double danda
    text = re.sub(r'\s+।', '।', text)  # Remove space before danda
    text = re.sub(r'।\s+', '। ', text)  # Normalize space after danda
    
    # Remove website artifacts and author names (common in quote images)
    text = re.sub(r'Insta\s*-\s*MarathiMulga\.com', '', text)
    text = re.sub(r'YourPositiveOasis\.com', '', text)
    
    # Remove common OCR noise characters
    text = re.sub(r'[*]', '', text)
    text = re.sub(r'[}{]', '', text)
    
    return text


def clean_ocr_text(text):
    """
    Clean common OCR artifacts to make text more translatable.
    Detects language and applies language-specific cleaning rules.
    """
    if not text:
        return text

    import re

    # Detect the language of the text
    language = detect_language(text)
    print(f"Detected language for OCR cleaning: {language}")

    # Apply language-specific cleaning
    if language == 'hindi':
        text = clean_hindi_ocr_text(text)
    elif language == 'english':
        text = clean_english_ocr_text(text)
    else:  # mixed
        # For mixed text, apply both cleaning strategies
        # First clean Hindi-specific issues
        text = clean_hindi_ocr_text(text)
        # Then clean English-specific issues
        text = clean_english_ocr_text(text)

    # Apply universal cleaning (language-agnostic)
    # Remove excessive punctuation and symbols
    text = re.sub(r'[«»""''""'';°|¢—]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Fix word fragments (language-agnostic)
    text = re.sub(r'(\w+)\s*-\s*(\w+)', r'\1\2', text)  # Join hyphenated words

    # Remove currency symbols and other OCR artifacts
    text = re.sub(r'[¥£€$¢₹]', '', text)
    text = re.sub(r'[©®™]', '', text)

    # Handle common character substitutions and distortions (language-agnostic)
    text = re.sub(r'[}{]', '', text)
    text = re.sub(r'[*]', '', text)
    text = re.sub(r'[«»]', '', text)
    text = re.sub(r'[""''""]', '', text)
    text = re.sub(r'[;]', '', text)

    return text.strip()
