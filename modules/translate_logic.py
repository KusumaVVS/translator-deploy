import csv
import os
import re
import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*layers were not sharded.*")

from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer
# ─── Dataset ──────────────────────────────────────────────────────────────────

en_to_hi = {}
hi_to_en = {}
en_to_te = {}
te_to_en = {}

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False

def load_dataset():
    global en_to_hi, hi_to_en
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(base_dir, '..', 'data', 'Dataset_English_Hindi.csv'))
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    en = re.sub(r'[^\w\s]', '', row[0]).strip().lower()
                    hi = re.sub(r'[^\w\s]', '', ','.join(row[1:])).strip()
                    if en and hi:
                        en_to_hi[en] = hi
                        hi_to_en[hi] = en

load_dataset()

# ─── Hindi pipelines (Helsinki + M2M100 418M) ─────────────────────────────────

translation_pipelines = {
    'en-hi': pipeline('translation_en_to_hi', model='Helsinki-NLP/opus-mt-en-hi'),
    'hi-en': pipeline('translation_hi_to_en', model='Helsinki-NLP/opus-mt-hi-en'),
}


# Lazy loading via get_translation_pipeline() handles all directions - no pre-initialization needed

# ─── M2M100 Pipeline Factory ─────────────────────────────────────────────────

def get_translation_pipeline(direction):
    """Return cached translation pipeline (NLLB-200 for Telugu, M2M100 for Hindi)."""
    if direction in translation_pipelines:
        return translation_pipelines[direction]
    
    src, tgt = direction.split('-')
    

    if direction in ('en-te', 'te-en'):
        # NLLB-200 for superior Telugu translation
        src_lang = 'eng_Latn' if src == 'en' else 'tel_Telu'
        tgt_lang = 'tel_Telu' if tgt == 'te' else 'eng_Latn'
        model_name = 'facebook/nllb-200-distilled-600M'

    elif direction in ('en-hi', 'hi-en'):
        # Keep existing M2M100 for Hindi (per user instruction)
        src_lang = src
        tgt_lang = tgt
        model_name = 'facebook/m2m100_418M'
    else:
        print(f"Unsupported direction: {direction}")
        return None
    
    try:
        pipe = pipeline(
            'translation',
            model=model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        translation_pipelines[direction] = pipe
        print(f"Loaded {model_name} pipeline for {direction}")
        return pipe
    except Exception as e:
        print(f"Pipeline creation failed for {direction}: {e}")
        return None

# ─── Single-word

SINGLE_WORD_TRANSLATIONS = {
    'NOW': 'अब', 'HOW': 'कैसे', 'WHAT': 'क्या', 'WHEN': 'कब',
    'WHERE': 'कहाँ', 'WHY': 'क्यों', 'WHO': 'कौन',
}

TELUGU_SINGLE_WORD_TRANSLATIONS = {
    'NOW': 'ఇప్పుడు', 'HOW': 'ఎలా', 'WHAT': 'ఏమిటి', 'WHEN': 'ఎప్పుడు',
    'WHERE': 'ఎక్కడ', 'WHY': 'ఎందుకు', 'WHO': 'ఎవరు', 'YES': 'అవును',
    'NO': 'కాదు', 'HELLO': 'నమస్కారం', 'PLEASE': 'దయచేసి',
    'SORRY': 'క్షమించండి', 'GOOD': 'మంచిది', 'BAD': 'చెడ్డది',
    'WATER': 'నీరు', 'FOOD': 'ఆహారం', 'LOVE': 'ప్రేమ',
    'FRIEND': 'స్నేహితుడు', 'FAMILY': 'కుటుంబం', 'HOME': 'ఇల్లు',
    'WORK': 'పని', 'TODAY': 'ఈరోజు', 'TOMORROW': 'రేపు', 'YESTERDAY': 'నిన్న',
}

TELUGU_PHRASE_MAPPINGS = {
    'hello': 'నమస్కారం', 'hi': 'నమస్కారం',
    'good morning': 'శుభోదయం', 'good evening': 'శుభ సాయంత్రం',
    'good night': 'శుభ రాత్రి', 'goodbye': 'వెళ్ళిపోతున్నాను',
    'thank you': 'ధన్యవాదాలు', 'thanks': 'ధన్యవాదాలు',
    'please': 'దయచేసి', 'sorry': 'క్షమించండి', 'excuse me': 'క్షమించండి',
    'how are you': 'మీరు ఎలా ఉన్నారు', 'i am fine': 'నేను బాగున్నాను',
    'what is your name': 'మీ పేరు ఏమిటి', 'my name is': 'నా పేరు',
    'i love you': 'నేను నిన్ను ప్రేమిస్తున్నాను',
    'what is this': 'ఇది ఏమిటి', 'where is': 'ఎక్కడ ఉంది',
    'how much': 'ఎంత', 'how many': 'ఎన్ని',
    'yes': 'అవును', 'no': 'కాదు', 'okay': 'సరే', 'ok': 'సరే',
}

# ─── Text normalisation / pre / post processing ───────────────────────────────

def normalize_text(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    text = re.sub(r'[.,!?;:]+$', '', text).strip()
    return re.sub(r'\s+', ' ', text)


def preprocess_text(text: str, direction: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if direction in ('en-hi', 'en-te'):
        abbrevs = {
            r"\b[uU]\b": "you", r"\b[uU][rR]\b": "your", r"\b[rR]\b": "are",
            r"\b[pP][lL][sS]\b": "please", r"\b[pP][lL][zZ]\b": "please",
            r"\b[tT][hH][xX]\b": "thanks", r"\b[bB][tT][wW]\b": "by the way",
            r"\b[iI][dD][kK]\b": "I don't know",
            r"\b[oO][mM][gG]\b": "oh my god",
            r"\b[aA][sS][aA][pP]\b": "as soon as possible",
        }
        for pat, rep in abbrevs.items():
            text = re.sub(pat, rep, text)
    return re.sub(r'\s+', ' ', text).strip()


def postprocess_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text).strip()
    for pat in (r'_+\s*$', r'\b_BAR\b', r'\b_PUNBAR\b',
                r'\b\d+DPDBAR\b', r'\bSIPPBAR\w*\b'):
        text = re.sub(pat, '', text)
    text = re.sub(r'_+', ' ', text).strip()
    # Suppress runaway word repetition
    words = text.split()
    if len(words) > 4:
        counts = {}
        for w in words:
            cw = re.sub(r'[^\w\u0900-\u097F\u0C00-\u0C7F]', '', w)
            counts[cw] = counts.get(cw, 0) + 1
        if counts and max(counts.values()) > 3:
            most = max(counts, key=counts.get)
            filtered, seen = [], 0
            for w in words:
                cw = re.sub(r'[^\w\u0900-\u097F\u0C00-\u0C7F]', '', w)
                if cw == most:
                    if seen < 2:
                        filtered.append(w); seen += 1
                else:
                    filtered.append(w)
            text = ' '.join(filtered)
    return text

# ─── Transliteration helpers — Hindi ──────────────────────────────────────────

def simple_en_to_hi_transliterate(text: str) -> str:
    if not text:
        return text
    word_map = {
        'anits': 'एनिट्स', 'mit': 'एमआईटी', 'iit': 'आईआईटी',
        'google': 'गूगल', 'microsoft': 'माइक्रोसॉफ्ट', 'apple': 'ऐपल',
        'amazon': 'अमेज़न', 'delhi': 'दिल्ली', 'mumbai': 'मुंबई',
        'hyderabad': 'हैदराबाद', 'gandhi': 'गांधी', 'mahatma': 'महात्मा',
        'python': 'पाइथन', 'john': 'जॉन',
    }
    tl = text.lower().strip()
    if tl in word_map:
        return word_map[tl]
    cv = {
        'sha':'शा','shi':'शी','shu':'शु','she':'शे','sho':'शो',
        'cha':'चा','chi':'ची','chu':'चु','che':'चे','cho':'चो',
        'tha':'था','thi':'थी','thu':'थु','the':'थे','tho':'थो',
        'dha':'धा','dhi':'धी','dhu':'धु','dhe':'धे','dho':'धो',
        'ka':'का','ki':'की','ku':'कु','ke':'के','ko':'को',
        'ga':'गा','gi':'गी','gu':'गु','ge':'गे','go':'गो',
        'ta':'ता','ti':'ती','tu':'तु','te':'ते','to':'तो',
        'da':'दा','di':'दी','du':'दु','de':'दे','do':'दो',
        'na':'ना','ni':'नी','nu':'नु','ne':'ने','no':'नो',
        'pa':'पा','pi':'पी','pu':'पु','pe':'पे','po':'पो',
        'ba':'बा','bi':'बी','bu':'बु','be':'बे','bo':'बो',
        'ma':'मा','mi':'मी','mu':'मु','me':'मे','mo':'मो',
        'ra':'रा','ri':'री','ru':'रु','re':'रे','ro':'रो',
        'la':'ला','li':'ली','lu':'लु','le':'ले','lo':'लो',
        'sa':'सा','si':'सी','su':'सु','se':'से','so':'सो',
        'ha':'हा','hi':'ही','hu':'हु','he':'हे','ho':'हो',
        'ya':'या','yu':'यु','ye':'ये','yo':'यो',
        'va':'वा','vi':'वी','vu':'वु','ve':'वे','vo':'वो',
        'ja':'जा','ji':'जी','ju':'जु','je':'जे','jo':'जो',
    }
    single = {
        'a':'अ','i':'इ','u':'उ','e':'ए','o':'ओ',
        'k':'क','g':'ग','t':'त','d':'द','n':'न',
        'p':'प','b':'ब','m':'म','y':'य','r':'र',
        'l':'ल','v':'व','s':'स','h':'ह','j':'ज',
        'f':'फ','z':'ज़','w':'व','c':'क','q':'क','x':'क्स',
    }
    result, i = '', 0
    while i < len(tl):
        matched = False
        for length in (3, 2):
            chunk = tl[i:i+length]
            if chunk in cv:
                result += cv[chunk]; i += length; matched = True; break
        if not matched:
            result += single.get(tl[i], tl[i])
            i += 1
    return result.strip() or text


def transliterate_en_to_hi(text: str) -> str:
    if not text:
        return text
    result = simple_en_to_hi_transliterate(text)
    if result != text:
        return result
    if TRANSLITERATION_AVAILABLE:
        try:
            r = transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
            if r and any('\u0900' <= c <= '\u097F' for c in r):
                return r
        except Exception:
            pass
    return simple_en_to_hi_transliterate(text)


def transliterate_hi_to_en(text: str) -> str:
    if not text:
        return text
    if TRANSLITERATION_AVAILABLE:
        try:
            r = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
            if r and any(c.isascii() and c.isalpha() for c in r):
                return r
        except Exception:
            pass
    simple_map = {
        'अ':'a','आ':'aa','इ':'i','ई':'ee','उ':'u','ऊ':'oo','ए':'e','ऐ':'ai',
        'ओ':'o','औ':'au','क':'k','ख':'kh','ग':'g','घ':'gh','च':'ch','छ':'chh',
        'ज':'j','झ':'jh','ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t',
        'थ':'th','द':'d','ध':'dh','न':'n','प':'p','फ':'ph','ब':'b','भ':'bh',
        'म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh','ष':'sh','स':'s','ह':'h',
    }
    return ''.join(simple_map.get(c, c) for c in text).strip() or text

# ─── Transliteration helpers — Telugu ─────────────────────────────────────────
# These are FALLBACKS only. Real translation uses _translate_telugu() above.

def transliterate_en_to_te(text: str) -> str:
    """Phonetic English→Telugu fallback (used only when M2M100 model unavailable)."""
    if not text:
        return text
    word_map = {
        'hyderabad': 'హైదరాబాద్', 'vijayawada': 'విజయవాడ',
        'tirupati': 'తిరుపతి', 'warangal': 'వరంగల్',
        'guntur': 'గుంటూరు', 'google': 'గూగుల్',
        'microsoft': 'మైక్రోసాఫ్ట్', 'amazon': 'అమెజాన్',
        'anits': 'ఏనిట్స్', 'hello': 'నమస్కారం',
        'thank you': 'ధన్యవాదాలు',
    }
    tl = text.lower().strip()
    if tl in word_map:
        return word_map[tl]
    cv = {
        'sha':'శ','shi':'శి','shu':'శు','she':'శె','sho':'శొ',
        'cha':'చ','chi':'చి','chu':'చు','che':'చె','cho':'చొ',
        'ka':'క','ki':'కి','ku':'కు','ke':'కె','ko':'కొ',
        'ga':'గ','gi':'గి','gu':'గు','ge':'గె','go':'గొ',
        'ta':'ట','ti':'టి','tu':'టు','te':'టె','to':'టొ',
        'da':'డ','di':'డి','du':'డు','de':'డె','do':'డొ',
        'na':'న','ni':'ని','nu':'ను','ne':'నె','no':'నొ',
        'pa':'ప','pi':'పి','pu':'పు','pe':'పె','po':'పొ',
        'ba':'బ','bi':'బి','bu':'బు','be':'బె','bo':'బొ',
        'ma':'మ','mi':'మి','mu':'ము','me':'మె','mo':'మొ',
        'ra':'ర','ri':'రి','ru':'రు','re':'రె','ro':'రొ',
        'la':'ల','li':'లి','lu':'లు','le':'లె','lo':'లొ',
        'sa':'స','si':'సి','su':'సు','se':'సె','so':'సొ',
        'ha':'హ','hi':'హి','hu':'హు','he':'హె','ho':'హొ',
        'ya':'య','yu':'యు','ye':'యె','yo':'యొ',
        'va':'వ','vi':'వి','vu':'వు','ve':'వె','vo':'వొ',
        'ja':'జ','ji':'జి','ju':'జు','je':'జె','jo':'జొ',
    }
    single = {
        'a':'అ','i':'ఇ','u':'ఉ','e':'ఎ','o':'ఒ',
        'k':'క్','g':'గ్','t':'ట్','d':'డ్','n':'న్',
        'p':'ప్','b':'బ్','m':'మ్','y':'య్','r':'ర్',
        'l':'ల్','v':'వ్','s':'స్','h':'హ్','f':'ఫ్','z':'జ్',
        'j':'జ్','w':'వ్','c':'క్','q':'క్',
    }
    result, i = '', 0
    while i < len(tl):
        matched = False
        for length in (3, 2):
            chunk = tl[i:i+length]
            if chunk in cv:
                result += cv[chunk]; i += length; matched = True; break
        if not matched:
            result += single.get(tl[i], tl[i])
            i += 1
    return result.strip() or text


def transliterate_te_to_en(text: str) -> str:
    """Phonetic Telugu→English fallback (used only when M2M100 model unavailable)."""
    if not text:
        return text
    te_map = {
        'అ':'a','ఆ':'aa','ఇ':'i','ఈ':'ee','ఉ':'u','ఊ':'oo',
        'ఎ':'e','ఏ':'ee','ఐ':'ai','ఒ':'o','ఓ':'oo','ఔ':'au',
        'క':'k','ఖ':'kh','గ':'g','ఘ':'gh','ఙ':'ng',
        'చ':'ch','ఛ':'chh','జ':'j','ఝ':'jh','ఞ':'ny',
        'ట':'t','ఠ':'th','డ':'d','ఢ':'dh','ణ':'n',
        'త':'t','థ':'th','ద':'d','ధ':'dh','న':'n',
        'ప':'p','ఫ':'ph','బ':'b','భ':'bh','మ':'m',
        'య':'y','ర':'r','ల':'l','వ':'v',
        'శ':'sh','ష':'sh','స':'s','హ':'h','ళ':'l','ఱ':'r',
        'ా':'aa','ి':'i','ీ':'ee','ు':'u','ూ':'oo',
        'ె':'e','ే':'ee','ై':'ai','ొ':'o','ో':'oo','ౌ':'au',
        'ం':'m','ః':'h','్':'',
    }
    return ''.join(te_map.get(c, ' ' if c.isspace() else c) for c in text).strip() or text

# ─── Proper noun masking ───────────────────────────────────────────────────────

_KNOWN_PROPER_NOUNS = {
    'anits','mit','iit','nit','iisc','iim','google','microsoft','apple',
    'samsung','tesla','amazon','harvard','delhi','mumbai','bangalore',
    'john','pizza','dominos','eiffel','wembley','taj','python','kusuma','gandhi',
}
_COMMON_WORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of',
    'with','by','from','is','am','are','was','were','i','he','she',
    'it','we','they','this','that','these','those',
}


def identify_proper_nouns_and_acronyms(text: str):
    words = text.split()
    nouns, result_words = {}, []
    for idx, word in enumerate(words):
        clean = re.sub(r'[^\w]', '', word)
        is_pn = False
        if clean.isupper() and len(clean) > 1:
            is_pn = True
        elif idx > 0 and clean and clean[0].isupper() and len(clean) > 1:
            if clean.lower() not in {"i", "i'm", "i'll", "i've", "i'd"}:
                is_pn = True
        elif clean.lower() in _KNOWN_PROPER_NOUNS:
            is_pn = True
        if clean.lower() in _COMMON_WORDS:
            is_pn = False
        if is_pn:
            ph = f"__PN{chr(65 + len(nouns))}__"
            nouns[ph] = clean
            result_words.append(ph)
        else:
            result_words.append(word)
    return ' '.join(result_words), nouns


def _restore_proper_nouns(result: str, nouns_map: dict, direction: str) -> str:
    for ph, word in nouns_map.items():
        if direction == 'en-te':
            tr = transliterate_en_to_te(word.lower())
        else:
            tr = transliterate_en_to_hi(word.lower())
        for variant in [ph, ph.lower(), ph.upper(),
                        ph.replace('__', '_'), ph.replace('_', '')]:
            result = re.sub(re.escape(variant), tr, result, flags=re.IGNORECASE)
    return result

# ─── Hindi phrase dictionary ───────────────────────────────────────────────────

_HINDI_PHRASES = {
    'संघर्ष प्रगति का आमंत्रण है जो इसे स्वीकारता है उसका जीवन निखर जाता है':
        'Struggle is an invitation to progress; the life of one who accepts it becomes bright.',
    'संघर्ष प्रगति का आमंत्रण है':
        'Struggle is an invitation to progress.',
    'मनुष्य जीवन में संघर्ष आवश्यक है':
        'Struggle is essential in human life.',
}

# ─── Public API ────────────────────────────────────────────────────────────────

def translate(text: str, direction: str, model_name: str = 'M2M100') -> str:
    """
    Translate between English, Hindi, and Telugu.

    direction values:
        'en-hi'  English  -> Hindi
        'hi-en'  Hindi    -> English
        'en-te'  English  -> Telugu   (uses M2M100 1.2B direct API)
        'te-en'  Telugu   -> English  (uses M2M100 1.2B direct API)
    """
    text = normalize_text(text)
    if not text:
        return ""

    # Single-word shortcuts
    upper = text.upper()
    if direction == 'en-hi' and upper in SINGLE_WORD_TRANSLATIONS:
        return SINGLE_WORD_TRANSLATIONS[upper]
    if direction == 'en-te' and upper in TELUGU_SINGLE_WORD_TRANSLATIONS:
        return TELUGU_SINGLE_WORD_TRANSLATIONS[upper]

    # Telugu phrase table
    if direction == 'en-te':
        mapped = TELUGU_PHRASE_MAPPINGS.get(text.lower().strip())
        if mapped:
            return mapped

    return _translate_core(text, direction, model_name)


def _translate_core(text: str, direction: str, model_name: str = 'M2M100') -> str:

    # Mask proper nouns before sending to model
    nouns_map = {}
    if direction in ('en-hi', 'en-te'):
        text, nouns_map = identify_proper_nouns_and_acronyms(text)

    text = preprocess_text(text, direction)
    if not text:
        return ""

    input_text = normalize_text(text)
    if not input_text:
        return ""

    # ── Hindi phrase shortcut (hi-en) ────────────────────────────────────────
    if direction == 'hi-en':
        clean = re.sub(r'[^\u0900-\u097F\s]', '', input_text).strip()
        for phrase, translation in _HINDI_PHRASES.items():
            if phrase in input_text or phrase in clean:
                return translation

    # ════════════════════════════════════════════════════════════════════════
    #  TELUGU — M2M100 Pipeline (same as Hindi)
    # ════════════════════════════════════════════════════════════════════════

    if direction in ('en-te', 'te-en'):
        try:
            pipe = get_translation_pipeline(direction)
            
            if pipe is None:
                return "Telugu pipeline unavailable"
            
            raw = pipe(input_text, max_length=512, truncation=True)
            
            if not raw:
                return "Translation failed: empty output"
            
            result = raw[0]['translation_text'].strip()
            
            if nouns_map:
                result = _restore_proper_nouns(result, nouns_map, direction)
            
            return postprocess_text(result)
            
        except Exception as e:
            print(f"{direction} pipeline error: {e}")
            
            if direction == 'en-te':
                return postprocess_text(transliterate_en_to_te(input_text))
            else:
                return postprocess_text(transliterate_te_to_en(input_text))

    # ════════════════════════════════════════════════════════════════════════
    #  HINDI — M2M100 418M pipeline
    # ════════════════════════════════════════════════════════════════════════

    try:
        pipe = get_translation_pipeline(direction)

        if pipe is None:
            return f"Translation failed: No pipeline available for '{direction}'"

        raw = pipe(input_text, max_length=512, truncation=True)
        if not raw:
            return "Translation failed: empty model output"

        first = raw[0]
        result = first.get('translation_text', '') if isinstance(first, dict) else str(first)
        result = result.strip()
        result = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', result)
        result = re.sub(r'[½¼¾⅓⅔⅛⅜⅝⅞]', '', result).strip()

        if not result:
            return "Translation failed: empty result after cleaning"

        # Hallucination check — fallback to Helsinki if needed
        bad_words = {'death','ecstasy','stimulant','nuclear','council',
                     'household','susceptible','effects','unclear'}
        if sum(1 for w in bad_words if w in result.lower()) >= 3:
            alt_pipe = translation_pipelines.get(direction)
            if alt_pipe:
                try:
                    alt_raw = alt_pipe(input_text, max_length=512, truncation=True)
                    if alt_raw:
                        alt = alt_raw[0]
                        alt_result = alt.get('translation_text', '') if isinstance(alt, dict) else str(alt)
                        if alt_result.strip():
                            result = alt_result.strip()
                except Exception:
                    pass

        # Restore proper nouns
        if nouns_map:
            result = _restore_proper_nouns(result, nouns_map, direction)

        # Transliterate any leftover untranslated English words in Hindi output
        if direction == 'en-hi':
            out_words = []
            for w in result.split():
                cw = re.sub(r'[^\w]', '', w)
                if cw and re.match(r'^[a-zA-Z]+$', cw):
                    out_words.append(transliterate_en_to_hi(cw.lower()))
                else:
                    out_words.append(w)
            result = ' '.join(out_words)

        return postprocess_text(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Translation failed: {e}"