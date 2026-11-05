"""Telugu text front-end (NEW - this is the main "Telugu support" addition).

ming024 only supports English (ARPAbet via g2p_en) and Mandarin (pinyin).
Telugu is written in an abugida whose spelling is close to phonemic, so a
rule-based grapheme-to-phoneme converter is accurate enough for TTS:

  * every consonant letter carries an inherent /a/ unless it is followed by a
    vowel sign (matra) or by the virama (్), which suppresses the vowel;
  * conjuncts (e.g. ప్ర, క్ష, ద్ధ) therefore fall out naturally as C + ్ + C;
  * the anusvara (ం) is realised as a nasal homorganic with the next consonant
    (ంగ -> ng g, ంచ -> ny ch, ండ -> nn dd, ంద -> n d, ంబ -> m b) and as /m/
    word-finally (సమావేశం -> ... sh a m);
  * visarga (ః) -> /h/; arasunna (ఁ) is silent in modern Telugu.

Numbers are expanded into Telugu words (cardinals in the Indian lakh/crore
system, ordinals with the -వ suffix, clock times), because the training data
contains no digits and the model has no symbol for them.

Phones are written in lowercase ASCII so they never collide with the
upper-case ARPAbet symbols used by the English model.
"""
import re
import unicodedata

# ----------------------------------------------------------------------------
# Phone inventory
# ----------------------------------------------------------------------------
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "ll", "ఴ": "ll",
    "వ": "v", "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ౘ": "ch", "ౙ": "j", "ౚ": "r",
}
NUKTA_MAP = {"ph": "f", "j": "z"}  # ఫ఼ -> f, జ఼ -> z (loanwords)

INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}
VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
CANDRABINDU = "ఁ"
NUKTA = "఼"
AI_LENGTH_MARK = "ౖ"
IGNORED = {"‌", "‍", "ఀ", "ఄ", "ఽ", "ౕ"}

VOWELS = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
PHONES = sorted(set(VOWELS) | set(CONSONANTS.values()) | {"f", "z", "ng", "ny", "nn", "n", "m"})
valid_symbols = PHONES  # used by text/symbols.py

_PLACE = {}
for p in ["k", "kh", "g", "gh"]:
    _PLACE[p] = "ng"
for p in ["ch", "chh", "j", "jh"]:
    _PLACE[p] = "ny"
for p in ["tt", "tth", "dd", "ddh", "nn"]:
    _PLACE[p] = "nn"
for p in ["t", "th", "d", "dh", "n"]:
    _PLACE[p] = "n"
for p in ["p", "ph", "b", "bh", "m", "f"]:
    _PLACE[p] = "m"

TELUGU_DIGITS = {chr(0x0C66 + i): str(i) for i in range(10)}
PAUSE_PUNCT = {",", ";", ":", ".", "?", "!"}


def word_to_phones(word):
    """Grapheme-to-phoneme conversion for a single Telugu word."""
    word = unicodedata.normalize("NFC", word)
    phones = []
    pending = False  # a consonant waiting for its vowel
    anusvara_at = []  # indices of placeholder nasals to resolve
    for ch in word:
        if ch in IGNORED:
            continue
        if ch in CONSONANTS:
            if pending:
                phones.append("a")
            phones.append(CONSONANTS[ch])
            pending = True
        elif ch == NUKTA:
            if phones and phones[-1] in NUKTA_MAP:
                phones[-1] = NUKTA_MAP[phones[-1]]
        elif ch in VOWEL_SIGNS:
            phones += VOWEL_SIGNS[ch]
            pending = False
        elif ch == AI_LENGTH_MARK:  # decomposed ై (ె + ౖ)
            if phones and phones[-1] == "e":
                phones[-1] = "ai"
        elif ch == VIRAMA:
            pending = False
        elif ch in INDEPENDENT_VOWELS:
            if pending:
                phones.append("a")
            phones += INDEPENDENT_VOWELS[ch]
            pending = False
        elif ch == ANUSVARA:
            if pending:
                phones.append("a")
                pending = False
            anusvara_at.append(len(phones))
            phones.append("<N>")
        elif ch == VISARGA:
            if pending:
                phones.append("a")
                pending = False
            phones.append("h")
        elif ch == CANDRABINDU:
            if pending:
                phones.append("a")
                pending = False
        else:
            # non-Telugu character inside a word: ignore it
            continue
    if pending:
        phones.append("a")
    for i in anusvara_at:
        nxt = phones[i + 1] if i + 1 < len(phones) else None
        phones[i] = _PLACE.get(nxt, "m")
    return phones


# ----------------------------------------------------------------------------
# Number normalisation
# ----------------------------------------------------------------------------
_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు",
          "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు",
          "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై",
         7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def _below100(n):
    if n < 20:
        return _UNITS[n]
    t, u = divmod(n, 10)
    return _TENS[t] + ("" if u == 0 else " " + _UNITS[u])


def _below1000(n):
    if n < 100:
        return _below100(n)
    h, r = divmod(n, 100)
    if h == 1:
        return "వంద" if r == 0 else "నూట " + _below100(r)
    return _UNITS[h] + (" వందలు" if r == 0 else " వందల " + _below100(r))


def telugu_number(n):
    """Cardinal number -> Telugu words (Indian numbering: వేలు, లక్షలు, కోట్లు)."""
    n = int(n)
    if n < 1000:
        return _below1000(n)
    if n < 100000:
        t, r = divmod(n, 1000)
        if t == 1:
            head = "వెయ్యి" if r == 0 else "వెయ్యి"
        else:
            head = _below100(t) + (" వేలు" if r == 0 else " వేల")
        return head if r == 0 else head + " " + _below1000(r)
    if n < 10000000:
        l, r = divmod(n, 100000)
        head = "ఒక లక్ష" if l == 1 else _below100(l) + (" లక్షలు" if r == 0 else " లక్షల")
        return head if r == 0 else head + " " + telugu_number(r)
    if n < 10 ** 9:
        c, r = divmod(n, 10000000)
        head = "ఒక కోటి" if c == 1 else _below100(c) + (" కోట్లు" if r == 0 else " కోట్ల")
        return head if r == 0 else head + " " + telugu_number(r)
    return " ".join(_UNITS[int(d)] for d in str(n))  # read very long numbers digit by digit


def telugu_ordinal_stem(n):
    """12 -> పన్నెండ (so that 12వ -> పన్నెండవ), 20 -> ఇరవయ్య (ఇరవయ్యవ)."""
    w = telugu_number(n)
    if w.endswith("ై"):
        return w[:-1] + "య్య"
    if w.endswith("ు") or w.endswith("ి"):
        return w[:-1]
    return w


def _time_te(m):
    h, mi = int(m.group(1)), int(m.group(2))
    out = telugu_number(h)
    if mi:
        out += " " + telugu_number(mi)
    return " " + out + " "


def normalize_telugu(text):
    """Raw Telugu text -> normalised text (Telugu words + pause punctuation)."""
    text = unicodedata.normalize("NFC", text)
    for k, v in TELUGU_DIGITS.items():
        text = text.replace(k, v)
    text = text.replace("।", ".").replace("॥", ".")
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time_te, text)
    text = re.sub(r"(\d+)\.(\d+)",
                  lambda m: " " + telugu_number(m.group(1)) + " పాయింట్ "
                  + " ".join(_UNITS[int(d)] for d in m.group(2)) + " ", text)
    text = text.replace("%", " శాతం ")
    # ordinals: 12వ, 12 వ, 12వది ...
    text = re.sub(r"(\d+)\s*వ", lambda m: " " + telugu_ordinal_stem(m.group(1)) + "వ", text)
    text = re.sub(r"\d+", lambda m: " " + telugu_number(m.group(0)) + " ", text)
    text = re.sub(r"[\"“”'‘’()\[\]{}<>*_/\\|~^`=+#&@-]", " ", text)
    text = re.sub(r"([,;:.?!])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(norm_text):
    return [t for t in norm_text.split() if t]


def has_latin(text):
    return re.search(r"[A-Za-z]", text) is not None


def words_for_aligner(norm_text):
    return [t for t in tokenize(norm_text) if t not in PAUSE_PUNCT]


class TeluguG2P:
    """Sentence -> phone list, with 'sp' for pause punctuation."""

    def word_prons(self, word):
        return [" ".join(word_to_phones(word))]

    def __call__(self, raw_text, return_words=False):
        tokens = tokenize(normalize_telugu(raw_text))
        while tokens and tokens[-1] in PAUSE_PUNCT:
            tokens.pop()
        phones, spans = [], []
        for tok in tokens:
            if tok in PAUSE_PUNCT:
                if phones and phones[-1] != "sp":
                    phones.append("sp")
                    spans.append(("<pause>", len(phones) - 1, len(phones)))
                continue
            p = word_to_phones(tok)
            if not p:
                continue
            start = len(phones)
            phones += p
            spans.append((tok, start, len(phones)))
        if return_words:
            return phones, spans
        return phones


# ---------------------------------------------------------------------------
# helpers used by data preparation
# ---------------------------------------------------------------------------
_ALLOWED_TE = re.compile(r"^[ఀ-౿‌‍0-9\s,;:.?!।'\"‘’“”()\-%]+$")


def _check_transcript_te(raw_text):
    if has_latin(raw_text):
        return False, None, "latin_letters"
    if not _ALLOWED_TE.match(unicodedata.normalize("NFC", raw_text)):
        return False, None, "unexpected_chars"
    norm = normalize_telugu(raw_text)
    words = words_for_aligner(norm)
    if len(words) < 1 or not any(word_to_phones(w) for w in words):
        return False, None, "too_short"
    return True, norm, ""


TeluguG2P.check_transcript = staticmethod(_check_transcript_te)
TeluguG2P.words_for_aligner = staticmethod(words_for_aligner)
TeluguG2P.normalize = staticmethod(normalize_telugu)
