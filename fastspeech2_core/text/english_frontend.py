"""English text front-end (NEW, replaces ming024's `preprocess_english`).

Why a new front-end?
  * ming024 called g2p_en word-by-word, so the POS tagger never saw sentence
    context and heteronyms ("conduct" noun vs. verb) always got one reading.
  * g2p_en's number handling breaks on times/dates: "10:45" -> "tenforty-five",
    "AM" -> "AE1 M" (the word "am"), "2026" -> "twenty twenty-six" (hyphen
    glued into one OOV token).

The SAME normaliser is used (a) to write the transcripts that the aligner sees,
(b) to build the aligner's pronunciation dictionary and (c) at inference, so
training and synthesis use identical phone sequences.
"""
import re
import unicodedata

import inflect
from unidecode import unidecode

_inflect = inflect.engine()

MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
_MONTH_RE = "|".join(m.capitalize() + "|" + m for m in MONTHS)

# Letter-name pronunciations (ARPAbet) used when an acronym is spelt out.
LETTER_PRONS = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1",
    "f": "EH1 F", "g": "JH IY1", "h": "EY1 CH", "i": "AY1", "j": "JH EY1",
    "k": "K EY1", "l": "EH1 L", "m": "EH1 M", "n": "EH1 N", "o": "OW1",
    "p": "P IY1", "q": "K Y UW1", "r": "AA1 R", "s": "EH1 S", "t": "T IY1",
    "u": "Y UW1", "v": "V IY1", "w": "D AH1 B AH0 L Y UW0", "x": "EH1 K S",
    "y": "W AY1", "z": "Z IY1",
}
LETTER_TOKEN = "{}_"  # "a_" marks "spell this letter" in the normalised text

ABBREVIATIONS = [
    (r"\bmrs\.", "missus"), (r"\bmr\.", "mister"), (r"\bdr\.", "doctor"),
    (r"\bst\.", "saint"), (r"\bjr\.", "junior"), (r"\bsr\.", "senior"),
    (r"\bvs\.?", "versus"), (r"\betc\.", "et cetera"), (r"\bi\.e\.", "that is"),
    (r"\be\.g\.", "for example"), (r"\bno\.(?=\s*\d)", "number"),
    (r"\bprof\.", "professor"),
]

PAUSE_PUNCT = {",", ";", ":", ".", "?", "!"}


def _num(n):
    """Cardinal number -> words without 'and' or hyphens."""
    return _inflect.number_to_words(int(n), andword="").replace("-", " ").replace(",", "")


def _ordinal(n):
    w = _inflect.number_to_words(_inflect.ordinal(int(n)), andword="")
    return w.replace("-", " ").replace(",", "")


def _year(n):
    n = int(n)
    if 2000 <= n <= 2009:
        return "two thousand" + ("" if n == 2000 else " " + _num(n % 100))
    if 1100 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return _num(hi) + " hundred"
        if lo < 10:
            return _num(hi) + " oh " + _num(lo)
        return _num(hi) + " " + _num(lo)
    return _num(n)


def _spell(word):
    return " ".join(LETTER_TOKEN.format(c) for c in word.lower() if c.isalpha())


def _time(m):
    h, mi, ampm = m.group(1), m.group(2), m.group(3)
    words = _num(h)
    if int(mi) == 0:
        words += " o'clock" if not ampm else ""
    elif int(mi) < 10:
        words += " oh " + _num(mi)
    else:
        words += " " + _num(mi)
    if ampm:
        words += " " + _spell(ampm.replace(".", ""))
    return " " + words + " "


def _date_dm(m):  # "12 November 2026", "12th November"
    day, month, year = m.group(1), m.group(2), m.group(3)
    out = _ordinal(day) + " " + month.lower()
    if year:
        out += " " + _year(year)
    return " " + out + " "


def _date_md(m):  # "November 12, 2026"
    month, day, year = m.group(1), m.group(2), m.group(3)
    out = month.lower() + " " + _ordinal(day)
    if year:
        out += " " + _year(year)
    return " " + out + " "


def normalize_english(text):
    """Raw text -> normalised text: lowercase words, letter tokens ('a_' = spell
    the letter), apostrophes, and pause punctuation (, ; : . ? !) as tokens."""
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    text = text.replace("_", " ")
    text = text.replace("&", " and ").replace("%", " percent ").replace("@", " at ")
    for pat, rep in ABBREVIATIONS:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # times: 10:45 AM / 10:45 a.m. / 7:05
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s?[Mm]\.?)?(?![\w])", _time, text)
    # dates
    text = re.sub(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTH_RE + r")\b,?(?:\s+(\d{4})\b)?",
        _date_dm, text)
    text = re.sub(
        r"\b(" + _MONTH_RE + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b,?(?:\s+(\d{4})\b)?",
        _date_md, text)
    # currency
    text = re.sub(r"\$\s?(\d+)", lambda m: " " + _num(m.group(1)) + " dollars ", text)
    text = re.sub(r"(?:\u20b9|\bRs\.?|\bINR)\s?(\d+)",
                  lambda m: " " + _num(m.group(1)) + " rupees ", text, flags=re.IGNORECASE)
    # remove thousands separators, then decimals / ordinals / years / cardinals
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"(\d+)\.(\d+)",
                  lambda m: " " + _num(m.group(1)) + " point " + " ".join(_num(d) for d in m.group(2)) + " ",
                  text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: " " + _ordinal(m.group(1)) + " ", text)
    text = re.sub(r"\b(1[1-9]\d0|20\d0)'?s\b",  # decades: 1990s -> nineteen nineties
                  lambda m: " " + re.sub(r"y$", "ies", _year(m.group(1))) + " ", text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: " " + _year(m.group(1)) + " ", text)
    text = re.sub(r"\d+", lambda m: " " + _num(m.group(0)) + " ", text)

    # acronyms written in capitals (AM, UPI, IIT) are spelt letter by letter;
    # the pronoun "I" and the article "A" are left alone.
    text = re.sub(r"\b([A-Z]{2,5})s?\b", lambda m: " " + _spell(m.group(1)) + " ", text)
    text = text.lower()

    text = re.sub(r"[\"()\[\]{}<>*/\\|~^`=+#]", " ", text)
    text = re.sub(r"[-]+", " ", text)  # hyphenated compounds -> separate words
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"([,;:.?!])", r" \1 ", text)  # pause punctuation -> own token
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)  # quotes, not apostrophes
    text = re.sub(r"[^a-z' .,;:?!_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(norm_text):
    """Normalised text -> list of tokens (words, letter tokens, punctuation)."""
    toks = []
    for t in norm_text.split():
        if re.fullmatch(r"[a-z]_", t):
            toks.append(t)
        elif t in PAUSE_PUNCT:
            toks.append(t)
        else:
            t = t.strip("'")
            if t:
                toks.append(t)
    return toks


def words_for_aligner(norm_text):
    """Word sequence written to the .lab files (letter tokens lose the dot)."""
    return [t.rstrip("_") for t in tokenize(norm_text) if t not in PAUSE_PUNCT]


class EnglishG2P:
    """Sentence-level G2P built on g2p_en's lexicon, homograph table and neural model."""

    def __init__(self):
        from g2p_en import G2p
        self.g2p = G2p()
        self.cmu = self.g2p.cmu
        self.homographs = self.g2p.homograph2features

    def word_prons(self, word):
        """All candidate pronunciations of a word (used for the MFA dictionary)."""
        prons = []
        if word in self.cmu:
            prons += [" ".join(p) for p in self.cmu[word]]
        if word in self.homographs:
            p1, p2, _ = self.homographs[word]
            prons += [" ".join(p1), " ".join(p2)]
        if len(word) == 1 and word in LETTER_PRONS:
            prons.append(LETTER_PRONS[word])
        if not prons:
            prons.append(" ".join(self.g2p.predict(word)))
        out = []
        for p in prons:
            if p not in out:
                out.append(p)
        return out

    def __call__(self, raw_text, return_words=False):
        """Raw sentence -> (phones, word_spans). Punctuation becomes 'sp'."""
        from nltk import pos_tag

        tokens = tokenize(normalize_english(raw_text))
        while tokens and tokens[-1] in PAUSE_PUNCT:  # drop sentence-final punctuation
            tokens.pop()
        word_toks = [t for t in tokens if t not in PAUSE_PUNCT]
        tags = dict()
        if word_toks:
            tagged = pos_tag([w.rstrip("_") for w in word_toks])
            tags = {i: tag for i, (_, tag) in enumerate(tagged)}

        phones, spans, wi = [], [], 0
        for tok in tokens:
            if tok in PAUSE_PUNCT:
                if phones and phones[-1] != "sp":
                    phones.append("sp")
                    spans.append(("<pause>", len(phones) - 1, len(phones)))
                continue
            pos = tags.get(wi, "NN")
            wi += 1
            if re.fullmatch(r"[a-z]_", tok):
                pron = LETTER_PRONS[tok[0]].split()
                label = tok[0].upper()
            elif tok in self.homographs:
                p1, p2, pos1 = self.homographs[tok]
                pron = p1 if pos.startswith(pos1) else p2
                label = tok
            elif tok in self.cmu:
                pron = self.cmu[tok][0]
                label = tok
            else:
                pron = self.g2p.predict(tok)
                label = tok
            start = len(phones)
            phones += list(pron)
            spans.append((label, start, len(phones)))
        if return_words:
            return phones, spans
        return phones


# ---------------------------------------------------------------------------
# helpers used by data preparation
# ---------------------------------------------------------------------------
def _check_transcript_en(raw_text):
    if re.search(r"[\[\]<>{}]", raw_text):
        return False, None, "annotation"
    if re.search(r"[^\x00-\x7F‘’“”–—₹]", raw_text):
        return False, None, "non_ascii"
    norm = normalize_english(raw_text)
    words = words_for_aligner(norm)
    if len(words) < 2:
        return False, None, "too_short"
    return True, norm, ""


EnglishG2P.check_transcript = staticmethod(_check_transcript_en)
EnglishG2P.words_for_aligner = staticmethod(words_for_aligner)
EnglishG2P.normalize = staticmethod(normalize_english)
