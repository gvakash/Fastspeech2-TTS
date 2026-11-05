""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

[MODIFIED] The pinyin (Mandarin) symbols of ming024 are removed and the Telugu
phone set from text/telugu_frontend.py is added. Telugu phones are lowercase
ASCII ("@aa", "@tt", "@ny"...), so they never collide with the upper-case
ARPAbet symbols ("@AA1", "@T"...). Both models share this table; each model
simply never sees the other language's symbols (their embedding rows stay
untrained and unused).
"""

from text import cmudict
from text.telugu_frontend import valid_symbols as telugu_symbols

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to phone symbols to ensure uniqueness (some are the same as letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_telugu = ["@" + s for s in telugu_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _telugu
    + _silences
)
