"""Language front-end factory (NEW)."""


def get_frontend(language):
    language = language.lower()
    if language in ("en", "english"):
        from text.english_frontend import EnglishG2P
        return EnglishG2P()
    if language in ("te", "telugu"):
        from text.telugu_frontend import TeluguG2P
        return TeluguG2P()
    raise ValueError("Unsupported language: " + language)
