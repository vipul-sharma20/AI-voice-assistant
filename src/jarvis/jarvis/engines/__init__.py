from .stt import STTGoogleEngine, STTVernacularEngine


SPEECH_ENGINES = {
    'vernacular': STTVernacularEngine,
    'google': STTGoogleEngine,
}
