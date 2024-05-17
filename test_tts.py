# test_tts.py
import pyttsx3

def test_tts():
    tts_engine = pyttsx3.init(driverName='nsss')
    tts_engine.say("Hello, this is a test.")
    tts_engine.runAndWait()

test_tts()


