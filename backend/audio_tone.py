import librosa
import numpy as np
import io

def extract_tone_features(wav_bytes):
    y, sr = librosa.load(wav_bytes, sr=16000)

    features = {
        "rms": float(librosa.feature.rms(y=y).mean()),
        "zcr": float(librosa.feature.zero_crossing_rate(y).mean()),
        "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
        "centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    }
    return features


def classify_tone(f):
    rms = f["rms"]
    zcr = f["zcr"]
    tempo = f["tempo"]
    centroid = f["centroid"]

    # Simple heuristic model
    if rms > 0.05 and zcr > 0.1:
        tone = "Angry / Aggressive"
    elif tempo > 140:
        tone = "Agitated"
    elif rms < 0.02:
        tone = "Sad / Whisper"
    else:
        tone = "Neutral"

    # Convert to numeric
    score = min(int((rms*200) + (zcr*100)), 100)

    return tone, score
