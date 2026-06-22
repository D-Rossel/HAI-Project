""" Extrahiert mit Essentia musikalische Merkmale wie BPM, Energie und Klangcharakter aus dem Audio. """

import numpy as np
import sounddevice as sd

import essentia.standard as es

# ===== Essentia Feature Extractors =====
frame_size = 2048
hop_size = 1024

window = es.Windowing(type='hann')
spectrum = es.Spectrum()
centroid = es.Centroid()

# optional: RMS Energy (für "Energy"-Visuals)
rms = es.RMS()


# ===== Audio Callback =====
def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata[:, 0].astype(np.float32)

    # Essentia erwartet Frames → wir schneiden manuell
    if len(audio) < frame_size:
        return

    frame = audio[:frame_size]

    # ===== Feature Extraction =====
    spec = spectrum(window(frame))
    c = centroid(spec)
    e = rms(frame)

    # ===== LIVE OUTPUT =====
    print(f"Centroid (Brightness): {c:.2f} | Energy: {e:.4f}")


# ===== STREAM START =====
samplerate = 44100
blocksize = hop_size

print("🎧 Live Audio Analysis running... (Ctrl+C to stop)")

with sd.InputStream(
    channels=1,
    samplerate=samplerate,
    blocksize=blocksize,
    callback=audio_callback
):
    while True:
        sd.sleep(1000)