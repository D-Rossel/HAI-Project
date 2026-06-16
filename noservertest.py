import queue
import sounddevice as sd
import numpy as np
import essentia.standard as es

TARGET_SR = 16000
BLOCKSIZE = 2048

audio_queue = queue.Queue()

window = es.Windowing(type="hann")
spectrum = es.Spectrum()
centroid = es.Centroid()
rms = es.RMS()

class MoodState:
    def __init__(self):
        self.mood = {"primary": "calm", "descriptors": []}

    def update(self, c, e):
        if e > 0.018:
            self.mood["primary"] = "energetic"
        elif e < 0.012:
            self.mood["primary"] = "calm"

        desc = []
        desc.append("bright" if c > 0.12 else "warm")
        desc.append("intense" if e > 0.015 else "soft")
        self.mood["descriptors"] = desc
        return self.mood

state = MoodState()

def process(audio):
    if len(audio) < 2048:
        return None
    frame = audio[:2048].astype(np.float32)
    spec = spectrum(window(frame))
    return {
        "centroid": float(centroid(spec)),
        "energy": float(rms(frame))
    }

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata[:, 0].copy())

print("🎬 Starting local audio analysis...")

with sd.InputStream(
    channels=1,
    samplerate=TARGET_SR,
    blocksize=BLOCKSIZE,
    dtype="float32",
    callback=callback
):
    while True:
        audio = audio_queue.get()
        features = process(audio)
        if features is None:
            continue
        mood = state.update(features["centroid"], features["energy"])
        print(mood)