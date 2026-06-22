import asyncio
import websockets
import numpy as np

import essentia.standard as es

# ===== Essentia =====
window = es.Windowing(type="hann")
spectrum = es.Spectrum()
centroid = es.Centroid()
rms = es.RMS()

model = es.TensorflowPredict2D(
    graphFilename="mtg_jamendo_moodtheme-discogs-effnet.pb",
    input="model/Placeholder",   # IMPORTANT (fixed for MTG models)
    output="model/Sigmoid"
)

class MoodState:
    def __init__(self):
        self.mood = {"primary": "calm", "descriptors": []}

    def update(self, c, e):
        if e > 0.018:
            self.mood["primary"] = "energetic"
        elif e < 0.012:
            self.mood["primary"] = "calm"

        self.mood["descriptors"] = []
        if c > 0.12:
            self.mood["descriptors"].append("bright")
        else:
            self.mood["descriptors"].append("warm")

        if e > 0.015:
            self.mood["descriptors"].append("intense")
        else:
            self.mood["descriptors"].append("soft")
        return self.mood


def process(audio):
    frame = audio[:2048]
    # frame = audio[:1024]  # smaller than before = safer
    
    spec = spectrum(window(frame))

    return {"centroid": float(centroid(spec)), "energy": float(rms(frame))}


async def handler(websocket):
    print("client connected")

    state = MoodState()

    async for message in websocket:
        audio = np.frombuffer(message, dtype=np.float32)

        features = process(audio)
        mood = state.update(features["centroid"], features["energy"])
        print(mood)

async def main():
    print("Essentia minimal server running...")

    async with websockets.serve(
        handler, "0.0.0.0", 8765, ping_interval=20, ping_timeout=60, max_size=2**20
    ):
        await asyncio.Future()


asyncio.run(main())
