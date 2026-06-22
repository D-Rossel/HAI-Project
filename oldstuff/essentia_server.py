import asyncio
import websockets
import numpy as np
import essentia.standard as es

# ===== Essentia =====
window = es.Windowing(type='hann')
spectrum = es.Spectrum()
centroid = es.Centroid()
rms = es.RMS()

FRAME_SIZE = 2048


def process(audio):
    if len(audio) < FRAME_SIZE:
        return None

    frame = audio[:FRAME_SIZE]
    spec = spectrum(window(frame))

    return {
        "centroid": float(centroid(spec)),
        "energy": float(rms(frame))
    }


async def handler(websocket):
    print("🧠 Windows connected")

    async for message in websocket:
        # 🚀 binary decode
        audio = np.frombuffer(message, dtype=np.float32)

        features = process(audio)

        if features:
            print("🎧", features)


async def main():
    print("🚀 Essentia server running on ws://0.0.0.0:8765")

    async with websockets.serve(
        handler,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=60,
        max_size=2**20  # prevents small buffer crashes
    ):
        await asyncio.Future()


asyncio.run(main())