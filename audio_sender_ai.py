import sounddevice as sd
import numpy as np
import websocket

print("🎬 Starting audio sender...")

WS_URL = "ws://localhost:8765"

ws = websocket.WebSocket()
ws.connect(WS_URL)

TARGET_SR = 16000
BLOCKSIZE = 2048  # smaller = more stable for streaming models


def callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata[:, 0].astype(np.float32)

    # send binary
    try:
        ws.send(audio.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
    except Exception as e:
        print("Send error:", e)


print("Streaming audio to WSL (16kHz compatible)...")

with sd.InputStream(
    channels=1,
    samplerate=TARGET_SR,
    blocksize=BLOCKSIZE,
    dtype="float32",
    callback=callback
):
    while True:
        sd.sleep(1000)