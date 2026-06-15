import sounddevice as sd
import numpy as np
import websocket

WS_URL = "ws://localhost:8765"

ws = websocket.WebSocket()
ws.connect(WS_URL)

samplerate = 44100
blocksize = 2048  # ~46ms latency


def callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata[:, 0].astype(np.float32)

    # 🚀 SEND AS BINARY (IMPORTANT FIX)
    ws.send(audio.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)


print("🎤 Streaming audio to WSL...")

with sd.InputStream(
    channels=1,
    samplerate=samplerate,
    blocksize=blocksize,
    callback=callback
):
    while True:
        sd.sleep(1000)