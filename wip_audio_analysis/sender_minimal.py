import websocket
import numpy as np
import time

ws = websocket.WebSocket()
ws.connect("ws://localhost:8765")

print("sending...")

while True:
    # fake audio buffer (IMPORTANT: constant size)
    fake_audio = np.random.rand(1024).astype(np.float32)

    ws.send(fake_audio.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)

    time.sleep(0.02)  # 50 FPS (~20ms)