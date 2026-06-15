import asyncio
import websockets
import numpy as np

async def handler(websocket):
    print("client connected")

    count = 0

    async for message in websocket:
        audio = np.frombuffer(message, dtype=np.float32)

        count += 1
        print("packet", count, "size:", len(audio))


async def main():
    print("server running...")

    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()

asyncio.run(main())