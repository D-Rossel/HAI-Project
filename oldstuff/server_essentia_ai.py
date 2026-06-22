import asyncio
from email.mime import audio
import websockets
import numpy as np
import essentia.standard as es

SR = 16000
FRAME_SIZE = 2048
HOP_SIZE = 1024
BUFFER_SECONDS = 20
BUFFER_SIZE = SR * BUFFER_SECONDS

window = es.Windowing(type="hann")
spectrum = es.Spectrum()
mel = es.MelBands(
    numberBands=96,
    sampleRate=SR,
    highFrequencyBound=7900
)
log = es.UnaryOperator(type="log")

embedding_model = es.TensorflowPredictEffnetDiscogs(
    graphFilename="discogs-effnet-bs64-1.pb",
    output="PartitionedCall:1"
)
# embedding_model = es.TensorflowPredictMusiCNN(
#     graphFilename="msd-musicnn-1.pb", 
#     output="model/dense/BiasAdd")

model = es.TensorflowPredict2D(
    graphFilename="mtg_jamendo_moodtheme-discogs-effnet.pb",
    input="model/Placeholder",
    output="model/Sigmoid"
)

LABELS = [
    "action", "adventure", "advertising", "background", "ballad", "calm",
    "children", "christmas", "commercial", "cool", "corporate", "dark",
    "deep", "documentary", "drama", "dramatic", "dream", "emotional",
    "energetic", "epic", "fast", "film", "fun", "funny", "game",
    "groovy", "happy", "heavy", "holiday", "hopeful", "inspiring",
    "love", "meditative", "melancholic", "melodic", "motivational",
    "movie", "nature", "party", "positive", "powerful", "relaxing",
    "retro", "romantic", "sad", "sexy", "slow", "soft", "soundscape",
    "space", "sport", "summer", "trailer", "travel", "upbeat", "uplifting"
]

class MoodState:
    def __init__(self):
        self.prev = None

    def update(self, preds):
        top_k = np.argsort(preds)[-5:][::-1]
        top_tags = [(LABELS[i], preds[i]) for i in top_k]
        # optional smoothing (reduces flicker)
        if self.prev is None:
            self.prev = top_tags
        elif self.prev != top_tags:
            # simple stability rule: require repetition
            self.prev = top_tags

        return self.prev

def extract_logmel(audio):
    frames = es.FrameGenerator(
        audio,
        frameSize=FRAME_SIZE,
        hopSize=HOP_SIZE
    )

    features = []

    for frame in frames:
        frame = window(frame)
        spec = spectrum(frame)
        mels = mel(spec)
        logm = log(mels)

        features.append(logm)

    return np.array(features, dtype=np.float32)

def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak

def process(audio_buffer):
    audio = audio_buffer.astype(np.float32)

    # mono safety
    if audio.ndim > 1:
        audio = audio[:, 0]

    # keep last 2 seconds
    audio = audio[-BUFFER_SIZE:]

    rms = np.sqrt(np.mean(audio**2))
    print("RMS:", rms)
    print("audio range:", audio.min(), audio.max())

    # if rms < 0.005:
    #     print("No meaningful audio")
    #     return None
    # logmel = extract_logmel(audio)
    # print("logmel:", logmel.shape)

    normalized_audio = normalize(audio)
    embeddings = embedding_model(normalized_audio)

    # print("embeddings:", np.array(embeddings).shape)

    preds = model(embeddings)
    preds = np.mean(preds, axis=0)
    
    # confidence thresholding
    top_idx = np.argmax(preds)
    top_score = preds[top_idx]

    # if top_score < 0.30:
    #     print("No confident mood detected")
    #     return

    # print("preds:", np.array(preds).shape)
    
    return preds

async def handler(websocket):
    print("Client connected")

    state = MoodState()

    # rolling buffer
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

    async for message in websocket:
        chunk = np.frombuffer(message, dtype=np.float32)

        # update buffer (rolling window)
        if len(chunk) > BUFFER_SIZE:
            chunk = chunk[-BUFFER_SIZE:]

        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk

        try:
            preds = process(buffer)
            mood = state.update(preds)

            print("Mood:", mood)

        except Exception as e:
            print("Processing error:", e)


# =========================
# 🚀 MAIN SERVER
# =========================
async def main():
    print("🚀 Essentia MTG-TF Mood Server running on ws://0.0.0.0:8765")

    async with websockets.serve(
        handler,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=60,
        max_size=2**20
    ):
        await asyncio.Future()


asyncio.run(main())