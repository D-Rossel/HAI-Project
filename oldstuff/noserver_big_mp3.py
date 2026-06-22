import numpy as np
import essentia.standard as es

SR = 16000
BUFFER_SECONDS = 20
BUFFER_SIZE = SR * BUFFER_SECONDS

embedding_model = es.TensorflowPredictEffnetDiscogs(
    graphFilename="discogs-effnet-bs64-1.pb",
    output="PartitionedCall:1"
)

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

def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak

def process(audio_buffer):
    audio = audio_buffer.astype(np.float32)
    audio = audio[-BUFFER_SIZE:]

    normalized_audio = normalize(audio)
    embeddings = embedding_model(normalized_audio)
    preds = model(embeddings)
    preds = np.mean(preds, axis=0)
    return preds

filename = "test.mp3"
audio = es.MonoLoader(filename=filename, sampleRate=SR)()

preds = process(audio)
top_k = np.argsort(preds)[-5:][::-1]

print("Top moods:")
for i in top_k:
    print(LABELS[i], float(preds[i]))