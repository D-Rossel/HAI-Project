import queue
import numpy as np
import sounddevice as sd
import essentia.standard as es

rms = es.RMS()
SR = 41000
BLOCKSIZE = 1024
"""BUFFER_SECONDS = 20
BUFFER_SIZE = SR * BUFFER_SECONDS
INFER_EVERY = 10
RMS_THRESHOLD = 0.001"""

BUFFER_SECONDS = 4
BUFFER_SIZE = SR * BUFFER_SECONDS
INFER_EVERY = 10
RMS_THRESHOLD = 0.01
SILENT_FRAMES_RESET = 6

audio_queue = queue.Queue(maxsize=4)

def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak

def process(audio_buffer):
    print("process")
    audio = audio_buffer.astype(np.float32)

    if audio.ndim > 1:
        audio = audio[:, 0]

    audio = audio[-BUFFER_SIZE:]
    
    es_rms = rms(audio)
    if es_rms < RMS_THRESHOLD:
        print("silence")
        return None

    energy = es.Energy()(audio)
    zcr = es.ZeroCrossingRate()(audio)
    centroid = es.SpectralCentroidTime()(audio)
    #rolloff = es.SpectralRolloff()(audio)
    #flux = es.SpectralFlux()(audio)
    key, scale, strength = es.KeyExtractor()(audio)

    print({
        "energy": energy,
        "zcr": zcr,
        "centroid": centroid,
        #"rolloff": rolloff,
        #"flux": flux,
        "key": key,
        "scale": scale,
        "strength": strength,
        "rms": es_rms
    })

    normalized_audio = normalize(audio)
    #embeddings = embedding_model(normalized_audio)
    #preds = model(embeddings)
    preds = np.mean(preds, axis=0)

    return preds

def callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata.mean(axis=1).astype(np.float32)

    try:
        audio_queue.put_nowait(audio.copy())
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            audio_queue.put_nowait(audio.copy())
        except queue.Full:
            pass

def main():
    print("🎬 Starting local Essentia mood analysis on macOS with BlackHole 2ch...")
    print(sd.query_devices())

    #state = MoodState(threshold=0.08, history_size=8, top_k=5)
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    counter = 0

    with sd.InputStream(
        device=2,
        channels=1,
        samplerate=SR,
        blocksize=BLOCKSIZE,
        dtype="float32",
        callback=callback
    ):
        while True:
            chunk = audio_queue.get()
            counter += 1
            if len(chunk) > BUFFER_SIZE:
                chunk = chunk[-BUFFER_SIZE:]

            buffer = np.roll(buffer, -len(chunk))
            buffer[-len(chunk):] = chunk

            if counter % INFER_EVERY != 0:
                continue

            try:
                process(buffer)

            except Exception as e:
                print("Processing error:", e)

if __name__ == "__main__":
    main()