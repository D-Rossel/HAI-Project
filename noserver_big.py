import queue
import threading
import numpy as np
import sounddevice as sd
import essentia.standard as es

from Moodstate import MoodState
from image_generator import ImageGenerator

rms = es.RMS()
SR = 41000
BLOCKSIZE = 1024

BUFFER_SECONDS = 4
BUFFER_SIZE = SR * BUFFER_SECONDS
INFER_EVERY = 40
RMS_THRESHOLD = 0.01
SILENT_FRAMES_RESET = 6

IMAGE_EVERY = INFER_EVERY * 10

audio_queue = queue.Queue(maxsize=4)

# Queue für Bild-Jobs: maxsize=1, damit sich keine Anfragen stapeln,
# falls die Generierung länger dauert als das Intervall.
image_job_queue = queue.Queue(maxsize=1)


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
    key, scale, strength = es.KeyExtractor()(audio)

    features = {
        "energy": energy,
        "zcr": zcr,
        "centroid": centroid,
        "key": key,
        "scale": scale,
        "strength": strength,
        "rms": es_rms,
    }

    print(features)
    return features


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


def image_worker():
    """
    Läuft in eigenem Thread. Nimmt Mood-Prompts aus image_job_queue
    und generiert Bilder, ohne den Audio-Analyse-Loop zu blockieren.
    """
    gen = ImageGenerator()
    image_counter = 0

    while True:
        mood_top = image_job_queue.get()  # blockiert, bis ein Job da ist

        if mood_top is None:
            # Sentinel zum Beenden des Threads
            break

        try:
            final_prompt = "Generate a picture that visualises music with the following adjactives: "
            for x in  mood_top:
                final_prompt += x + ", "
            print("🎨 Generiere Bild für Mood:", mood_top)
            image = gen.generate_image(
                prompt=final_prompt,
                steps=5,
                width=256,
                height=256
            )
            image_counter += 1
            image.save(f"output.png")
            #image.save(f"output_{image_counter}.png")
            print("✅ Bild gespeichert: output_%d.png" % image_counter)
        except Exception as e:
            print("Image generation error:", e)


def main():
    print("🎬 Starting local Essentia mood analysis on macOS with BlackHole 2ch...")
    print(sd.query_devices())

    state = MoodState(history_size=8, top_k=5)
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    counter = 0
    silent_frames = 0

    # Worker-Thread für Bildgenerierung starten
    worker_thread = threading.Thread(target=image_worker, daemon=True)
    worker_thread.start()

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
                features = process(buffer)

                if features is None:
                    silent_frames += 1
                    if silent_frames >= SILENT_FRAMES_RESET:
                        state.reset()
                    continue

                silent_frames = 0
                mood = state.update(features)
                print("Mood (aktuell):", mood["current"])
                print("Mood (Top):", mood["top"])

                if counter % IMAGE_EVERY == 0:
                    try:
                        # Nicht-blockierend: falls der Worker noch beschäftigt ist,
                        # wird dieser Job einfach übersprungen statt zu warten.
                        image_job_queue.put_nowait(mood["top"])
                    except queue.Full:
                        print("⏳ Bild-Worker noch beschäftigt, Job übersprungen.")

            except Exception as e:
                print("Processing error:", e)


if __name__ == "__main__":
    main()