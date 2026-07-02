import queue
import threading
import numpy as np
import sounddevice as sd
import essentia.standard as es
import tkinter as tk

from Moodstate import MoodState
from image_generator2 import ImageGenerator

rms = es.RMS()
SR = 44100
BLOCKSIZE = 1024

BUFFER_SECONDS = 4
BUFFER_SIZE = SR * BUFFER_SECONDS
INFER_EVERY = 40
RMS_THRESHOLD = 0.01
SILENT_FRAMES_RESET = 6
IMAGE_EVERY = INFER_EVERY * 10

user_input = []

audio_queue = queue.Queue(maxsize=4)
image_job_queue = queue.Queue(maxsize=1)


def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak


def process(audio_buffer):
    audio = audio_buffer.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio[-BUFFER_SIZE:]

    es_rms = rms(audio)
    if es_rms < RMS_THRESHOLD:
        return None

    energy = es.Energy()(audio)
    zcr = es.ZeroCrossingRate()(audio)
    centroid = es.SpectralCentroidTime()(audio)

    features = {
        "energy": energy,
        "zcr": zcr,
        "centroid": centroid,
        "rms": es_rms,
    }
    #print(features)
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
    gen = ImageGenerator()
    image_counter = 0
    while True:
        mood_top = image_job_queue.get()
        if mood_top is None:
            break
        try:
            final_prompt = build_prompt(mood_top)
            image = gen.generate_image(
                prompt=final_prompt,
                steps=4,
                width=512,
                height=512
            )
            image_counter += 1
            image.save("output.png")
        except Exception as e:
            print("Image generation error:", e)


def build_prompt(mood_top: list[str]) -> str:
    unique_adjectives = list(dict.fromkeys(mood_top))
    if len(unique_adjectives) > 1:
        adjective_str = ", ".join(unique_adjectives[:-1]) + " and " + unique_adjectives[-1]
    else:
        adjective_str = unique_adjectives[0] if unique_adjectives else "neutral"

    user_part = f" The picture should be influenced by: {', '.join(user_input)}." if user_input else ""

    erg = (
        f"An abstract piece of generative art "
        f"with a {adjective_str} mood.{user_part} Flowing shapes, expressive color "
        f"palette, atmospheric lighting, high detail, artstation quality."
    )
    print(erg)
    return erg


def audio_loop():
    """Läuft im Hintergrund-Thread — der komplette Audio+Mood-Loop."""
    state = MoodState(history_size=8, top_k=5)
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    counter = 0
    silent_frames = 0

    with sd.InputStream(
        device=1,
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
                #print("Mood (aktuell):", mood["current"])
                #print("Mood (Top):", mood["top"])
                #print("____________________________")

                if counter % IMAGE_EVERY == 0:
                    try:
                        image_job_queue.put_nowait(mood["top"])
                    except queue.Full:
                        print("Bild-Worker noch beschäftigt, Job übersprungen.")

            except Exception as e:
                print("Processing error:", e)


def main():
    print(sd.query_devices())

    # Audio- und Bild-Worker als Daemon-Threads
    threading.Thread(target=audio_loop, daemon=True).start()
    threading.Thread(target=image_worker, daemon=True).start()

    # ----------------------------
    # Tkinter im MAIN THREAD
    # ----------------------------
    def add_word():
        word = entry.get().strip()
        if word:
            user_input.append(word)
            entry.delete(0, tk.END)
            update_label()

    def clear_words():
        user_input.clear()
        update_label()

    def update_label():
        label_words.config(
            text="Words: " + ", ".join(user_input) if user_input else "Words: (none)"
        )

    root = tk.Tk()
    root.title("Mood Input")
    root.geometry("350x160")
    root.resizable(False, False)

    tk.Label(root, text="Add mood word:", font=("Helvetica", 12)).pack(pady=(12, 4))

    entry = tk.Entry(root, font=("Helvetica", 12), width=28)
    entry.pack()
    entry.bind("<Return>", lambda e: add_word())

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)
    tk.Button(btn_frame, text="Add", width=10, command=add_word).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_frame, text="Clear", width=10, command=clear_words).pack(side=tk.LEFT, padx=4)

    label_words = tk.Label(root, text="Words: (none)", font=("Helvetica", 10), fg="gray")
    label_words.pack()

    root.mainloop()


if __name__ == "__main__":
    main()