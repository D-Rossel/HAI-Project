import queue
import threading
import tkinter as tk

import essentia.standard as es
import numpy as np
import sounddevice as sd

from mood_state import MoodState
from image_generator import ImageGenerator


rms = es.RMS()
SR = 44100
BLOCK_SIZE = 1024

BUFFER_SECONDS = 4
BUFFER_SIZE = SR * BUFFER_SECONDS
INFER_EVERY = 40
RMS_THRESHOLD = 0.01
SILENT_FRAMES_RESET = 6
IMAGE_EVERY = INFER_EVERY * 10

user_input: list[str] = []

audio_queue = queue.Queue(maxsize=4)
image_job_queue = queue.Queue(maxsize=1)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio by peak amplitude."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak


def process(audio_buffer: np.ndarray):
    """Extract features from the rolling audio buffer."""
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
    return features


def callback(indata, frames, time, status) -> None:
    """Audio input callback that pushes the latest chunk into the queue."""
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


def build_prompt(top_moods: list[str]) -> str:
    """Build a text prompt from detected moods and optional user words."""
    unique_adjectives = list(dict.fromkeys(top_moods))
    if len(unique_adjectives) > 1:
        adjective_text = ", ".join(unique_adjectives[:-1]) + " and " + unique_adjectives[-1]
    else:
        adjective_text = unique_adjectives[0] if unique_adjectives else "neutral"

    user_part = f" The picture should be influenced by: {', '.join(user_input)}." if user_input else ""
    prompt = (
        f"An abstract piece of generative art "
        f"with a {adjective_text} mood.{user_part} Flowing shapes, expressive color "
        f"palette, atmospheric lighting, high detail, artstation quality."
    )

    print(prompt)
    return prompt


def image_worker() -> None:
    """Background worker that generates images from queued mood descriptors."""
    generator = ImageGenerator()

    while True:
        top_moods = image_job_queue.get()
        if top_moods is None:
            break

        try:
            final_prompt = build_prompt(top_moods)
            image = generator.generate_image(
                prompt=final_prompt,
                steps=4,
                width=512,
                height=512,
            )
            image.save("output.png")
        except Exception as error:
            print("Image generation error:", error)


def audio_loop() -> None:
    """Run the complete audio and mood processing loop in a background thread."""
    state = MoodState(history_size=8, top_k=5)
    buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    counter = 0
    silent_frames = 0

    with sd.InputStream(
        device=1,
        channels=1,
        samplerate=SR,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback,
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

                if counter % IMAGE_EVERY == 0:
                    try:
                        image_job_queue.put_nowait(mood["top"])
                    except queue.Full:
                        print("Image worker is still busy, skipped job.")

            except Exception as error:
                print("Processing error:", error)


def main() -> None:
    """Start the audio and image workers and run the Tkinter UI."""
    print(sd.query_devices())

    threading.Thread(target=audio_loop, daemon=True).start()
    threading.Thread(target=image_worker, daemon=True).start()

    def add_word() -> None:
        word = entry.get().strip()
        if word:
            user_input.append(word)
            entry.delete(0, tk.END)
            update_label()

    def clear_words() -> None:
        user_input.clear()
        update_label()

    def update_label() -> None:
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
    entry.bind("<Return>", lambda event: add_word())

    button_frame = tk.Frame(root)
    button_frame.pack(pady=8)
    tk.Button(button_frame, text="Add", width=10, command=add_word).pack(side=tk.LEFT, padx=4)
    tk.Button(button_frame, text="Clear", width=10, command=clear_words).pack(side=tk.LEFT, padx=4)

    label_words = tk.Label(root, text="Words: (none)", font=("Helvetica", 10), fg="gray")
    label_words.pack()

    root.mainloop()


if __name__ == "__main__":
    main()