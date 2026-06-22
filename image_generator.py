import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        local_path=None,
        device=None,
        use_auth_token=None
    ):
        """
        Lädt Stable Diffusion entweder aus HF oder lokalem Pfad.
        """

        # ----------------------------
        # Device Auswahl
        # ----------------------------
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device

        # ----------------------------
        # Modellquelle
        # ----------------------------
        model_source = local_path if local_path else model_id

        print(f"Loading model from: {model_source}")
        print(f"Using device: {self.device}")

        # ----------------------------
        # Pipeline laden
        # ----------------------------
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_source,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            token=use_auth_token
        )

        self.pipe = self.pipe.to(self.device)

        # Optional: Speed tweak (Achtung: kann Qualität beeinflussen)
        self.pipe.safety_checker = None

    def generate_image(self, prompt, steps=25, width=512, height=512):
        """
        Generiert ein Bild aus einem Prompt.
        """

        result = self.pipe(
            prompt,
            num_inference_steps=steps,
            width=width,
            height=height
        )

        return result.images[0]


# ----------------------------------------------------
# TEST / DEMO (wird NICHT beim Import ausgeführt)
# ----------------------------------------------------
'''if __name__ == "__main__":

    gen = ImageGenerator()

    prompt = "transparent black backround with square"

    image = gen.generate_image(
        prompt=prompt,
        steps=50,
        width=256,
        height=256
    )

    
    image.save("output1.png")

    print("Bild gespeichert: output1.png")

    import time
    time.sleep(1)
    prompt = "circle"

    image = gen.generate_image(
        prompt=prompt,
        steps=50,
        width=256,
        height=256
    )

    
    image.save("output2.png")

    print("Bild gespeichert: output2.png")'''