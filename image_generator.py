""" Erzeugt aus dem Prompt mithilfe von Stable Diffusion ein Bild. """

import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """
        Lädt das Stable-Diffusion-Modell einmal beim Start.
        """
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        )

        self.pipe = self.pipe.to(self.device)

        # optional: schneller machen
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

        image = result.images[0]
        return image
    
    
    
'''
Nutzung:

from image_generator import ImageGenerator

gen = ImageGenerator()

prompt = "cyberpunk city, neon lights, rainy atmosphere, cinematic, ultra detailed"

image = gen.generate_image(prompt)

image.save("output.png")

print("Bild gespeichert!")'''