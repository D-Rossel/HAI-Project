import torch
from diffusers import AutoPipelineForText2Image

class ImageGenerator:
    def __init__(self, model_id="stabilityai/sdxl-turbo", device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        print(f"Loading SDXL-Turbo on: {self.device}")

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)

    def generate_image(self, prompt, steps=4, width=512, height=512, guidance=0.0):
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,  # Turbo-Modell: CFG deaktiviert
            width=width,
            height=height,
        )
        return result.images[0]
    
def build_prompt(mood_top: list[str]) -> str:
    unique_adjectives = list(dict.fromkeys(mood_top))

    if len(unique_adjectives) > 1:
        adjective_str = ", ".join(unique_adjectives[:-1]) + " and " + unique_adjectives[-1]
    else:
        adjective_str = unique_adjectives[0] if unique_adjectives else "neutral"
    erg = (
        f"An abstract piece of generative art that visually expresses music "
        f"with a {adjective_str} mood. Flowing shapes, expressive color "
        f"palette, atmospheric lighting, high detail, artstation quality."
    )
    print(erg)
    return erg

if __name__ == "__main__":

    gen = ImageGenerator()
    try:
                final_prompt = build_prompt(["dark", "sweet", "calm"])
                image = gen.generate_image(
                    prompt=final_prompt,
                    steps=4,
                    width=512,
                    height=512
                )
                image.save("output.png")

    except Exception as e:
        print("Image generation error:", e)