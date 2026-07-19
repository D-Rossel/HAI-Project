import torch
from diffusers import AutoPipelineForText2Image


class ImageGenerator:
    """Generate images from text prompts using an SDXL Turbo pipeline."""

    def __init__(self, model_id: str = "stabilityai/sdxl-turbo", device: str | None = None) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        print(f"Loading SDXL-Turbo on: {self.device}")

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)

    def generate_image(
        self,
        prompt: str,
        steps: int = 4,
        width: int = 512,
        height: int = 512,
        guidance: float = 0.0,
    ):
        """Generate a single image for the given prompt."""
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        )
        return result.images[0]


def build_prompt(top_moods: list[str]) -> str:
    """Build a prompt for abstract music-inspired generative art."""
    unique_adjectives = list(dict.fromkeys(top_moods))

    if len(unique_adjectives) > 1:
        adjective_text = ", ".join(unique_adjectives[:-1]) + " and " + unique_adjectives[-1]
    else:
        adjective_text = unique_adjectives[0] if unique_adjectives else "neutral"

    prompt = (
        f"An abstract piece of generative art that visually expresses music "
        f"with a {adjective_text} mood. Flowing shapes, expressive color "
        f"palette, atmospheric lighting, high detail, artstation quality."
    )

    print(prompt)
    return prompt


if __name__ == "__main__":
    generator = ImageGenerator()
    try:
        final_prompt = build_prompt(["dark", "sweet", "calm"])
        image = generator.generate_image(
            prompt=final_prompt,
            steps=4,
            width=512,
            height=512,
        )
        image.save("output.png")
    except Exception as error:
        print("Image generation error:", error)