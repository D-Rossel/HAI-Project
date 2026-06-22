""" Wandelt die extrahierten Audio-Features in einen beschreibenden Stable-Diffusion-Prompt um. """

class PromptEngine:
    def __init__(self):
        pass

    def build_prompt(self, features):
        """
        features ist ein Dictionary z.B.:
        {
            "bpm": 128,
            "energy": 0.8,
            "brightness": 0.6,
            "mode": "minor"
        }
        """

        prompt_parts = []

        # 🎵 Tempo → Bewegung
        bpm = features.get("bpm", 100)
        if bpm > 130:
            prompt_parts.append("fast motion, dynamic composition, intense movement")
        elif bpm < 90:
            prompt_parts.append("slow motion, calm atmosphere, minimal movement")
        else:
            prompt_parts.append("balanced motion, smooth rhythm")

        # 🔥 Energie → Intensität
        energy = features.get("energy", 0.5)
        if energy > 0.7:
            prompt_parts.append("high energy, chaotic scene, powerful visuals")
        else:
            prompt_parts.append("soft energy, stable composition")

        # 💡 Helligkeit → Lichtstil
        brightness = features.get("brightness", 0.5)
        if brightness > 0.6:
            prompt_parts.append("bright neon lighting, glowing highlights")
        else:
            prompt_parts.append("dark cinematic shadows, low-key lighting")

        # 🎼 Stimmung (Dur/Moll)
        mode = features.get("mode", "major")
        if mode == "minor":
            prompt_parts.append("melancholic mood, emotional atmosphere")
        else:
            prompt_parts.append("uplifting mood, positive atmosphere")

        # 🎨 Standard Quality Boost (immer dabei)
        prompt_parts.append("ultra detailed, cinematic, 8k, highly realistic")

        return ", ".join(prompt_parts)