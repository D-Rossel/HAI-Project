from collections import deque, Counter

class MoodState:
    """
    Mappt rohe Audio-Features (Energy, ZCR, Spectral Centroid, Key/Scale/Strength)
    auf beschreibende Adjektive für Klang/Musik-Charakter.
    """

    # Schwellenwerte – ggf. an dein Material anpassen
    ENERGY_THRESHOLDS = (1000, 4000)        # low < t0 <= mid <= t1 < high
    ZCR_THRESHOLDS = (0.05, 0.15)
    CENTROID_THRESHOLDS = (1500, 4000)     # Hz
    STRENGTH_THRESHOLDS = (0.4, 0.7)       # Key-Strength (Tonalitäts-Konfidenz)

    ADJECTIVES = {
        "energy": {
            "low":  ["calm", "gentle", "subdued"],
            "mid":  ["balanced", "flowing"],
            "high": ["energetic", "driving", "intense"],
        },
        "zcr": {
            "low":  ["warm", "soft", "round"],
            "mid":  ["balanced"],
            "high": ["sharp", "percussive", "noisy"],
        },
        "centroid": {
            "low":  ["dark", "bass-heavy", "weighty"],
            "mid":  ["balanced", "natural"],
            "high": ["bright", "brilliant", "airy"],
        },
        "scale": {
            "major": ["happy", "cheerful", "open"],
            "minor": ["melancholic", "dark", "introspective"],
        },
        "strength": {
            "low":  ["diffuse", "atonal", "floating"],
            "mid":  ["slightly tonal"],
            "high": ["clear", "tonal", "focused"],
        },
    }

    def __init__(self, history_size: int = 8, top_k: int = 5):
        self.history_size = history_size
        self.top_k = top_k
        self.history = deque(maxlen=history_size)

    # ---------- Hilfsfunktionen ----------

    @staticmethod
    def _bucket(value: float, thresholds: tuple[float, float]) -> str:
        low_t, high_t = thresholds
        if value < low_t:
            return "low"
        elif value < high_t:
            return "mid"
        else:
            return "high"

    def _features_to_adjectives(self, features: dict) -> list[str]:
        adjectives: list[str] = []

        energy_bucket = self._bucket(features["energy"], self.ENERGY_THRESHOLDS)
        adjectives += self.ADJECTIVES["energy"][energy_bucket]

        zcr_bucket = self._bucket(features["zcr"], self.ZCR_THRESHOLDS)
        adjectives += self.ADJECTIVES["zcr"][zcr_bucket]

        centroid_bucket = self._bucket(features["centroid"], self.CENTROID_THRESHOLDS)
        adjectives += self.ADJECTIVES["centroid"][centroid_bucket]

        scale = features.get("scale", "").lower()
        if scale in ("major", "minor"):
            adjectives += self.ADJECTIVES["scale"][scale]

        strength_bucket = self._bucket(features["strength"], self.STRENGTH_THRESHOLDS)
        adjectives += self.ADJECTIVES["strength"][strength_bucket]

        return adjectives

    # ---------- Public API ----------

    def update(self, features: dict) -> dict:
        """
        Nimmt ein Feature-Dict (energy, zcr, centroid, key, scale, strength, rms)
        und gibt ein Dict mit Einzel-Kategorien + geglätteten Top-Adjektiven zurück.
        """
        adjectives = self._features_to_adjectives(features)
        self.history.extend(adjectives)

        counts = Counter(self.history)
        top_adjectives = [adj for adj, _ in counts.most_common(self.top_k)]

        return {
            "current": adjectives,
            "top": top_adjectives,
            "key": features.get("key"),
            "scale": features.get("scale"),
        }

    def reset(self):
        self.history.clear()