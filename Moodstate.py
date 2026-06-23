from collections import deque, Counter

class MoodState:
    ENERGY_THRESHOLDS = (1000, 4000)        # low < t0 <= mid <= t1 < high
    ZCR_THRESHOLDS = (0.05, 0.15)
    CENTROID_THRESHOLDS = (1500, 4000)     # Hz

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

        return adjectives

    # ---------- Public API ----------

    def update(self, features: dict) -> dict:
        adjectives = self._features_to_adjectives(features)
        self.history.extend(adjectives)

        counts = Counter(self.history)
        top_adjectives = [adj for adj, _ in counts.most_common(self.top_k)]

        return {
            "current": adjectives,
            "top": top_adjectives,
        }

    def reset(self):
        self.history.clear()