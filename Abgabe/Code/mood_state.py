from collections import Counter, deque


class MoodState:
    """Track recent audio-derived mood descriptors and return dominant adjectives."""

    ENERGY_THRESHOLDS = (1000, 4000)  # low < t0 <= mid <= t1 < high
    ZCR_THRESHOLDS = (0.05, 0.15)
    CENTROID_THRESHOLDS = (1500, 4000)  # Hz

    ADJECTIVES = {
        "energy": {
            "low": ["calm", "gentle", "subdued"],
            "mid": ["balanced", "flowing"],
            "high": ["energetic", "driving", "intense"],
        },
        "zcr": {
            "low": ["warm", "soft", "round"],
            "mid": ["balanced"],
            "high": ["sharp", "percussive", "noisy"],
        },
        "centroid": {
            "low": ["dark", "bass-heavy", "weighty"],
            "mid": ["balanced", "natural"],
            "high": ["bright", "brilliant", "airy"],
        },
    }

    def __init__(self, history_size: int = 8, top_k: int = 5) -> None:
        self.history_size = history_size
        self.top_k = top_k
        self.history = deque(maxlen=history_size)

    @staticmethod
    def _bucket(value: float, thresholds: tuple[float, float]) -> str:
        """Map a numeric feature value to a low/mid/high bucket."""
        low_threshold, high_threshold = thresholds
        if value < low_threshold:
            return "low"
        if value < high_threshold:
            return "mid"
        return "high"

    def _features_to_adjectives(self, features: dict) -> list[str]:
        """Convert extracted audio features into descriptive mood adjectives."""
        adjectives: list[str] = []

        energy_bucket = self._bucket(features["energy"], self.ENERGY_THRESHOLDS)
        adjectives += self.ADJECTIVES["energy"][energy_bucket]

        zcr_bucket = self._bucket(features["zcr"], self.ZCR_THRESHOLDS)
        adjectives += self.ADJECTIVES["zcr"][zcr_bucket]

        centroid_bucket = self._bucket(features["centroid"], self.CENTROID_THRESHOLDS)
        adjectives += self.ADJECTIVES["centroid"][centroid_bucket]

        return adjectives

    def update(self, features: dict) -> dict:
        """Update the mood history and return current and dominant descriptors."""
        adjectives = self._features_to_adjectives(features)
        self.history.extend(adjectives)

        counts = Counter(self.history)
        top_adjectives = [adjective for adjective, _ in counts.most_common(self.top_k)]

        return {
            "current": adjectives,
            "top": top_adjectives,
        }

    def reset(self) -> None:
        """Clear the rolling mood history."""
        self.history.clear()