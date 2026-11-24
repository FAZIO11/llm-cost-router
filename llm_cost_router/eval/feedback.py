"""Adaptive feedback loop for threshold adjustment."""

from typing import Dict

from ..storage import Database, ThresholdAdjustment


class FeedbackLoop:
    """Adaptive feedback loop that adjusts complexity thresholds based on quality."""

    def __init__(
        self,
        database: Database,
        target_quality: float = 0.78,
        window_size: int = 20,
        adjustment_step: float = 0.05,
    ):
        """
        Initialize feedback loop.

        Args:
            database: Database for quality metrics
            target_quality: Target quality score for each tier
            window_size: Number of recent requests to consider
            adjustment_step: How much to adjust thresholds
        """
        self.database = database
        self.target_quality = target_quality
        self.window_size = window_size
        self.adjustment_step = adjustment_step

        # Current thresholds
        self.simple_threshold = 0.35
        self.moderate_threshold = 0.65

    def check_and_adjust(self) -> Dict[str, float]:
        """
        Check quality metrics and adjust thresholds if needed.

        Returns:
            Current thresholds
        """
        # Check quality for each tier
        tiers_to_check = ["local", "budget", "balanced"]

        for tier in tiers_to_check:
            rolling_quality = self.database.get_rolling_quality_by_tier(tier, self.window_size)

            # If quality is below target, tighten threshold (escalate more prompts to higher tier)
            if rolling_quality < self.target_quality:
                self._tighten_threshold(tier, rolling_quality)

        return {
            "simple_threshold": self.simple_threshold,
            "moderate_threshold": self.moderate_threshold,
        }

    def _tighten_threshold(self, tier: str, rolling_quality: float):
        """Tighten threshold for a tier to escalate more prompts."""
        if tier == "local":
            # Tighten simple threshold (move more prompts from local to budget)
            old_threshold = self.simple_threshold
            self.simple_threshold = max(0.20, self.simple_threshold - self.adjustment_step)

            if old_threshold != self.simple_threshold:
                self.database.log_threshold_adjustment(
                    ThresholdAdjustment(
                        tier="local",
                        old_threshold=old_threshold,
                        new_threshold=self.simple_threshold,
                        reason=f"Quality below target ({rolling_quality:.3f} < {self.target_quality})",
                        rolling_quality_score=rolling_quality,
                    )
                )

        elif tier == "budget":
            # Tighten moderate threshold (move more prompts from budget to balanced)
            old_threshold = self.moderate_threshold
            self.moderate_threshold = max(0.50, self.moderate_threshold - self.adjustment_step)

            if old_threshold != self.moderate_threshold:
                self.database.log_threshold_adjustment(
                    ThresholdAdjustment(
                        tier="budget",
                        old_threshold=old_threshold,
                        new_threshold=self.moderate_threshold,
                        reason=f"Quality below target ({rolling_quality:.3f} < {self.target_quality})",
                        rolling_quality_score=rolling_quality,
                    )
                )

    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return {
            "simple_threshold": self.simple_threshold,
            "moderate_threshold": self.moderate_threshold,
        }
