from dataclasses import dataclass
from personalityClass import PersonalityVector
from dialogueClass import FinalAction


@dataclass
class BeliefState:
    """
    Per-game mutable read on the opponent and the agent's own current stance.
    Resets at the start of each game. Continuous fields only for now;
    discrete trackers (unanswered_questions, contradiction_count, etc.) come later.
    """
    trust_in_opponent: float = 0.5     # how much the agent currently believes opponent will split
    suspicion_level: float = 0.0       # accumulated wariness from the dialogue so far
    intention: FinalAction = FinalAction.SPLIT  # current leaning for the final action

    @classmethod
    def initialize_from_personality(cls, personality: PersonalityVector) -> "BeliefState":
        """
        At the start of a game, seed beliefs from the agent's stable traits.
        A trusting agent starts with higher trust and a split-leaning intention;
        an aggressive agent starts more suspicious and may already lean steal.
        """
        trust = personality.trust_baseline
        suspicion = personality.aggression * 0.5  # aggressive agents start a bit wary

        # Initial intention: lean split if trust outweighs aggression, else steal.
        if trust >= personality.aggression:
            intention = FinalAction.SPLIT
        else:
            intention = FinalAction.STEAL

        return cls(
            trust_in_opponent=trust,
            suspicion_level=suspicion,
            intention=intention,
        )

    def clamp(self) -> None:
        """Keep continuous belief values within [0, 1] after updates."""
        self.trust_in_opponent = max(0.0, min(1.0, self.trust_in_opponent))
        self.suspicion_level = max(0.0, min(1.0, self.suspicion_level))
