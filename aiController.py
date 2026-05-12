from __future__ import annotations

import math
from random import choices
from typing import TYPE_CHECKING

from beliefStateClass import BeliefState
from dialogueClass import DialogueMove, FinalAction, SpeechAct
from personalityClass import PersonalityVector
from playerClass import Controller, Player

if TYPE_CHECKING:
    from gameClass import Game


class AIController(Controller):
    """
    Trainable controller. Uses a fixed PersonalityVector and a per-game
    BeliefState to score available speech acts and sample a dialogue move.
    """

    def __init__(self, player: Player, personality: PersonalityVector):
        super().__init__(player)
        self.personality = personality
        self.beliefs = BeliefState.initialize_from_personality(personality)
        self._move_counter = 0

    def reset_for_new_game(self) -> None:
        """Call between games to wipe per-game beliefs but keep personality."""
        self.beliefs = BeliefState.initialize_from_personality(self.personality)
        self._move_counter = 0

    def observe_dialogue_move(self, game: "Game", move: DialogueMove) -> None:
        # Belief-update logic on opponent moves comes later.
        # For now, beliefs only reflect their initial personality-derived values.
        return None

    def choose_dialogue_move(self, game: "Game") -> DialogueMove:
        scores = self._score_speech_acts()
        act = self._sample(scores)
        payload = self._build_payload(act)

        self._move_counter += 1
        move_id = int(f"{self.player.player_id}{self._move_counter:03d}")

        return DialogueMove(
            move_id=move_id,
            speaker_id=self.player.player_id,
            act=act,
            payload=payload,
            turn_number=game.current_turn,
        )

    def choose_final_action(self, game: "Game") -> FinalAction:
        # Final action reads directly off the current intention in the belief state.
        return self.beliefs.intention

    # ---------- internal scoring ----------

    def _score_speech_acts(self) -> dict[SpeechAct, float]:
        ## HAVE NOT INTEGRATED EVERY SPEECH ACT YET
        """
        Each speech act gets a score that combines:
          - personality traits (stable)
          - belief state (mutable, currently personality-seeded)
          - current intention (whether honesty or deception is in play)
        Scores are turned into a probability distribution via softmax in _sample.
        """
        p = self.personality
        b = self.beliefs
        intends_split = b.intention == FinalAction.SPLIT

        # PROMISE: cooperative tone, claim "split".
        # Honest if intention is split. If intention is steal, only attractive
        # when lie_propensity is high (false promise as bait).
        promise_score = (
            0.5
            + p.trust_baseline
            + b.trust_in_opponent
            + (1.0 if intends_split else p.lie_propensity)
        )

        # THREATEN: aggressive tone, claim "steal".
        # Most natural when intention is steal; suspicion and aggression boost it.
        threaten_score = (
            0.2
            + p.aggression
            + b.suspicion_level
            + (1.0 if not intends_split else 0.0)
        )

        # ACCUSE: confrontational; mostly driven by suspicion and some aggression.
        # Agent claims the moral high ground ("split").
        accuse_score = (
            0.2
            + b.suspicion_level
            + 0.5 * p.aggression
        )

        return {
            SpeechAct.PROMISE: promise_score,
            SpeechAct.THREATEN: threaten_score,
            SpeechAct.ACCUSE: accuse_score,
        }

    def _sample(self, scores: dict[SpeechAct, float]) -> SpeechAct:
        """Softmax over scores, then sample one speech act."""
        acts = list(scores.keys())
        raw = [scores[a] for a in acts]
        # Subtract max for numerical stability.
        m = max(raw)
        weights = [math.exp(s - m) for s in raw]
        return choices(acts, weights=weights, k=1)[0]

    def _build_payload(self, act: SpeechAct) -> dict[str, str]:
        """Map the chosen speech act to a payload with text and claimed action."""
        if act == SpeechAct.PROMISE:
            return {
                "text": "I'll split. We both walk away with something.",
                "claim_action": "split",
            }
        if act == SpeechAct.THREATEN:
            return {
                "text": "If you don't split, I'll steal and you get nothing.",
                "claim_action": "steal",
            }
        # ACCUSE
        return {
            "text": "You're not being straight with me. I'm splitting; you should too.",
            "claim_action": "split",
        }
