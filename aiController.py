from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from beliefStateClass import BeliefState
from dialogueClass import DialogueMove, FinalAction, PROMPT_ACTS, SpeechAct
from personalityClass import PersonalityVector
from playerClass import Controller, Player

if TYPE_CHECKING:
    from gameClass import Game


# Tunable constants for the new prompt-tracking mechanics. Pulled out so they
# can be moved into PersonalityVector or made trainable later.
# TRUST_DECAY_PER_UNANSWERED = 0.05  # per turn, per open prompt of our own
# TRUST_BONUS_ON_ANSWER = 0.15       # one-shot bump when opponent answers us


class AIController(Controller):
    """
    Trainable controller. Uses a fixed PersonalityVector and a per-game
    BeliefState to score available speech acts and sample a dialogue move.

    Discrete back-and-forth tracking:
      - "Prompt" speech acts (QUESTION, OFFER, ACCUSE) made by either player
        get added to the agent's belief state (own_open_prompts or
        opponent_open_prompts).
      - On the agent's turn, it rolls against its evasiveness trait to decide
        whether to respond to the opponent's most recent open prompt. If it
        responds, the new move's ref_move_id points to that prompt.
      - When opponent answers one of our prompts, trust_in_opponent jumps.
      - Each turn, trust_in_opponent decays proportional to our outstanding
        unanswered prompt count.
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
        act = move.act
        p = self.personality
        b = self.beliefs

        is_self = move.speaker_id == self.player.player_id
        is_prompt = act in PROMPT_ACTS

        # -------- prompt bookkeeping (new) --------
        if is_self:
            # Our own prompt becomes an outstanding prompt waiting on the opponent.
            if is_prompt:
                b.own_open_prompts.append(move.move_id)
            # We just answered one of the opponent's prompts.
            if move.ref_move_id is not None and move.ref_move_id in b.opponent_open_prompts:
                b.opponent_open_prompts.remove(move.ref_move_id)
        else:
            # Opponent just made a prompt; track it so we can decide later
            # whether to respond.
            if is_prompt:
                b.opponent_open_prompts.append(move.move_id)
            # Opponent referenced one of our prompts -> they answered us.
            # Bump trust on top of whatever the standard updates below do.
            if move.ref_move_id is not None and move.ref_move_id in b.own_open_prompts:
                b.own_open_prompts.remove(move.ref_move_id)
                b.trust_in_opponent += 0.1 * p.cooperativeness

        # -------- standard speech-act-driven belief updates --------
        # Only opponent moves should shift our read on the opponent. Our own
        # moves don't tell us anything new about them. (The prompt-bookkeeping
        # block above is allowed to touch our own moves because it's just
        # maintaining the open-prompts data structure, not updating beliefs.)
        if not is_self:
            if act == SpeechAct.PROMISE:
                b.suspicion_level -= 0.2 * b.trust_in_opponent
                b.trust_in_opponent += 0.1 * p.trust_baseline

            if act == SpeechAct.THREATEN:
                b.suspicion_level += 0.2 * b.trust_in_opponent

            if act == SpeechAct.ACCUSE:
                b.suspicion_level += 0.2 * p.aggression
                b.trust_in_opponent += 0.1 * p.cooperativeness

            if act == SpeechAct.OFFER:
                b.suspicion_level -= 0.2 * p.cooperativeness
                b.trust_in_opponent += 0.1 * p.trust_baseline

            if act == SpeechAct.QUESTION:
                b.trust_in_opponent -= (p.trust_baseline - p.aggression)

        b.clamp()
        return None

    def choose_dialogue_move(self, game: "Game") -> DialogueMove:
        # Per-turn trust decay for every unanswered prompt of ours.
        # Applied before scoring so any response decision uses the decayed value.
        if self.beliefs.own_open_prompts:
            self.beliefs.trust_in_opponent -= (
                (1 - self.personality.trust_baseline) * len(self.beliefs.own_open_prompts)
            )
            self.beliefs.clamp()

        # Decide whether to respond to the opponent's most-recent open prompt.
        # evasiveness in [0,1] is the probability of NOT responding.
        ref_move_id: int | None = None
        if self.beliefs.opponent_open_prompts:
            if random.random() > self.personality.evasiveness:
                ref_move_id = self.beliefs.opponent_open_prompts[-1]

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
            ref_move_id=ref_move_id,
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

        # PROMISE: claim "split".
        # Honest if intention is split. If intention is steal, only attractive
        # when lie_propensity is high (false promise as bait).
        promise_score = (
                p.trust_baseline if intends_split else
                p.lie_propensity
        )

        # THREATEN: aggressive tone, claim "steal".
        # Most natural when intention is steal; suspicion and aggression boost it.
        threaten_score = (
            + p.aggression
            + b.suspicion_level
            + (1.0 - p.lie_propensity if not intends_split else 0.0)
        )

        # ACCUSE: prompt, claim "steal"
        # Agent claims steal and prompts opponent to convince them not to.
        accuse_score = (
            + b.suspicion_level
            + p.aggression
        )

        ## OFFER: prompt, claim split
        # Agent claims split and prompts opponent to agree
        offer_score = (
            p.cooperativeness - p.aggression if intends_split else p.lie_propensity
        )

        ## QUESTION: prompt, no claim
        # Agent asks what opponent will do
        question_score = (
            1.0 - abs(p.trust_baseline - b.suspicion_level)
        )


        return {
            SpeechAct.PROMISE: promise_score,
            SpeechAct.THREATEN: threaten_score,
            SpeechAct.ACCUSE: accuse_score,
            SpeechAct.OFFER: offer_score,
            SpeechAct.QUESTION: question_score
        }

    def _sample(self, scores: dict[SpeechAct, float]) -> SpeechAct:
        """Softmax over scores, then sample one speech act."""
        acts = list(scores.keys())
        raw = [scores[a] for a in acts]
        # Subtract max for numerical stability.
        m = max(raw)
        weights = [math.exp(s - m) for s in raw]
        return random.choices(acts, weights=weights, k=1)[0]

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
        if act == SpeechAct.ACCUSE:
            return {
                "text": "You're lying about splitting, so I'll steal too. Why should I believe you?",
                "claim_action": "steal",
            }
        if act == SpeechAct.OFFER:
            return {
                "text": "If we both steal, we both get nothing. Why don't we both just split it?",
                "claim_action": "split",
            }
        if act == SpeechAct.QUESTION:
            return {
                "text": "What are you actually going to do? Split or steal?",
            }
        return {}