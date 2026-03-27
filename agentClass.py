from __future__ import annotations

from enum import Enum
from random import choice
from typing import TYPE_CHECKING

from dialogueClass import DialogueMove, FinalAction, SpeechAct
from playerClass import Controller, Player

if TYPE_CHECKING:
    from gameClass import Game


class AgentPersonality(Enum):
    HONEST = "honest"
    DECEPTIVE_THREATEN = "deceptive_threaten"
    TIT_FOR_TAT = "tit_for_tat"
    ALWAYS_SPLIT = "always_split"
    ALWAYS_STEAL = "always_steal"


class AgentController(Controller):
    def __init__(self, player: Player, personality: AgentPersonality):
        super().__init__(player)
        self.personality = personality
        self._move_counter = 0

    def observe_dialogue_move(self, game: Game, move: DialogueMove) -> None:
        # Agents are stateless observers for now; strategy is computed from history each turn.
        return None

    def choose_dialogue_move(self, game: Game) -> DialogueMove:
        self._move_counter += 1
        move_id = int(f"{self.player.player_id}{self._move_counter:03d}")

        if self.personality == AgentPersonality.HONEST:
            act, payload = SpeechAct.PROMISE, {
                "text": "I will split and you should too. We both do better that way.",
                "claim_action": "split",
            }
        elif self.personality == AgentPersonality.DECEPTIVE_THREATEN:
            act, payload = SpeechAct.THREATEN, {
                "text": "If you do not split, I will steal and you get nothing.",
                "claim_action": "steal",
            }
        elif self.personality == AgentPersonality.TIT_FOR_TAT:
            mirrored = self._infer_opponent_last_claim(game)
            claim = "split" if mirrored is None else mirrored
            if claim == "split":
                act, payload = SpeechAct.PROMISE, {
                    "text": "I'll split if you split. It's a win-win.",
                    "claim_action": "split",
                }
            else:
                act, payload = SpeechAct.THREATEN, {
                    "text": "If you steal, I'll steal and we'll both get nothing.",
                    "claim_action": "steal",
                }
        elif self.personality == AgentPersonality.ALWAYS_SPLIT:
            act, payload = self._random_split_dialogue()
        elif self.personality == AgentPersonality.ALWAYS_STEAL:
            act, payload = self._random_steal_dialogue()
        else:
            act, payload = SpeechAct.PROMISE, {
                "text": "I will split.",
                "claim_action": "split",
            }

        return DialogueMove(
            move_id=move_id,
            speaker_id=self.player.player_id,
            act=act,
            payload=payload,
            turn_number=game.current_turn,
        )

    def choose_final_action(self, game: Game) -> FinalAction:
        if self.personality == AgentPersonality.HONEST:
            return FinalAction.SPLIT

        if self.personality == AgentPersonality.DECEPTIVE_THREATEN:
            return FinalAction.STEAL

        if self.personality == AgentPersonality.TIT_FOR_TAT:
            mirrored = self._infer_opponent_last_claim(game)
            if mirrored == "steal":
                return FinalAction.STEAL
            return FinalAction.SPLIT

        if self.personality == AgentPersonality.ALWAYS_SPLIT:
            return FinalAction.SPLIT

        if self.personality == AgentPersonality.ALWAYS_STEAL:
            return FinalAction.STEAL

        return FinalAction.SPLIT

    # human dialogue currently doesn't set payload.claim_action, so we're only 'inferring' using the speech action
    def _infer_opponent_last_claim(self, game: Game) -> str | None:
        opponent_id = game.other_player(self.player.player_id).player_id
        for move in reversed(game.dialogue_history.moves):
            if move.speaker_id != opponent_id:
                continue
            claim = move.payload.get("claim_action")
            if claim in {"split", "steal"}:
                return claim

            # Fallback for moves that don't include an explicit claim payload.
            if move.act == SpeechAct.PROMISE:
                return "split"
            if move.act == SpeechAct.THREATEN or move.act == SpeechAct.ACCUSE:
                return "steal"

        return None

    def _random_split_dialogue(self) -> tuple[SpeechAct, dict[str, str]]:
        options = [
            (
                SpeechAct.PROMISE,
                {"text": "I am choosing split.", "claim_action": "split"},
            ),
            (
                SpeechAct.ACCUSE,
                {
                    "text": "Stealing wrecks both of us. Split is smarter.",
                    "claim_action": "split",
                },
            ),
            (
                SpeechAct.THREATEN,
                {
                    "text": "If you push for steal, we both lose trust. I still split.",
                    "claim_action": "split",
                },
            ),
        ]
        return choice(options)

    def _random_steal_dialogue(self) -> tuple[SpeechAct, dict[str, str]]:
        options = [
            (
                SpeechAct.THREATEN,
                {"text": "I am taking the jackpot.", "claim_action": "steal"},
            ),
            (
                SpeechAct.PROMISE,
                {
                    "text": "Trust me, split is fine.",
                    "claim_action": "steal",
                },
            ),
            (
                SpeechAct.ACCUSE,
                {
                    "text": "You cannot be trusted, so I steal.",
                    "claim_action": "steal",
                },
            ),
        ]
        return choice(options)