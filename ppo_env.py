from __future__ import annotations

import hashlib
import random
from typing import Any, Callable

import numpy as np
import torch

from agentClass import AgentController
from beliefStateClass import BeliefState
from dialogueClass import DialogueMove, FinalAction, PROMPT_ACTS, SpeechAct
from personalityClass import PersonalityVector
from playerClass import Controller, Player

from ppo_buffer import RolloutBuffer
from ppo_model import ActorCritic


DIALOGUE_ACTS: tuple[SpeechAct, ...] = tuple(SpeechAct)


def _stable_float(value: Any) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64 - 1)


def _append_feature(features: list[float], value: Any) -> None:
    if value is None:
        features.append(0.0)
        return
    if isinstance(value, bool):
        features.append(1.0 if value else 0.0)
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        features.append(float(value))
        return
    if hasattr(value, "value"):
        raw = getattr(value, "value")
        if isinstance(raw, (int, float)):
            features.append(float(raw))
            return
        features.append(_stable_float(raw))
        return
    features.append(_stable_float(value))


def build_payload(act: SpeechAct) -> dict[str, str]:
    if act == SpeechAct.PROMISE:
        return {"text": "I will split. We both do better that way.", "claim_action": "split"}
    if act == SpeechAct.THREATEN:
        return {"text": "If you do not split, I will steal.", "claim_action": "steal"}
    if act == SpeechAct.ACCUSE:
        return {"text": "I do not trust you, so I am leaning steal.", "claim_action": "steal"}
    if act == SpeechAct.OFFER:
        return {"text": "Let us both split and walk away with something.", "claim_action": "split"}
    if act == SpeechAct.QUESTION:
        return {"text": "What are you actually going to do?"}
    return {}


def action_to_speech_act(action: int) -> SpeechAct:
    return DIALOGUE_ACTS[action % len(DIALOGUE_ACTS)]


def int_to_final_action(action: int) -> FinalAction:
    return FinalAction.SPLIT if action == 0 else FinalAction.STEAL


def encode_game_state(game: Any, player: Player, opponent: Player | None, obs_dim: int) -> np.ndarray:
    features: list[float] = []

    for attr in (
        "jackpot",
        "max_turns",
        "current_turn",
        "turn_index",
        "turn",
        "dialogue_step",
        "dialogue_turn",
        "phase_index",
        "current_player_index",
    ):
        _append_feature(features, getattr(game, attr, None))

    history = getattr(game, "dialogue_history", None)
    _append_feature(features, len(history) if history is not None else 0)

    for actor in (player, opponent):
        if actor is None:
            features.extend([0.0] * 8)
            continue
        for attr in ("player_id", "money", "score", "stake", "bid", "last_reward"):
            _append_feature(features, getattr(actor, attr, None))
        _append_feature(features, getattr(actor, "name", None))

    controller = getattr(player, "controller", None)
    if controller is not None:
        personality = getattr(controller, "personality", None)
        if personality is not None:
            for attr in ("aggression", "trust_baseline", "lie_propensity", "evasiveness", "cooperativeness"):
                _append_feature(features, getattr(personality, attr, None))

    if hasattr(controller, "beliefs"):
        beliefs = getattr(controller, "beliefs")
        for attr in ("trust_in_opponent", "suspicion_level"):
            _append_feature(features, getattr(beliefs, attr, None))
        _append_feature(features, len(getattr(beliefs, "own_open_prompts", [])))
        _append_feature(features, len(getattr(beliefs, "opponent_open_prompts", [])))

    if history is not None and getattr(history, "moves", None):
        last_move = history.moves[-1]
        _append_feature(features, last_move.act)
        _append_feature(features, last_move.speaker_id)
        _append_feature(features, last_move.ref_move_id is not None)
    else:
        features.extend([0.0, 0.0, 0.0])

    if len(features) < obs_dim:
        features.extend([0.0] * (obs_dim - len(features)))
    return np.asarray(features[:obs_dim], dtype=np.float32)


class PPOController(Controller):
    def __init__(
        self,
        player: Player,
        model: ActorCritic,
        buffer: RolloutBuffer,
        *,
        obs_dim: int,
        device: str | None = None,
        seed_personality: PersonalityVector | None = None,
    ) -> None:
        super().__init__(player)
        self.model = model
        self.buffer = buffer
        self.obs_dim = obs_dim
        self.device = device or next(model.parameters()).device.type
        self.personality = seed_personality or PersonalityVector()
        self.beliefs = BeliefState.initialize_from_personality(self.personality)
        self._move_counter = 0
        self._cached_final_action: FinalAction | None = None

    def reset_for_new_game(self) -> None:
        # Update personality from model if learning is enabled
        if hasattr(self.model, 'learn_personality') and self.model.learn_personality:
            personality_dict = self.model.get_personality_dict()
            self.personality.aggression = personality_dict['aggression']
            self.personality.trust_baseline = personality_dict['trust_baseline']
            self.personality.lie_propensity = personality_dict['lie_propensity']
            self.personality.evasiveness = personality_dict['evasiveness']
            self.personality.cooperativeness = personality_dict['cooperativeness']
        self.beliefs = BeliefState.initialize_from_personality(self.personality)
        self._move_counter = 0
        self._cached_final_action = None

    def begin_episode(self, game: Any) -> None:
        self.reset_for_new_game()

    def _opponent_for(self, game: Any) -> Player | None:
        for candidate in getattr(game, "players", []):
            if candidate.player_id != self.player.player_id:
                return candidate
        return None

    def _observe(self, game: Any) -> np.ndarray:
        return encode_game_state(game, self.player, self._opponent_for(game), self.obs_dim)

    def _record_transition(self, observation: np.ndarray, action: int, logprob: float, value: float, head: str, reward: float) -> None:
        self.buffer.add(
            observation=observation,
            action=action,
            logprob=logprob,
            reward=reward,
            done=False,
            value=value,
            head=head,
        )

    def _dialogue_reward(self, act: SpeechAct) -> float:
        reward = 0.01
        intends_split = self.beliefs.intention == FinalAction.SPLIT
        if act in {SpeechAct.PROMISE, SpeechAct.OFFER}:
            reward += 0.04 if intends_split else -0.04
        elif act in {SpeechAct.THREATEN, SpeechAct.ACCUSE}:
            reward += 0.04 if not intends_split else -0.02
        elif act == SpeechAct.QUESTION:
            reward += 0.02 if self.beliefs.opponent_open_prompts else 0.0
        if act in PROMPT_ACTS:
            reward += 0.01
        reward += 0.01 * (self.beliefs.trust_in_opponent - self.beliefs.suspicion_level)
        return reward

    def observe_dialogue_move(self, game: Any, move: DialogueMove) -> None:
        p = self.personality
        b = self.beliefs
        is_self = move.speaker_id == self.player.player_id
        is_prompt = move.act in PROMPT_ACTS

        if is_self:
            if is_prompt:
                b.own_open_prompts.append(move.move_id)
            if move.ref_move_id is not None and move.ref_move_id in b.opponent_open_prompts:
                b.opponent_open_prompts.remove(move.ref_move_id)
        else:
            if is_prompt:
                b.opponent_open_prompts.append(move.move_id)
            if move.ref_move_id is not None and move.ref_move_id in b.own_open_prompts:
                b.own_open_prompts.remove(move.ref_move_id)
                b.trust_in_opponent += 0.1 * p.cooperativeness

        if not is_self:
            if move.act == SpeechAct.PROMISE:
                b.suspicion_level -= 0.2 * b.trust_in_opponent
                b.trust_in_opponent += 0.1 * p.trust_baseline
            if move.act == SpeechAct.THREATEN:
                b.suspicion_level += 0.2 * b.trust_in_opponent
            if move.act == SpeechAct.ACCUSE:
                b.suspicion_level += 0.2 * p.aggression
                b.trust_in_opponent += 0.1 * p.cooperativeness
            if move.act == SpeechAct.OFFER:
                b.suspicion_level -= 0.2 * p.cooperativeness
                b.trust_in_opponent += 0.1 * p.trust_baseline
            if move.act == SpeechAct.QUESTION:
                b.trust_in_opponent -= (p.trust_baseline - p.aggression)

        b.clamp()
        b.intention = FinalAction.STEAL if b.suspicion_level > p.trust_baseline else FinalAction.SPLIT

    def choose_dialogue_move(self, game: Any) -> DialogueMove:
        if self.beliefs.own_open_prompts:
            self.beliefs.trust_in_opponent -= (1.0 - self.personality.trust_baseline) * len(self.beliefs.own_open_prompts)
            self.beliefs.clamp()

        ref_move_id: int | None = None
        if self.beliefs.opponent_open_prompts and random.random() > self.personality.evasiveness:
            ref_move_id = self.beliefs.opponent_open_prompts[-1]

        observation = self._observe(game)
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        action, logprob, _, value = self.model.act(obs_tensor, head="dialogue", deterministic=False)
        speech_act = action_to_speech_act(action)

        self._move_counter += 1
        move_id = int(f"{self.player.player_id}{self._move_counter:03d}")
        self._record_transition(observation, action, logprob, value, "dialogue", self._dialogue_reward(speech_act))

        return DialogueMove(
            move_id=move_id,
            speaker_id=self.player.player_id,
            act=speech_act,
            payload=build_payload(speech_act),
            turn_number=game.current_turn,
            ref_move_id=ref_move_id,
        )

    def choose_final_action(self, game: Any) -> FinalAction:
        if self._cached_final_action is not None:
            return self._cached_final_action

        observation = self._observe(game)
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        action, logprob, _, value = self.model.act(obs_tensor, head="final", deterministic=False)
        final_action = int_to_final_action(action)

        self._record_transition(observation, action, logprob, value, "final", 0.0)
        self._cached_final_action = final_action
        return final_action

    def apply_terminal_reward(self, reward: float) -> None:
        self.buffer.set_last_reward(reward, done=True)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name in {"choose_action", "choose_speech_act", "make_speech_act", "respond", "generate_response", "choose_dialogue_turn", "choose_move", "speak", "talk", "select_action"}:
            return self.choose_dialogue_move
        raise AttributeError(name)
