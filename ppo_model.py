from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.distributions import Categorical


HeadName = Literal["dialogue", "final"]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 128, dialogue_action_dim: int = 5, final_action_dim: int = 2, learn_personality: bool = False) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.dialogue_action_dim = dialogue_action_dim
        self.final_action_dim = final_action_dim
        self.learn_personality = learn_personality

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.dialogue_head = nn.Linear(hidden_size, dialogue_action_dim)
        self.final_head = nn.Linear(hidden_size, final_action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Learnable personality parameters (5 dims: aggression, trust_baseline, lie_propensity, evasiveness, cooperativeness)
        if learn_personality:
            self.personality_params = nn.Parameter(torch.zeros(5))
        else:
            self.register_buffer('personality_params', torch.zeros(5))

    def _logits_for_head(self, hidden: torch.Tensor, head: HeadName) -> torch.Tensor:
        if head == "dialogue":
            return self.dialogue_head(hidden)
        if head == "final":
            return self.final_head(hidden)
        raise ValueError(f"Unknown head: {head}")

    def forward(self, obs: torch.Tensor, head: HeadName) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        hidden = self.encoder(obs)
        logits = self._logits_for_head(hidden, head)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, *, head: HeadName, deterministic: bool = False) -> tuple[int, float, float, float]:
        logits, value = self.forward(obs, head=head)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), float(logprob.item()), float(entropy.item()), float(value.squeeze(0).item())

    def evaluate_action(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        *,
        head: HeadName,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, head=head)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy, value
    
    def get_personality_dict(self) -> dict[str, float]:
        """Extract personality as dict with sigmoid activation to keep values in [0,1]."""
        p = torch.sigmoid(self.personality_params).detach().cpu().numpy()
        return {
            'aggression': float(p[0]),
            'trust_baseline': float(p[1]),
            'lie_propensity': float(p[2]),
            'evasiveness': float(p[3]),
            'cooperativeness': float(p[4]),
        }
    
    def set_personality_from_vector(self, pv) -> None:
        """Initialize personality parameters from a PersonalityVector using inverse sigmoid (logit)."""
        import math
        def logit(x: float) -> float:
            x = max(0.001, min(0.999, x))
            return math.log(x / (1.0 - x))
        
        vals = [
            logit(pv.aggression),
            logit(pv.trust_baseline),
            logit(pv.lie_propensity),
            logit(pv.evasiveness),
            logit(pv.cooperativeness),
        ]
        with torch.no_grad():
            self.personality_params.copy_(torch.tensor(vals, dtype=torch.float32, device=self.personality_params.device))
