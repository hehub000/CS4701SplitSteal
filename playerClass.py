from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from dialogueClass import DialogueMove, FinalAction

if TYPE_CHECKING:
    from gameClass import Game

@dataclass
class Player: #this class is not the AI agent itself, it just represents the existence of a player within a game instance
    player_id: int
    name: str
    money: float = 0.0
    private_info: dict[str, Any] = field(default_factory=dict)
    controller: Optional["Controller"] = None

class Controller(ABC): #later we we will write AIController, HumanController, and baseline AI such as AlwaysStealController, AlwaysSplitController etc.
    def __init__(self, player: Player):
        self.player = player

    @abstractmethod
    def observe_dialogue_move(self, game: "Game", move: DialogueMove) -> None:
        pass

    @abstractmethod
    def choose_dialogue_move(self, game: "Game") -> DialogueMove:
        pass

    @abstractmethod
    def choose_final_action(self, game: "Game") -> FinalAction:
        pass