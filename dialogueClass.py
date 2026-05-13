from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum, auto

class SpeechAct(Enum):
  PROMISE = auto()
  THREATEN = auto()
  ACCUSE = auto()
  OFFER = auto()
  QUESTION = auto()

class FinalAction(Enum):
  SPLIT = auto()
  STEAL = auto()

@dataclass
class DialogueMove:
    move_id: int
    speaker_id: int
    act: SpeechAct
    payload: dict[str, Any] = field(default_factory=dict)
    ref_move_id: Optional[int] = None
    turn_number: int = 0

@dataclass
class DialogueHistory:
    moves: list[DialogueMove] = field(default_factory=list)

    def add(self, move: DialogueMove) -> None:
        self.moves.append(move)

    def get(self, move_id: int) -> Optional[DialogueMove]:
        for move in self.moves:
            if move.move_id == move_id:
                return move
        return None

    def __len__(self) -> int:
        return len(self.moves)