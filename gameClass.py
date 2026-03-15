from dataclasses import dataclass, field
from playerClass import Player
from dialogueClass import DialogueMove, DialogueHistory, FinalAction

@dataclass
class Game:
    players: list[Player]
    jackpot: float
    max_turns: int = 6
    dialogue_history: DialogueHistory = field(default_factory=DialogueHistory)
    current_turn: int = 0

    def get_player(self, player_id: int) -> Player:
        for player in self.players:
            if player.player_id == player_id:
                return player
        raise ValueError(f"No player with id {player_id}")

    def other_player(self, player_id: int) -> Player:
        for player in self.players:
            if player.player_id != player_id:
                return player
        raise ValueError("No opposing player found")

    def record_dialogue_move(self, move: DialogueMove) -> None:
        self.dialogue_history.add(move)
        for player in self.players:
            if player.controller is not None:
                player.controller.observe_dialogue_move(self, move)

    def run_dialogue_phase(self) -> None:
        for turn in range(self.max_turns):
            self.current_turn = turn
            current_player = self.players[turn % len(self.players)]

            if current_player.controller is None:
                raise ValueError(f"Player {current_player.name} has no controller")

            move = current_player.controller.choose_dialogue_move(self)
            self.record_dialogue_move(move)

    def resolve_final_actions(self) -> dict[int, float]:
        actions = {}
        for player in self.players:
            if player.controller is None:
                raise ValueError(f"Player {player.name} has no controller")
            actions[player.player_id] = player.controller.choose_final_action(self)

        p1, p2 = self.players
        a1 = actions[p1.player_id]
        a2 = actions[p2.player_id]

        payouts = {p1.player_id: 0.0, p2.player_id: 0.0}

        if a1 == FinalAction.SPLIT and a2 == FinalAction.SPLIT:
            payouts[p1.player_id] = self.jackpot / 2
            payouts[p2.player_id] = self.jackpot / 2
        elif a1 == FinalAction.STEAL and a2 == FinalAction.SPLIT:
            payouts[p1.player_id] = self.jackpot
        elif a1 == FinalAction.SPLIT and a2 == FinalAction.STEAL:
            payouts[p2.player_id] = self.jackpot

        p1.money += payouts[p1.player_id]
        p2.money += payouts[p2.player_id]

        return payouts
