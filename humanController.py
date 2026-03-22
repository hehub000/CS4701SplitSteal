from __future__ import annotations
from typing import TYPE_CHECKING
from playerClass import Player, Controller
from dialogueClass import DialogueMove, SpeechAct, FinalAction

if TYPE_CHECKING:
    from gameClass import Game


class HumanController(Controller):
    def __init__(self, player: Player):
        super().__init__(player)
        self._move_counter = 0

    def observe_dialogue_move(self, game: Game, move: DialogueMove) -> None:
        speaker = game.get_player(move.speaker_id)
        ref_str = f" [refs move {move.ref_move_id}]" if move.ref_move_id is not None else ""
        payload_str = f" {move.payload}" if move.payload else ""
        print(f"  [{move.move_id}] {speaker.name}: {move.act.name}{payload_str}{ref_str}")

    def choose_dialogue_move(self, game: Game) -> DialogueMove:
        print(f"\n--- {self.player.name}'s turn (turn {game.current_turn}) ---")

        acts = list(SpeechAct)
        for i, act in enumerate(acts):
            print(f"  {i + 1}. {act.name}")

        while True:
            try:
                choice = int(input("Choose a speech act (number): ")) - 1
                if 0 <= choice < len(acts):
                    chosen_act = acts[choice]
                    break
                print(f"  Please enter a number between 1 and {len(acts)}.")
            except ValueError:
                print("  Invalid input, enter a number.")

        ref_move_id = None
        if len(game.dialogue_history) > 0:
            print("  Prior moves:")
            for m in game.dialogue_history.moves:
                speaker = game.get_player(m.speaker_id)
                print(f"    [{m.move_id}] {speaker.name}: {m.act.name}")
            ref_input = input("  Reference a prior move by ID? (leave blank to skip): ").strip()
            if ref_input:
                try:
                    ref_id = int(ref_input)
                    if game.dialogue_history.get(ref_id) is not None:
                        ref_move_id = ref_id
                    else:
                        print("  Move ID not found, skipping reference.")
                except ValueError:
                    print("  Invalid ID, skipping reference.")

        self._move_counter += 1
        move_id = int(f"{self.player.player_id}{self._move_counter:03d}")

        return DialogueMove(
            move_id=move_id,
            speaker_id=self.player.player_id,
            act=chosen_act,
            turn_number=game.current_turn,
            ref_move_id=ref_move_id,
        )

    def choose_final_action(self, game: Game) -> FinalAction:
        print(f"\n--- {self.player.name}: choose your final action ---")
        print("  1. SPLIT")
        print("  2. STEAL")
        while True:
            try:
                choice = int(input("Your choice (1 or 2): "))
                if choice == 1:
                    return FinalAction.SPLIT
                elif choice == 2:
                    return FinalAction.STEAL
                print("  Enter 1 or 2.")
            except ValueError:
                print("  Invalid input, enter 1 or 2.")
