from pathlib import Path

from playerClass import Player
from gameClass import Game
from humanController import HumanController
from agentClass import AgentController, AgentPersonality
from aiController import AIController
from personalityClass import PersonalityVector


def choose_personality_file() -> PersonalityVector:
    """Prompt for a personality CSV path and load it into a PersonalityVector."""
    # If a `personalities/` directory exists alongside the script, show its
    # contents as a hint. Purely cosmetic — the user can still type any path.
    personalities_dir = Path("personalities")
    if personalities_dir.is_dir():
        csv_files = sorted(personalities_dir.glob("*.csv"))
        if csv_files:
            print("\n  Personality files found in personalities/:")
            for p in csv_files:
                print(f"    - {p}")

    while True:
        path_str = input("  Path to personality CSV: ").strip()
        if not path_str:
            print("  Path cannot be empty.")
            continue
        try:
            return PersonalityVector.from_csv(path_str)
        except FileNotFoundError:
            print(f"  File not found: {path_str}")
        except ValueError as e:
            print(f"  Could not load personality: {e}")


def choose_controller_for_player(player: Player):
    print(f"\nConfigure controller for {player.name}:")
    print("  1. Human")
    print("  2. Agent (rule-based baseline)")
    print("  3. AI (trainable, loads personality from file)")

    while True:
        controller_choice = input("Choose controller type (1, 2, or 3): ").strip()

        if controller_choice == "1":
            return HumanController(player)

        if controller_choice == "2":
            personalities = list(AgentPersonality)
            print("\n  Agent personalities:")
            for i, personality in enumerate(personalities, start=1):
                label = personality.name.replace("_", " ").title()
                print(f"  {i}. {label}")

            while True:
                pick = input(f"Choose personality (1-{len(personalities)}): ").strip()
                try:
                    index = int(pick) - 1
                    if 0 <= index < len(personalities):
                        return AgentController(player, personalities[index])
                    print(f"  Please enter a number between 1 and {len(personalities)}.")
                except ValueError:
                    print("  Invalid input, enter a number.")

        if controller_choice == "3":
            personality = choose_personality_file()
            return AIController(player, personality)

        print("  Invalid choice. Enter 1, 2, or 3.")


def main():
    print("=== Split or Steal ===\n")

    name1 = input("Enter name for Player 1: ").strip() or "Player 1"
    name2 = input("Enter name for Player 2: ").strip() or "Player 2"

    jackpot = 100.0
    max_turns = 12

    p1 = Player(player_id=1, name=name1)
    p2 = Player(player_id=2, name=name2)

    p1.controller = choose_controller_for_player(p1)
    p2.controller = choose_controller_for_player(p2)

    game = Game(players=[p1, p2], jackpot=jackpot, max_turns=max_turns)

    print(f"\nJackpot: ${jackpot:.2f} | Turns: {max_turns}")
    print("=" * 40)
    print("\n--- Dialogue Phase ---")

    game.run_dialogue_phase()

    print("\n--- Final Actions ---")
    print("(Each player chooses privately — don't let the other see!)\n")

    payouts = game.resolve_final_actions()

    p1_action = p1.controller.player  # just for payout display
    p2_action = p2.controller.player

    print("\n=== Results ===")
    print(f"  {p1.name} walked away with: ${payouts[p1.player_id]:.2f}")
    print(f"  {p2.name} walked away with: ${payouts[p2.player_id]:.2f}")

    if payouts[p1.player_id] == 0 and payouts[p2.player_id] == 0:
        print("  Both players stole — nobody wins!")
    elif payouts[p1.player_id] == jackpot or payouts[p2.player_id] == jackpot:
        winner = p1.name if payouts[p1.player_id] == jackpot else p2.name
        print(f"  {winner} stole the jackpot!")
    else:
        print("  Both players split — everyone wins!")


if __name__ == "__main__":
    main()