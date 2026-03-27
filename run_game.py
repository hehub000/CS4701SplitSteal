from playerClass import Player
from gameClass import Game
from humanController import HumanController
from agentClass import AgentController, AgentPersonality


def choose_controller_for_player(player: Player):
    print(f"\nConfigure controller for {player.name}:")
    print("  1. Human")
    print("  2. Agent")

    while True:
        controller_choice = input("Choose controller type (1 or 2): ").strip()
        if controller_choice == "1":
            return HumanController(player)
        if controller_choice == "2":
            break
        print("  Invalid choice. Enter 1 or 2.")

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


def main():
    print("=== Split or Steal ===\n")

    name1 = input("Enter name for Player 1: ").strip() or "Player 1"
    name2 = input("Enter name for Player 2: ").strip() or "Player 2"

    jackpot = 100.0
    max_turns = 6

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
