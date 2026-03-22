from playerClass import Player
from gameClass import Game
from humanController import HumanController


def main():
    print("=== Split or Steal ===\n")

    name1 = input("Enter name for Player 1: ").strip() or "Player 1"
    name2 = input("Enter name for Player 2: ").strip() or "Player 2"

    jackpot = 100.0
    max_turns = 6

    p1 = Player(player_id=1, name=name1)
    p2 = Player(player_id=2, name=name2)

    p1.controller = HumanController(p1)
    p2.controller = HumanController(p2)

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
