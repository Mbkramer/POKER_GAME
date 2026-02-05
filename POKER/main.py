from core.player import Player
from core.table_state import TableState
from core.hand_evaluator import HandEvaluator
from ui.pygame_ui import PygameUI


def main():
    ui = PygameUI(None)
    ui.run(ui.screen)

if __name__ == "__main__":
    main()
