# ui/ui_events.py
class PlayerAction:
    def __init__(self, action_type: str, amount: int = 0):
        self.type = action_type
        self.amount = amount
