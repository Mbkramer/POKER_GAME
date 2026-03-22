"""
Hybrid Poker Bot: Neural CFR hand strength + behavioral betting logic

Combines the best of both worlds:
- Neural net provides hand buckets (0-7) and EV estimates
- Behavioral logic handles sizing and street-by-street strategy
- Result: less shove-happy, more human-like play
"""
from __future__ import annotations
from typing import Optional
import random

import torch
import torch.nn.functional as F
from core.card import Card
from core.table_state import TableState
from core.player_action import PlayerAction, ActionType
from engine.game_state import GamePhase

from bots.cfr_bots.neural.cfr_net import CFRNet
from bots.cfr_bots.cfr.preflop_abstraction import hand_to_bucket as _cfr_hand_to_bucket
from bots.cfr_bots.neural.combined_state_encoder import encode_state, mask_logits, N_FEATURES, N_ACTIONS, ALL_ACTIONS

# ── Hand bucket mapping ───────────────────────────────────────────────────────
_ENGINE_TO_CFR_RANK = {
    'A': 'A', 'K': 'K', 'Q': 'Q', 'J': 'J', '10': 'T',
    '9': '9', '8': '8', '7': '7', '6': '6', '5': '5', '4': '4', '3': '3', '2': '2',
}

def _hand_bucket(card1: Card, card2: Card) -> int:
    """Map two engine Cards to 0-7 preflop strength bucket."""
    r1 = _ENGINE_TO_CFR_RANK[card1.value]
    r2 = _ENGINE_TO_CFR_RANK[card2.value]
    suited = (card1.suit == card2.suit)
    return _cfr_hand_to_bucket(r1, r2, suited)


# ── Proxy for state encoder ───────────────────────────────────────────────────
_PHASE_TO_STREET = {
    GamePhase.PREFLOP: 0,
    GamePhase.FLOP: 1,
    GamePhase.TURN: 2,
    GamePhase.RIVER: 3,
}

MAX_RAISES = 2  

class EngineStateProxy:
    """Minimal proxy for combined_state_encoder.encode_state()"""
    def __init__(self, table: TableState, phase: GamePhase, acting_seat: int,
                 n_raises: int, action_history: list, hand_buckets: list):
        self.to_move         = acting_seat
        self.n_players       = table.num_players
        self.hands           = hand_buckets
        self.wallet          = table.wallet
        self.pot             = table.pot
        self.current_bet     = table.current_bet
        self.stacks          = [p.cash for p in table.players]
        self.bets            = [p.bet for p in table.players]
        self.n_raises        = n_raises
        self.folded          = {i for i, p in enumerate(table.players) if p.folded}
        self.street          = _PHASE_TO_STREET.get(phase, 0)
        self.community_cards = list(table.community_cards)
        self.hole_cards      = [p.hand for p in table.players]
        self.buyin           = table.buy_in
        self.last_raise_size = table.last_raise_size

        # Normalise action_history to (seat, action_str) tuples regardless of
        # what the call site passes. Handles three formats:
        #   plain strings:  "RAISE_2"          → (acting_seat, "RAISE_2")
        #   2-tuples:       (seat, "RAISE_2")  → unchanged
        #   3-tuples:       (seat, "RAISE_2", amount) → (seat, "RAISE_2")
        normalised = []
        for entry in action_history:
            if isinstance(entry, str):
                normalised.append((acting_seat, entry))
            else:
                normalised.append((entry[0], entry[1]))
        self.action_history = normalised

# ── Hybrid Bot ─────────────────────────────────────────────────────────────────
class HybridPokerBot:
    """
    Combines neural hand-strength assessment with behavioral betting patterns.
    
    Strategy tiers based on neural bucket:
      0-1: Premium (AA-TT, AK)     → aggressive
      2-3: Strong (99-66, AQ-AJ)   → value-oriented
      4-5: Medium (suited broadway) → positional
      6-7: Trash                   → tight
    """
    
    def __init__(self, net: CFRNet, player_index: int, table: TableState, 
                 phase: GamePhase, aggression: float = 1.0):
        self.net = net
        self.player_index = player_index
        self.table = table
        self.aggression = aggression  # 0.5 = tight, 1.0 = normal, 1.5 = loose
        
        # Per-street tracking
        self._current_street = phase
        self._n_raises = 0
        self._action_history = []
        self._hand_buckets = [7] * table.num_players

        #per action tracking
        self.actions_probs = {}
        
    def reset_hand(self, phase: GamePhase = GamePhase.PREFLOP):
        """Call at hand start after hole cards dealt"""
        self._current_street = phase
        self._n_raises = 0
        self._action_history = []
        self.actions_probs = {}
        self.table.last_raise_size = self.table.buy_in
        
        for i, player in enumerate(self.table.players):
            if len(player.hand) == 2:
                self._hand_buckets[i] = _hand_bucket(player.hand[0], player.hand[1])
            else:
                self._hand_buckets[i] = 7
    
    def update(self, table: TableState, new_phase: GamePhase):
        """Call on street transitions"""
        self._current_street = new_phase
        self.table = table
        self._n_raises = 0
        self._action_history = []
    
    def notify_action(self, action_str: str):
        """Call after every action to update history"""
        self._action_history.append(action_str)
        if action_str in ("RAISE_2", "RAISE_4", "ALLIN"):
            self._n_raises += 1
    
    # ── Core decision logic ────────────────────────────────────────────────────
    
    def decide(self) -> tuple[str, PlayerAction]:
        """Choose directly from the CFR average strategy."""
        player  = self.table.players[self.player_index]
        to_call = self.table.current_bet - player.bet
        stack   = player.cash

        legal = []
        if to_call > 0:
            legal += ["FOLD", "CALL"]
        else:
            legal += ["CHECK"]

        if self._n_raises < MAX_RAISES:
            legal += ["RAISE_2", "RAISE_4"]

        if stack > 0 and self._current_street != GamePhase.PREFLOP:
            legal.append("ALLIN")

        policy = self._get_avg_policy(legal)
        action_name = self._sample_action(policy, legal)

        chosen = self._execute_action(action_name, player, to_call, stack)
        print(f"[BOT DEBUG] abstract={action_name} concrete={chosen[1].action_type} amount={chosen[1].raise_amount}")
        return chosen


    def _get_avg_policy(self, legal_actions: list[str]) -> dict[str, float]:
        """
        Run the CFR net, mask illegal actions, softmax over legal actions only.
        Returns avg strategy as a name->probability dict.
        """
        proxy = EngineStateProxy(
            table=self.table,
            phase=self._current_street,
            acting_seat=self.player_index,
            n_raises=self._n_raises,
            action_history=self._action_history,
            hand_buckets=self._hand_buckets,
        )
        features = encode_state(proxy).unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            policy_logits, _ = self.net(features)   # (1, N_ACTIONS)
        policy_logits = policy_logits.squeeze(0)    # (N_ACTIONS,)

        masked = mask_logits(policy_logits, legal_actions)
        probs  = F.softmax(masked, dim=-1)          # (N_ACTIONS,)

        # Store for diagnostics / PHH logging
        self.actions_probs = probs

        return {
            a: float(probs[i])
            for i, a in enumerate(ALL_ACTIONS)
        }


    def _sample_action(self, policy: dict[str, float], legal_actions: list[str]) -> str:
        names = [a for a in ALL_ACTIONS if a in legal_actions]
        ranked = sorted(names, key=lambda a: policy.get(a, 0.0), reverse=True)

        best = ranked[0]
        if len(ranked) == 1:
            return best

        second = ranked[1]
        if policy.get(best, 0.0) - policy.get(second, 0.0) >= 0.10:
            return best

        return best

    def _execute_action(
        self,
        action_name: str,
        player,
        to_call: float,
        stack: float,
    ) -> tuple[str, PlayerAction]:
        """
        Translate sampled action name into a PlayerAction with pot-geometry sizing.
        """
        if action_name == "FOLD":
            return self._make_fold("FOLD")

        if action_name == "CHECK":
            return self._make_check("CHECK")

        if action_name == "CALL":
            return self._make_call("CALL")

        if action_name == "ALLIN":
            return self._make_allin(player, "ALLIN")

        if action_name in ("RAISE_2", "RAISE_4"):
            multiplier = 2 if action_name == "RAISE_2" else 4
            min_inc    = max(self.table.last_raise_size, self.table.buy_in)
            raise_to   = self.table.current_bet + multiplier * min_inc
            max_commit = player.bet + stack
            raise_to   = min(raise_to, max_commit)

            min_legal  = self.table.current_bet + self.table.last_raise_size
            if raise_to < min_legal:
                return self._make_allin(player, action_name)

            return self._make_raise(raise_to, stack, action_name)

        return self._make_check("CHECK")

    # ── Action constructors ────────────────────────────────────────────────────
    
    def _make_fold(self, cfr_action: str) -> tuple[str, PlayerAction]:
        return (cfr_action, PlayerAction(
            action_type=ActionType.FOLD,
            player_index=self.player_index,
            raise_amount=0
        ))
    
    def _make_check(self, cfr_action: str) -> tuple[str, PlayerAction]:
        return (cfr_action, PlayerAction(
        action_type=ActionType.CHECK,
        player_index=self.player_index,
        raise_amount=0
    ))

    def _make_call(self, cfr_action: str) -> tuple[str, PlayerAction]:
        return (cfr_action, PlayerAction(
            action_type=ActionType.CALL,
            player_index=self.player_index,
            raise_amount=0
        ))
    
    def _make_raise(self, raise_to: float, stack: float, cfr_action: str) -> tuple[str, PlayerAction]:
        """
        Emit a RAISE PlayerAction. raise_to is an absolute chip total already
        validated by _execute_action (>= engine min, <= player stack).
        """
        player = self.table.players[self.player_index]
        raise_to = min(int(raise_to), int(player.bet + stack))  # integer chips, stack-capped
        return (cfr_action, PlayerAction(
            action_type=ActionType.RAISE,
            player_index=self.player_index,
            raise_amount=int(raise_to)
        ))
    
    def _make_allin(self, player, cfr_action: str) -> tuple[str, PlayerAction]:
        """All-in shove"""
        raise_to = player.bet + player.cash
        return (cfr_action, PlayerAction(
            action_type=ActionType.RAISE,
            player_index=self.player_index,
            raise_amount=int(raise_to)
        ))
    
    def _print_action_probs(self):
        if not hasattr(self.actions_probs, '__len__') or len(self.actions_probs) == 0:
            print("Policy: (no decision yet)")
            return
        print(f"Policy: ", end="")
        for a, p in zip(ALL_ACTIONS, self.actions_probs):
            print(f"{a}={p:.2f}", end="  ")
        print()


# ── Factory function ───────────────────────────────────────────────────────────

def get_hybrid_bot(checkpoint_path: str = "POKER/bots/cfr_bots/checkpoints/best_nlh_4P_10B_500W.pt",
                   player_index: int = 0,
                   table: TableState = None,
                   phase: GamePhase = GamePhase.PREFLOP,
                   aggression: float = 1.0) -> HybridPokerBot:
    """
    Load hybrid bot with neural CFR weights.
    
    Args:
        checkpoint_path: Path to trained CFR model
        player_index: Seat index (0-based)
        table: TableState reference
        phase: Current game phase
        aggression: 0.5=tight, 1.0=normal, 1.5=loose
    """
    
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    net = CFRNet(
        n_features=ckpt.get("n_features", N_FEATURES),
        n_actions=ckpt.get("n_actions", N_ACTIONS),
        hidden_dim=128,
        n_blocks=3,
    ).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    
    return HybridPokerBot(net, player_index, table, phase, aggression)