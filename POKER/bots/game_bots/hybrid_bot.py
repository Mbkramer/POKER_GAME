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
from bots.cfr_bots.neural.state_encoder import encode_state, mask_logits, N_FEATURES, N_ACTIONS, ALL_ACTIONS

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

class EngineStateProxy:
    """Minimal proxy for state_encoder.encode_state()"""
    def __init__(self, table: TableState, phase: GamePhase, acting_seat: int,
                 n_raises: int, action_history: list, hand_buckets: list):
        self.to_move = acting_seat
        self.n_players = table.num_players
        self.hands = hand_buckets
        self.wallet = table.wallet
        self.pot = table.pot
        self.current_bet = table.current_bet
        self.stacks = [p.cash for p in table.players]
        self.bets = [p.bet for p in table.players]
        self.n_raises = n_raises
        self.folded = {p.id for p in table.players if p.folded}
        self.action_history = action_history
        self.street = _PHASE_TO_STREET.get(phase, 0)
        self.community_cards = list(table.community_cards)


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
        self.actions_probs = []
        
    def reset_hand(self, phase: GamePhase = GamePhase.PREFLOP):
        """Call at hand start after hole cards dealt"""
        self._current_street = phase
        self._n_raises = 0
        self._action_history = []
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
        """Main decision point: assess hand, choose action, size bet"""
        player = self.table.players[self.player_index]
        bucket = self._hand_buckets[self.player_index]
        
        # Get neural assessment
        ev = self._get_neural_ev()

        # Compute pot odds and stack depth
        pot = self.table.pot
        to_call = self.table.current_bet - player.bet
        stack = player.cash
        spr = stack / max(self.table.buy_in, 1)  # stack-to-pot ratio
        
        # Street-specific strategy
        if self._current_street == GamePhase.PREFLOP:
            return self._decide_preflop(bucket, ev, to_call, pot, stack, spr)
        else:
            return self._decide_postflop(bucket, ev, to_call, pot, stack, spr)
    
    def _get_neural_ev(self) -> float:
        """Get value head estimate from neural net"""
        proxy = EngineStateProxy(
            table=self.table, phase=self._current_street,
            acting_seat=self.player_index, n_raises=self._n_raises,
            action_history=self._action_history, hand_buckets=self._hand_buckets
        )
        features = encode_state(proxy).unsqueeze(0)

        with torch.no_grad():
            policy_logits, _ = self.net(features)     # (1, N_ACTIONS)
            policy_logits    = policy_logits.squeeze(0)

        masked = mask_logits(policy_logits, ['FOLD', 'CALL', 'RAISE_2', 'RAISE_4', 'ALL_IN'])
        self.actions_probs = F.softmax(masked, dim=-1)

        self.net.eval()
        with torch.no_grad():
            _, value = self.net(features)
        return float(value.squeeze())
    
    # ── Preflop strategy ───────────────────────────────────────────────────────
    
    def _decide_preflop(self, bucket: int, ev: float, to_call: float, 
                       pot: float, stack: float, spr: float) -> tuple[str, PlayerAction]:
        """Preflop: hand-strength-dependent ranges"""
        player = self.table.players[self.player_index]
        
        # Premium (0-1): always aggressive
        if bucket <= 1:
            if to_call == 0:
                return self._make_raise(pot * 0.7, stack, "RAISE_2")  # open 3.5bb
            elif spr < 15:
                return self._make_allin(player, "ALLIN")  # shove short stacks
            else:
                return self._make_raise(to_call + pot * 0.8, stack, "RAISE_2")  # 3bet
        
        # Strong (2-3): value-oriented
        elif bucket <= 3:
            if to_call == 0:
                if random.random() < 0.7 * self.aggression:
                    return self._make_raise(pot * 0.6, stack, "RAISE_2")
                return self._make_call(to_call, "CALL")
            elif to_call < pot * 0.15:  # cheap to call
                return self._make_call(to_call, "CALL")
            elif ev > 0.3 and random.random() < 0.4 * self.aggression:
                return self._make_raise(to_call + pot * 0.7, stack, "RAISE_2")
            elif ev > 0.1:
                return self._make_call(to_call, "CALL")
            return self._make_fold("FOLD")
        
        # Medium (4-5): positional
        elif bucket <= 5:
            if to_call == 0:
                if random.random() < 0.3 * self.aggression:
                    return self._make_raise(pot * 0.5, stack, "RAISE_2")
                return self._make_call(0, "CALL")  # check
            elif to_call < pot * 0.1:
                return self._make_call(to_call, "CALL")
            elif ev > 0.2:
                return self._make_call(to_call, "CALL")
            return self._make_fold("FOLD")
        
        # Trash (6-7): tight
        else:
            if to_call == 0:
                return self._make_call(0, "CALL")  # check
            elif to_call < pot * 0.05 and ev > 0:
                return self._make_call(to_call, "CALL")
            return self._make_fold("FOLD")
    
    # ── Postflop strategy ──────────────────────────────────────────────────────
    
    def _decide_postflop(self, bucket: int, ev: float, to_call: float,
                        pot: float, stack: float, spr: float) -> tuple[str, PlayerAction]:
        """Postflop: EV-driven with pot geometry"""
        player = self.table.players[self.player_index]
        
        # Facing a bet
        if to_call > 0:
            pot_odds = to_call / (pot + to_call)
            
            if ev > pot_odds + 0.2:  # strong hand, raise
                if spr < 3:
                    return self._make_allin(player, "ALLIN")
                return self._make_raise(to_call + pot * 0.8, stack, "RAISE_2")
            
            elif ev > pot_odds:  # call
                return self._make_call(to_call, "CALL")
            
            elif ev > pot_odds - 0.15 and random.random() < 0.3:  # bluff raise
                return self._make_raise(to_call + pot * 1.2, stack, "RAISE_4")
            
            else:  # fold
                return self._make_fold("FOLD")
        
        # First to act (no bet facing)
        else:
            if ev > 0.4:  # strong, bet for value
                size = pot * (0.5 + ev * 0.3)  # 50-80% pot
                return self._make_raise(size, stack, "RAISE_2")
            
            elif ev > 0.2 and random.random() < 0.5 * self.aggression:  # probe
                return self._make_raise(pot * 0.4, stack, "RAISE_2")
            
            elif ev < -0.1 and random.random() < 0.2:  # bluff
                return self._make_raise(pot * 0.6, stack, "RAISE_2")
            
            else:  # check
                return self._make_call(0, "CALL")
    
    # ── Action constructors ────────────────────────────────────────────────────
    
    def _make_fold(self, cfr_action: str) -> tuple[str, PlayerAction]:
        return (cfr_action, PlayerAction(
            action_type=ActionType.FOLD,
            player_index=self.player_index,
            raise_amount=0
        ))
    
    def _make_call(self, to_call: float, cfr_action: str) -> tuple[str, PlayerAction]:
        if to_call <= 0:
            return (cfr_action, PlayerAction(
                action_type=ActionType.CHECK,
                player_index=self.player_index,
                raise_amount=0
            ))
        return (cfr_action, PlayerAction(
            action_type=ActionType.CALL,
            player_index=self.player_index,
            raise_amount=0
        ))
    
    def _make_raise(self, size: float, stack: float, cfr_action: str) -> tuple[str, PlayerAction]:
        player = self.table.players[self.player_index]
        min_raise = self.table.current_bet + max(self.table.last_raise_size, self.table.buy_in)
        raise_to = max(int(size), min_raise)           # enforce minimum
        raise_to = min(raise_to, int(player.bet + stack))  # cap at stack
        return (cfr_action, PlayerAction(
            action_type=ActionType.RAISE,
            player_index=self.player_index,
            raise_amount=raise_to
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