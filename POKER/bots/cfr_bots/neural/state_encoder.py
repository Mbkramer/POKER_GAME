"""
state_encoder.py  (v4)
---------------------------------------------
Encodes an NLHGameState into a fixed-size feature vector for CFRNet.

v4 change:
  CHANGE  Board bucket one-hot expanded from 8 → 12 buckets to match
          postflop_equity_bucket() in preflop_abstraction.py v4.
          This shifts every feature at index ≥ 39 up by 4, and increases
          N_FEATURES from 73 → 77. N_BOARD_BUCKETS added as named constant.

v3 change (preserved):
  CHANGE  Preflop hand bucket one-hot expanded from 8 → 13 buckets.

v2 fixes (preserved):
  FIX 1  [26] history length clamped to [0,1].
  FIX 2  [63] stack-to-call ratio replaces duplicate seat feature.
  FIX 3  [67] aggressor flag replaces duplicate tilt feature.
  FIX 4  [20][21] pot and current_bet clamped to [0,1].

Feature vector (77 dims):

  [0]      acting seat index          (normalised by n_players-1)
  [1-6]    seat one-hot               (6 dims, seats 0-5)
  [7-19]   hand bucket one-hot        (13 preflop buckets)
  [20]     pot size                   (normalised by wallet, clamped [0,1])
  [21]     current bet                (normalised by wallet, clamped [0,1])
  [22]     acting player's stack      (normalised by wallet)
  [23]     amount owed to call        (normalised by wallet)
  [24]     n_raises this street       (normalised by MAX_RAISES=4)
  [25]     n_players still active     (normalised by n_players)
  [26]     history length             (normalised by 10, clamped [0,1])
  [27-30]  street one-hot             (PRE/FLP/TRN/RVR -- 4 dims)
  [31-42]  board bucket one-hot       (12 postflop buckets)
  [43]     n_community_cards          (normalised by 5)
  [44]     postflop hand strength     (0-1, phevaluator-based)
  [45]     flush draw                 (0 or 1)
  [46]     straight draw              (0 or 1)
  [47]     bias                       (always 1.0)
  [48]     pot odds                   (call / (pot + call))
  [49]     stack-to-pot ratio         (capped at 10x, normalised)
  [50]     avg opponent stack         (normalised by wallet)
  [51]     min opponent stack         (normalised by wallet)
  [52]     players yet to act         (normalised by n_players)
  [53]     actor VPIP
  [54]     actor PFR
  [55]     actor postflop aggression factor
  [56]     actor tilt factor
  [57]     actor fold-to-cbet
  [58]     avg opp VPIP
  [59]     avg opp PFR
  [60]     avg opp postflop aggression
  [61]     avg opp tilt factor
  [62]     avg opp fold-to-cbet
  [63]     stack-to-call ratio        (to_call / stack, clamped [0,1])
  [64]     is_last_to_act             (0 or 1)
  [65]     pot_committed ratio        (bets[seat] / wallet)
  [66]     hands played confidence    (normalised, 0-1)
  [67]     aggressor flag             (last raiser seat, normalised)
  [68]     bias                       (always 1.0)
  [69]     actor 3-bet %
  [70]     actor C-bet %
  [71]     actor ATS (attempt to steal)
  [72]     actor WTSD%
  [73]     avg opp 3-bet %
  [74]     avg opp C-bet %
  [75]     avg opp fold-to-3bet
  [76]     bias                       (always 1.0)

N_ACTIONS = 5: [FOLD, CALL, RAISE_2, RAISE_4, ALLIN]
"""

from __future__ import annotations
import torch
import numpy as np
from collections import Counter, defaultdict

N_FEATURES      = 77 # 
N_ACTIONS       = 5
ALL_ACTIONS     = ["FOLD", "CALL", "RAISE_2", "RAISE_4", "ALLIN"]
N_SEATS         = 6
N_BUCKETS       = 13   # preflop buckets
N_BOARD_BUCKETS = 12   # postflop board equity buckets
MAX_RAISES      = 4
N_STREETS       = 4

# ── Player profile tracker ────────────────────────────────────────────────────

class PlayerProfile:
    def __init__(self):
        self.hands_played         = 0
        self.saw_flop_count       = 0
        self.vpip_count           = 0
        self.pfr_count            = 0
        self.three_bet_count      = 0
        self.three_bet_opps       = 0
        self.fold_to_3bet_count   = 0
        self.faced_3bet_count     = 0
        self.ats_count            = 0
        self.ats_opps             = 0
        self.postflop_bets        = 0
        self.postflop_checks      = 0
        self.cbet_count           = 0
        self.cbet_opps            = 0
        self.fold_to_cbet_count   = 0
        self.cbet_faced_count     = 0
        self.wtsd_count           = 0
        self.starting_stack       = 0.0
        self.recent_stacks        = []

    @property
    def vpip(self) -> float:
        if self.hands_played < 5: return 0.5
        return min(self.vpip_count / self.hands_played, 1.0)

    @property
    def pfr(self) -> float:
        if self.hands_played < 5: return 0.3
        return min(self.pfr_count / self.hands_played, 1.0)

    @property
    def three_bet_pct(self) -> float:
        if self.three_bet_opps < 5: return 0.08
        return min(self.three_bet_count / self.three_bet_opps, 1.0)

    @property
    def fold_to_3bet(self) -> float:
        if self.faced_3bet_count < 3: return 0.55
        return min(self.fold_to_3bet_count / self.faced_3bet_count, 1.0)

    @property
    def ats(self) -> float:
        if self.ats_opps < 5: return 0.3
        return min(self.ats_count / self.ats_opps, 1.0)

    @property
    def postflop_agg(self) -> float:
        total = self.postflop_bets + self.postflop_checks
        if total < 5: return 0.5
        return min(self.postflop_bets / total, 1.0)

    @property
    def cbet_pct(self) -> float:
        if self.cbet_opps < 3: return 0.6
        return min(self.cbet_count / self.cbet_opps, 1.0)

    @property
    def fold_to_cbet_rate(self) -> float:
        if self.cbet_faced_count < 3: return 0.5
        return min(self.fold_to_cbet_count / self.cbet_faced_count, 1.0)

    @property
    def wtsd(self) -> float:
        if self.saw_flop_count < 5: return 0.35
        return min(self.wtsd_count / self.saw_flop_count, 1.0)

    @property
    def tilt_factor(self) -> float:
        if len(self.recent_stacks) < 3: return 0.0
        losses = sum(
            max(0, self.recent_stacks[i] - self.recent_stacks[i + 1])
            for i in range(len(self.recent_stacks) - 1)
        )
        return min(losses / max(self.starting_stack * 2, 1.0), 1.0)

    def record_hand_start(self, stack):
        self.starting_stack = stack
        self.hands_played  += 1

    def record_hand_end(self, stack, reached_showdown=False):
        self.recent_stacks.append(stack)
        if len(self.recent_stacks) > 5:
            self.recent_stacks.pop(0)
        if reached_showdown:
            self.wtsd_count += 1

    def record_vpip(self):        self.vpip_count += 1
    def record_pfr(self):         self.pfr_count  += 1
    def record_3bet(self):        self.three_bet_count += 1; self.three_bet_opps += 1
    def record_3bet_opp(self):    self.three_bet_opps  += 1
    def record_faced_3bet(self, folded):
        self.faced_3bet_count += 1
        if folded: self.fold_to_3bet_count += 1
    def record_ats_opp(self, attempted):
        self.ats_opps += 1
        if attempted: self.ats_count += 1
    def record_saw_flop(self):    self.saw_flop_count += 1
    def record_postflop_action(self, aggressive):
        if aggressive: self.postflop_bets   += 1
        else:          self.postflop_checks += 1
    def record_cbet_opp(self, fired):
        self.cbet_opps += 1
        if fired: self.cbet_count += 1
    def record_cbet_faced(self, folded):
        self.cbet_faced_count += 1
        if folded: self.fold_to_cbet_count += 1


_PLAYER_PROFILES: dict[int, PlayerProfile] = defaultdict(PlayerProfile)

def get_profile(seat: int) -> PlayerProfile:
    return _PLAYER_PROFILES[seat]

def reset_profiles():
    _PLAYER_PROFILES.clear()


# ── Postflop hand strength ────────────────────────────────────────────────────

def _postflop_hand_strength(hole_cards, community_cards) -> float:
    if not community_cards or not hole_cards or len(hole_cards) < 2:
        return 0.0
    try:
        from phevaluator import evaluate_cards

        _SUIT_MAP = {
            'SPADES': 's', 'HEARTS': 'h', 'DIAMONDS': 'd', 'CLUBS': 'c',
            's': 's', 'h': 'h', 'd': 'd', 'c': 'c',
        }

        def _to_ph(card):
            if hasattr(card, 'value') and hasattr(card, 'suit'):
                rank = 'T' if card.value == '10' else card.value
                suit = _SUIT_MAP.get(card.suit, card.suit[0].lower())
                return f"{rank}{suit}"
            rank, suit = card[0], card[1]
            rank = 'T' if rank == '10' else rank
            suit = _SUIT_MAP.get(suit, suit[0].lower())
            return f"{rank}{suit}"

        all_cards  = list(hole_cards) + list(community_cards)
        ph_strings = [_to_ph(c) for c in all_cards]
        rank       = evaluate_cards(*ph_strings)   # 1=best, 7462=worst
        return float(1.0 - (rank - 1) / 7461.0)
    except Exception:
        return 0.0

def _get_rank_suit(card):
    if isinstance(card, tuple):
        return card[0], card[1]
    return getattr(card, 'rank', getattr(card, 'value', '')), card.suit

def _draw_features(hole_cards, community_cards):
    if len(community_cards) < 3 or len(hole_cards) < 2:
        return 0.0, 0.0
    try:
        all_cards = list(hole_cards) + list(community_cards)
        suits = [_get_rank_suit(c)[1] for c in all_cards]
        suit_counts = Counter(suits)
        flush_draw = 1.0 if max(suit_counts.values()) == 4 else 0.0

        _RANK_VAL = {
            '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,
            '9':9,'T':10,'J':11,'Q':12,'K':13,'A':14,
        }
        ranks = sorted(set(
            _RANK_VAL.get(_get_rank_suit(c)[0], 0)
            for c in all_cards
        ))
        straight_draw = 0.0
        for i in range(len(ranks) - 3):
            if ranks[i + 3] - ranks[i] == 3:
                straight_draw = 1.0
                break
        return flush_draw, straight_draw
    except Exception:
        return 0.0, 0.0


# ── Main encoder ──────────────────────────────────────────────────────────────

def _normalize(value, min_value, max_value):
    """Normalize a value to the range [0, 1]."""
    if max_value == min_value:
        return 0.0
    return (value - min_value) / (max_value - min_value)


def encode_state(state) -> torch.Tensor:
    """
    Encode an NLHGameState (or EngineStateProxy) into a 73-dim float32 tensor.
    """
    fv     = np.zeros(N_FEATURES, dtype=np.float32)
    seat   = state.to_move
    n      = state.n_players
    wallet = max(state.wallet, 1.0)

    # ── [0] Seat index ────────────────────────────────────────────────────────
    fv[0] = seat / max(n - 1, 1)

    # ── [1-6] Seat one-hot ────────────────────────────────────────────────────
    if 0 <= seat < 6:
        fv[1 + seat] = 1.0

    # ── [7-19] Hand bucket one-hot (13 buckets) ───────────────────────────────
    bucket = state.hands[seat]
    if 0 <= bucket < N_BUCKETS:
        fv[7 + bucket] = 1.0

    # ── [20-23] Chip counts ───────────────────────────────────────────────────
    to_call = max(state.current_bet - state.bets[seat], 0.0)
    fv[20]  = min(state.pot / wallet, 1.0)
    fv[21]  = min(state.current_bet / wallet, 1.0)
    fv[22]  = min(max(state.stacks[seat] / wallet, 0.0), 1.0)
    fv[23]  = min(to_call / wallet, 1.0)

    # ── [24] Raises this street ───────────────────────────────────────────────
    fv[24] = state.n_raises / MAX_RAISES

    # ── [25] Active players ───────────────────────────────────────────────────
    n_active = sum(
        1 for i in range(n)
        if i not in state.folded and state.stacks[i] > 0
    )
    fv[25] = n_active / max(n, 1)

    # ── [26] History length ───────────────────────────────────────────────────
    fv[26] = min(len(state.action_history) / 10.0, 1.0)

    # ── [27-30] Street one-hot ────────────────────────────────────────────────
    street = getattr(state, 'street', 0)
    if 0 <= street < N_STREETS:
        fv[27 + street] = 1.0

    # ── [31-42] Board texture bucket one-hot (12 buckets) ────────────────────
    community = getattr(state, 'community_cards', [])
    if community:
        bb = N_BOARD_BUCKETS - 1   # default: weakest bucket
        try:
            from cfr_bots.cfr.nlh_gamestate import _board_bucket
            bb = _board_bucket(community)
        except ImportError:
            pass
        if 0 <= bb < N_BOARD_BUCKETS:
            fv[31 + bb] = 1.0

    # ── [43] Community card count ─────────────────────────────────────────────
    fv[43] = len(community) / 5.0

    # ── [44-46] Postflop hand strength + draws ────────────────────────────────
    seat = state.to_move
    hole_cards_all = getattr(state, 'hole_cards', None)
    if hole_cards_all and community:
        hole_cards_seat = hole_cards_all[seat]
        fv[44] = _postflop_hand_strength(hole_cards_seat, community)
        flush_draw, straight_draw = _draw_features(hole_cards_seat, community)
        fv[45] = flush_draw
        fv[46] = straight_draw

    # ── [47] Bias ─────────────────────────────────────────────────────────────
    fv[47] = 1.0

    # ── [48] Pot odds ─────────────────────────────────────────────────────────
    pot_plus_call = state.pot + to_call
    fv[48] = (to_call / pot_plus_call) if pot_plus_call > 0 else 0.0

    # ── [49] Stack-to-pot ratio (capped at 10x) ───────────────────────────────
    fv[49] = min(state.stacks[seat] / max(state.pot, 1.0), 10.0) / 10.0

    # ── [50-51] Opponent stacks ───────────────────────────────────────────────
    opp_stacks = [
        state.stacks[i] for i in range(n)
        if i != seat and i not in state.folded
    ]
    if opp_stacks:
        fv[50] = sum(opp_stacks) / len(opp_stacks) / wallet
        fv[51] = min(opp_stacks) / wallet

    # ── [52] Players yet to act ───────────────────────────────────────────────
    acted      = len(state.action_history)
    yet_to_act = max(n_active - acted - 1, 0)
    fv[52]     = yet_to_act / max(n, 1)

    # ── [53-57] Actor HUD stats ───────────────────────────────────────────────
    actor = get_profile(seat)
    fv[53] = actor.vpip
    fv[54] = actor.pfr
    fv[55] = actor.postflop_agg
    fv[56] = actor.tilt_factor
    fv[57] = actor.fold_to_cbet_rate

    # ── [58-62] Average opponent HUD stats ────────────────────────────────────
    opp_seats    = [i for i in range(n) if i != seat and i not in state.folded]
    opp_profiles = [get_profile(i) for i in opp_seats]

    if opp_profiles:
        fv[58] = sum(p.vpip              for p in opp_profiles) / len(opp_profiles)
        fv[59] = sum(p.pfr               for p in opp_profiles) / len(opp_profiles)
        fv[60] = sum(p.postflop_agg      for p in opp_profiles) / len(opp_profiles)
        fv[61] = sum(p.tilt_factor       for p in opp_profiles) / len(opp_profiles)
        fv[62] = sum(p.fold_to_cbet_rate for p in opp_profiles) / len(opp_profiles)
    else:
        fv[58] = 0.5; fv[59] = 0.3; fv[60] = 0.5; fv[61] = 0.0; fv[62] = 0.5

    # ── [63] Stack-to-call ratio ──────────────────────────────────────────────
    fv[63] = min(to_call / max(state.stacks[seat], 1.0), 1.0)

    # ── [64] Is last to act ───────────────────────────────────────────────────
    active_seats = [
        i for i in range(n)
        if i not in state.folded and state.stacks[i] > 0
    ]
    fv[64] = 1.0 if active_seats and seat == active_seats[-1] else 0.0

    # ── [65] Pot committed ratio ──────────────────────────────────────────────
    fv[65] = state.bets[seat] / wallet

    # ── [66] Hands played confidence ─────────────────────────────────────────
    fv[66] = min(actor.hands_played / 100.0, 1.0)

    # ── [67] Aggressor flag ───────────────────────────────────────────────────
    last_raiser_seat = -1
    if hasattr(state, 'last_raiser_index'):
        last_raiser_seat = state.last_raiser_index
    fv[67] = _normalize(last_raiser_seat, 0, state.n_players - 1) if last_raiser_seat != -1 else 0.0

    # ── [68] Bias ─────────────────────────────────────────────────────────────
    fv[68] = 1.0

    # ── [69-72] Actor extended HUD ────────────────────────────────────────────
    fv[69] = actor.three_bet_pct
    fv[70] = actor.cbet_pct
    fv[71] = actor.ats
    fv[72] = actor.wtsd

    # ── [73-75] Opponent extended HUD ────────────────────────────────────────
    if opp_profiles:
        fv[73] = sum(p.three_bet_pct for p in opp_profiles) / len(opp_profiles)
        fv[74] = sum(p.cbet_pct      for p in opp_profiles) / len(opp_profiles)
        fv[75] = sum(p.fold_to_3bet  for p in opp_profiles) / len(opp_profiles)
    else:
        fv[73] = 0.08; fv[74] = 0.60; fv[75] = 0.55

    # ── [76] Bias ─────────────────────────────────────────────────────────────
    fv[76] = 1.0

    return torch.tensor(fv, dtype=torch.float32)


# ── Policy helpers ────────────────────────────────────────────────────────────

def policy_tensor(sigma: dict, legal_actions: list) -> torch.Tensor:
    """
    Build a full N_ACTIONS policy tensor from a CFR sigma dict.
    Legal actions get their CFR probability; illegal actions get 0.
    """
    pi = torch.zeros(N_ACTIONS, dtype=torch.float32)
    for i, a in enumerate(ALL_ACTIONS):
        if a in sigma:
            pi[i] = sigma[a]
    return pi


def mask_logits(logits: torch.Tensor, legal_actions: list) -> torch.Tensor:
    """
    At inference: mask illegal action logits to -inf before softmax.
    logits: shape (N_ACTIONS,) or (batch, N_ACTIONS)
    """
    mask = torch.full_like(logits, float("-inf"))
    for i, a in enumerate(ALL_ACTIONS):
        if a in legal_actions:
            if logits.dim() == 1:
                mask[i] = logits[i]
            else:
                mask[:, i] = logits[:, i]
    return mask