"""
combined_state_encoder.py
------------------------------------------------
Merges range_equity_state_encoder.py (GTO/CFR features) with
state_encoder.py (HUD/exploitative features) into one unified encoder.

Villain range equity is estimated via a precomputed board-hit lookup table —
zero Monte Carlo, zero evaluator calls for the villain side, O(range_size)
per encode call (~microseconds).

Feature vector (80 dims):

IDENTITY (7)
  [0]      acting seat index              (normalised by n_players-1)
  [1-6]    seat one-hot                  (6 dims)

HAND STRENGTH (16)
  [7-22]   preflop bucket one-hot        (16 buckets, 0=strongest)

GAME STATE SCALARS (10)
  [23]     pot size                      (normalised by wallet)
  [24]     to_call                       (normalised by wallet)
  [25]     stack size                    (normalised by wallet)
  [26]     log SPR                       (log1p scale, capped)
  [27]     raises this street            (normalised by MAX_RAISES)
  [28]     active players                (normalised by n_players)
  [29]     history length                (normalised by 10)
  [30]     pot odds                      (call / (pot + call))
  [31]     pot committed ratio           (bets[seat] / wallet)
  [32]     stack-to-call ratio           (to_call / stack)

STREET / BOARD (11)
  [33-36]  street one-hot                (PRE/FLP/TRN/RVR)
  [37]     community card count          (normalised by 5)
  [38]     hand strength                 (phevaluator, raw [0,1])
  [39]     flush draw
  [40]     straight draw
  [41]     draw equity                   (blended fd+sd)
  [42]     is_last_to_act
  [43]     aggressor flag                (last raiser seat, normalised)

BOARD TEXTURE (5)
  [44]     paired board
  [45]     monotone board
  [46]     two-tone board
  [47]     connected board
  [48]     high-card board

PREFLOP CONTEXT (3)
  [49]     pf_unopened
  [50]     pf_vs_open
  [51]     pf_vs_3bet

POSITION / DEPTH (6)
  [52]     is_ip
  [53]     is_oop
  [54]     eff_bb_norm
  [55]     depth_short
  [56]     depth_mid
  [57]     depth_deep

HERO HUD (7)
  [58]     actor VPIP
  [59]     actor PFR
  [60]     actor 3-bet %
  [61]     actor C-bet %
  [62]     actor ATS
  [63]     actor postflop aggression
  [64]     actor WTSD%

VILLAIN HUD (6)
  [65]     avg opp VPIP
  [66]     avg opp PFR
  [67]     avg opp 3-bet %
  [68]     avg opp C-bet %
  [69]     avg opp fold-to-3bet
  [70]     avg opp fold-to-cbet

META (5)
  [71]     actor tilt factor
  [72]     hands-played confidence
  [73]     iteration_progress
  [74]     players yet to act
  [75]     bias

N_FEATURES = 76
N_ACTIONS  = 6  ["FOLD", "CHECK", "CALL", "RAISE_2", "RAISE_4", "ALLIN"]
"""

from __future__ import annotations
import math
import torch
import numpy as np
from collections import Counter, defaultdict
from phevaluator import evaluate_cards   

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES = 76
N_ACTIONS  = 6
ALL_ACTIONS = ["FOLD", "CHECK", "CALL", "RAISE_2", "RAISE_4", "ALLIN"]

N_SEATS    = 6
N_BUCKETS  = 16        # must match N_PREFLOP_BUCKETS in preflop_abstraction.py
N_STREETS  = 4
MAX_RAISES = 2

IDX_STREET_START  = 33
IDX_STREET_END    = 37
IDX_HAND_STRENGTH = 38
STREET_SLICE      = slice(IDX_STREET_START, IDX_STREET_END)

# ── Player profile tracker ────────────────────────────────────────────────────

class PlayerProfile:
    def __init__(self):
        self.hands_played       = 0
        self.saw_flop_count     = 0
        self.vpip_count         = 0
        self.pfr_count          = 0
        self.three_bet_count    = 0
        self.three_bet_opps     = 0
        self.fold_to_3bet_count = 0
        self.faced_3bet_count   = 0
        self.ats_count          = 0
        self.ats_opps           = 0
        self.postflop_bets      = 0
        self.postflop_checks    = 0
        self.cbet_count         = 0
        self.cbet_opps          = 0
        self.fold_to_cbet_count = 0
        self.cbet_faced_count   = 0
        self.wtsd_count         = 0
        self.starting_stack     = 0.0
        self.recent_stacks      = []

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

    @property
    def has_sufficient_data(self) -> bool:
        """True once we have enough hands to trust profile stats."""
        return self.hands_played >= 10

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


# ── Card utilities ────────────────────────────────────────────────────────────

RANK_MAP = {
    '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
    '8':8,'9':9,'T':10,'10':10,  
    'J':11,'Q':12,'K':13,'A':14
}

def rank(card):
    return RANK_MAP.get(card[0], 0) if isinstance(card, tuple) else RANK_MAP.get(card.value, 0)

def suit(card):
    return card[1] if isinstance(card, tuple) else card.suit

def _get_rank_suit(card):
    if isinstance(card, tuple):
        return card[0], card[1]
    return getattr(card, 'rank', getattr(card, 'value', '')), card.suit


# ── Hand strength ─────────────────────────────────────────────────────────────

def _postflop_hand_strength(hole_cards, community_cards) -> float:
    if not community_cards or not hole_cards or len(hole_cards) < 2:
        return 0.0
    try:

        _SUIT_MAP = {
            'SPADES': 's', 'HEARTS': 'h', 'DIAMONDS': 'd', 'CLUBS': 'c',
            's': 's', 'h': 'h', 'd': 'd', 'c': 'c',
        }

        def _to_ph(card):
            if hasattr(card, 'value') and hasattr(card, 'suit'):
                r = 'T' if card.value == '10' else card.value
                s = _SUIT_MAP.get(card.suit, card.suit[0].lower())
                return f"{r}{s}"
            r, s = card[0], card[1]
            r = 'T' if r == '10' else r
            s = _SUIT_MAP.get(s, s[0].lower())
            return f"{r}{s}"

        all_cards  = list(hole_cards) + list(community_cards)
        ph_strings = [_to_ph(c) for c in all_cards]
        score      = evaluate_cards(*ph_strings)
        return float(1.0 - (score - 1) / 7461.0)
    except Exception:
        return 0.0


# ── Draw detection ────────────────────────────────────────────────────────────

def _draw_features(hole_cards, community_cards):
    """Returns (flush_draw, straight_draw, draw_equity)."""
    if len(community_cards) < 3 or len(hole_cards) < 2:
        return 0.0, 0.0, 0.0
    try:
        all_cards   = list(hole_cards) + list(community_cards)
        suits_list  = [_get_rank_suit(c)[1] for c in all_cards]
        suit_counts = Counter(suits_list)
        flush_draw  = 1.0 if max(suit_counts.values()) == 4 else 0.0

        _RANK_VAL = {
            '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,
            '9':9,'T':10,'10':10,'J':11,'Q':12,'K':13,'A':14,
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

        draw_equity = flush_draw * 0.5 + straight_draw * 0.5
        return flush_draw, straight_draw, draw_equity
    except Exception:
        return 0.0, 0.0, 0.0


# ── Stack / position helpers ──────────────────────────────────────────────────

def _effective_stack_bb(state, seat: int) -> float:
    opps  = [i for i in range(state.n_players) if i != seat and i not in state.folded]
    if not opps:
        return 10.0
    eff   = min([state.stacks[seat]] + [state.stacks[i] for i in opps])
    buyin = getattr(state, 'buyin', None) or getattr(state, 'buy_in', None) or 10.0
    return eff / max(buyin, 1.0)


def _position_context(state, seat: int) -> float:
    """Returns continuous position score [0=OOP, 1=IP]."""
    n      = state.n_players
    street = getattr(state, 'street', 0)
    folded = getattr(state, 'folded', set())

    if street == 0:
        full_order = list(range(0, n))
    else:
        full_order = list(range(n - 2, n)) + list(range(0, n - 2))

    active_order = [s for s in full_order if s not in folded]

    if seat not in active_order:
        return 0.5

    position_index = active_order.index(seat)
    return position_index / max(len(active_order) - 1, 1)


# ── Board texture ─────────────────────────────────────────────────────────────

def board_texture(board):
    """Returns (paired, monotone, two_tone, connected, high_card) as floats."""
    if not board:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    suits_list  = [suit(c) for c in board]
    ranks_list  = [rank(c) for c in board]
    counts      = Counter(ranks_list)
    suit_counts = Counter(suits_list)

    paired    = 1.0 if max(counts.values()) >= 2 else 0.0
    monotone  = 1.0 if len(suit_counts) == 1 else 0.0
    two_tone  = 1.0 if len(suit_counts) == 2 else 0.0
    ranks_s   = sorted(ranks_list)
    connected = 1.0 if (len(ranks_s) >= 2 and max(ranks_s) - min(ranks_s) <= 4) else 0.0
    high_card = 1.0 if max(ranks_list) >= 12 else 0.0

    return paired, monotone, two_tone, connected, high_card


# ── Villain range equity via board-hit lookup ─────────────────────────────────
#
# The key insight: instead of Monte Carlo, we precompute for each abstract
# hand class a "board hit score" — how strongly that hand class connects
# with each board texture type. Villain equity is then a weighted sum over
# the range using the actual board texture as weights.  Pure arithmetic,
# zero sampling, ~microseconds per call.
#
# Board hit weights per hand property:
#   pair_affinity    — how likely to hit a paired board (high cards pair more)
#   flush_affinity   — suited hands spike on monotone/two-tone boards
#   straight_affinity— connected hands spike on connected boards
#   high_affinity    — high card hands stay strong on high-card boards
#
# These weights are precomputed once at module load.

_ALL_RANKS     = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']
_RANK_STRENGTH = {r: 12 - i for i, r in enumerate(_ALL_RANKS)}

# Each entry: (preflop_strength, rank1, rank2, suited, pair_aff, flush_aff, straight_aff, high_aff)
_HAND_CLASS_TABLE: list = []


def _init_hand_class_table():
    global _HAND_CLASS_TABLE
    if _HAND_CLASS_TABLE:
        return

    for i, r1 in enumerate(_ALL_RANKS):
        for r2 in _ALL_RANKS[i:]:
            r1v = _RANK_STRENGTH[r1]
            r2v = _RANK_STRENGTH[r2]
            hi  = max(r1v, r2v)
            lo  = min(r1v, r2v)
            gap = hi - lo

            if r1 == r2:
                # Pocket pair
                pf_str    = (r1v + r2v) / 24.0 + 0.15
                pair_aff  = 0.35 + r1v / 36.0    # sets — more likely on any board
                flush_aff = 0.0                   # pairs don't care about suit texture
                str_aff   = 0.05                  # low-grade straight potential
                high_aff  = 0.5 + r1v / 24.0     # high pairs dominate on high boards
                _HAND_CLASS_TABLE.append((pf_str, r1, r2, False,
                                          pair_aff, flush_aff, str_aff, high_aff))
            else:
                base = (r1v + r2v) / 24.0
                # Connectedness bonus for straight affinity
                str_aff = max(0.0, (4 - gap) / 4.0) * 0.4   # 0 if gap>4, 0.4 if connector

                # Suited variant
                _HAND_CLASS_TABLE.append((
                    base + 0.03, r1, r2, True,
                    0.08,                     # pair_aff: low, but can pair either card
                    0.35 + 0.15 * (hi / 12), # flush_aff: high suited cards have nut flush potential
                    str_aff,
                    0.4 + hi / 24.0,          # high_aff
                ))
                # Offsuit variant
                _HAND_CLASS_TABLE.append((
                    base, r1, r2, False,
                    0.08,
                    0.0,                      # flush_aff: none
                    str_aff * 0.7,            # straight potential slightly lower (no backdoor fd)
                    0.4 + hi / 24.0,
                ))

    _HAND_CLASS_TABLE.sort(reverse=True)


_init_hand_class_table()

# ── Main encoder ──────────────────────────────────────────────────────────────

def encode_state(state, iteration_progress: float = 0.0) -> torch.Tensor:
    """
    Encode game state into an 80-dim feature vector.

    cfr_mode=True  — used during CFR self-play training.
        Zeros out features that are always constant during training
        (range equity, HUD stats, tilt, hands-played confidence) so
        the net learns to ignore those feature slots rather than
        fitting to uninformative constants.  The net stays 80-dim so
        the same checkpoint works for live play without any changes.

    cfr_mode=False (default) — used during live play.
        All 80 features populated, including villain range equity
        (when profile has sufficient data) and full HUD stats.
    """
    fv   = np.zeros(N_FEATURES, dtype=np.float32)
    seat = state.to_move
    n    = state.n_players

    hand_bucket = state.hands[seat]
    wallet      = max(state.wallet, 1.0)

    # ── [0] Seat index ────────────────────────────────────────────────────────
    fv[0] = seat / max(n - 1, 1)

    # ── [1-6] Seat one-hot ────────────────────────────────────────────────────
    if 0 <= seat < N_SEATS:
        fv[1 + seat] = 1.0

    # ── [7-22] Preflop bucket one-hot (16 buckets) ────────────────────────────
    # Written conditionally after street + eff_bb are computed (see below).
    # Zeroed by default via np.zeros — feature vector size unchanged (76-dim).

    # ── [23-32] Game state scalars ────────────────────────────────────────────
    to_call = max(state.current_bet - state.bets[seat], 0.0)

    fv[23] = min(state.pot / wallet, 1.0)
    fv[24] = min(to_call / wallet, 1.0)
    fv[25] = min(state.stacks[seat] / wallet, 1.0)

    spr    = state.stacks[seat] / max(state.pot, 1.0)
    fv[26] = min(math.log1p(spr) / math.log1p(50), 1.0)

    fv[27] = state.n_raises / MAX_RAISES

    active = sum(1 for i in range(n)
                 if i not in state.folded and state.stacks[i] > 0)
    fv[28] = active / n

    fv[29] = min(len(state.action_history) / 10.0, 1.0)

    pot_plus_call = state.pot + to_call
    fv[30] = (to_call / pot_plus_call) if pot_plus_call > 0 else 0.0  # pot odds

    fv[31] = min(state.bets[seat] / wallet, 1.0)                       # pot committed

    fv[32] = min(to_call / max(state.stacks[seat], 1.0), 1.0)          # stack-to-call ratio

    # ── [33-36] Street one-hot ────────────────────────────────────────────────
    street = getattr(state, 'street', 0)
    if 0 <= street < N_STREETS:
        fv[33 + street] = 1.0

    # ── [37] Community card count ─────────────────────────────────────────────
    board  = getattr(state, 'community_cards', [])
    fv[37] = len(board) / 5.0

    # ── [38-41] Hand strength + draws ─────────────────────────────────────────
    hole_cards = getattr(state, 'hole_cards', None)

    if hole_cards and board:
        hole = hole_cards[seat]
        hs   = _postflop_hand_strength(hole, board)
        fv[38] = hs

        fd, sd, de = _draw_features(hole, board)
        fv[39] = fd
        fv[40] = sd
        fv[41] = de

    # ── [42] Is last to act (action-order aware) ──────────────────────────────
    # Build action order for this street: postflop SB-first, preflop UTG-first
    if street == 0:
        action_order = list(range(n))
    else:
        action_order = list(range(n - 2, n)) + list(range(0, n - 2))
    active_in_order = [s for s in action_order
                       if s not in state.folded and state.stacks[s] > 0]
    fv[42] = 1.0 if active_in_order and seat == active_in_order[-1] else 0.0

    # ── [43] Aggressor flag ───────────────────────────────────────────────────
    last_raiser = -1
    if hasattr(state, 'last_raiser_index'):
        last_raiser = state.last_raiser_index
    elif state.action_history:
        for entry in reversed(state.action_history):
            s, a = entry[0], entry[1]
            if a in ('RAISE_2', 'RAISE_4', 'ALLIN'):
                last_raiser = s
                break
    fv[43] = (last_raiser / max(n - 1, 1)) if last_raiser >= 0 else 0.0

    # ── [44-48] Board texture ─────────────────────────────────────────────────
    paired, mono, two_tone, conn, high = board_texture(board)
    fv[44] = paired
    fv[45] = mono
    fv[46] = two_tone
    fv[47] = conn
    fv[48] = high

    # ── Preflop context flags stored at [53-55] ──
    pf_unopened = 1.0 if street == 0 and state.n_raises == 0 else 0.0
    pf_vs_open  = 1.0 if street == 0 and state.n_raises == 1 else 0.0
    pf_vs_3bet  = 1.0 if street == 0 and state.n_raises >= 2 else 0.0

    is_ip  = _position_context(state, seat)
    is_oop = 1.0 - is_ip
    eff_bb = _effective_stack_bb(state, seat)

    # All active opponents — N=2: single element list, N>2: all remaining players
    opp_seats    = [i for i in range(n) if i != seat and i not in state.folded]
    opp_profiles = [get_profile(i) for i in opp_seats]

    # ── [53-55] Preflop context ───────────────────────────────────────────────
    fv[49] = pf_unopened
    fv[50] = pf_vs_open
    fv[51] = pf_vs_3bet

    # ── [56-61] Position + stack depth ───────────────────────────────────────
    fv[52] = is_ip
    fv[53] = is_oop
    
    eff_bb_norm = min(eff_bb / 100.0, 1.0)
    fv[54] = eff_bb_norm
    fv[55] = 1.0 if eff_bb <= 8 else 0.0
    fv[56] = 1.0 if 8 < eff_bb <= 20 else 0.0
    fv[57] = 1.0 if eff_bb > 20 else 0.0

    # ── [7-22] Preflop bucket one-hot — conditional write ─────────────────────
    # Active preflop always. Active postflop only when stack depth gives the
    # preflop hand class genuine strategic relevance: SPR >= 2.0 AND eff_bb >= 20.
    # Below these thresholds (e.g. 10BB games) the infoset is blind to hb and
    # keeping it active postflop would let the net learn spurious strategies the
    # CFR regret table doesn't support. Feature vector stays 80-dim: fv[7-22]
    # are simply left as zeros when the condition is false (76-dim unchanged).
    SPR_THRESHOLD = 2.0
    BB_THRESHOLD  = 20.0
    hb_active = (street == 0) or (spr >= SPR_THRESHOLD and eff_bb >= BB_THRESHOLD)
    if hb_active:
        fv[7 + min(hand_bucket, N_BUCKETS - 1)] = 1.0

    # ── [62-68] Hero HUD (gated on sufficient data) ───────────────────────────
    actor = get_profile(seat)
    fv[58] = actor.vpip
    fv[59] = actor.pfr
    fv[60] = actor.three_bet_pct
    fv[61] = actor.cbet_pct
    fv[62] = actor.ats
    fv[63] = actor.postflop_agg
    fv[64] = actor.wtsd

    # ── [69-74] Villain HUD (opp_seats/opp_profiles already built above) ────────
    if opp_profiles:
        fv[65] = sum(p.vpip              for p in opp_profiles) / len(opp_profiles)
        fv[66] = sum(p.pfr               for p in opp_profiles) / len(opp_profiles)
        fv[67] = sum(p.three_bet_pct     for p in opp_profiles) / len(opp_profiles)
        fv[68] = sum(p.cbet_pct          for p in opp_profiles) / len(opp_profiles)
        fv[69] = sum(p.fold_to_3bet      for p in opp_profiles) / len(opp_profiles)
        fv[70] = sum(p.fold_to_cbet_rate for p in opp_profiles) / len(opp_profiles)
    else:
        fv[65] = 0.5
        fv[66] = 0.3
        fv[67] = 0.08
        fv[68] = 0.6
        fv[69] = 0.55
        fv[70] = 0.5

    # ── [75-79] Meta ──────────────────────────────────────────────────────────
    fv[71] = actor.tilt_factor
    fv[72] = min(actor.hands_played / 100.0, 1.0)
    fv[73] = float(iteration_progress)

    acted      = len(state.action_history)
    yet_to_act = max(active - acted - 1, 0)
    fv[74] = yet_to_act / max(n, 1)

    fv[75] = 1.0

    return torch.tensor(fv, dtype=torch.float32)


# ── Policy helpers ────────────────────────────────────────────────────────────

def policy_tensor(sigma: dict, legal_actions: list):
    pi         = torch.zeros(N_ACTIONS, dtype=torch.float32)
    legal_mask = torch.zeros(N_ACTIONS, dtype=torch.bool)

    for i, a in enumerate(ALL_ACTIONS):
        if a in legal_actions:
            legal_mask[i] = True
        if a in sigma:
            pi[i] = sigma[a]

    return pi, legal_mask


def mask_logits(logits, legal_actions):
    mask = torch.full_like(logits, float('-inf'))
    for i, a in enumerate(ALL_ACTIONS):
        if a in legal_actions:
            if logits.dim() == 1:
                mask[i] = logits[i]
            else:
                mask[:, i] = logits[:, i]
    return mask


# ── Smoke test ────────────────────────────────────────────────────────────────

class _FakeCard:
    def __init__(self, value, suit):
        self.value = value
        self.suit  = suit

if __name__ == '__main__':
    hole  = [_FakeCard('A', 'SPADES'), _FakeCard('K', 'SPADES')]
    board = [_FakeCard('Q', 'SPADES'), _FakeCard('J', 'HEARTS'), _FakeCard('2', 'CLUBS')]

    hs = _postflop_hand_strength(hole, board)
    fd, sd, de = _draw_features(hole, board)
    texture = board_texture(board)

    print(f"Hand strength   : {hs:.4f}")
    print(f"Flush draw      : {fd}  Straight draw: {sd}  Draw equity: {de:.4f}")
    print(f"Board texture   : paired={texture[0]} mono={texture[1]} "
          f"two_tone={texture[2]} conn={texture[3]} high={texture[4]}")
    print(f"N_FEATURES      : {N_FEATURES}")
    print(f"N_BUCKETS       : {N_BUCKETS}")
    print(f"IDX_STREET_START: {IDX_STREET_START}")
    print(f"IDX_HAND_STRENGTH: {IDX_HAND_STRENGTH}")
    print(f"Hand class table: {len(_HAND_CLASS_TABLE)} entries precomputed")

    # Test board-hit equity
    dummy_entries = _HAND_CLASS_TABLE[:20]  # top 20 hands (premiums)