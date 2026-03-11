"""
range_equity_state_encoder.py
------------------------------------------------

Adds range and board texture features while keeping the
encoder compact and game-theoretically consistent.

Feature vector (58 dims)

[0]      acting seat index
[1-6]    seat one-hot
[7-22]   preflop bucket one-hot (16)  ← expanded from 13 to match preflop_abstraction.py

[23]     pot size
[24]     to_call
[25]     stack size
[26]     log SPR
[27]     raises this street
[28]     active players
[29]     history length

[30-33]  street one-hot
[34]     community card count
[35]     hand strength
[36]     flush draw
[37]     straight draw

[38]     hero_range_equity
[39]     villain_range_equity
[40]     range_advantage
[41]     draw_equity

[42]     paired_board
[43]     monotone_board
[44]     two_tone
[45]     connected_board
[46]     high_card_board

[47]     iteration_progress

[48]     pf_unopened
[49]     pf_vs_open
[50]     pf_vs_3bet

[51]     is_ip
[52]     is_oop

[53]     eff_bb_norm
[54]     depth_short
[55]     depth_mid
[56]     depth_deep

[57]     bias
"""

from __future__ import annotations
import torch
import numpy as np
from collections import Counter, defaultdict
import math

N_FEATURES = 58       # expanded from 55 (3 extra bucket slots)
N_ACTIONS  = 5

ALL_ACTIONS = ["FOLD","CALL","RAISE_2","RAISE_4","ALLIN"]

N_SEATS    = 6
N_BUCKETS  = 16       # must match N_PREFLOP_BUCKETS in preflop_abstraction.py
N_STREETS  = 4
MAX_RAISES = 4

# ── Derived index constants (all shifted +3 vs old 55-dim layout) ─────────────
IDX_STREET_START  = 30
IDX_STREET_END    = 34
IDX_HAND_STRENGTH = 35
STREET_SLICE      = slice(IDX_STREET_START, IDX_STREET_END)

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


# --------------------------------------------------
# Card utilities
# --------------------------------------------------

RANK_MAP = {
    '2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
    '8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14
}

def rank(card):
    return RANK_MAP.get(card[0], 0) if isinstance(card, tuple) else RANK_MAP.get(card.value, 0)

def suit(card):
    return card[1] if isinstance(card, tuple) else card.suit


# --------------------------------------------------
# Hand strength
# --------------------------------------------------

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
                r = 'T' if card.value == '10' else card.value
                s = _SUIT_MAP.get(card.suit, card.suit[0].lower())
                return f"{r}{s}"
            r, s = card[0], card[1]
            r = 'T' if r == '10' else r
            s = _SUIT_MAP.get(s, s[0].lower())
            return f"{r}{s}"

        all_cards  = list(hole_cards) + list(community_cards)
        ph_strings = [_to_ph(c) for c in all_cards]
        score      = evaluate_cards(*ph_strings)   # 1=best, 7462=worst
        return float(1.0 - (score - 1) / 7461.0)
    except Exception:
        return 0.0


# --------------------------------------------------
# Draw detection
# --------------------------------------------------

def _get_rank_suit(card):
    if isinstance(card, tuple):
        return card[0], card[1]
    return getattr(card, 'rank', getattr(card, 'value', '')), card.suit


def _draw_features(hole_cards, community_cards):
    if len(community_cards) < 3 or len(hole_cards) < 2:
        return 0.0, 0.0, 0.0
    try:
        all_cards   = list(hole_cards) + list(community_cards)
        suits       = [_get_rank_suit(c)[1] for c in all_cards]
        suit_counts = Counter(suits)
        flush_draw  = 1.0 if max(suit_counts.values()) == 4 else 0.0

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

        draw_equity = flush_draw * 0.5 + straight_draw * 0.5
        return flush_draw, straight_draw, draw_equity
    except Exception:
        return 0.0, 0.0, 0.0


# --------------------------------------------------
# Stack / position helpers
# --------------------------------------------------

def _effective_stack_bb(state, seat: int) -> float:
    opps = [i for i in range(state.n_players) if i != seat and i not in state.folded]
    if not opps:
        return 10.0
    eff   = min([state.stacks[seat]] + [state.stacks[i] for i in opps])
    buyin = getattr(state, 'buyin', None) or getattr(state, 'buy_in', None) or 10.0
    return eff / max(buyin, 1.0)


def _position_context(state, seat: int) -> float:
    n      = state.n_players
    street = getattr(state, 'street', 0)
    folded = getattr(state, 'folded', set())

    if street == 0:
        full_order = list(range(0, n))                               # preflop: UTG first
    else:
        full_order = list(range(n - 2, n)) + list(range(0, n - 2))  # postflop: SB first

    active_order = [s for s in full_order if s not in folded]

    if seat not in active_order:
        return 0.5

    position_index = active_order.index(seat)
    return position_index / max(len(active_order) - 1, 1)


# --------------------------------------------------
# Board texture
# --------------------------------------------------

def board_texture(board):
    if not board:
        return (0, 0, 0, 0, 0)

    suits       = [suit(c) for c in board]
    ranks       = [rank(c) for c in board]
    counts      = Counter(ranks)
    suit_counts = Counter(suits)

    paired    = 1.0 if max(counts.values()) >= 2 else 0.0
    monotone  = 1.0 if len(suit_counts) == 1 else 0.0
    two_tone  = 1.0 if len(suit_counts) == 2 else 0.0
    ranks_s   = sorted(ranks)
    connected = 1.0 if max(ranks_s) - min(ranks_s) <= 4 else 0.0
    high_card = 1.0 if max(ranks) >= 12 else 0.0

    return paired, monotone, two_tone, connected, high_card


# --------------------------------------------------
# Fast range approximations
# --------------------------------------------------

# Precompute abstract hand classes once.
_ALL_RANKS     = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']
_RANK_STRENGTH = {r: 12 - i for i, r in enumerate(_ALL_RANKS)}

_PRECOMPUTED_HAND_CLASSES: list = []


def _init_hand_classes():
    global _PRECOMPUTED_HAND_CLASSES
    if _PRECOMPUTED_HAND_CLASSES:
        return
    for i, r1 in enumerate(_ALL_RANKS):
        for r2 in _ALL_RANKS[i:]:
            if r1 == r2:
                strength = (_RANK_STRENGTH[r1] + _RANK_STRENGTH[r2]) / 24.0 + 0.15
                _PRECOMPUTED_HAND_CLASSES.append((strength, r1, r2, False))
            else:
                base = (_RANK_STRENGTH[r1] + _RANK_STRENGTH[r2]) / 24.0
                _PRECOMPUTED_HAND_CLASSES.append((base + 0.03, r1, r2, True))
                _PRECOMPUTED_HAND_CLASSES.append((base,        r1, r2, False))
    _PRECOMPUTED_HAND_CLASSES.sort(reverse=True)


_init_hand_classes()


def _build_villain_range(profile, pf_unopened, pf_vs_open, pf_vs_3bet, street, n_raises):
    """Fast abstract opponent range builder. Returns list of (rank1, rank2, suited)."""
    if street == 0:
        if pf_vs_3bet:
            width = max(0.05, 1.0 - profile.fold_to_3bet)
        elif pf_vs_open:
            width = profile.pfr * 2.2
        else:
            width = profile.vpip
    else:
        width = 0.28 + profile.postflop_agg * 0.35
        if n_raises >= 2:
            width -= 0.10
        elif n_raises == 1:
            width -= 0.04

    width  = min(max(width, 0.05), 1.0)
    cutoff = max(1, int(len(_PRECOMPUTED_HAND_CLASSES) * width))
    return [(r1, r2, suited) for _, r1, r2, suited in _PRECOMPUTED_HAND_CLASSES[:cutoff]]


def _range_equity_vs_villain(hole_cards, community_cards, villain_range,
                              n_samples=None, n_rollouts=None):
    """
    Fast deterministic equity proxy. Returns (hero_eq, villain_eq) independently.
    No Monte Carlo, no evaluator calls for villain side.
    """
    try:
        # ── Hero preflop hand quality ──────────────────────────────────────
        if hole_cards is not None and len(hole_cards) >= 2:
            r1 = rank(hole_cards[0])
            r2 = rank(hole_cards[1])
            s1 = suit(hole_cards[0])
            s2 = suit(hole_cards[1])
            hi, lo   = max(r1, r2), min(r1, r2)
            hero_pre = ((hi - 2) + (lo - 2)) / 24.0
            if r1 == r2:                        hero_pre += 0.18
            if s1 == s2 and r1 != r2:           hero_pre += 0.03
            if abs(r1 - r2) <= 1 and r1 != r2: hero_pre += 0.02
            hero_pre = min(max(hero_pre, 0.0), 1.0)
        else:
            hero_pre = 0.5

        # ── Villain average range strength ─────────────────────────────────
        if villain_range:
            villain_avg = sum(
                (_RANK_STRENGTH[r1c] + _RANK_STRENGTH[r2c]) / 24.0
                + (0.15 if r1c == r2c else 0.03 if suited else 0.0)
                for r1c, r2c, suited in villain_range
            ) / len(villain_range)
        else:
            villain_avg = 0.5

        # ── Preflop: raw hand quality as independent signals ───────────────
        if not community_cards or len(community_cards) == 0:
            hero_eq    = min(max(hero_pre,    0.0), 1.0)
            villain_eq = min(max(villain_avg, 0.0), 1.0)
            return hero_eq, villain_eq

        # ── Postflop: blend made-hand + draw, villain from range + board ───
        hs         = _postflop_hand_strength(hole_cards, community_cards)
        fd, sd, de = _draw_features(hole_cards, community_cards)

        if hs <= 0.0:
            hs = hero_pre   # evaluator failed, fall back to preflop proxy

        hero_post = 0.78 * hs + 0.22 * de

        paired, mono, two, conn, high = board_texture(community_cards)
        board_pressure = (
            0.04 * paired +
            0.05 * mono   +
            0.03 * two    +
            0.06 * conn   +
            0.02 * high
        )
        villain_post = villain_avg + board_pressure

        hero_eq    = min(max(hero_post,    0.0), 1.0)
        villain_eq = min(max(villain_post, 0.0), 1.0)
        return hero_eq, villain_eq

    except Exception:
        return 0.5, 0.5


def range_features(bucket, pf_unopened, pf_vs_open, pf_vs_3bet,
                   is_ip, eff_bb, street, board, n_raises,
                   hole_cards=None, villain_seat=None, profile=None):
    """
    Fast deterministic range features. Returns (hero_equity, villain_equity, advantage).
    Uses profile-based range when available, falls back to heuristic.
    """
    base_hero = 1.0 - (bucket / max(N_BUCKETS - 1, 1))

    if profile is not None:
        villain_range = _build_villain_range(
            profile     = profile,
            pf_unopened = pf_unopened,
            pf_vs_open  = pf_vs_open,
            pf_vs_3bet  = pf_vs_3bet,
            street      = street,
            n_raises    = n_raises,
        )
        hero_equity, villain_equity = _range_equity_vs_villain(
            hole_cards      = hole_cards,
            community_cards = board if board else [],
            villain_range   = villain_range,
        )
    else:
        # Heuristic fallback (no profile available)
        hero_equity = base_hero
        if street == 0:
            if pf_unopened:
                villain_equity = 0.50 if is_ip else 0.54
            elif pf_vs_open:
                villain_equity = 0.60 if is_ip else 0.64
            elif pf_vs_3bet:
                villain_equity = 0.68 if is_ip else 0.72
            else:
                villain_equity = 0.56
            if eff_bb <= 8:    villain_equity += 0.03
            elif eff_bb >= 25: villain_equity -= 0.02
        else:
            paired, mono, two, conn, high = board_texture(board)
            villain_equity = 0.50
            if n_raises >= 2:   villain_equity += 0.08
            elif n_raises >= 1: villain_equity += 0.04
            if mono:    villain_equity += 0.06
            if two:     villain_equity += 0.03
            if conn:    villain_equity += 0.07
            if paired:  villain_equity += 0.04
            if high:    villain_equity += 0.02

    hero_equity    = min(max(hero_equity,    0.0), 1.0)
    villain_equity = min(max(villain_equity, 0.0), 1.0)
    advantage      = hero_equity - villain_equity
    return hero_equity, villain_equity, advantage


# --------------------------------------------------
# Main encoder
# --------------------------------------------------

def encode_state(state, iteration_progress=0.0):

    fv   = np.zeros(N_FEATURES, dtype=np.float32)
    seat = state.to_move
    n    = state.n_players

    hand_bucket = state.hands[seat]
    wallet      = max(state.wallet, 1)

    # [0] acting seat index
    fv[0] = seat / max(n - 1, 1)

    # [1-6] seat one-hot
    if 0 <= seat < N_SEATS:
        fv[1 + seat] = 1

    # [7-22] preflop bucket one-hot (16 buckets, 0=strongest)
    fv[7 + min(hand_bucket, N_BUCKETS - 1)] = 1

    # [23-29] game state scalars
    to_call = max(state.current_bet - state.bets[seat], 0)

    fv[23] = min(state.pot / wallet, 1)
    fv[24] = min(to_call / wallet, 1)
    fv[25] = min(state.stacks[seat] / wallet, 1)

    spr    = state.stacks[seat] / max(state.pot, 1)
    fv[26] = min(math.log1p(spr) / math.log1p(50), 1)

    fv[27] = state.n_raises / MAX_RAISES

    active = sum(
        1 for i in range(n)
        if i not in state.folded and state.stacks[i] > 0
    )
    fv[28] = active / n

    fv[29] = min(len(state.action_history) / 10, 1)

    # [30-33] street one-hot
    street = getattr(state, "street", 0)
    if 0 <= street < N_STREETS:
        fv[30 + street] = 1

    # [34] community card count
    board  = getattr(state, "community_cards", [])
    fv[34] = len(board) / 5

    # [35] hand strength  [36] flush draw  [37] straight draw  (set below)
    hole_cards = getattr(state, "hole_cards", None)

    if hole_cards and board:
        hole   = hole_cards[seat]
        hs     = _postflop_hand_strength(hole, board)
        fv[35] = hs   # raw equity, no compression

        flush_draw, straight_draw, draw_eq = _draw_features(hole, board)
        fv[36] = flush_draw
        fv[37] = straight_draw
        fv[41] = draw_eq   # draw_equity stored at [41]

    # preflop context flags
    pf_unopened = 1.0 if state.street == 0 and state.n_raises == 0 else 0.0
    pf_vs_open  = 1.0 if state.street == 0 and state.n_raises == 1 else 0.0
    pf_vs_3bet  = 1.0 if state.street == 0 and state.n_raises >= 2 else 0.0

    is_ip  = float(_position_context(state, seat))
    is_oop = 1.0 - is_ip

    eff_bb = _effective_stack_bb(state, seat)

    # find villain
    villain_seat = next(
        (i for i in range(state.n_players)
         if i != seat and i not in state.folded),
        None
    )
    villain_profile = get_profile(villain_seat) if villain_seat is not None else None
    hero_hole_cards = hole_cards[seat] if hole_cards else None

    # [38-40] range equity features
    hero_eq, vill_eq, adv = range_features(
        bucket       = hand_bucket,
        pf_unopened  = pf_unopened,
        pf_vs_open   = pf_vs_open,
        pf_vs_3bet   = pf_vs_3bet,
        is_ip        = is_ip,
        eff_bb       = eff_bb,
        street       = street,
        board        = board,
        n_raises     = state.n_raises,
        hole_cards   = hero_hole_cards,
        villain_seat = villain_seat,
        profile      = villain_profile,
    )

    fv[38] = hero_eq
    fv[39] = vill_eq
    fv[40] = adv
    # fv[41] draw_equity already assigned above

    # [42-46] board texture
    paired, mono, two, conn, high = board_texture(board)
    fv[42] = paired
    fv[43] = mono
    fv[44] = two
    fv[45] = conn
    fv[46] = high

    # [47] iteration progress
    fv[47] = iteration_progress

    # [48-50] preflop context
    fv[48] = pf_unopened
    fv[49] = pf_vs_open
    fv[50] = pf_vs_3bet

    # [51-52] position
    fv[51] = is_ip
    fv[52] = is_oop

    # [53-56] stack depth
    depth_short = 1.0 if eff_bb <= 8 else 0.0
    depth_mid   = 1.0 if 8 < eff_bb <= 20 else 0.0
    depth_deep  = 1.0 if eff_bb > 20 else 0.0
    eff_bb_norm = min(eff_bb / 40.0, 1.0)

    fv[53] = eff_bb_norm
    fv[54] = depth_short
    fv[55] = depth_mid
    fv[56] = depth_deep

    # [57] bias
    fv[57] = 1.0

    return torch.tensor(fv, dtype=torch.float32)


# --------------------------------------------------
# Policy helpers
# --------------------------------------------------

def policy_tensor(sigma: dict, legal_actions: list):
    """
    Build:
      - pi: target policy over all N_ACTIONS
      - legal_mask: boolean mask over all N_ACTIONS
    """
    pi         = torch.zeros(N_ACTIONS, dtype=torch.float32)
    legal_mask = torch.zeros(N_ACTIONS, dtype=torch.bool)

    for i, a in enumerate(ALL_ACTIONS):
        if a in legal_actions:
            legal_mask[i] = True
        if a in sigma:
            pi[i] = sigma[a]

    return pi, legal_mask


def mask_logits(logits, legal_actions):
    mask = torch.full_like(logits, float("-inf"))

    for i, a in enumerate(ALL_ACTIONS):
        if a in legal_actions:
            if logits.dim() == 1:
                mask[i] = logits[i]
            else:
                mask[:, i] = logits[:, i]

    return mask


# --------------------------------------------------
# Quick smoke test
# --------------------------------------------------

class FakeCard:
    def __init__(self, value, suit):
        self.value = value
        self.suit  = suit

if __name__ == '__main__':
    hole  = [FakeCard('5', 'D'), FakeCard('7', 'H')]
    board = [FakeCard('4', 'C'), FakeCard('3', 'H'), FakeCard('Q', 'S')]
    result = _postflop_hand_strength(hole, board)
    print("Hand strength:", result)
    print("N_FEATURES:", N_FEATURES)
    print("N_BUCKETS:", N_BUCKETS)
    print("IDX_STREET_START:", IDX_STREET_START)
    print("IDX_HAND_STRENGTH:", IDX_HAND_STRENGTH)