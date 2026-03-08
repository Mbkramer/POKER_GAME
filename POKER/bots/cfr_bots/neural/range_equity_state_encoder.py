"""
range_equity_state_encoder.py
------------------------------------------------

Adds range and board texture features while keeping the
encoder compact and game-theoretically consistent.

Feature vector (55 dims)

[0]      acting seat index
[1-6]    seat one-hot
[7-19]   preflop bucket one-hot (13)

[20]     pot size
[21]     to_call
[22]     stack size
[23]     log SPR
[24]     raises this street
[25]     active players
[26]     history length

[27-30]  street one-hot
[31]     community card count
[32]     hand strength
[33]     flush draw
[34]     straight draw

[35]     hero_range_equity
[36]     villain_range_equity
[37]     range_advantage
[38]     draw_equity

[39]     paired_board
[40]     monotone_board
[41]     two_tone
[42]     connected_board
[43]     high_card_board

[44]     iteration_progress

[45]     pf_unopened
[46]     pf_vs_open
[47]     pf_vs_3bet

[48]     is_ip
[49]     is_oop

[50]     eff_bb_norm
[51]     depth_short
[52]     depth_mid
[53]     depth_deep

[54]     bias
"""

from __future__ import annotations
import torch
import numpy as np
from collections import Counter
import math

N_FEATURES = 55
N_ACTIONS  = 5

ALL_ACTIONS = ["FOLD","CALL","RAISE_2","RAISE_4","ALLIN"]

N_SEATS   = 6
N_BUCKETS = 13
N_STREETS = 4
MAX_RAISES = 4

IDX_STREET_START = 27
IDX_STREET_END = 31
IDX_HAND_STRENGTH = 32
STREET_SLICE = slice(IDX_STREET_START, IDX_STREET_END)

# --------------------------------------------------
# Card utilities
# --------------------------------------------------

RANK_MAP = {
'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14
}

def rank(card):
    return RANK_MAP.get(card[0],0) if isinstance(card,tuple) else RANK_MAP.get(card.value,0)

def suit(card):
    return card[1] if isinstance(card,tuple) else card.suit


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

# --------------------------------------------------
# Draw detection
# --------------------------------------------------

def _get_rank_suit(card):
    if isinstance(card, tuple):
        return card[0], card[1]
    return getattr(card, 'rank', getattr(card, 'value', '')), card.suit


def _draw_features(hole_cards, community_cards):
    if len(community_cards) < 3 or len(hole_cards) < 2:
        return 0.0, 0.0, 0.0  # Return 0 for flush_draw, straight_draw, and draw_equity
    try:
        all_cards = list(hole_cards) + list(community_cards)
        suits = [_get_rank_suit(c)[1] for c in all_cards]
        suit_counts = Counter(suits)
        flush_draw = 1.0 if max(suit_counts.values()) == 4 else 0.0

        _RANK_VAL = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
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

        # Calculate draw_equity based on flush_draw and straight_draw
        draw_equity = flush_draw * 0.5 + straight_draw * 0.5

        return flush_draw, straight_draw, draw_equity
    except Exception:
        return 0.0, 0.0, 0.0


def _effective_stack_bb(state, seat: int) -> float:
    opps = [i for i in range(state.n_players) if i != seat and i not in state.folded]
    if not opps:
        return 10.0
    eff = min([state.stacks[seat]] + [state.stacks[i] for i in opps])

    buyin = getattr(state, 'buyin', None) or getattr(state, 'buy_in', None) or 10.0
    return eff / max(buyin, 1.0)


def _position_context(state, seat: int):
    """
    Returns 1 if hero is in-position relative to the opponent,
    else 0.
    """
    if state.n_players != 2:
        return 0

    # In HU: big blind acts last postflop
    bb = state.n_players - 1
    return 1 if seat == bb else 0

# --------------------------------------------------
# Board texture
# --------------------------------------------------

def board_texture(board):

    if not board:
        return (0,0,0,0,0)

    suits = [suit(c) for c in board]
    ranks = [rank(c) for c in board]

    counts = Counter(ranks)

    paired = 1.0 if max(counts.values()) >= 2 else 0.0

    suit_counts = Counter(suits)

    monotone = 1.0 if len(suit_counts) == 1 else 0.0
    two_tone = 1.0 if len(suit_counts) == 2 else 0.0

    ranks_sorted = sorted(ranks)

    connected = 1.0 if max(ranks_sorted)-min(ranks_sorted) <= 4 else 0.0

    high_card = 1.0 if max(ranks) >= 12 else 0.0

    return paired,monotone,two_tone,connected,high_card


# --------------------------------------------------
# Range approximations
# --------------------------------------------------

def range_features(
    bucket,
    pf_unopened,
    pf_vs_open,
    pf_vs_3bet,
    is_ip,
    eff_bb,
    street,
    board,
    n_raises,
):
    """
    Lightweight range strength approximation using only public information.
    No cheating: uses bucket + public betting/board context only.
    """

    # Hero proxy: lower bucket = stronger hand
    hero_equity = 1.0 - (bucket / max(N_BUCKETS - 1, 1))

    if street == 0:
        if pf_unopened:
            villain_equity = 0.50 if is_ip else 0.54
        elif pf_vs_open:
            villain_equity = 0.60 if is_ip else 0.64
        elif pf_vs_3bet:
            villain_equity = 0.68 if is_ip else 0.72
        else:
            villain_equity = 0.56

        if eff_bb <= 8:
            villain_equity += 0.03
        elif eff_bb >= 25:
            villain_equity -= 0.02

    else:
        paired, mono, two, conn, high = board_texture(board)
        
        villain_equity = 0.50
        if n_raises >= 2:
            villain_equity += 0.08  # heavy aggression = strong range
        elif n_raises >= 1:
            villain_equity += 0.04
        if mono:
            villain_equity += 0.06  # flush possible
        if two:
            villain_equity += 0.03  # flush draw possible, moderate threat
        if conn:
            villain_equity += 0.07  # straight heavy board
        if paired:
            villain_equity += 0.04  # trips/boat possible
        if high:
            villain_equity += 0.02  # broadway hands in range

    villain_equity = min(max(villain_equity, 0.0), 1.0)
    advantage = hero_equity - villain_equity
    return hero_equity, villain_equity, advantage


# --------------------------------------------------
# Main encoder
# --------------------------------------------------

def encode_state(state, iteration_progress=0.0):

    fv = np.zeros(N_FEATURES,dtype=np.float32)

    seat = state.to_move

    hand_bucket = state.hands[seat]

    n = state.n_players
    wallet = max(state.wallet,1)

    fv[0] = seat/max(n-1,1)

    if 0 <= seat < N_SEATS:
        fv[1 + seat] = 1

    fv[7 + min(hand_bucket, N_BUCKETS - 1)] = 1

    to_call = max(state.current_bet - state.bets[seat],0)

    fv[20] = min(state.pot/wallet,1)
    fv[21] = min(to_call/wallet,1)
    fv[22] = min(state.stacks[seat]/wallet,1)

    spr = state.stacks[seat]/max(state.pot,1)
    fv[23] = min(math.log1p(spr)/math.log1p(50),1)

    fv[24] = state.n_raises/MAX_RAISES

    active = sum(
        1 for i in range(n)
        if i not in state.folded and state.stacks[i] > 0
    )

    fv[25] = active/n

    fv[26] = min(len(state.action_history)/10,1)

    street = getattr(state,"street",0)

    if 0 <= street < N_STREETS:
        fv[27+street] = 1

    board = getattr(state,"community_cards",[])
    fv[31] = len(board)/5

    hole_cards = getattr(state,"hole_cards",None)

    if hole_cards and board:

        hole = hole_cards[seat]

        hs = _postflop_hand_strength(hole, board)

        if street >= 2:
            # boost strong hands, don't inflate weak ones
            fv[32] = hs if hs < 0.5 else min(0.5 + (hs - 0.5) * 2.0, 1.0)
        else:
            fv[32] = hs

        flush_draw,straight_draw,draw_eq = _draw_features(hole,board)

        fv[33] = flush_draw
        fv[34] = straight_draw

        fv[38] = draw_eq

    # compute range features based on preflop context
    pf_unopened = 1.0 if state.street == 0 and state.n_raises == 0 else 0.0
    pf_vs_open  = 1.0 if state.street == 0 and state.n_raises == 1 else 0.0
    pf_vs_3bet  = 1.0 if state.street == 0 and state.n_raises >= 2 else 0.0

    is_ip  = float(_position_context(state, seat))
    is_oop = 1.0 - is_ip

    eff_bb = _effective_stack_bb(state, seat)

    hero_eq, vill_eq, adv = range_features(
        bucket=hand_bucket,
        pf_unopened=pf_unopened,
        pf_vs_open=pf_vs_open,
        pf_vs_3bet=pf_vs_3bet,
        is_ip=is_ip,
        eff_bb=eff_bb,
        street=street,
        board=board,
        n_raises=state.n_raises,
    )

    fv[35] = hero_eq
    fv[36] = vill_eq
    fv[37] = adv

    paired,mono,two,conn,high = board_texture(board)

    fv[39] = paired
    fv[40] = mono
    fv[41] = two
    fv[42] = conn
    fv[43] = high

    fv[44] = iteration_progress

    eff_bb = _effective_stack_bb(state, seat)
    depth_short = 1.0 if eff_bb <= 8 else 0.0
    depth_mid   = 1.0 if 8 < eff_bb <= 20 else 0.0
    depth_deep  = 1.0 if eff_bb > 20 else 0.0
    eff_bb_norm = min(eff_bb / 40.0, 1.0)

    fv[45] = pf_unopened
    fv[46] = pf_vs_open
    fv[47] = pf_vs_3bet

    fv[48] = is_ip
    fv[49] = is_oop

    fv[50] = eff_bb_norm
    fv[51] = depth_short
    fv[52] = depth_mid
    fv[53] = depth_deep
    fv[54] = 1.0  # bias

    return torch.tensor(fv,dtype=torch.float32)


# --------------------------------------------------
# Policy helpers
# --------------------------------------------------

def policy_tensor(sigma: dict, legal_actions: list):
    """
    Build:
      - pi: target policy over all N_ACTIONS
      - legal_mask: boolean mask over all N_ACTIONS
    """
    pi = torch.zeros(N_ACTIONS, dtype=torch.float32)
    legal_mask = torch.zeros(N_ACTIONS, dtype=torch.bool)

    for i, a in enumerate(ALL_ACTIONS):
        if a in legal_actions:
            legal_mask[i] = True
        if a in sigma:
            pi[i] = sigma[a]

    return pi, legal_mask


def mask_logits(logits,legal_actions):

    mask = torch.full_like(logits,float("-inf"))

    for i,a in enumerate(ALL_ACTIONS):

        if a in legal_actions:

            if logits.dim()==1:
                mask[i] = logits[i]
            else:
                mask[:,i] = logits[:,i]

    return mask