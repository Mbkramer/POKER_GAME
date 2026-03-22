"""
nlh_gamestate.py 
----------------------------------------------
Full NLH GameState: preflop → flop → turn → river.

Key design principles:
  - Cards are identified by .id string (e.g. "SA", "H10") -- works for both
    Card objects (core.card.Card) and (rank, suit) tuples
  - full_deck is the canonical 52-card list; never mutated, never deepcopied
  - community_cards grow each street via _deal_next_street
  - action_history resets each street so _next_to_act works cleanly
  - evaluation() uses river direct eval, flop/turn Monte Carlo, fold shortcut
  - evaluation() returns a list of per-player payoffs [p0, p1, ...]
"""

from __future__ import annotations
from typing import List, Optional
import random

from .constants import CHANCE

# ── Action constants ──────────────────────────────────────────────────────────

FOLD    = "FOLD"
CHECK   = "CHECK"
CALL    = "CALL"
RAISE_2 = "RAISE_2"
RAISE_4 = "RAISE_4"
ALLIN   = "ALLIN"

_ACTION_ABBR = {
    FOLD: "F",
    CHECK: "X",
    CALL: "C",
    RAISE_2: "R2",
    RAISE_4: "R4",
    ALLIN: "AI",
}

ALL_ACTIONS      = [FOLD, CHECK, CALL, RAISE_2, RAISE_4, ALLIN]
SEAT_NAMES       = {0:"UTG", 1:"UTG1", 2:"HJ", 3:"CO", 4:"BTN", 5:"SB", 6:"BB"}
STREET_NAMES     = {0:"PRE", 1:"FLP", 2:"TRN", 3:"RVR"}
CARDS_PER_STREET = {0: 3, 1: 1, 2: 1}   # cards dealt on transition

RANK_MAP = {
    'A':14,'K':13,'Q':12,'J':11,'T':10,'10':10,  # support both 'T' and '10'
    '9':9,'8':8,'7':7,'6':6,'5':5,'4':4,'3':3,'2':2,
}

# ── Card identity helper ──────────────────────────────────────────────────────

def _card_id(card) -> str:
    """
    Unique string ID for a card.  Works for:
      - core.card.Card objects  (has .id attribute)
      - (rank, suit) tuples     e.g. ('A', 'SPADES')
    """
    if hasattr(card, 'id'):
        return card.id
    rank, suit = card[0], card[1]
    return f"{suit[0]}{rank}"          # e.g. 'SA' for ('A','SPADES')

def _card_val(card) -> tuple:
    """(int_rank, suit_str) -- works for Card objects and tuples."""
    if hasattr(card, 'value'):
        return (RANK_MAP.get(card.value, 0), card.suit)
    return (RANK_MAP.get(card[0], 0), card[1])

# ── Standalone hand evaluator ─────────────────────────────────────────────────

def _eval5(cards) -> tuple:
    from collections import Counter
    vals   = sorted([_card_val(c)[0] for c in cards], reverse=True)
    suits  = [_card_val(c)[1] for c in cards]
    vc     = Counter(vals)
    groups = sorted(vc.items(), key=lambda x: (-x[1], -x[0]))

    flush    = len(set(suits)) == 1
    uvals    = sorted(set(vals), reverse=True)
    straight = False
    s_high   = 0
    for i in range(len(uvals) - 4):
        if uvals[i] - uvals[i+4] == 4:
            straight = True; s_high = uvals[i]; break
    if not straight and {14,2,3,4,5}.issubset(set(vals)):
        straight = True; s_high = 5

    if straight and flush:                  return (9, s_high)
    if groups[0][1] == 4:                   return (8, groups[0][0], groups[1][0])
    if groups[0][1]==3 and groups[1][1]==2: return (7, groups[0][0], groups[1][0])
    if flush:                               return (6,) + tuple(vals)
    if straight:                            return (5, s_high)
    if groups[0][1] == 3:                   return (4, groups[0][0]) + tuple(v for v,_ in groups[1:])
    if groups[0][1]==2 and groups[1][1]==2:
        hi = max(groups[0][0], groups[1][0])
        lo = min(groups[0][0], groups[1][0])
        return (3, hi, lo, groups[2][0])
    if groups[0][1] == 2:                   return (2, groups[0][0]) + tuple(v for v,_ in groups[1:])
    return (1,) + tuple(vals)

def _eval7(cards) -> tuple:
    from itertools import combinations
    best = None
    for combo in combinations(cards, 5):
        v = _eval5(combo)
        if best is None or v > best: best = v
    return best

# ── Equity functions ──────────────────────────────────────────────────────────

def _river_equity(p0_cards, p1_cards, community) -> float:
    """Direct evaluation at river. Returns P0 equity (1.0/0.5/0.0)."""
    v0 = _eval7(list(community) + list(p0_cards))
    v1 = _eval7(list(community) + list(p1_cards))
    if v0 > v1:  return 1.0
    if v0 == v1: return 0.5
    return 0.0

def _mc_equity(p0_cards, p1_cards, community, full_deck, pre_board=None, n=5000) -> float:
    """
    Equity for flop/turn showdown.

    If pre_board is provided (fixed 5-card runout), use it directly --
    no simulation needed, result is deterministic and fast.

    Otherwise, Monte Carlo over remaining unknown cards.
    """
    needed = 5 - len(community)
    if needed <= 0:
        return _river_equity(p0_cards, p1_cards, community)

    # Fast path: use pre-dealt board
    if pre_board is not None:
        full_board = list(pre_board[:5])
        return _river_equity(p0_cards, p1_cards, full_board)

    # Slow path: Monte Carlo over remaining deck
    used = set()
    for c in list(p0_cards) + list(p1_cards) + list(community):
        used.add(_card_id(c))

    remaining = [c for c in full_deck if _card_id(c) not in used]
    if needed > len(remaining):
        raise ValueError(
            f"MC equity: need {needed} cards but only {len(remaining)} available"
        )

    wins = 0.0
    for _ in range(n):
        board = list(community) + random.sample(remaining, needed)
        v0    = _eval7(list(p0_cards) + board)
        v1    = _eval7(list(p1_cards) + board)
        if v0 > v1:  wins += 1.0
        elif v0==v1: wins += 0.5
    return wins / n

# ── Preflop Abstraction Helper ──────────────────────────────────────────────────────

def _preflop_action_context(state) -> str:
    if state.street != 0:
        return "POST"

    if state.n_raises == 0:
        return "UNOPENED"
    elif state.n_raises == 1:
        return "VS_OPEN"
    else:
        return "VS_3BET"

# ── Board texture ──────────────────────────────────────────────────────

def _board_bucket(community) -> int:
    if not community: return 7
    from collections import Counter
    vals  = [_card_val(c)[0] for c in community]
    suits = [_card_val(c)[1] for c in community]
    vc, sc = Counter(vals), Counter(suits)

    if max(vc.values()) >= 2: return 0   # paired
    if max(sc.values()) >= 3: return 1   # monotone
    if max(sc.values()) == 2: return 2   # two-tone
    uvals = sorted(set(vals))
    for i in range(len(uvals)):
        if len([v for v in uvals if uvals[i] <= v <= uvals[i]+4]) >= 3:
            return 3                      # connected
    high = max(vals)
    if high >= 12: return 4
    if high >= 9:  return 5
    return 6


def _hand_strength_bucket(hole_cards, community, full_deck) -> int:
    """
    Postflop hand strength bucket (0=strongest, 11=weakest).

    Uses postflop_equity_bucket() from preflop_abstraction — MC equity
    vs a random opponent, result is cached by (hole_key, board_key) so
    each unique (hand, board) pair is only computed once per training run.

    Returns 6 (neutral mid-bucket) when hole_cards or community are absent
    so the infoset key degrades gracefully on preflop nodes (where this is
    never called) or in edge cases.
    """
    if not hole_cards or not community:
        return 6
    try:
        from .preflop_abstraction import postflop_equity_bucket
        return postflop_equity_bucket(hole_cards, community, full_deck)
    except Exception:
        return 6

def _position_context(state, seat: int) -> str:
    if state.n_players == 2:
        bb = state.n_players - 1
        return "IP" if seat == bb else "OOP"
    return "IP" if seat >= state.n_players // 2 else "OOP"

def _effective_stack_bucket(state, seat: int) -> str:
    active = [i for i in range(state.n_players) if i not in state.folded and i != seat]
    if not active:
        return "DEPTH_MID"

    eff = min([state.stacks[seat]] + [state.stacks[i] for i in active])
    bb = max(state.buyin, 1.0)
    eff_bb = eff / bb

    if eff_bb <= 8:
        return "DEPTH_SHORT"
    elif eff_bb <= 20:
        return "DEPTH_MID"
    else:
        return "DEPTH_DEEP"

# ── Chance node ───────────────────────────────────────────────────────────────

class NLHChanceNode:
    """
    Root chance node. Each child is an NLHGameState for one deal.

    Each deal dict must have:
      'buckets'   : tuple of int bucket IDs per player
      'hole_cards': tuple of (card, card) per player
      'full_deck' : list of all 52 card objects (never mutated)
      'equity_p0' : float (precomputed preflop equity, optional)
    """

    def __init__(self, hand_deals, wallet, buyin, n_players=6):
        self.to_move      = CHANCE
        self.parent       = None
        self.actions      = list(range(len(hand_deals)))
        self.hand_deals   = hand_deals
        self.wallet       = wallet
        self.buyin        = buyin
        self.n_players    = n_players
        self.children     = {}
        self._chance_prob = 1.0 / len(self.actions)

    def is_terminal(self): return False
    def is_chance(self):   return True
    def chance_prob(self): return self._chance_prob
    def inf_set(self):     return "."

    def play(self, action):
        if action not in self.children:
            deal = self.hand_deals[action]
            if not isinstance(deal, dict):
                raise ValueError("hand_deals must contain dicts with 'buckets', 'hole_cards', 'full_deck'")

            full_deck = deal.get('full_deck')
            if full_deck is None or len(full_deck) != 52:
                raise RuntimeError(
                    f"Deal {action}: full_deck must be 52 cards, got "
                    f"{0 if full_deck is None else len(full_deck)}"
                )

            self.children[action] = NLHGameState(
                parent     = self,
                hands      = deal['buckets'],
                hole_cards = deal['hole_cards'],
                full_deck  = full_deck,
                pre_board  = deal.get('board'),
                equity_p0  = deal.get('equity_p0', 0.5),
                wallet     = self.wallet,
                buyin      = self.buyin,
                n_players  = self.n_players,
                street     = 0,
            )
        return self.children[action]

    def sample_one(self):
        return self.play(random.choice(self.actions))


# ── Street chance node ────────────────────────────────────────────────────────

class StreetChanceNode:
    """
    Chance node for community card deals at street transitions.
    Uses pre_board for deterministic CFR, or random sample as fallback.
    """

    CARDS_PER_STREET = {0: 3, 1: 1, 2: 1}

    def __init__(self, parent, next_street, hands, hole_cards, full_deck,
                 pre_board, equity_p0, wallet, buyin, n_players,
                 community_cards, stacks, pot, folded, last_raise_size, 
                 current_bet, n_raises, all_in_runout=False):
        self.to_move         = CHANCE
        self.parent          = parent
        self.next_street     = next_street
        self.hands           = hands
        self.hole_cards      = hole_cards
        self.full_deck       = full_deck
        self.pre_board       = pre_board
        self.equity_p0       = equity_p0
        self.wallet          = wallet
        self.buyin           = buyin
        self.n_players       = n_players
        self.community_cards = community_cards
        self.stacks          = stacks
        self.pot             = pot
        self.folded          = folded
        self.last_raise_size = last_raise_size
        self.current_bet     = current_bet
        self.n_raises        = n_raises
        self.all_in_runout   = all_in_runout
        self._cache          = {}

    def is_terminal(self):  return False
    def is_chance(self):    return True
    def chance_prob(self):  return 1.0
    def inf_set(self):      return "."
    actions = []

    def sample_one(self):
        n = self.CARDS_PER_STREET[self.next_street - 1]

        if self.pre_board is not None:
            start = len(self.community_cards)
            drawn = self.pre_board[start: start + n]
        else:
            used = set()
            for hc in self.hole_cards:
                for c in hc: used.add(_card_id(c))
            for c in self.community_cards:
                used.add(_card_id(c))
            available = [c for c in self.full_deck if _card_id(c) not in used]
            drawn = random.sample(available, n)

        new_community = list(self.community_cards) + drawn
        board_key     = tuple(_card_id(c) for c in drawn)

        if board_key not in self._cache:
            if self.all_in_runout:
                if self.next_street >= 3 and len(new_community) >= 5:
                    result = NLHGameState(
                        parent=self, hands=self.hands, hole_cards=self.hole_cards,
                        full_deck=self.full_deck, pre_board=self.pre_board,
                        equity_p0=self.equity_p0, wallet=self.wallet,
                        buyin=self.buyin, n_players=self.n_players,
                        street=3, community_cards=new_community,
                        stacks=list(self.stacks), pot=self.pot,
                        bets=[0.0]*self.n_players, folded=set(self.folded),
                        action_history=[], current_bet=0.0, last_raise_size=0.0, n_raises=0,
                        to_move=None,
                    )
                else:
                    result = StreetChanceNode(
                        parent=self, next_street=self.next_street + 1,
                        hands=self.hands, hole_cards=self.hole_cards,
                        full_deck=self.full_deck, pre_board=self.pre_board,
                        equity_p0=self.equity_p0, wallet=self.wallet,
                        buyin=self.buyin, n_players=self.n_players,
                        community_cards=new_community,
                        stacks=list(self.stacks), pot=self.pot,
                        folded=set(self.folded), current_bet=0.0,
                        last_raise_size=self.last_raise_size, n_raises=0,
                        all_in_runout=True,
                    )
            else:
                sb_seat = self.n_players - 2
                first   = None
                for offset in range(self.n_players):
                    c = (sb_seat + offset) % self.n_players
                    if c not in self.folded and self.stacks[c] > 0:
                        first = c
                        break

                result = NLHGameState(
                    parent=self, hands=self.hands, hole_cards=self.hole_cards,
                    full_deck=self.full_deck, pre_board=self.pre_board,
                    equity_p0=self.equity_p0, wallet=self.wallet,
                    buyin=self.buyin, n_players=self.n_players,
                    street=self.next_street, community_cards=new_community,
                    stacks=list(self.stacks), pot=self.pot,
                    bets=[0.0]*self.n_players, folded=set(self.folded),
                    action_history=[], current_bet=0.0, last_raise_size=self.buyin, n_raises=0,
                    to_move=first,
                )

            self._cache[board_key] = result

        return self._cache[board_key]


# ── Game state ────────────────────────────────────────────────────────────────

class NLHGameState:
    """
    Multi-street NLH game state for CFR.

    Immutable design: play(action) returns a new child state.
    Children are cached lazily in _children_cache.

    action_history is reset to [] at each street transition so
    _next_to_act can tell whether each player has acted this street.
    """

    MAX_RAISES = 2

    def __init__(
        self,
        parent,
        hands,
        wallet,
        buyin,
        n_players       = 2,
        street          = 0,
        hole_cards      = None,
        full_deck       = None,
        pre_board       = None,
        equity_p0       = 0.5,
        community_cards = None,
        stacks          = None,
        pot             = 0.0,
        bets            = None,
        folded          = None,
        action_history  = None,
        current_bet     = 0.0,
        last_raise_size = 0.0, 
        n_raises        = 0,
        to_move         = None,
    ):
        self.parent          = parent
        self.hands           = hands
        self.hole_cards      = hole_cards
        self.full_deck       = full_deck
        self.pre_board       = pre_board
        self.equity_p0       = equity_p0
        self.wallet          = wallet
        self.buyin           = buyin
        self.n_players       = n_players
        self.street          = street
        self.community_cards = community_cards if community_cards is not None else []

        # ── Fresh game: post blinds ───────────────────────────────────────
        if stacks is None:
            sb      = buyin * 0.5
            stacks  = [float(wallet)] * n_players
            bets    = [0.0]           * n_players
            folded  = set()

            sb_seat = n_players - 2
            bb_seat = n_players - 1

            sb_post = min(sb,    stacks[sb_seat])
            bb_post = min(buyin, stacks[bb_seat])
            stacks[sb_seat] -= sb_post;  bets[sb_seat] = sb_post
            stacks[bb_seat] -= bb_post;  bets[bb_seat] = bb_post

            pot            = sb_post + bb_post
            current_bet    = bb_post
            last_raise_size= bb_post
            n_raises       = 0
            action_history = []
            to_move        = 0 if n_players > 2 else sb_seat

        self.stacks         = stacks
        self.pot            = pot
        self.bets           = bets
        self.folded         = folded
        self.action_history = action_history
        self.current_bet    = current_bet
        self.last_raise_size= last_raise_size
        self.n_raises       = n_raises
        self.to_move        = to_move

        self.actions         = self._legal_actions()
        self._children_cache = {}
        self._inf_set_str    = self._build_inf_set()

    # ── Flags ─────────────────────────────────────────────────────────────────

    def is_terminal(self):  return self.to_move is None
    def is_chance(self):    return False
    def chance_prob(self):  return 1.0

    def _effective_stack_remaining(self) -> float:
        live = [
            self.stacks[p]
            for p in range(self.n_players)
            if p not in self.folded
        ]
        return min(live) if live else 0.0

    def _pot_size_now(self) -> float:
        return float(self.pot)

    def _spr_now(self) -> float:
        pot_now = max(self._pot_size_now(), 1.0)
        eff = self._effective_stack_remaining()
        return eff / pot_now

    def _should_force_preflop_commit_resolution(self) -> bool:
        if self.street != 0:
            return False

        live_players = [p for p in range(self.n_players) if p not in self.folded]
        if len(live_players) != 2:
            return False

        if self.n_raises < 2:
            return False

        return self._spr_now() <= 1.5

    # ── Legal actions ─────────────────────────────────────────────────────────

    def _legal_actions(self):
        if self.is_terminal():
            return []

        seat  = self.to_move
        stack = self.stacks[seat]
        owed  = max(self.current_bet - self.bets[seat], 0.0)
        acts  = []

        # Facing a bet: fold/call are meaningful
        if owed > 0:
            acts.append(FOLD)
            acts.append(CALL)
        else:
            # Unopened / checked-to node
            acts.append(CHECK)

        # Raise / bet logic
        max_raises = self.MAX_RAISES
        can_raise = self.n_raises < max_raises and stack > owed

        if can_raise:
            min_inc = max(self.last_raise_size, self.buyin)
            min_raise_to = self.current_bet + min_inc if self.current_bet > 0 else min_inc
            max_raise_to = self.bets[seat] + stack

            r2_to = min_raise_to
            r4_to = min_raise_to + 2 * min_inc

            if r2_to <= max_raise_to:
                acts.append(RAISE_2)

            if self.n_raises > 0 and r4_to <= max_raise_to:
                acts.append(RAISE_4)

        # All-in availability
        if stack > 0:
            allow_allin = False
            if self.street == 0:
                eff_bb = min(self.stacks) / max(self.buyin, 1.0)
                commit_frac = self.bets[seat] / max(self.wallet, 1.0)
                big_action = owed >= 2.5 * self.buyin

                if eff_bb <= 10:
                    allow_allin = True
                elif big_action and commit_frac >= 0.25:
                    allow_allin = True
            else:
                allow_allin = True

            if allow_allin:
                acts.append(ALLIN)

        return acts

    # ── Child creation ────────────────────────────────────────────────────────

    def play(self, action):
        if self.is_terminal():
            raise RuntimeError("play() on terminal state")
        if action not in self._children_cache:
            self._children_cache[action] = self._make_child(action)
        return self._children_cache[action]
    
    def _advance_street(self, stacks, pot, bets, folded, next_street, all_in=False):
        return StreetChanceNode(
            parent=self,
            next_street=next_street,
            hands=self.hands,
            hole_cards=self.hole_cards,
            full_deck=self.full_deck,
            pre_board=self.pre_board,
            equity_p0=self.equity_p0,
            wallet=self.wallet,
            buyin=self.buyin,
            n_players=self.n_players,
            community_cards=self.community_cards,
            stacks=stacks,
            pot=pot,
            folded=folded,
            last_raise_size=self.buyin,
            current_bet=0.0,
            n_raises=0,
            all_in_runout=all_in,
        )
    

    def _make_child(self, action):
        seat    = self.to_move
        stacks  = list(self.stacks)
        bets    = list(self.bets)
        folded  = set(self.folded)
        pot     = self.pot
        cb      = self.current_bet
        lr      = self.last_raise_size
        raises  = self.n_raises
        history = list(self.action_history) + [(seat, action)]

        if action == FOLD:
            folded.add(seat)

        elif action == CHECK:
            # Explicit no-op pass action when not facing a bet
            pass

        elif action == CALL:
            owed = max(cb - bets[seat], 0.0)
            paid = min(owed, stacks[seat])
            stacks[seat] -= paid
            bets[seat]   += paid
            pot          += paid

        elif action in (RAISE_2, RAISE_4):
            min_raise_size = max(lr, self.buyin)
            min_raise_to   = cb + min_raise_size

            if action == RAISE_2:
                # Smallest legal raise
                target = min_raise_to
            else:
                # Larger raise tier
                target = cb + 3 * min_raise_size

            target = min(target, bets[seat] + stacks[seat])


            # If player cannot make a full legal raise, treat as a call/all-in-call.
            if target < min_raise_to:
                owed = max(cb - bets[seat], 0.0)
                paid = min(owed, stacks[seat])
                stacks[seat] -= paid
                bets[seat]   += paid
                pot          += paid
            else:
                inc = target - bets[seat]
                stacks[seat] -= inc
                bets[seat]   += inc
                pot          += inc

                old_cb = cb
                cb     = bets[seat]
                lr     = cb - old_cb
                raises += 1

        elif action == ALLIN:
            total_commit = bets[seat] + stacks[seat]

            # Case 1: all-in does not exceed current bet -> it's just a call for less / call all-in
            if total_commit <= cb:
                paid = stacks[seat]
                stacks[seat] = 0.0
                bets[seat]  += paid
                pot         += paid

            else:
                raise_size      = total_commit - cb
                min_raise_size  = max(lr, self.buyin)

                paid = stacks[seat]
                stacks[seat] = 0.0
                bets[seat]   = total_commit
                pot         += paid

                # Only reopen betting if this is a full legal raise
                if raise_size >= min_raise_size:
                    old_cb = cb
                    cb     = total_commit
                    lr     = cb - old_cb
                    raises += 1
                # else: short all-in overcall; do not change current_bet / last_raise_size

        next_seat = self._next_to_act(seat, folded, stacks, bets, cb, history, raises)

        if next_seat is not None:
            return NLHGameState(
                parent=self, hands=self.hands, hole_cards=self.hole_cards,
                full_deck=self.full_deck, pre_board=self.pre_board,
                equity_p0=self.equity_p0, wallet=self.wallet,
                buyin=self.buyin, n_players=self.n_players,
                street=self.street, community_cards=self.community_cards,
                stacks=stacks, pot=pot, bets=bets, folded=folded,
                action_history=history, current_bet=cb,
                last_raise_size=lr, n_raises=raises, to_move=next_seat,
            )

        active = [i for i in range(self.n_players) if i not in folded]

        if len(active) == 1:
            return NLHGameState(
                parent=self, hands=self.hands, hole_cards=self.hole_cards,
                full_deck=self.full_deck, pre_board=self.pre_board,
                equity_p0=self.equity_p0, wallet=self.wallet,
                buyin=self.buyin, n_players=self.n_players,
                street=self.street, community_cards=self.community_cards,
                stacks=stacks, pot=pot, bets=bets, folded=folded,
                action_history=history, current_bet=0.0,
                last_raise_size=self.buyin, n_raises=0, to_move=None,
            )

        all_allin = all(stacks[i] == 0 for i in active)
        if all_allin:
            next_street = self.street + 1
            if next_street > 3:
                return NLHGameState(
                    parent=self, hands=self.hands, hole_cards=self.hole_cards,
                    full_deck=self.full_deck, pre_board=self.pre_board,
                    equity_p0=self.equity_p0, wallet=self.wallet,
                    buyin=self.buyin, n_players=self.n_players,
                    street=self.street, community_cards=self.community_cards,
                    stacks=stacks, pot=pot, bets=bets, folded=folded,
                    action_history=history, current_bet=0.0,
                    last_raise_size=self.buyin, n_raises=0, to_move=None,
                )
            return self._advance_street(
                stacks=stacks, pot=pot, bets=bets, folded=folded,
                next_street=next_street, all_in=True
            )

        next_street = self.street + 1

        # New street starts with fresh betting state
        cb = 0.0
        raises = 0
        lr = self.buyin

        if next_street > 3:
            return NLHGameState(
                parent=self, hands=self.hands, hole_cards=self.hole_cards,
                full_deck=self.full_deck, pre_board=self.pre_board,
                equity_p0=self.equity_p0, wallet=self.wallet,
                buyin=self.buyin, n_players=self.n_players,
                street=3, community_cards=self.community_cards,
                stacks=stacks, pot=pot, bets=[0.0]*self.n_players,
                folded=folded, action_history=[], current_bet=0.0,
                last_raise_size=self.buyin, n_raises=0, to_move=None,
            )

        return self._advance_street(
            stacks=stacks, pot=pot, bets=bets, folded=folded,
            next_street=next_street, all_in=False
        )

    # ── Next to act ───────────────────────────────────────────────────────────

    def _next_to_act(self, last_seat, folded, stacks, bets, current_bet, history, raises):
        active = [i for i in range(self.n_players) if i not in folded]
        if len(active) <= 1:
            return None

        # Pass 1: anyone who still owes chips
        for offset in range(1, self.n_players + 1):
            c = (last_seat + offset) % self.n_players
            if c in folded or stacks[c] == 0:
                continue
            if bets[c] < current_bet:
                return c

        # Pass 2: anyone who hasn't acted yet this street
        acted = {s for s, _ in history}
        for offset in range(1, self.n_players + 1):
            c = (last_seat + offset) % self.n_players
            if c in folded or stacks[c] == 0:
                continue
            if c not in acted:
                return c

        # BB option: preflop, no raises, BB hasn't acted yet
        if self.street == 0 and raises == 0:
            bb = self.n_players - 1
            if (
                bb not in folded
                and stacks[bb] > 0
                and bets[bb] == current_bet
                and last_seat != bb
                and not any(s == bb for s, _ in history) 
            ):
                return bb

        return None

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluation(self):
        if not self.is_terminal():
            raise RuntimeError("evaluation() called on non-terminal state")

        active = [i for i in range(self.n_players) if i not in self.folded]
        contributed = [self.wallet - self.stacks[i] for i in range(self.n_players)]

        # Fold terminal
        if len(active) == 1:
            winner = active[0]
            payoffs = [-contributed[i] for i in range(self.n_players)]
            payoffs[winner] += self.pot
            return payoffs

        # Heads-up showdown
        if self.hole_cards is not None and len(active) == 2 and 0 in active:
            p1   = [i for i in active if i != 0][0]
            p0c  = self.hole_cards[0]
            p1c  = self.hole_cards[p1]
            comm = list(self.community_cards)

            payoffs = [-contributed[i] for i in range(self.n_players)]

            # Exact river showdown
            if self.street == 3 and len(comm) == 5:
                v0 = _eval7(list(p0c) + comm)
                v1 = _eval7(list(p1c) + comm)

                if v0 > v1:
                    payoffs[0] += self.pot
                elif v1 > v0:
                    payoffs[p1] += self.pot
                else:
                    payoffs[0]  += self.pot / 2.0
                    payoffs[p1] += self.pot / 2.0
                return payoffs

            # Unfinished board: equity approximation
            if self.pre_board is not None:
                equity = _mc_equity(p0c, p1c, comm, self.full_deck,
                                    pre_board=self.pre_board, n=100)
            else:
                equity = _mc_equity(p0c, p1c, comm, self.full_deck, n=100)

            payoffs[0]  += equity * self.pot
            payoffs[p1] += (1.0 - equity) * self.pot
            return payoffs

        # Multiway showdown
        if self.hole_cards is not None and len(active) > 2:
            comm = list(self.community_cards)
            if self.street == 3 and len(comm) == 5:
                board = comm
            elif self.pre_board is not None:
                board = list(self.pre_board[:5])
            else:
                board = comm

            hand_vals = {}
            for idx in active:
                cards = list(self.hole_cards[idx]) + board
                hand_vals[idx] = _eval7(cards)

            best_val = max(hand_vals[idx] for idx in active)
            winners = [idx for idx in active if hand_vals[idx] == best_val]

            payoffs = [-contributed[i] for i in range(self.n_players)]
            share = self.pot / len(winners)
            for w in winners:
                payoffs[w] += share
            return payoffs

        # Fallback approximation
        payoffs = [-contributed[i] for i in range(self.n_players)]
        payoffs[0] += self.equity_p0 * self.pot
        return payoffs

    # ── Info set / repr ───────────────────────────────────────────────────────

    def inf_set(self): return self._inf_set_str

    def _build_inf_set(self):
        seat   = self.to_move
        sname  = SEAT_NAMES.get(seat, str(seat))
        hb     = self.hands[seat] if seat is not None else "X"
        street = STREET_NAMES.get(self.street, str(self.street))
        bb     = _board_bucket(self.community_cards)
        hist = "_".join(f"{s}{_ACTION_ABBR[a]}" for s, a in self.action_history)

        if self.street == 0 and seat is not None:
            pf_ctx    = _preflop_action_context(self)
            pos_ctx   = _position_context(self, seat)
            depth_ctx = _effective_stack_bucket(self, seat)
            return f"{sname}.{hb}.{street}.{pf_ctx}.{pos_ctx}.{depth_ctx}.{hist}"

        # Postflop: include hand strength bucket so CFR differentiates
        # "trash hand on a paired board" from "top pair on a paired board".
        # Without this, all hands with the same preflop bucket collapse into
        # a single infoset regardless of how they connected with the board.
        hole  = self.hole_cards[seat] if (self.hole_cards and seat is not None) else None
        hsb   = _hand_strength_bucket(hole, self.community_cards, self.full_deck)
        return f"{sname}.{street}.{bb}.{hsb}.{hist}" # Hand bucket removed from infoset for now to reduce size and focus equity

    def __repr__(self):
        seat = SEAT_NAMES.get(self.to_move, str(self.to_move))
        return (f"NLHGameState(street={self.street}, seat={seat}, "
                f"pot={self.pot:.1f}, bet={self.current_bet:.1f}, "
                f"community={len(self.community_cards)}, folded={self.folded})")