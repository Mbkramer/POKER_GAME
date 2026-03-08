"""
preflop_abstraction.py
----------------------
Preflop and postflop card abstraction for 6-max NLH CFR.

Converts real hole cards → one of 16 preflop strength buckets (0 = strongest),
or postflop equity → one of 12 postflop strength buckets (0 = strongest).

────────────────────────────────────────────────────────────────────────────────
PREFLOP BUCKETS  (0 = strongest, 15 = weakest)
────────────────────────────────────────────────────────────────────────────────
Buckets 0–11 are assigned via an explicit lookup table (_BUCKET_TABLE).
Buckets 12–15 are assigned by _classify_trash() for all hands not in the table.

  0   Monster pairs             AA, KK
  1   Strong pairs              QQ, JJ
  2   Medium pairs              TT, 99
  3   Small pairs               88–55
  4   Tiny pairs                44–22
  5   Premium suited            AKs, AQs, AJs, ATs
  6   Premium offsuit + KQs     AKo, AQo, KQs
  7   Strong broadways          AJo, ATo, KJs, KQo, QJs
  8   Weak broadways            KJo, KTo, QJo, QTs, JTs
  9   Suited weak aces          A9s–A2s
 10   Suited connectors         K9s, Q9s, J9s, T9s, 98s, 87s, 76s, 65s, 54s
 11   Weak playable             KTs, offsuit connectors T9o–65o, K9o, Q9o, JTo
 12   Trash – low disconnected  wide-gap, low-rank, offsuit (e.g. 72o, 93o)
 13   Trash – Jx/Qx offsuit     J or Q as high card, offsuit, not in table above
 14   Trash – weak suited       suited hands not covered by buckets 5–10
 15   Trash – offsuit connectors gap <= 2, low-rank, offsuit (e.g. 64o, 53o)

────────────────────────────────────────────────────────────────────────────────
POSTFLOP BUCKETS  (0 = strongest, 11 = weakest)
────────────────────────────────────────────────────────────────────────────────
Equity is estimated by Monte Carlo simulation against one random opponent.
11 thresholds produce 12 buckets (n thresholds -> n+1 buckets).
Finer resolution is concentrated in the 0.33-0.65 band where draws, middle
pairs, and top-pair-weak-kicker hands cluster. An earlier 8-bucket scheme
compressed that entire band into two buckets, causing CFR to treat
strategically distinct situations identically.

  0   equity >= 0.85   near-certain winner
  1   equity >= 0.75   strong made hand
  2   equity >= 0.65   good made hand
  3   equity >= 0.58   top pair territory
  4   equity >= 0.52   slight favourite
  5   equity >= 0.46   coinflip / middle
  6   equity >= 0.40   slight underdog
  7   equity >= 0.33   weak / draw-dependent
  8   equity >= 0.25   dominated / weak draw
  9   equity >= 0.15   near-dead
 10   equity >= 0.08   bluff territory only
 11   equity <  0.08   near-certain loser
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from typing import List, Tuple

# phevaluator is required only for postflop equity bucketing.
# The import is deferred inside postflop_equity_bucket() so the module remains
# usable for preflop-only workflows without installing phevaluator.

# ── Rank / suit constants ─────────────────────────────────────────────────────

RANKS    = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_VAL = {r: i for i, r in enumerate(RANKS)}   # 2 -> 0, A -> 12
SUITS    = ["c", "d", "h", "s"]

# ── Bucket count constants ────────────────────────────────────────────────────

# 16 preflop buckets: 0-11 from the explicit lookup table, 12-15 from _classify_trash().
N_PREFLOP_BUCKETS  = 16

# 12 postflop buckets: derived from the 11 thresholds in _POSTFLOP_BOUNDARIES.
N_POSTFLOP_BUCKETS = 12

# ── Preflop bucket assignment table ──────────────────────────────────────────
# Keys are (high_rank, low_rank, suited) tuples, e.g. ("A", "K", True) -> 5.
# high_rank is always >= low_rank by RANK_VAL ordering.
# Pairs are stored with suited=False (a pair can never be suited).
# Any hand not in this table falls through to _classify_trash().

def _build_bucket_table() -> dict:
    table = {}

    def add(hi, lo, suited, bucket):
        table[(hi, lo, suited)] = bucket

    def add_pair(rank, bucket):
        table[(rank, rank, False)] = bucket

    # Bucket 0: monster pairs
    add_pair("A", 0); add_pair("K", 0)

    # Bucket 1: strong pairs
    add_pair("Q", 1); add_pair("J", 1)

    # Bucket 2: medium pairs
    add_pair("T", 2); add_pair("9", 2)

    # Bucket 3: small pairs
    for r in ["8", "7", "6", "5"]:
        add_pair(r, 3)

    # Bucket 4: tiny pairs
    for r in ["4", "3", "2"]:
        add_pair(r, 4)

    # Bucket 5: premium suited
    add("A", "K", True, 5)
    add("A", "Q", True, 5)
    add("A", "J", True, 5)
    add("A", "T", True, 5)

    # Bucket 6: premium offsuit + KQs
    add("A", "K", False, 6)
    add("A", "Q", False, 6)
    add("K", "Q", True,  6)

    # Bucket 7: strong broadways
    add("A", "J", False, 7)
    add("A", "T", False, 7)
    add("K", "J", True,  7)
    add("K", "Q", False, 7)
    add("Q", "J", True,  7)

    # Bucket 8: weak broadways
    add("K", "J", False, 8)
    add("K", "T", False, 8)
    add("Q", "J", False, 8)
    add("Q", "T", True,  8)
    add("J", "T", True,  8)

    # Bucket 9: suited weak aces (A9s down to A2s)
    for lo in ["9", "8", "7", "6", "5", "4", "3", "2"]:
        add("A", lo, True, 9)

    # Bucket 10: suited connectors and one-gappers
    for hi, lo in [("K", "9"), ("Q", "9"), ("J", "9"), ("T", "9"),
                   ("9", "8"), ("8", "7"), ("7", "6"), ("6", "5"), ("5", "4")]:
        add(hi, lo, True, 10)

    # Bucket 11: weak playable -- KTs, offsuit connectors T9o-65o, weak offsuit broadways
    add("K", "T", True, 11)
    for hi, lo in [("T", "9"), ("9", "8"), ("8", "7"), ("7", "6"), ("6", "5")]:
        add(hi, lo, False, 11)
    add("K", "9", False, 11)
    add("Q", "9", False, 11)
    add("J", "T", False, 11)

    return table


def _rank_gap(hi: str, lo: str) -> int:
    """Absolute rank distance between two cards using RANK_VAL (0-based: 2=0, A=12)."""
    return abs(RANK_VAL[hi] - RANK_VAL[lo])


def _classify_trash(rank1: str, rank2: str, suited: bool) -> int:
    """
    Assign a bucket (12-15) to hands not covered by _BUCKET_TABLE.
    Inputs must already be normalised so rank1 >= rank2 by RANK_VAL.

    The trash band is split into four sub-buckets to give CFR finer resolution
    over marginal and garbage holdings:

      13  Jx / Qx offsuit   -- J or Q as the high card, offsuit, not in the table.
                               K-high and A-high hands never reach here: all Ax
                               suited are bucket 9, and Kx/Ax offsuit trash is
                               rare enough not to warrant its own sub-bucket.
      14  Weak suited        -- any suited hand not assigned by buckets 5-10.
      15  Offsuit connectors -- gap <= 2, low-rank, offsuit (e.g. 64o, 53o).
      12  Disconnected low   -- everything else: wide gap, low rank, offsuit.
    """
    gap      = _rank_gap(rank1, rank2)
    hi_is_jq = rank1 in {"J", "Q"}   # K/A-high trash does not reach this function

    if hi_is_jq and not suited:
        return 13   # Jx/Qx offsuit trash

    if suited:
        return 14   # weak suited garbage

    if gap <= 2:
        return 15   # offsuit connected-ish trash

    return 12       # disconnected low offsuit trash


_BUCKET_TABLE = _build_bucket_table()


def hand_to_bucket(rank1: str, rank2: str, suited: bool) -> int:
    """
    Return the preflop bucket (0-15) for a two-card hand.

    Args:
        rank1:  rank string, e.g. "A", "T", "2"
        rank2:  rank string
        suited: True if both cards share the same suit

    Normalises so rank1 is always the higher card by RANK_VAL, then looks up
    the explicit table before falling back to _classify_trash().
    Pairs are forced to suited=False (a pair cannot be suited).
    """
    if RANK_VAL[rank1] < RANK_VAL[rank2]:
        rank1, rank2 = rank2, rank1

    if rank1 == rank2:
        suited = False

    base = _BUCKET_TABLE.get((rank1, rank2, suited))
    if base is not None:
        return base

    return _classify_trash(rank1, rank2, suited)


# ── Full deck and deal generation ─────────────────────────────────────────────

def _make_deck() -> List[Tuple[str, str]]:
    """Return a list of (rank, suit) tuples for all 52 cards."""
    return [(r, s) for r in RANKS for s in SUITS]


def _cards_to_bucket(card1: Tuple[str, str], card2: Tuple[str, str]) -> int:
    """Convert a two-card hand expressed as (rank, suit) tuples to its preflop bucket."""
    r1, s1 = card1
    r2, s2 = card2
    return hand_to_bucket(r1, r2, suited=(s1 == s2))


class PreflopAbstraction:
    """
    Samples preflop deals as sequences of bucket IDs, one per seat.

    The CFR chance node uses this class to draw from the abstracted deal space
    rather than the full combinatorial space of real hole cards.
    """

    def __init__(self, n_players: int = 6):
        self.n_players = n_players
        self.deck      = _make_deck()

    def sample_deal(self) -> dict:
        """
        Draw one random deal from a freshly shuffled deck.

        Returns a dict with:
          'buckets':    tuple of preflop bucket IDs, one per seat
          'hole_cards': tuple of ((rank, suit), (rank, suit)) per seat
          'full_deck':  the shuffled 52-card list used for this deal
                        (passed downstream for postflop equity lookups)
        """
        deck = list(self.deck)
        random.shuffle(deck)
        buckets    = []
        hole_cards = []
        for i in range(self.n_players):
            c1, c2 = deck[i * 2], deck[i * 2 + 1]
            buckets.append(_cards_to_bucket(c1, c2))
            hole_cards.append((c1, c2))
        return {
            "buckets":    tuple(buckets),
            "hole_cards": tuple(hole_cards),
            "full_deck":  deck,
        }

    def all_deals(self, max_deals: int = 10_000) -> List[dict]:
        """
        Sample up to max_deals *unique* deals (unique by hole-card identity,
        not bucket identity). Uniqueness is enforced via a seen-set, so this
        is Monte Carlo sampling with deduplication, not exhaustive enumeration.

        Terminates early if max_attempts (20x max_deals) is reached without
        collecting enough unique deals -- this can happen at very small table
        sizes where the deal space is small.
        """
        seen         = set()
        deals        = []
        attempts     = 0
        max_attempts = max_deals * 20

        while len(deals) < max_deals and attempts < max_attempts:
            d = self.sample_deal()
            # Canonical key: sort cards within each hand, then sort hands
            # across seats so deal order does not create false duplicates.
            key = tuple(
                tuple(sorted(hand)) for hand in sorted(d["hole_cards"])
            )
            if key not in seen:
                seen.add(key)
                deals.append(d)
            attempts += 1

        return deals

    def bucket_name(self, bucket: int) -> str:
        """Human-readable label for a preflop bucket index (0-15)."""
        names = [
            "Monster pairs (AA/KK)",            # 0
            "Strong pairs (QQ/JJ)",             # 1
            "Medium pairs (TT/99)",             # 2
            "Small pairs (88-55)",              # 3
            "Tiny pairs (44-22)",               # 4
            "Premium suited",                   # 5
            "Premium offsuit + KQs",            # 6
            "Strong broadways",                 # 7
            "Weak broadways",                   # 8
            "Suited weak aces",                 # 9
            "Suited connectors",                # 10
            "Weak playable",                    # 11
            "Trash - low disconnected offsuit", # 12
            "Trash - Jx/Qx offsuit",            # 13
            "Trash - weak suited",              # 14
            "Trash - offsuit connectors",       # 15
        ]
        return names[bucket] if 0 <= bucket < len(names) else f"bucket_{bucket}"


# ── Preflop equity simulation ─────────────────────────────────────────────────
#
# Self-contained Monte Carlo evaluator for preflop equity estimates.
# Does NOT use phevaluator -- hand strength is computed via a pure-Python
# eval5/eval7 pair, so this function works without any external dependency.
#
# Note on RANK_VAL: this function defines a local rank map (face values 2-14,
# Ace=14) used only inside card_val() comparison tuples. This is intentionally
# different from the module-level RANK_VAL (0-indexed, 2=0 A=12), which is used
# solely for gap arithmetic in the preflop bucket table. The two maps are never
# mixed and must not be merged.

def preflop_equity_vs_random(hole_cards_p0, n_simulations: int = 200) -> float:
    """
    Estimate P0's heads-up equity against one random opponent via Monte Carlo.

    Args:
        hole_cards_p0: iterable of 2 (rank, suit) tuples, e.g. [("A","s"),("K","h")]
        n_simulations: number of MC runouts (default 200, ~+-3% accuracy)

    Returns:
        float in [0.0, 1.0] -- fraction of pots won (ties count as 0.5)
    """
    # Local rank values 2-14 (Ace=14) used only for card_val() hand-strength tuples.
    # See module-level note above for why this differs from RANK_VAL.
    _local_rank_val = {
        "2": 2,  "3": 3,  "4": 4,  "5": 5,  "6": 6,  "7": 7,
        "8": 8,  "9": 9,  "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }

    def card_val(r, s):
        return (_local_rank_val[r], s)

    def eval5(cards):
        """
        Evaluate a 5-card hand. Returns a comparable tuple; higher = better.
        Hand rank encoding: 8=straight flush, 7=quads, 6=full house, 5=flush,
                            4=straight, 3=trips, 2=two pair, 1=pair, 0=high card.
        """
        vals   = sorted([c[0] for c in cards], reverse=True)
        suits  = [c[1] for c in cards]
        vc     = Counter(vals)
        groups = sorted(vc.items(), key=lambda x: (-x[1], -x[0]))

        is_flush    = len(set(suits)) == 1
        uvals       = sorted(set(vals), reverse=True)
        is_straight = False
        s_high      = 0
        for i in range(len(uvals) - 4):
            if uvals[i] - uvals[i + 4] == 4:
                is_straight = True
                s_high = uvals[i]
                break
        # Wheel: A-2-3-4-5
        if not is_straight and {14, 2, 3, 4, 5}.issubset(set(vals)):
            is_straight = True
            s_high = 5

        if is_straight and is_flush:             return (8, s_high)
        if groups[0][1] == 4:                    return (7, groups[0][0], groups[1][0])
        if groups[0][1] == 3 and groups[1][1] == 2:
                                                 return (6, groups[0][0], groups[1][0])
        if is_flush:                             return (5,) + tuple(vals)
        if is_straight:                          return (4, s_high)
        if groups[0][1] == 3:                    return (3, groups[0][0]) + tuple(v for v, _ in groups[1:])
        if groups[0][1] == 2 and groups[1][1] == 2:
            hi = max(groups[0][0], groups[1][0])
            lo = min(groups[0][0], groups[1][0])
            return (2, hi, lo, groups[2][0])
        if groups[0][1] == 2:                    return (1, groups[0][0]) + tuple(v for v, _ in groups[1:])
        return (0,) + tuple(vals)

    def eval7(cards):
        """Return the best 5-card hand value from 7 cards."""
        return max(eval5(combo) for combo in combinations(cards, 5))

    # Build a full deck and remove P0's known cards
    full_deck = [(r, s) for r in RANKS for s in SUITS]
    p0_set    = set(map(tuple, hole_cards_p0))
    deck      = [c for c in full_deck if tuple(c) not in p0_set]

    wins = 0.0
    for _ in range(n_simulations):
        random.shuffle(deck)
        opp_cards = deck[:2]
        board     = deck[2:7]

        # Safety check: skip if opponent cards collide with P0 (should not occur)
        if set(map(tuple, opp_cards)) & p0_set:
            continue

        p0_hand  = [card_val(*c) for c in hole_cards_p0] + [card_val(*c) for c in board]
        opp_hand = [card_val(*c) for c in opp_cards]     + [card_val(*c) for c in board]

        p0_val  = eval7(p0_hand)
        opp_val = eval7(opp_hand)

        if   p0_val > opp_val:  wins += 1.0
        elif p0_val == opp_val: wins += 0.5

    return wins / n_simulations


# ── Postflop equity bucketing ─────────────────────────────────────────────────
#
# Uses phevaluator for fast, accurate hand evaluation. The import is deferred
# inside postflop_equity_bucket() so preflop-only callers do not need it.
#
# Cache: module-level dict keyed by (hole_card_ids, community_card_ids).
# CFR traversal hits the same infoset from many traversal paths; the cache
# avoids redundant MC runs. The cache grows to at most
#   n_deals x n_streets x n_active_players unique keys per training run.
# Call clear_postflop_cache() between runs if deal pools or board cards change.

_postflop_bucket_cache: dict = {}

# 11 thresholds define 12 buckets: bucket b requires equity >= _POSTFLOP_BOUNDARIES[b].
# The final bucket (11) catches all equity values below the last threshold (0.08).
_POSTFLOP_BOUNDARIES = [0.85, 0.75, 0.65, 0.58, 0.52, 0.46, 0.40, 0.33, 0.25, 0.15, 0.08]

# Suit string normalisation for _card_to_ph().
# Accepts both full suit names (from Card objects) and single-character strings.
_PF_SUIT_MAP = {
    "SPADES": "s", "HEARTS": "h", "DIAMONDS": "d", "CLUBS": "c",
    "s": "s",      "h": "h",      "d": "d",         "c": "c",
}


def _card_to_ph(card) -> str:
    """
    Convert a card to a phevaluator-format string, e.g. "As", "Td", "2c".
    Accepts either a (rank, suit) tuple or a Card object with .value/.suit attrs.
    Normalises "10" -> "T" for phevaluator compatibility.
    """
    if hasattr(card, "value") and hasattr(card, "suit"):
        rank = "T" if card.value == "10" else card.value
        suit = _PF_SUIT_MAP.get(card.suit, card.suit[0].lower())
        return f"{rank}{suit}"
    rank, suit = card[0], card[1]
    rank = "T" if rank == "10" else rank
    suit = _PF_SUIT_MAP.get(suit, suit[0].lower())
    return f"{rank}{suit}"


def _card_id_pf(card) -> str:
    """
    Return a stable string identifier for a card, used as a cache key component.
    Accepts Card objects (uses .id if present, else .suit/.value) or (rank, suit) tuples.
    """
    if hasattr(card, "id"):
        return card.id
    if hasattr(card, "value") and hasattr(card, "suit"):
        return f"{card.suit[0]}{card.value}"
    rank, suit = card[0], card[1]
    return f"{suit}{rank}"


def postflop_equity_bucket(hole_cards, community_cards, full_deck,
                           n_simulations: int = 200) -> int:
    """
    Estimate a player's heads-up equity given hole cards and board, then map
    to a strength bucket 0-11 (0 = strongest, 11 = weakest).

    Equity is estimated by Monte Carlo: opponent hole cards and any remaining
    board cards are sampled randomly from the cards not already in play.

    Args:
        hole_cards:      iterable of 2 cards (tuple or Card object)
        community_cards: iterable of 3-5 cards (current board)
        full_deck:       full 52-card deck in the same card format
        n_simulations:   MC samples; 200 gives ~+-3% accuracy, sufficient
                         for 12-bucket resolution (~6-7% bucket width)

    Returns:
        int 0-11

    Requires:
        phevaluator  (pip install phevaluator)
    """
    from phevaluator import evaluate_cards as _ph_eval

    # ── Cache lookup ──────────────────────────────────────────────────────────
    hole_key  = tuple(_card_id_pf(c) for c in hole_cards)
    comm_key  = tuple(_card_id_pf(c) for c in community_cards)
    cache_key = (hole_key, comm_key)

    if cache_key in _postflop_bucket_cache:
        return _postflop_bucket_cache[cache_key]

    # ── Build the pool of cards available for sampling ────────────────────────
    known     = set(hole_key) | set(comm_key)
    available = [c for c in full_deck if _card_id_pf(c) not in known]

    needed_board = 5 - len(community_cards)   # runout cards still to come
    needed_total = needed_board + 2            # runout + opponent's 2 hole cards

    if len(available) < needed_total:
        # Degenerate state (should not occur in a well-formed game tree).
        # Return the neutral mid-bucket rather than raising an exception.
        _postflop_bucket_cache[cache_key] = 5
        return 5

    ph_hole = [_card_to_ph(c) for c in hole_cards]
    ph_comm = [_card_to_ph(c) for c in community_cards]

    # ── Monte Carlo equity estimation ─────────────────────────────────────────
    wins       = 0.0
    valid_sims = 0

    for _ in range(n_simulations):
        sample    = random.sample(available, needed_total)
        runout    = sample[:needed_board]
        opp_cards = sample[needed_board:]

        full_board = ph_comm + [_card_to_ph(c) for c in runout]
        ph_opp     = [_card_to_ph(c) for c in opp_cards]

        hero_rank = _ph_eval(*(ph_hole + full_board))
        opp_rank  = _ph_eval(*(ph_opp  + full_board))

        # phevaluator convention: lower rank number = stronger hand
        if   hero_rank < opp_rank:  wins += 1.0
        elif hero_rank == opp_rank: wins += 0.5
        valid_sims += 1

    equity = wins / max(valid_sims, 1)

    # ── Map equity -> bucket (scan thresholds high-to-low) ───────────────────
    bucket = N_POSTFLOP_BUCKETS - 1   # default: weakest bucket
    for b, threshold in enumerate(_POSTFLOP_BOUNDARIES):
        if equity >= threshold:
            bucket = b
            break

    _postflop_bucket_cache[cache_key] = bucket
    return bucket


def clear_postflop_cache() -> None:
    """
    Clear the postflop equity bucket cache.
    Call between training runs whenever deal pools or board assignments change,
    to avoid stale cached values being returned for recycled card combinations.
    """
    _postflop_bucket_cache.clear()


# ─────────────────────────────────────────────────────────────────
# Run directly to verify bucket assignments and deal sampling:
#   python preflop_abstraction.py

if __name__ == "__main__":
    abst = PreflopAbstraction(n_players=6)

    # Spot-check known hands against expected bucket IDs
    checks = [
        ("A", "A", False,  0),
        ("K", "K", False,  0),
        ("Q", "Q", False,  1),
        ("T", "T", False,  2),
        ("9", "9", False,  2),
        ("7", "7", False,  3),
        ("3", "3", False,  4),
        ("A", "K", True,   5),
        ("A", "J", True,   5),
        ("A", "K", False,  6),
        ("K", "Q", True,   6),
        ("A", "J", False,  7),
        ("K", "J", True,   7),
        ("K", "J", False,  8),
        ("J", "T", True,   8),
        ("A", "5", True,   9),
        ("A", "2", True,   9),
        ("T", "9", True,  10),
        ("6", "5", True,  10),
        ("J", "T", False, 11),
        ("T", "9", False, 11),
        ("7", "2", False, 12),
        ("9", "3", False, 12),
    ]
    print("=== Bucket spot checks ===")
    for r1, r2, s, expected in checks:
        got        = hand_to_bucket(r1, r2, s)
        tag        = "OK" if got == expected else f"FAIL (expected {expected})"
        suited_str = "s" if s else "o"
        print(f"  {r1}{r2}{suited_str}: bucket {got}  {tag}")

    # Sample one deal
    print("\n=== Sample deal ===")
    deal = abst.sample_deal()
    for seat, b in enumerate(deal["buckets"]):
        print(f"  Seat {seat}: bucket {b} ({abst.bucket_name(b)})")

    # Collect unique deals and show bucket distribution
    print("\n=== Sampling 1,000 unique deals ===")
    deals = abst.all_deals(max_deals=1_000)
    print(f"  Got {len(deals)} unique deals")

    flat = [b for d in deals for b in d["buckets"]]
    dist = Counter(flat)
    print("\n=== Bucket distribution across all seats ===")
    for b in range(N_PREFLOP_BUCKETS):
        print(f"  Bucket {b:2d} ({abst.bucket_name(b)}): {dist[b]:,}")