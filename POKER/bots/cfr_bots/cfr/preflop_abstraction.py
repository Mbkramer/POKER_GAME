"""
preflop_abstraction.py
----------------------
Preflop card abstraction for 6-max NLH CFR.

Converts real hole cards → one of 8 strength buckets.
No external library needed -- pure rank/suit logic.

Bucket definitions (0 = strongest):
  0  Premium pairs:      AA, KK
  1  Strong pairs:       QQ, JJ, AKs
  2  Medium pairs+:      TT, AQs, AKo, AJs
  3  Small pairs+:       99-77, KQs, ATs, AQo
  4  Broadways+:         66-55, AJo, KJs, QJs, KQo, ATo
  5  Suited connectors:  44-22, KTs, QTs, JTs, T9s, 98s, 87s, 76s, 65s
  6  Weak suited:        A2s-A9s, K9s, Q9s, J9s, KJo, QJo, KTo
  7  Trash:              everything else

"""

from __future__ import annotations
import random
from itertools import combinations
from typing import List, Tuple

# ── Rank ordering ─────────────────────────────────────────────────────────────

RANKS     = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
RANK_VAL  = {r: i for i, r in enumerate(RANKS)}   # 2=0 ... A=12
SUITS     = ["c","d","h","s"]

N_BUCKETS = 8


# ── Bucket assignment table ───────────────────────────────────────────────────
# Key: (high_rank, low_rank, suited)  e.g. ("A","K",True) → bucket
# Built programmatically below.

def _build_bucket_table() -> dict:
    table = {}

    def add(hi, lo, suited, bucket):
        table[(hi, lo, suited)] = bucket

    def add_pair(rank, bucket):
        table[(rank, rank, False)] = bucket   # pairs are never "suited"

    # Bucket 0: premium pairs
    add_pair("A", 0); add_pair("K", 0)

    # Bucket 1: strong pairs + top suited
    add_pair("Q", 1); add_pair("J", 1)
    add("A","K", True,  1)

    # Bucket 2: medium pairs + strong broadways
    add_pair("T", 2)
    add("A","Q", True,  2); add("A","K", False, 2); add("A","J", True,  2)

    # Bucket 3: small-medium pairs + strong suited
    for r in ["9","8","7"]:
        add_pair(r, 3)
    add("K","Q", True,  3); add("A","T", True,  3); add("A","Q", False, 3)

    # Bucket 4: small pairs + broadway combos
    for r in ["6","5"]:
        add_pair(r, 4)
    add("A","J", False, 4)
    add("K","J", True,  4); add("Q","J", True,  4)
    add("K","Q", False, 4); add("A","T", False, 4)

    # Bucket 5: tiny pairs + suited connectors / one-gappers
    for r in ["4","3","2"]:
        add_pair(r, 5)
    suited_connectors = [
        ("K","T"), ("Q","T"), ("J","T"),
        ("T","9"), ("9","8"), ("8","7"), ("7","6"), ("6","5"),
    ]
    for hi, lo in suited_connectors:
        add(hi, lo, True, 5)

    # Bucket 6: weak suited aces/kings + offsuit broadways
    weak_suited = [
        ("A","9"),("A","8"),("A","7"),("A","6"),("A","5"),
        ("A","4"),("A","3"),("A","2"),
        ("K","9"),("Q","9"),("J","9"),
    ]
    for hi, lo in weak_suited:
        add(hi, lo, True, 6)
    add("K","J", False, 6); add("Q","J", False, 6); add("K","T", False, 6)

    # Everything else → bucket 7
    return table


_BUCKET_TABLE = _build_bucket_table()


def hand_to_bucket(rank1: str, rank2: str, suited: bool) -> int:
    """
    Convert two card ranks + suited flag to a bucket ID (0-7).

    rank1, rank2: one of "2"-"9","T","J","Q","K","A"
    suited:       True if both cards share the same suit
    """
    # Normalise: high rank first
    if RANK_VAL[rank1] < RANK_VAL[rank2]:
        rank1, rank2 = rank2, rank1

    if rank1 == rank2:
        suited = False   # pairs can't be suited

    return _BUCKET_TABLE.get((rank1, rank2, suited), 7)


# ── Full deck and deal generation ─────────────────────────────────────────────

def _make_deck() -> List[Tuple[str, str]]:
    """Returns list of (rank, suit) for all 52 cards."""
    return [(r, s) for r in RANKS for s in SUITS]


def _cards_to_bucket(card1: Tuple[str,str], card2: Tuple[str,str]) -> int:
    r1, s1 = card1
    r2, s2 = card2
    return hand_to_bucket(r1, r2, suited=(s1 == s2))


class PreflopAbstraction:
    """
    Generates and samples preflop deals as bucket-ID tuples.

    Each deal is a tuple of n_players bucket IDs, one per seat.
    The chance node samples from this space.

    """

    def __init__(self, n_players: int = 6):
        self.n_players = n_players
        self.deck      = _make_deck()

    def sample_deal(self) -> dict:
        """
        Sample one random deal.
        Returns dict with:
          'buckets':    tuple of bucket IDs per seat
          'hole_cards': tuple of ((rank,suit),(rank,suit)) per seat
          'full_deck':  full list of (rank,suit) tuples (all 52 cards, same objects)
        """
        deck   = list(self.deck)
        random.shuffle(deck)
        buckets    = []
        hole_cards = []
        for i in range(self.n_players):
            c1, c2 = deck[i*2], deck[i*2 + 1]
            buckets.append(_cards_to_bucket(c1, c2))
            hole_cards.append((c1, c2))
        return {
            'buckets':    tuple(buckets),
            'hole_cards': tuple(hole_cards),
            'full_deck':  deck,
        }

    def all_deals(self, max_deals: int = 10_000) -> List[dict]:
        """
        Enumerate unique deals up to max_deals.
        Each deal is a dict with 'buckets' and 'hole_cards'.

        """
        seen  = set()
        deals = []
        attempts = 0
        max_attempts = max_deals * 20

        while len(deals) < max_deals and attempts < max_attempts:
            d = self.sample_deal()
            # Canonical key: sort each hand, then sort hands across seats
            key = tuple(
                tuple(sorted(hand)) for hand in sorted(d['hole_cards'])
            )
            if key not in seen:
                seen.add(key)
                deals.append(d)
            attempts += 1

        return deals

    def bucket_name(self, bucket: int) -> str:
        names = [
            "Premium (AA/KK)",
            "Strong (QQ/JJ/AKs)",
            "Medium (TT/AQs/AKo)",
            "Small pair+ (99-77/KQs)",
            "Broadway (66-55/AJo/KJs)",
            "Suited conn (44-22/T9s)",
            "Weak suited (A9s-A2s)",
            "Trash",
        ]
        return names[bucket] if 0 <= bucket < len(names) else f"bucket_{bucket}"


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    abst = PreflopAbstraction(n_players=6)

    # Spot-check known hands
    checks = [
        ("A", "A", False, 0),
        ("K", "K", False, 0),
        ("A", "K", True,  1),
        ("A", "K", False, 2),
        ("T", "9", True,  5),
        ("7", "2", False, 7),
    ]
    print("=== Bucket spot checks ===")
    for r1, r2, s, expected in checks:
        got = hand_to_bucket(r1, r2, s)
        tag = "OK" if got == expected else f"FAIL (expected {expected})"
        suited_str = "s" if s else "o"
        print(f"  {r1}{r2}{suited_str}: bucket {got}  {tag}")

    # Sample a deal
    print("\n=== Sample deal ===")
    deal = abst.sample_deal()
    for seat, b in enumerate(deal):
        print(f"  Seat {seat}: bucket {b} ({abst.bucket_name(b)})")

    # Enumerate deals
    print("\n=== Enumerating 1,000 unique deals ===")
    deals = abst.all_deals(max_deals=1_000)
    print(f"  Got {len(deals)} unique bucket-tuple deals")

    # Bucket distribution
    from collections import Counter
    flat = [b for d in deals for b in d]
    dist = Counter(flat)
    print("\n=== Bucket distribution across all seats ===")
    for b in range(N_BUCKETS):
        print(f"  Bucket {b} ({abst.bucket_name(b)}): {dist[b]:,}")


# ── Equity simulation ─────────────────────────────────────────────────────────

def preflop_equity_vs_random(hole_cards_p0, n_simulations=200):
    """
    Estimate P0's equity against one random opponent using Monte Carlo.

    """
    import random
    from itertools import combinations

    RANK_VAL  = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,
                 "9":9,"T":10,"J":11,"Q":12,"K":13,"A":14}

    def card_val(r, s): return (RANK_VAL[r], s)

    def eval5(cards):
        """Evaluate 5 cards. Returns comparable tuple (higher = better)."""
        vals  = sorted([c[0] for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        from collections import Counter
        vc = Counter(vals)
        groups = sorted(vc.items(), key=lambda x: (-x[1], -x[0]))

        is_flush    = len(set(suits)) == 1
        uvals       = sorted(set(vals), reverse=True)
        is_straight = False
        s_high      = 0
        for i in range(len(uvals)-4):
            if uvals[i] - uvals[i+4] == 4:
                is_straight = True; s_high = uvals[i]; break
        if not is_straight and {14,2,3,4,5}.issubset(set(vals)):
            is_straight = True; s_high = 5

        if is_straight and is_flush: return (8, s_high)
        if groups[0][1] == 4:        return (7, groups[0][0], groups[1][0])
        if groups[0][1]==3 and groups[1][1]==2: return (6, groups[0][0], groups[1][0])
        if is_flush:                 return (5,) + tuple(vals)
        if is_straight:              return (4, s_high)
        if groups[0][1] == 3:        return (3, groups[0][0]) + tuple(v for v,_ in groups[1:])
        if groups[0][1]==2 and groups[1][1]==2:
            hi = max(groups[0][0],groups[1][0])
            lo = min(groups[0][0],groups[1][0])
            return (2, hi, lo, groups[2][0])
        if groups[0][1] == 2:        return (1, groups[0][0]) + tuple(v for v,_ in groups[1:])
        return (0,) + tuple(vals)

    def eval7(cards):
        best = None
        for combo in combinations(cards, 5):
            v = eval5(combo)
            if best is None or v > best: best = v
        return best

    # Build full deck excluding hole cards
    all_ranks = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"]
    all_suits = ["c","d","h","s"]
    full_deck = [(r,s) for r in all_ranks for s in all_suits]
    p0_set    = set(map(tuple, hole_cards_p0))
    deck      = [c for c in full_deck if tuple(c) not in p0_set]

    wins = 0.0
    for _ in range(n_simulations):
        random.shuffle(deck)
        opp_cards  = deck[:2]
        board      = deck[2:7]
        opp_set    = set(map(tuple, opp_cards))

        # Redraw if collision (shouldn't happen but safety check)
        if opp_set & p0_set:
            continue

        p0_hand  = [card_val(*c) for c in hole_cards_p0] + [card_val(*c) for c in board]
        opp_hand = [card_val(*c) for c in opp_cards]     + [card_val(*c) for c in board]

        p0_val  = eval7(p0_hand)
        opp_val = eval7(opp_hand)

        if   p0_val > opp_val: wins += 1.0
        elif p0_val == opp_val: wins += 0.5

    return wins / n_simulations