"""
state_encoder.py  
---------------------------------------------
Encodes an NLHGameState into a fixed-size feature vector for CFRNet.

Feature vector (68 dims):

EXISTING:
  [0]      acting seat index          (normalised by n_players)
  [1-6]    seat one-hot               (6 dims, seats 0-5)
  [7-14]   hand bucket one-hot        (8 buckets)
  [15]     pot size                   (normalised by wallet)
  [16]     current bet                (normalised by wallet)
  [17]     acting player's stack      (normalised by wallet)
  [18]     amount owed to call        (normalised by wallet)
  [19]     n_raises this street       (normalised by MAX_RAISES=4)
  [20]     n_players still active     (normalised by n_players)
  [21]     history length             (normalised by 10)
  [22-25]  street one-hot             (PRE/FLP/TRN/RVR -- 4 dims)
  [26-33]  board bucket one-hot       (8 buckets)
  [34]     n_community_cards          (normalised by 5)
  [35]     postflop hand strength     (0-1)
  [36]     flush draw                 (0 or 1)
  [37]     straight draw              (0 or 1)
  [38]     bias                       (always 1.0)

DECISION CONTEXT:
  [39]     pot odds                   (call / (pot + call))
  [40]     stack-to-pot ratio         (capped at 10x, normalised)
  [41]     avg opponent stack         (normalised by wallet)
  [42]     min opponent stack         (normalised by wallet)
  [43]     players yet to act         (normalised by n_players)
  [54]     position                   (0=early, 1=last)
  [55]     is_last_to_act             (0 or 1)
  [56]     pot_committed ratio        (bets[seat] / wallet)
  [57]     hands played confidence    (normalised, 0-1)
  [58]     recent loss factor         (tilt proxy)

ACTOR HUD STATS:
  [44]     actor VPIP
  [45]     actor PFR
  [46]     actor postflop aggression factor
  [47]     actor tilt factor
  [48]     actor fold-to-cbet
  [60]     actor 3-bet %
  [61]     actor C-bet %
  [62]     actor ATS (attempt to steal)
  [63]     actor WTSD%

OPPONENT HUD STATS (averages across active opponents):
  [49]     avg opp VPIP
  [50]     avg opp PFR
  [51]     avg opp postflop aggression
  [52]     avg opp tilt factor
  [53]     avg opp fold-to-cbet
  [64]     avg opp 3-bet %
  [65]     avg opp C-bet %
  [66]     avg opp fold-to-3bet

  [67]     bias                       (always 1.0)

N_ACTIONS = 5: [FOLD, CALL, RAISE_2, RAISE_4, ALLIN]
"""

from __future__ import annotations
import torch
import numpy as np
from collections import Counter, defaultdict

N_FEATURES = 68
N_ACTIONS  = 5
ALL_ACTIONS = ["FOLD", "CALL", "RAISE_2", "RAISE_4", "ALLIN"]
N_SEATS     = 6
N_BUCKETS   = 8
MAX_RAISES  = 4
N_STREETS   = 4


# ── Player profile tracker ────────────────────────────────────────────────────

class PlayerProfile:
    """
    Tracks per-player HUD stats across hands for opponent modeling.
    One instance per seat, persisted across hands during live play.
    During CFR training all stats return sensible priors.
    """

    def __init__(self):
        # Hand counts
        self.hands_played         = 0
        self.saw_flop_count       = 0

        # Preflop
        self.vpip_count           = 0
        self.pfr_count            = 0
        self.three_bet_count      = 0
        self.three_bet_opps       = 0
        self.fold_to_3bet_count   = 0
        self.faced_3bet_count     = 0
        self.ats_count            = 0   # raise when folded to in steal position
        self.ats_opps             = 0   # times in steal position (BTN/CO/SB)

        # Postflop
        self.postflop_bets        = 0
        self.postflop_checks      = 0
        self.cbet_count           = 0
        self.cbet_opps            = 0   # times preflop aggressor saw flop
        self.fold_to_cbet_count   = 0
        self.cbet_faced_count     = 0
        self.wtsd_count           = 0   # reached showdown

        # Tilt tracking
        self.starting_stack       = 0.0
        self.recent_stacks        = []  # last 5 end-of-hand stacks

    # ── Preflop stats ─────────────────────────────────────────────────────────

    @property
    def vpip(self) -> float:
        if self.hands_played < 5:
            return 0.5
        return min(self.vpip_count / self.hands_played, 1.0)

    @property
    def pfr(self) -> float:
        if self.hands_played < 5:
            return 0.3
        return min(self.pfr_count / self.hands_played, 1.0)

    @property
    def three_bet_pct(self) -> float:
        if self.three_bet_opps < 5:
            return 0.08
        return min(self.three_bet_count / self.three_bet_opps, 1.0)

    @property
    def fold_to_3bet(self) -> float:
        if self.faced_3bet_count < 3:
            return 0.55
        return min(self.fold_to_3bet_count / self.faced_3bet_count, 1.0)

    @property
    def ats(self) -> float:
        if self.ats_opps < 5:
            return 0.3
        return min(self.ats_count / self.ats_opps, 1.0)

    # ── Postflop stats ────────────────────────────────────────────────────────

    @property
    def postflop_agg(self) -> float:
        total = self.postflop_bets + self.postflop_checks
        if total < 5:
            return 0.5
        return min(self.postflop_bets / total, 1.0)

    @property
    def cbet_pct(self) -> float:
        if self.cbet_opps < 3:
            return 0.6
        return min(self.cbet_count / self.cbet_opps, 1.0)

    @property
    def fold_to_cbet_rate(self) -> float:
        if self.cbet_faced_count < 3:
            return 0.5
        return min(self.fold_to_cbet_count / self.cbet_faced_count, 1.0)

    @property
    def wtsd(self) -> float:
        if self.saw_flop_count < 5:
            return 0.35
        return min(self.wtsd_count / self.saw_flop_count, 1.0)

    # ── Tilt ─────────────────────────────────────────────────────────────────

    @property
    def tilt_factor(self) -> float:
        """0=calm, 1=tilting. Based on recent stack trajectory."""
        if len(self.recent_stacks) < 3:
            return 0.0
        losses = sum(
            max(0, self.recent_stacks[i] - self.recent_stacks[i + 1])
            for i in range(len(self.recent_stacks) - 1)
        )
        return min(losses / max(self.starting_stack * 2, 1.0), 1.0)

    # ── Recording methods (call from game loop) ───────────────────────────────

    def record_hand_start(self, stack: float):
        self.starting_stack = stack
        self.hands_played  += 1

    def record_hand_end(self, stack: float, reached_showdown: bool = False):
        self.recent_stacks.append(stack)
        if len(self.recent_stacks) > 5:
            self.recent_stacks.pop(0)
        if reached_showdown:
            self.wtsd_count += 1

    def record_vpip(self):
        self.vpip_count += 1

    def record_pfr(self):
        self.pfr_count += 1

    def record_3bet(self):
        self.three_bet_count += 1
        self.three_bet_opps  += 1

    def record_3bet_opp(self):
        """Call when player had opportunity to 3-bet but didn't."""
        self.three_bet_opps += 1

    def record_faced_3bet(self, folded: bool):
        self.faced_3bet_count += 1
        if folded:
            self.fold_to_3bet_count += 1

    def record_ats_opp(self, attempted: bool):
        self.ats_opps += 1
        if attempted:
            self.ats_count += 1

    def record_saw_flop(self):
        self.saw_flop_count += 1

    def record_postflop_action(self, aggressive: bool):
        if aggressive:
            self.postflop_bets   += 1
        else:
            self.postflop_checks += 1

    def record_cbet_opp(self, fired: bool):
        """Call when player was preflop aggressor and saw flop."""
        self.cbet_opps += 1
        if fired:
            self.cbet_count += 1

    def record_cbet_faced(self, folded: bool):
        self.cbet_faced_count += 1
        if folded:
            self.fold_to_cbet_count += 1


# ── Global profile registry ───────────────────────────────────────────────────
# Keyed by seat index. Persists across hands during live play.
# During CFR training all methods return prior values (no history).

_PLAYER_PROFILES: dict[int, PlayerProfile] = defaultdict(PlayerProfile)


def get_profile(seat: int) -> PlayerProfile:
    return _PLAYER_PROFILES[seat]

def reset_profiles():
    """Call between sessions or when player lineup changes."""
    _PLAYER_PROFILES.clear()


# ── Postflop hand strength ────────────────────────────────────────────────────

def _postflop_hand_strength(hole_cards, community_cards) -> float:
    """Returns normalised hand rank 0.0 (high card) to 1.0 (straight flush)."""
    if not community_cards or not hole_cards or len(hole_cards) < 2:
        return 0.0
    try:
        from core.hand_evaluator import evaluate_hand
        all_cards = list(hole_cards) + list(community_cards)
        rank = evaluate_hand(all_cards)   # returns 0-8
        return rank / 8.0
    except Exception:
        return 0.0


def _draw_features(hole_cards, community_cards):
    """Returns (flush_draw, straight_draw) as 0.0 or 1.0."""
    if len(community_cards) < 3 or len(hole_cards) < 2:
        return 0.0, 0.0
    try:
        all_cards   = list(hole_cards) + list(community_cards)
        suits       = [c.suit for c in all_cards]
        suit_counts = Counter(suits)
        flush_draw  = 1.0 if max(suit_counts.values()) == 4 else 0.0

        _RANK_VAL = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
        }
        ranks = sorted(set(
            _RANK_VAL.get(getattr(c, 'rank', getattr(c, 'value', '')), 0)
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

def encode_state(state) -> torch.Tensor:
    """
    Encode an NLHGameState (or EngineStateProxy) into a 68-dim float32 tensor.
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

    # ── [7-14] Hand bucket one-hot ────────────────────────────────────────────
    bucket = state.hands[seat]
    if 0 <= bucket < N_BUCKETS:
        fv[7 + bucket] = 1.0

    # ── [15-18] Chip counts ───────────────────────────────────────────────────
    to_call = max(state.current_bet - state.bets[seat], 0.0)
    fv[15]  = state.pot / wallet
    fv[16]  = state.current_bet / wallet
    fv[17]  = state.stacks[seat] / wallet
    fv[18]  = to_call / wallet

    # ── [19] Raises this street ───────────────────────────────────────────────
    fv[19] = state.n_raises / MAX_RAISES

    # ── [20] Active players ───────────────────────────────────────────────────
    n_active = sum(
        1 for i in range(n)
        if i not in state.folded and state.stacks[i] > 0
    )
    fv[20] = n_active / max(n, 1)

    # ── [21] History length ───────────────────────────────────────────────────
    fv[21] = len(state.action_history) / 10.0

    # ── [22-25] Street one-hot ────────────────────────────────────────────────
    street = getattr(state, 'street', 0)
    if 0 <= street < N_STREETS:
        fv[22 + street] = 1.0

    # ── [26-33] Board texture bucket one-hot ──────────────────────────────────
    community = getattr(state, 'community_cards', [])
    if community:
        bb = 7
        try:
            from cfr_bots.cfr.nlh_gamestate import _board_bucket
            bb = _board_bucket(community)
        except ImportError:
            pass
        if 0 <= bb < N_BUCKETS:
            fv[26 + bb] = 1.0

    # ── [34] Community card count ─────────────────────────────────────────────
    fv[34] = len(community) / 5.0

    # ── [35-37] Postflop hand strength + draws ────────────────────────────────
    hole_cards = getattr(state, 'hole_cards', None)
    if hole_cards and community:
        fv[35] = _postflop_hand_strength(hole_cards, community)
        flush_draw, straight_draw = _draw_features(hole_cards, community)
        fv[36] = flush_draw
        fv[37] = straight_draw

    # ── [38] Bias ─────────────────────────────────────────────────────────────
    fv[38] = 1.0

    # ── [39] Pot odds: call / (pot + call) ───────────────────────────────────
    pot_plus_call = state.pot + to_call
    fv[39] = (to_call / pot_plus_call) if pot_plus_call > 0 else 0.0

    # ── [40] Stack-to-pot ratio (capped at 10x) ───────────────────────────────
    fv[40] = min(state.stacks[seat] / max(state.pot, 1.0), 10.0) / 10.0

    # ── [41-42] Opponent stacks ───────────────────────────────────────────────
    opp_stacks = [
        state.stacks[i] for i in range(n)
        if i != seat and i not in state.folded
    ]
    if opp_stacks:
        fv[41] = sum(opp_stacks) / len(opp_stacks) / wallet
        fv[42] = min(opp_stacks) / wallet
    
    # ── [43] Players yet to act ───────────────────────────────────────────────
    acted      = len(state.action_history)
    yet_to_act = max(n_active - acted - 1, 0)
    fv[43]     = yet_to_act / max(n, 1)

    # ── [44-48] Actor HUD stats ───────────────────────────────────────────────
    actor = get_profile(seat)
    fv[44] = actor.vpip
    fv[45] = actor.pfr
    fv[46] = actor.postflop_agg
    fv[47] = actor.tilt_factor
    fv[48] = actor.fold_to_cbet_rate

    # ── [49-53] Average opponent HUD stats ───────────────────────────────────
    opp_seats    = [i for i in range(n) if i != seat and i not in state.folded]
    opp_profiles = [get_profile(i) for i in opp_seats]

    if opp_profiles:
        fv[49] = sum(p.vpip            for p in opp_profiles) / len(opp_profiles)
        fv[50] = sum(p.pfr             for p in opp_profiles) / len(opp_profiles)
        fv[51] = sum(p.postflop_agg    for p in opp_profiles) / len(opp_profiles)
        fv[52] = sum(p.tilt_factor     for p in opp_profiles) / len(opp_profiles)
        fv[53] = sum(p.fold_to_cbet_rate for p in opp_profiles) / len(opp_profiles)
    else:
        fv[49] = 0.5
        fv[50] = 0.3
        fv[51] = 0.5
        fv[52] = 0.0
        fv[53] = 0.5

    # ── [54] Position (0=first, 1=last) ──────────────────────────────────────
    fv[54] = seat / max(n - 1, 1)

    # ── [55] Is last to act this street ──────────────────────────────────────
    active_seats = [
        i for i in range(n)
        if i not in state.folded and state.stacks[i] > 0
    ]
    fv[55] = 1.0 if active_seats and seat == active_seats[-1] else 0.0

    # ── [56] Pot committed ratio ──────────────────────────────────────────────
    fv[56] = state.bets[seat] / wallet

    # ── [57] Hands played confidence (0-1) ───────────────────────────────────
    fv[57] = min(actor.hands_played / 100.0, 1.0)

    # ── [58] Recent loss / tilt factor ───────────────────────────────────────
    fv[58] = actor.tilt_factor

    # ── [59] Bias ─────────────────────────────────────────────────────────────
    fv[59] = 1.0

    # ── [60-63] Actor extended HUD ────────────────────────────────────────────
    fv[60] = actor.three_bet_pct
    fv[61] = actor.cbet_pct
    fv[62] = actor.ats
    fv[63] = actor.wtsd

    # ── [64-66] Opponent extended HUD ────────────────────────────────────────
    if opp_profiles:
        fv[64] = sum(p.three_bet_pct    for p in opp_profiles) / len(opp_profiles)
        fv[65] = sum(p.cbet_pct         for p in opp_profiles) / len(opp_profiles)
        fv[66] = sum(p.fold_to_3bet     for p in opp_profiles) / len(opp_profiles)
    else:
        fv[64] = 0.08
        fv[65] = 0.60
        fv[66] = 0.55

    # ── [67] Bias ─────────────────────────────────────────────────────────────
    fv[67] = 1.0

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