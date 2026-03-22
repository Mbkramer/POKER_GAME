"""
self_play_train_nlh.py 
------------------------------------
Self-play training loop for NLH multi-street CFR.

Usage:
    python self_play_train_nlh_fixed.py \\
        --wallet wallet_amount --buyin buy_in_amount --players num_players
        --deals num_deals --iters num_outer_iterations --cfr num_cfr_tree_traversals ( >= num_deals * 3) --epochs num_net_epochs_per_cycle
"""

from __future__ import annotations

import random
import numpy as np
import sys
import os
import math
import re
import argparse
import warnings
import pickle
import hashlib
import json
import psutil
from collections import Counter

# Suppress a known PyTorch false-positive: SequentialLR / epoch-level schedulers
# trigger "scheduler.step() before optimizer.step()" on the very first call
# because optimizer._step_count is 0 at that point. Our ordering is correct:
# train_net_on_samples() calls optimizer.step() many times per outer iteration,
# then scheduler.step() advances the lr once -- exactly as intended.
warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`",
    category=UserWarning,
)
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
CFR_BOTS_DIR = SCRIPT_DIR.parent
BOTS_DIR     = CFR_BOTS_DIR.parent
sys.path.insert(0, str(BOTS_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from cfr_bots.cfr.cfrm import CounterfactualRegretMinimizationBase
from cfr_bots.cfr.nlh_gamestate import NLHChanceNode, _card_id
from cfr_bots.cfr.export_dataset import CFRDatasetCollector
from cfr_bots.cfr.preflop_abstraction import PreflopAbstraction

from cfr_net import CFRNet
#from state_encoder import encode_state, policy_tensor, N_FEATURES, N_ACTIONS, ALL_ACTIONS
from combined_state_encoder import (
    encode_state, policy_tensor, N_FEATURES, N_ACTIONS, ALL_ACTIONS,
    STREET_SLICE, IDX_HAND_STRENGTH
)

# ── VanillaCFR (FIXED) ────────────────────────────────────────────────────────

class VanillaCFR(CounterfactualRegretMinimizationBase):
    """
    External-sampling MCCFR.

    Chance and opponent actions are sampled.
    Traverser actions are fully expanded and regret-updated.
    """

    def __init__(self, root, sample_collector=None):
        super().__init__(root=root, chance_sampling=True,
                         sample_collector=sample_collector)

    def _player_index(self, state):
        raw = state.to_move
        n   = self._n_players
        if raw == -1:           return 0
        if raw == 1 and n == 2: return 1
        if raw >= 0:            return raw
        return 0

    # External Sampling Helper: ensure sigma entries exist for all actions at this infoset, then normalize.
    def _normalize_sigma_for_actions(self, inf_set, actions):
        self._ensure_info_set(inf_set, actions)

        for a in actions:
            if a not in self.sigma[inf_set]:
                self.sigma[inf_set][a]              = 1.0 / len(actions)
                self.cumulative_regrets[inf_set][a] = 0.0
                self.cumulative_sigma[inf_set][a]   = 0.0
                self.nash_equilibrium[inf_set][a]   = 1.0 / len(actions)

        total_sigma = sum(self.sigma[inf_set][a] for a in actions)
        if total_sigma <= 0:
            u = 1.0 / len(actions)
            for a in actions:
                self.sigma[inf_set][a] = u
        else:
            for a in actions:
                self.sigma[inf_set][a] /= total_sigma

    # External Sampling Helper: sample an action according to sigma at this infoset, restricted to given actions.
    def _sample_action_from_sigma(self, inf_set, actions):
        probs = [self.sigma[inf_set][a] for a in actions]
        r = random.random()
        cum = 0.0
        for a, p in zip(actions, probs):
            cum += p
            if r <= cum:
                return a
        return actions[-1]
    
    # External-sampling MCCFR recursive traversal.
    def _cfr_external_sampling(self, state, traverser, reaches, _depth=0):
        if _depth > 200:
            raise RecursionError("CFR external-sampling depth exceeded 200")

        if state.is_terminal():
            return state.evaluation()[traverser]

        if state.is_chance():
            child = state.sample_one()
            return self._cfr_external_sampling(child, traverser, reaches, _depth + 1)

        inf_set = state.inf_set()
        actions = state.actions
        self._normalize_sigma_for_actions(inf_set, actions)

        player = self._player_index(state)

        opp_reach = 1.0
        for j, rr in enumerate(reaches):
            if j != traverser:
                opp_reach *= rr
        opp_reach = max(opp_reach, 1e-12)

        if player == traverser:
            action_utils = {}
            node_util = 0.0

            for a in actions:
                child = state.play(a)
                child_reaches = list(reaches)

                u = self._cfr_external_sampling(
                    child,
                    traverser,
                    child_reaches,
                    _depth + 1,
                )

                action_utils[a] = u
                node_util += self.sigma[inf_set][a] * u

            for a in actions:
                regret = action_utils[a] - node_util
                self.cumulative_regrets[inf_set][a] = max(
                    0.0,
                    self.cumulative_regrets[inf_set][a] + opp_reach * regret
                )

            for a in actions:
                self.cumulative_sigma[inf_set][a] += opp_reach * self.sigma[inf_set][a]

            if self.sample_collector is not None:
                self.sample_collector(state, self.sigma[inf_set], node_util)

            return node_util

        sampled_action = self._sample_action_from_sigma(inf_set, actions)
        child_reaches = list(reaches)
        child_reaches[player] *= self.sigma[inf_set][sampled_action]

        return self._cfr_external_sampling(
            state.play(sampled_action),
            traverser,
            child_reaches,
            _depth + 1,
        )

    def run(self, iterations=1, progress_interval=0):
        n = self.root.n_players
        sigma_update_interval = 10  

        for i in range(iterations):
            for traverser in range(n):
                self._cfr_external_sampling(
                    self.root,
                    traverser=traverser,
                    reaches=[1.0] * n,
                    _depth=0,
                )

            if (i + 1) % sigma_update_interval == 0:
                for inf_set in self.cumulative_regrets:
                    self._update_sigma(inf_set)

            if progress_interval and (i + 1) % progress_interval == 0:
                pct = (i + 1) / iterations
                bar = int(pct * 20)
                print(
                    f"\r    CFR [{'X' * bar}{'.' * (20 - bar)}] {i+1}/{iterations}",
                    end="",
                    flush=True,
                )

        for inf_set in self.cumulative_regrets:
            self._update_sigma(inf_set)

        if progress_interval:
            print("\r" + " " * 55 + "\r", end="", flush=True)

    def compute_nash_equilibrium(self):
        for inf_set in self.cumulative_sigma:
            total = sum(self.cumulative_sigma[inf_set].values())
            for a in self.nash_equilibrium[inf_set]:
                self.nash_equilibrium[inf_set][a] = (
                    self.cumulative_sigma[inf_set][a] / total
                    if total > 0
                    else 1.0 / len(self.nash_equilibrium[inf_set])
                )

    def value_of_the_game(self, n_samples: int = 50):
        
        #Monte-Carlo estimate of P0 game value using average strategy.
        if not self.nash_equilibrium:
            return 0.0

        def _eval_one_path(state):
            if state.is_terminal():
                result = state.evaluation()
                return result if isinstance(result, list) else [result, -result]
            if state.is_chance():
                return _eval_one_path(state.sample_one())
            avg = self.nash_equilibrium.get(state.inf_set(), {})
            if not avg:
                return [0.0] * self.root.n_players
            actions = state.actions
            probs   = [avg.get(a, 0.0) for a in actions]
            total   = sum(probs)
            if total <= 0:
                return [0.0] * self.root.n_players
            r   = random.random()
            cum = 0.0
            for a, p in zip(actions, probs):
                cum += p / total
                if r <= cum:
                    return _eval_one_path(state.play(a))
            return _eval_one_path(state.play(actions[-1]))

        return sum(_eval_one_path(self.root)[0] for _ in range(n_samples)) / n_samples


# ── CFR dataset collector ─────────────────────────────────────────────────────

class NLHDatasetCollector(CFRDatasetCollector):
    """
    Collects CFR samples with two key improvements over the original:

    """

    def __init__(self, encode_state_fn, cfr_ref=None):
        super().__init__(encode_state_fn)
        self.cfr_ref = cfr_ref

        from collections import defaultdict
        self._infoset_data = defaultdict(lambda: {
            'features': None,
            'legal_actions': set(),
            'sigmas': [],
            'values': [],
        })
        
    def __call__(self, state, sigma, value):
        inf_set  = state.inf_set()
        features = self.encode_state(state)

        if (self.cfr_ref is not None
                and inf_set in self.cfr_ref.nash_equilibrium
                and self.cfr_ref.nash_equilibrium[inf_set]):
            target_sigma = dict(self.cfr_ref.nash_equilibrium[inf_set])
        else:
            target_sigma = dict(sigma)

        data = self._infoset_data[inf_set]
        if data['features'] is None:
            data['features'] = features

        data['legal_actions'].update(state.actions)
        data['sigmas'].append(target_sigma)
        data['values'].append(float(value))

    def get_dataset(self):
        """
        Emit one averaged sample per unique info set visited this iteration.
        Policy target:  mean of all sigma observations for this info set.
        Value target:   mean of all sampled payoffs for this info set.
        """
        samples = []
        for inf_set, data in self._infoset_data.items():
            if data['features'] is None or not data['values']:
                continue

            legal_actions = sorted(data['legal_actions'], key=ALL_ACTIONS.index)
            if not legal_actions:
                continue

            avg_sigma = {
                a: sum(s.get(a, 0.0) for s in data['sigmas']) / len(data['sigmas'])
                for a in legal_actions
            }

            pi, legal_mask = policy_tensor(avg_sigma, legal_actions)

            v = torch.tensor(
                sum(data['values']) / len(data['values']),
                dtype=torch.float32,
            )

            samples.append((data['features'], pi, v, legal_mask))

        return samples

    def reset(self):
        """Clear accumulated state. Call between iterations."""
        self._infoset_data.clear()
        self.samples.clear()

# ── Neural net training  ───────────────────────────────────────────────

def train_net_on_samples(net, optimizer, samples, device, epochs=10,
                         batch_size=512, buyin=10.0,
                         policy_weight=0.5, value_weight=1.0):
    """
    Train network on CFR samples.

    """
    if not samples:
        return 0.0, 0.0, 0.0

    from collections import defaultdict
    groups: dict = defaultdict(lambda: [[], [], [], []])  # [feats, pis, vs, legal_masks]
    for feat, pi, v, legal_mask in samples:
        
        feat_key = feat.detach().cpu().numpy().tobytes()
        mask_key = legal_mask.detach().cpu().numpy().tobytes()
        key = (feat_key, mask_key)

        groups[key][0].append(feat)
        groups[key][1].append(pi)
        groups[key][2].append(v.item())
        groups[key][3].append(legal_mask)

    agg_samples = []
    for feats, pis, vs, legal_masks in groups.values():
        if not feats:
            continue

        # legality should be identical for the same infoset; use the first
        legal_mask = legal_masks[0]

        agg_samples.append((
            feats[0],
            torch.stack(pis).mean(dim=0),
            torch.tensor(sum(vs) / len(vs), dtype=torch.float32),
            legal_mask
        ))

    # Use aggregated samples for training; fall back to raw if too few
    train_samples = agg_samples if len(agg_samples) >= 32 else samples
    # ─────────────────────────────────────────────────────────────────────

    features    = torch.stack([s[0] for s in train_samples]).to(device)
    policies    = torch.stack([s[1] for s in train_samples]).to(device)
    values      = torch.stack([s[2] for s in train_samples]).to(device)
    legal_masks = torch.stack([s[3] for s in train_samples]).to(device)

    # every sample must have at least one legal action
    if not legal_masks.any(dim=-1).all():
        bad_rows = (~legal_masks.any(dim=-1)).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"Encountered samples with no legal actions in legal_mask: rows={bad_rows[:10]}")


    # ── Fixed-scale value normalisation ────────────────────────────
    norm_scale  = buyin * 10.0
    values_norm = (values / norm_scale).clamp(-1.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────

    actual_batch = min(batch_size, len(train_samples))
    dataset = TensorDataset(features, policies, values_norm, legal_masks)

    street_ids = []
    for s in train_samples:

        feat = s[0]
        street = feat[STREET_SLICE]
        sid =  torch.argmax(street).item()
        street_ids.append(sid)

    street_counts = Counter(street_ids)

    target_mix = {
        0: 0.30,  # PRE  ← change from 0.15
        1: 0.30,  # FLP  ← change from 0.40
        2: 0.20,  # TRN  ← change from 0.27
        3: 0.20,  # RVR  ← change from 0.18
    }

    sample_weights = torch.tensor(
        [target_mix[sid] / max(street_counts[sid], 1) for sid in street_ids],
        dtype=torch.double
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_samples),
        replacement=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=max(actual_batch, 1),
        sampler=sampler,
        drop_last=False,
    )

    net.train()
    total_pol, total_val, n_batches = 0.0, 0.0, 0

    for _ in range(epochs):
        for X, pi, v, legal_mask in loader:
            optimizer.zero_grad()
            policy_logits, value_pred = net(X)

            # Mask truly illegal actions using the dataset-provided legal action mask
            # concentrates entirely within the legal action subspace.
            # This removes the irreducible loss floor from illegal-action mass.
            masked_logits = policy_logits.masked_fill(~legal_mask, float('-inf'))
            log_probs = F.log_softmax(masked_logits, dim=-1)

            # Target policy should already be zero outside legal actions,
            # but clamp defensively and renormalize over legal space.
            pi_legal = pi * legal_mask.float()
            pi_legal = pi_legal / pi_legal.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            raw_terms = pi_legal * log_probs
            policy_loss = -raw_terms.nan_to_num(0.0).sum(dim=-1).mean()
            # ─────────────────────────────────────────────────────────────

            # MSE squares large errors, so a single outlier payoff (e.g. rare
            # deep all-in) dominates the gradient and destabilizes value learning.
            # Huber = MSE for small errors, L1 for large ones: robust to outliers.
            value_loss    = F.smooth_l1_loss(value_pred, v, beta=0.5)
            loss = policy_weight * policy_loss + value_weight * value_loss

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_pol += policy_loss.item()
            total_val += value_loss.item()
            n_batches += 1

    n = max(n_batches, 1)
    return (total_pol + total_val) / n, total_pol / n, total_val / n


def _safe_entropy(probs_dict):
    probs = [float(v) for v in probs_dict.values()]
    s = sum(probs)
    if s <= 0:
        return 0.0
    probs = [p / s for p in probs if p > 0]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def _print_active_infoset_summary(cfr, collector, top_k=10, street_filter=".PRE."):
    """
    Print the most-visited infosets from THIS iteration, not hard-coded nodes
    and not arbitrary first dict entries.
    """
    rows = []

    for inf_set, data in collector._infoset_data.items():
        if street_filter not in inf_set:
            continue

        n_visits = len(data.get("values", []))
        if n_visits <= 0:
            continue

        sigma_now   = dict(cfr.sigma.get(inf_set, {}))
        regrets_now = dict(cfr.cumulative_regrets.get(inf_set, {}))
        avg_now     = dict(cfr.nash_equilibrium.get(inf_set, {}))
        cumsig_now  = dict(cfr.cumulative_sigma.get(inf_set, {}))
        dbg_now     = dict(getattr(cfr, "_sigma_debug", {}).get(inf_set, {}))

        avg_v = sum(data["values"]) / max(n_visits, 1)

        legal_actions = (
            sorted(data.get("legal_actions", []), key=ALL_ACTIONS.index)
            if data.get("legal_actions") else []
        )

        rows.append({
            "inf_set": inf_set,
            "visits": n_visits,
            "avg_value": avg_v,
            "sigma_now": sigma_now,
            "avg_strategy": avg_now,
            "regrets": regrets_now,
            "cumulative_sigma": cumsig_now,
            "legal_actions": legal_actions,
            "entropy_now": _safe_entropy(sigma_now) if sigma_now else 0.0,
            "entropy_avg": _safe_entropy(avg_now) if avg_now else 0.0,
            "rm_raw_regrets": dbg_now.get("raw_regrets", {}),
            "rm_pos_regrets": dbg_now.get("positive_regrets", {}),
            "rm_pos_sum": dbg_now.get("positive_sum", 0.0),
            "rm_uniform_fb": dbg_now.get("used_uniform_fallback", False),
        })

    rows.sort(key=lambda r: (r["visits"], r["entropy_avg"]), reverse=True)

    label = street_filter.strip(".")
    print(f"  [DIAG] Top {min(top_k, len(rows))} active infosets [{label}]:")

    for row in rows[:top_k]:
        print(f"    [NODE] {row['inf_set']}")
        print(f"      visits_this_iter : {row['visits']}")
        print(f"      avg_value_target : {row['avg_value']:+.4f}")
        print(f"      legal_actions    : {row['legal_actions']}")
        print(f"      sigma_now        : {row['sigma_now']}")
        print(f"      avg_strategy     : {row['avg_strategy']}")
        print(f"      regrets          : {row['regrets']}")
        print(f"      cumulative_sigma : {row['cumulative_sigma']}")
        print(f"      entropy_now      : {row['entropy_now']:.4f}")
        print(f"      entropy_avg      : {row['entropy_avg']:.4f}")
        print(f"      rm_raw_regrets   : {row['rm_raw_regrets']}")
        print(f"      rm_pos_regrets   : {row['rm_pos_regrets']}")
        print(f"      rm_pos_sum       : {row['rm_pos_sum']:.6f}")
        print(f"      rm_uniform_fb    : {row['rm_uniform_fb']}")


def _print_preflop_bucket_summary(collector, top_k=12):
    """
    Shows which PRE buckets were actually active this iteration.
    This directly answers whether bucket IDs like 12/13/5 are dominating.
    """
    bucket_visits = Counter()
    bucket_infosets = Counter()

    for inf_set, data in collector._infoset_data.items():
        if ".PRE." not in inf_set:
            continue

        n_visits = len(data.get("values", []))
        if n_visits <= 0:
            continue

        parts = inf_set.split(".")
        if len(parts) < 2:
            continue

        bucket = parts[1]
        bucket_visits[bucket] += n_visits
        bucket_infosets[bucket] += 1

    print(f"  [DIAG] Top PRE buckets by activity:")
    for bucket, visits in bucket_visits.most_common(top_k):
        print(f"    bucket={bucket:>4} | visits={visits:>5} | active_infosets={bucket_infosets[bucket]}")


def _print_preflop_coverage_summary(cfr, collector):
    total_pre = sum(1 for k in cfr.cumulative_regrets if ".PRE." in k)
    active_pre = 0
    total_pre_visits = 0
    one_visit = 0
    multi_visit = 0

    for inf_set, data in collector._infoset_data.items():
        if ".PRE." not in inf_set:
            continue
        n = len(data.get("values", []))
        if n > 0:
            active_pre += 1
            total_pre_visits += n
            if n == 1:
                one_visit += 1
            else:
                multi_visit += 1

    avg_visits = total_pre_visits / max(active_pre, 1)

    print("  [DIAG] Preflop coverage:")
    print(f"    total_pre_infosets   : {total_pre}")
    print(f"    active_this_iter     : {active_pre}")
    print(f"    one_visit_infosets   : {one_visit}")
    print(f"    multi_visit_infosets : {multi_visit}")
    print(f"    avg_visits_active    : {avg_visits:.2f}")

# ── Main self-play loop ───────────────────────────────────────────────────────

def self_play_train(
    wallet:         float = 200.0,   
    buyin:          float = 10.0,
    n_players:      int   = 2,
    n_deals:        int   = 1_000,
    n_iterations:   int   = 50,
    cfr_iterations: int   = 500,
    net_epochs:     int   = 10,
    checkpoint_dir: str   = str(CFR_BOTS_DIR / "checkpoints"),
    resume_path:    str   = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # New File Name Snippet
    ckpt_stem = f"{n_players}P_{int(buyin)}B_{int(wallet)}W"
    last_path  = out_dir / f"last_{ckpt_stem}.pt"
    best_path  = out_dir / f"best_{ckpt_stem}.pt"
    best_txt   = out_dir / f"best_{ckpt_stem}.txt"

    print(f"\n{'='*60}")
    print(f"  NLH CFR + Neural Net Self-Play Trainer  [FIXED v5]")
    print(f"  Device:            {device}")
    print(f"  Players:           {n_players}")
    print(f"  Wallet / Buyin:    {wallet} / {buyin}  (SPR~{wallet/15:.1f})")
    print(f"  Deal pool:         {n_deals:,}")
    print(f"  Outer iterations:  {n_iterations}")
    print(f"  CFR iters/cycle:   {cfr_iterations}")
    print(f"  Net epochs/cycle:  {net_epochs}")
    print(f"{'='*60}\n")

    # ── Build deal pool ───────────────────────────────────────────────────────
    print("Building deal pool...", end=" ", flush=True)
    random.seed(42)          # ← add this
    np.random.seed(42)       # ← add this
    abst  = PreflopAbstraction(n_players=n_players)
    deals = abst.all_deals(max_deals=n_deals)
    print(f"{len(deals):,} unique deals ready.\n")

    # ── Deals hash for CFR state file naming ─────────────────────────────────
    def _deals_hash(deals):
        return hashlib.md5(
            json.dumps([str(d) for d in deals], sort_keys=True).encode()
        ).hexdigest()[:8]

    cfr_state_path = out_dir / f"cfr_state_{n_players}P_{int(buyin)}B_{int(wallet)}W_{_deals_hash(deals)}.pkl"

    # ── Network + persistent optimizer ────────────────────────────────────────

    net = CFRNet(
        n_features=N_FEATURES,
        n_actions=N_ACTIONS,
        hidden_dim=128,
        n_blocks=3,
    ).to(device)

    lr        = 2e-4      # FIX D: lower lr prevents overshoot at start
    best_loss = float("inf")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Warmup + CosineAnnealing schedule.
    warmup_steps = max(3, n_iterations // 20)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, n_iterations - warmup_steps),
        eta_min=lr * 0.05,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    print(f"CFRNet parameters: {sum(p.numel() for p in net.parameters()):,}\n")

    # ── Resume from checkpoint ──────────────────────────────────────────────────
    if resume_path:
        path = Path(resume_path)
        if path.exists():
            ckpt      = torch.load(path, map_location=device, weights_only=True)
            print(ckpt.keys())
            net.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print(f"Optimizer state loaded, LR reset to {lr}")
                print(f"Optimizer state loaded. Keys: {list(ckpt['optimizer_state'].keys())}")
            else:
                print("WARNING: no optimizer_state in checkpoint!")

            # Read true best val_loss from txt, fall back to payload
            txt_path = path.with_suffix('.txt')
            best_loss = float(txt_path.read_text()) if txt_path.exists() else ckpt.get("net_loss", float("inf"))

            print(f"Resumed from {path} (saved loss {best_loss:.4f})\n")

    # ── Pre-compute equity and board for every deal ───────────────────────────
    from cfr_bots.cfr.preflop_abstraction import preflop_equity_vs_random, _make_deck

    canonical_deck = _make_deck()
    equity_sims    = 50
    n_valid        = sum(1 for d in deals if isinstance(d, dict) and 'hole_cards' in d)
    processed      = 0
    rng = random.Random(42)

    for deal in deals:
        if not isinstance(deal, dict) or 'hole_cards' not in deal:
            continue
        processed += 1
        if processed % 100 == 0 or processed == 1 or processed == n_valid:
            print(f"Pre-computing equities/boards... {processed}/{n_valid}",
                  end="\r", flush=True)

        deal['full_deck'] = canonical_deck
        deal['equity_p0'] = preflop_equity_vs_random(
            list(deal['hole_cards'][0]), n_simulations=equity_sims
        )

        hole_ids  = {_card_id(c) for hc in deal['hole_cards'] for c in hc}
        remaining = [c for c in canonical_deck if _card_id(c) not in hole_ids]

        rng.shuffle(remaining)
        deal['board'] = remaining[:5]

    print(f"Pre-computing equities/boards... {n_valid}/{n_valid} done.    \n")

    # ── CFR root ──────────────────────────────────────────────────────────────
    root = NLHChanceNode(
        hand_deals=deals,
        wallet=wallet,
        buyin=buyin,
        n_players=n_players,
    )
    cfr = VanillaCFR(root=root)

    # ── Load persisted CFR state if available ─────────────────────────────────
    if cfr_state_path.exists():
        print(f"Loading CFR state from {cfr_state_path}...", end=" ", flush=True)
        try:
            with open(cfr_state_path, 'rb') as f:
                cfr_state = pickle.load(f)
            cfr.cumulative_regrets = cfr_state['cumulative_regrets']
            cfr.cumulative_sigma   = cfr_state['cumulative_sigma']
            cfr.nash_equilibrium   = cfr_state['nash_equilibrium']
            cfr.sigma              = cfr_state['sigma']
            print(f"done. ({len(cfr.cumulative_regrets):,} info sets loaded)")
        except Exception as e:
            print(f"failed ({e}), starting fresh.")
    else:
        print(f"No CFR state found at {cfr_state_path}, starting fresh.")

    # Sliding-window replay buffer instead of full accumulation.
    iter_sample_history = []     # list-of-lists, one entry per outer iteration
    replay_window = 20_000       # New strat moved from 90000 to 20000 samples, not thats like 15+
    val_loss = float("inf")      # initialize before loop

    print(f"{'Iter':>5} | {'New':>6}  | {'States':>7} | {'Pol Loss':>9} | {'Val Loss':>9} | {'Avg Strat Entropy':>9} | {'Game Val':>9} | Ep/LR             | RAM USED ")
    print("-" * 115)

    for iteration in range(1, n_iterations + 1):

        # Step 1: CFR self-play
        collector = NLHDatasetCollector(
            encode_state_fn=encode_state,
            cfr_ref=cfr,
        )

        cfr.sample_collector = collector
        cfr.run(iterations=cfr_iterations, progress_interval=max(1, min(50, cfr_iterations // 10)))
        cfr.sample_collector = None

        cfr.compute_nash_equilibrium()
        game_value = cfr.value_of_the_game(n_samples=250)

        entropies = []
        for inf_set, actions in cfr.nash_equilibrium.items():
            probs = list(actions.values())
            total = sum(probs)
            if total > 0:
                probs = [p/total for p in probs]
                h = -sum(p * math.log(p+1e-10) for p in probs)
                entropies.append(h)
        avg_entropy = sum(entropies) / max(len(entropies), 1)

        # ── Save CFR state after every iteration ──────────────────────────────
        try:
            with open(cfr_state_path, 'wb') as f:
                pickle.dump({
                    'cumulative_regrets': cfr.cumulative_regrets,
                    'cumulative_sigma':   cfr.cumulative_sigma,
                    'nash_equilibrium':   cfr.nash_equilibrium,
                    'sigma':              cfr.sigma,
                }, f)
        except Exception as e:
            print(f"  WARNING: CFR state save failed: {e}")

        new_samples = collector.get_dataset()

        _print_active_infoset_summary(cfr, collector, top_k=3, street_filter=".PRE.")
        _print_active_infoset_summary(cfr, collector, top_k=3, street_filter=".FLP.")

        collector.reset()

        # ── DIAGNOSTIC ───────────────────────────────────────────────────────────
        action_counts = {a: 0 for a in ALL_ACTIONS}
        postflop_count = 0
        postflop_strength_sum = 0.0

        for feat, pi, v, legal_mask in new_samples:
            for i, a in enumerate(ALL_ACTIONS):
                if pi[i] > 0:
                    action_counts[a] += 1

            street = feat[STREET_SLICE].tolist()
            s = street.index(max(street)) if max(street) > 0 else 0

            if s > 0:
                postflop_count += 1
                postflop_strength_sum += feat[IDX_HAND_STRENGTH].item()

        total = max(len(new_samples), 1)
        print(f"\n  [DIAG] {total} raw samples this iteration:")
        for a, c in action_counts.items():
            print(f"    {a} support: {c/total:.2%}")

        print(f"  [DIAG] Postflop samples: {postflop_count}/{total} = {postflop_count/total:.2%}")

        if postflop_count > 0:
            avg_strength = postflop_strength_sum / postflop_count
            print(f"  [DIAG] Avg postflop hand strength fv[{IDX_HAND_STRENGTH}]: {avg_strength:.4f}")
            if avg_strength < 0.05:
                print("  [DIAG] *** WARNING: hand strength near zero — phevaluator fallback likely ***")

        street_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for feat, pi, v, _ in new_samples:
            street = feat[STREET_SLICE].tolist()
            s = street.index(max(street)) if max(street) > 0 else 0
            street_counts[s] += 1

        print(f"  [DIAG] Streets: PRE={street_counts[0]/total:.1%} "
            f"FLP={street_counts[1]/total:.1%} "
            f"TRN={street_counts[2]/total:.1%} "
            f"RVR={street_counts[3]/total:.1%}")

        # ─────────────────────────────────────────────────────────────────────────

        iter_sample_history.append(new_samples)

        # Rebuild buffer from most-recent iterations until replay_window is reached
        max_history_iters = (replay_window // max(len(new_samples), 1)) + 2
        iter_sample_history = iter_sample_history[-max_history_iters:]

        # ------------------------------------------------------------------
        # Street-balanced replay rebuild with recency bias
        # Target mix is deliberate, not frequency-matching.
        # ------------------------------------------------------------------
        street_targets = {
            0: 0.30,  # PRE
            1: 0.30,  # FLP
            2: 0.20,  # TRN
            3: 0.20,  # RVR
        }

        street_quotas = {
            s: int(replay_window * frac)
            for s, frac in street_targets.items()
        }

        # Collect samples by street, preserving recency weighting across iterations
        street_buckets = {0: [], 1: [], 2: [], 3: []}

        weighted_batches = []
        for i, batch in enumerate(iter_sample_history):
            age = len(iter_sample_history) - i   # 1=newest, N=oldest
            weight = 1.0 / age
            weighted_batches.append((batch, weight))

        for batch, weight in reversed(weighted_batches):
            for sample in batch:
                feat = sample[0]
                street = feat[STREET_SLICE].tolist()
                s = street.index(max(street)) if max(street) > 0 else 0
                street_buckets[s].append((sample, weight))

        accumulated_samples = []
        for s in range(4):
            bucket = street_buckets[s]
            if not bucket:
                continue

            quota = street_quotas[s]

            # Favor recent samples within each street bucket
            weights = np.array([w for _, w in bucket], dtype=np.float64)
            weights = weights / weights.sum()

            take = min(quota, len(bucket))
            idxs = np.random.choice(len(bucket), size=take, replace=False, p=weights)
            accumulated_samples.extend(bucket[i][0] for i in idxs)

        # Backfill if any street quota could not be met
        if len(accumulated_samples) < replay_window:
            all_samples = []
            for batch, weight in reversed(weighted_batches):
                for sample in batch:
                    all_samples.append((sample, weight))

            remaining = replay_window - len(accumulated_samples)
            if all_samples:
                weights = np.array([w for _, w in all_samples], dtype=np.float64)
                weights = weights / weights.sum()
                take = min(remaining, len(all_samples))
                idxs = np.random.choice(len(all_samples), size=take, replace=False, p=weights)
                accumulated_samples.extend(all_samples[i][0] for i in idxs)

        # Final trim just in case
        if len(accumulated_samples) > replay_window:
            accumulated_samples = accumulated_samples[:replay_window]

        # Adaptive epoch count — fewer epochs when buffer is large.
        target_grad_steps = net_epochs * (replay_window // 512)
        actual_batches    = max(len(accumulated_samples) // 512, 1)
        adaptive_epochs   = max(3, min(net_epochs, target_grad_steps // actual_batches))

        # Dynamic weight shifting based on previous iteration's val_loss
        if val_loss < 0.15:
            policy_weight = 1.0
            value_weight  = 0.7
        elif val_loss < 0.20:
            policy_weight = 0.7
            value_weight  = 0.8
        else:
            policy_weight = 0.5
            value_weight  = 1.0

        # Step 2: Train net with fixed buyin-scale normalisation
        net_loss, pol_loss, val_loss = train_net_on_samples(
            net, optimizer, accumulated_samples, device,
            epochs=adaptive_epochs,
            batch_size=512,   # keep it simple and fixed; train_net handles capping internally
            buyin=buyin,
            policy_weight=policy_weight,
            value_weight=value_weight
        )

        # Advance LR schedule AFTER optimizer.step() inside train_net (PyTorch rule).
        scheduler.step()

        saved_tag    = ""
        ckpt_payload = {
            "iteration":      iteration,
            "model_state":    net.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "scheduler_state":scheduler.state_dict(),
            "game_value":     game_value,
            "avg_entropy":    avg_entropy,
            "net_loss":       net_loss,
            "pol_loss":       pol_loss,
            "val_loss":       val_loss,
            "n_features":     N_FEATURES,
            "n_actions":      N_ACTIONS,
            "wallet":         wallet,
            "buyin":          buyin,
            "n_players":      n_players,
        }

        # Always save latest weights so --resume never loses progress
        torch.save(ckpt_payload,last_path)

        # Save best checkpoint when val_loss strictly improves
        if val_loss < best_loss:
            best_loss = val_loss
            # Read stored best loss if the file exists, otherwise treat as inf
            stored_loss = float(best_txt.read_text()) if best_txt.exists() else float("inf")
            if best_loss < stored_loss:
                torch.save(ckpt_payload, best_path)
                best_txt.write_text(str(best_loss))
                saved_tag = "    [BEST SAVED]"

        print(f"{iteration:>5} | {len(new_samples):>6,} | {len(accumulated_samples):>7,} | "
              f"{pol_loss:>9.4f} | {val_loss:>9.4f} |         {avg_entropy:>9.4f} | {game_value:>9.4f} | ep={adaptive_epochs} lr={scheduler.get_last_lr()[0]:.2e} |  {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB   "  
              f"{saved_tag}")

    # Fix the end-of-run print to show actual paths
    print(f"\n  Best val loss: {best_loss:.4f}")
    print(f"  Best model:  {best_path}")
    print(f"  Last model:  {last_path}")
    print(f"{'='*60}\n")
    return net, cfr


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # FIX 4: default wallet changed to 200 for manageable SPR
    parser.add_argument("--wallet",  type=float, default=200.0,
                        help="Starting stack (default 200 = 20bb effective depth)")
    parser.add_argument("--buyin",   type=float, default=10.0)
    parser.add_argument("--players", type=int,   default=2)
    parser.add_argument("--deals",   type=int,   default=1_000)
    parser.add_argument("--iters",   type=int,   default=50)
    parser.add_argument("--cfr",     type=int,   default=200)
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--resume",  type=str,   default=None,
                        help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()

    self_play_train(
        wallet         = args.wallet,
        buyin          = args.buyin,
        n_players      = args.players,
        n_deals        = args.deals,
        n_iterations   = args.iters,
        cfr_iterations = args.cfr,
        net_epochs     = args.epochs,
        resume_path    = args.resume,
    )