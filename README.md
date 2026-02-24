Intelligent Poker Engine

A modular, extensible poker game built in Python with a long‑term vision: create believable, intelligent opponents that play, learn, and adapt like real humans at the table.
This project blends game engine architecture, probability modeling, decision theory, and AI‑driven behavior to simulate authentic poker gameplay rather than simple scripted bots.
The current verison uses MCCFR to train 4 poker bot variants. Play ranges from 2 Player, 10 Buy-in, 100 Wallet to tournament style 6 Player, 10000 Buy-in, 600000 Wallet. 
Each bot is trained by collecting data from the current table state and preflop hand equities, post flop hand equity and dynamic play metrics, and by tracking openent play metrics to self train on a large set of predetermined hands. Within the model, decisions are analysed by agregating "regret" or similarly the oppurtunity cost of taking another action given a certain game circumstance. These decisions and regrets are aggregated accross a vast collection of different hands, and from which are used to determine a nash equilibirum and optimal game strategy. 

Primary objective:
Build a complete poker environment where opponents reason under uncertainty, adapt to player behavior, and evolve strategically over time.

Key principles:
Authenticity over shortcuts — gameplay should feel like a real table.
Transparent architecture — clean separation between engine, UI, and AI logic.
Research‑driven intelligence — probability, game theory, and machine learning inform decision making.
Iterative evolution — start with rule‑based bots → progress toward adaptive and learning agents.

Current Capabilities:
 - Core Game Engine
 - Full card, deck, and hand‑ranking system
 - Round progression (deal → betting streets → showdown)
 - Pot management and player state tracking
 - Deterministic game flow suitable for simulation and AI training
 - User Interface
 - Playable table built with Pygame
 - Visual rendering of cards, chips, and player actions
 - Input handling for betting, folding, and raising
 - Designed to remain decoupled from engine logic
 - Simulation & Evaluation
 - Hand utilities evaluation
 - Prosepctive hand probability with Monte Carlo sampling
 - Competive solo play against 4 variants of Monte Carlo Sampled Counter Factual Regret (MCCFR) self trained bots
 - Local hand tracking and store of game play hands via PHH json files

Sources:

cfrm algo
https://github.com/int8/counterfactual-regret-minimization/blob/master/common/utils.py

PHH data structures
Cattral, R. & Oppacher, F. (2002). Poker Hand [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5KW38.

MIT License
Copyright (c) 2024-2025 Universal, Open, Free, and Transparent Computer Poker Research Group
