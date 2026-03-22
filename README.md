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

Getting started:

This version currently holds all the elements you need to get started with the
poker engine and play against a bot, or potentially multiple bots.

1. Create a local workspace with virtual environment. Ensure that
you use python 3.11 for pytorch

2. Download all contents of this repository

3. You will have to train your own bots.. look to self_play_train_nlh.py.
This file is responsible for training anf storing your models into the /checkpoints folder. 
This may take awhile.

    To play with 2 players the recommended training run terminal command is:
    python self_play_train_nlh.py --wallet 200 --buyin 10 --players 2
    --deals 8000 --iters 100 --cfr 30000 (should be at least 2x deals) --epochs 8
    
    To play with more than two players you will need to run a similar 4 player and 
    6 player command. Change the wallet and buyin ratio for different styles of play.
    
    Note - If you want less iters you can add a --resume pathtolastcheckpoint at the end of your 
    training command to continue off of an old train run. 
    Make sure that all other components of the run are the same. 

4. Run the main method to play against your bot. 
Use the terminal logs to watch live bot decision making.
Review your games played in data/live_play_phh_store

Sources:

cfrm algo
https://github.com/int8/counterfactual-regret-minimization/blob/master/common/utils.py

PHH data structures
Cattral, R. & Oppacher, F. (2002). Poker Hand [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5KW38.

MIT License
Copyright (c) 2024-2025 Universal, Open, Free, and Transparent Computer Poker Research Group
