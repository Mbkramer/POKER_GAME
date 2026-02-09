Intelligent Poker Engine

A modular, extensible poker game built in Python with a long‑term vision: create believable, intelligent opponents that play, learn, and adapt like real humans at the table.

This project blends game engine architecture, probability modeling, decision theory, and AI‑driven behavior to simulate authentic poker gameplay rather than simple scripted bots.

Vision

Most poker implementations stop at rules and rendering.
This project goes further.

Primary objective:

Build a complete poker environment where opponents reason under uncertainty, adapt to player behavior, and evolve strategically over time.

Key principles guiding development:

Authenticity over shortcuts — gameplay should feel like a real table, not a toy simulator.

Transparent architecture — clean separation between engine, UI, and AI logic.

Research‑driven intelligence — probability, game theory, and machine learning inform decision making.

Iterative evolution — start with rule‑based bots → progress toward adaptive and learning agents.

Current Capabilities

The project already includes foundational components required for a full poker experience:

Core Game Engine

Full card, deck, and hand‑ranking system

Round progression (deal → betting streets → showdown)

Pot management and player state tracking

Deterministic game flow suitable for simulation and AI training

User Interface

Playable table built with Pygame

Visual rendering of cards, chips, and player actions

Input handling for betting, folding, and raising

Designed to remain decoupled from engine logic

Simulation & Evaluation

Hand evaluation utilities

Probability‑ready architecture for future Monte Carlo or solver integration

Cattral, R. & Oppacher, F. (2002). Poker Hand [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5KW38.
