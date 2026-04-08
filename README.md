---
title: A to B Travel Survival
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# A-to-B Travel Survival
This is a custom OpenEnv reinforcement learning environment where an AI agent must navigate a multi-hop transit network while managing a budget and surviving random real-world disruptions like storms and train strikes.

## Setup
This environment runs natively via the included `Dockerfile`.

## Actions and Observations
* **Actions:** `take_route`, `wait`
* **State:** Tracks current city, budget, time elapsed, weather conditions, and active strike/storm events.