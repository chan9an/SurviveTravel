import random
from dataclasses import dataclass
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class RouteOption(BaseModel):
    id: str
    mode: str  # bus | train | flight
    destination: str
    cost: float
    duration_hours: int
    status: str  # on-time | delayed | cancelled


class TravelObservation(BaseModel):
    current_city: str
    destination_city: str
    current_time_hours: int
    remaining_budget: float
    weather_condition: str  # clear | rain | storm
    active_events: List[str]  # e.g. ["Transit Strike active"]
    available_routes: List[RouteOption]


class TravelAction(BaseModel):
    action_type: str = Field(..., pattern=r"^(take_route|wait)$")
    target_route_id: Optional[str] = None
    wait_hours: Optional[int] = None


@dataclass
class StepResult:
    observation: TravelObservation
    reward: float
    done: bool


def _base_routes() -> List[RouteOption]:
    return [
        RouteOption(id="bus-A-C",    mode="bus",    destination="City C", cost=30.0,  duration_hours=4, status="on-time"),
        RouteOption(id="bus-C-B",    mode="bus",    destination="City B", cost=30.0,  duration_hours=4, status="on-time"),
        RouteOption(id="train-A-C",  mode="train",  destination="City C", cost=80.0,  duration_hours=2, status="on-time"),
        RouteOption(id="train-C-B",  mode="train",  destination="City B", cost=80.0,  duration_hours=2, status="on-time"),
        RouteOption(id="flight-A-B", mode="flight", destination="City B", cost=600.0, duration_hours=2, status="on-time"),
        RouteOption(id="flight-C-B", mode="flight", destination="City B", cost=200.0, duration_hours=1, status="on-time"),
    ]


class TravelEnv:
    def __init__(self, task_id: str = "easy-clear-skies", seed: int = 42):
        self.task_id = task_id
        self.random = random.Random(seed)
        self.state_data: TravelObservation = None

    async def reset(self) -> StepResult:
        if self.task_id == "easy-clear-skies":
            weather = "clear"
            events = []
        elif self.task_id == "medium-strike":
            weather = "clear"
            events = ["Train Strike"]
        else:  # hard-storm
            weather = "storm"
            events = ["Train Strike", "Storm in City C"]

        self.state_data = TravelObservation(
            current_city="City A",
            destination_city="City B",
            current_time_hours=0,
            remaining_budget=1000.0,
            weather_condition=weather,
            active_events=events,
            available_routes=_base_routes(),
        )
        return StepResult(observation=self.state_data, reward=0.0, done=False)

    async def step(self, action: TravelAction) -> StepResult:
        reward = 0.0
        done = False
        obs = self.state_data

        # --- Surge Pricing ---
        if "Train Strike" in obs.active_events:
            for r in obs.available_routes:
                if r.mode in ("flight", "bus"):
                    r.cost *= 2.5

        if obs.weather_condition == "storm":
            for r in obs.available_routes:
                if r.mode in ("train", "bus"):
                    r.cost *= 1.5

        # --- Process action ---
        if action.action_type == "wait":
            hours = action.wait_hours or 1
            obs.current_time_hours += hours

        elif action.action_type == "take_route":
            route = next((r for r in obs.available_routes if r.id == action.target_route_id), None)
            if route:
                # Validate route starts from current city
                route_start = route.id.split("-")[1] if "-" in route.id else None
                if route_start and route_start != obs.current_city.split()[-1]:
                    reward -= 0.1  # wrong city
                elif route.status == "cancelled":
                    reward -= 0.1  # tried cancelled route
                elif obs.remaining_budget < route.cost:
                    reward -= 0.1  # can't afford it
                else:
                    obs.remaining_budget -= route.cost
                    obs.current_time_hours += route.duration_hours
                    obs.current_city = route.destination
                    reward += 0.1  # successfully moved

        # --- Chaos engine ---
        # 20% chance weather worsens
        if self.random.random() < 0.20:
            if obs.weather_condition == "clear":
                obs.weather_condition = "rain"
            elif obs.weather_condition == "rain":
                obs.weather_condition = "storm"

        # Storm disrupts flights
        if obs.weather_condition == "storm":
            for r in obs.available_routes:
                if r.mode == "flight":
                    r.status = "cancelled" if self.random.random() < 0.5 else "delayed"

        # Strike cancels trains
        if "Train Strike" in obs.active_events:
            for r in obs.available_routes:
                if r.mode == "train":
                    r.status = "cancelled"

        # Stranded mechanic: Storm in City C
        if "Storm in City C" in obs.active_events and obs.current_city == "City C":
            for r in obs.available_routes:
                if r.mode == "flight" and self.random.random() < 0.5:
                    r.status = "cancelled"

        # --- Win / loss evaluation ---
        if obs.current_city == obs.destination_city:
            reward += 1.0
            done = True
        elif obs.remaining_budget < 0 or obs.current_time_hours > 72:
            reward -= 1.0
            done = True

        self.state_data = obs
        return StepResult(observation=self.state_data, reward=reward, done=done)

    async def grade(self) -> float:
        obs = self.state_data
        reached = obs.current_city == obs.destination_city

        if self.task_id == "easy-clear-skies":
            score = 0.95 if reached else 0.05

        elif self.task_id == "medium-strike":
            score = 0.0
            if reached:
                score += 1.0
            if obs.remaining_budget < 300:
                score -= 0.5
            if obs.remaining_budget > 200:
                score += 0.2
            score = min(0.95, max(0.05, score))

        elif self.task_id == "hard-storm":
            score = 0.0
            if reached:
                score += 0.5
            if obs.remaining_budget > 200:
                score += 0.25
            if obs.current_time_hours < 48:
                score += 0.25
            score = min(0.95, max(0.05, score))

        else:
            score = 0.05

        return score

    async def state(self) -> TravelObservation:
        return self.state_data

    async def close(self):
        pass
