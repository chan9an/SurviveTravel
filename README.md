# A-to-B Travel Survival

An [OpenEnv](https://openenv.dev) environment where an LLM agent must navigate from **City A** to **City B** while managing budget, time, and real-world disruptions like transit strikes and severe weather.

---

## Environment

| Property | Value |
|---|---|
| Entrypoint | `travel_env:TravelEnv` |
| Action model | `travel_env:TravelAction` |
| Observation model | `travel_env:TravelObservation` |

---

## Observation Space (`TravelObservation`)

| Field | Type | Description |
|---|---|---|
| `current_city` | str | Agent's current location |
| `destination_city` | str | Target destination |
| `current_time_hours` | int | Hours elapsed since start |
| `remaining_budget` | float | Remaining funds in USD |
| `weather_condition` | str | `clear` / `rain` / `storm` |
| `active_events` | List[str] | Active disruptions (e.g. `"Train Strike"`) |
| `available_routes` | List[RouteOption] | Bookable routes from current city |

### RouteOption fields
`id`, `mode` (bus/train/flight), `destination`, `cost`, `duration_hours`, `status` (on-time/delayed/cancelled)

---

## Action Space (`TravelAction`)

| Field | Type | Description |
|---|---|---|
| `action_type` | str | `take_route` or `wait` |
| `target_route_id` | str \| null | Route `id` to book (required for `take_route`) |
| `wait_hours` | int \| null | Hours to wait (required for `wait`) |

---

## Tasks

| ID | Difficulty | Description |
|---|---|---|
| `easy-clear-skies` | Easy | Straightforward route with minor delays |
| `medium-strike` | Medium | Reroute around a train strike on a limited budget |
| `hard-storm` | Hard | Multi-modal transit collapse from severe weather |

---

## Scoring

Each task returns a deterministic score between `0.0` and `1.0`:

- **easy**: `1.0` if destination reached, else `0.0`
- **medium**: `1.0` for reaching destination, `-0.5` penalty if budget drops below $300
- **hard**: `0.5` for reaching destination + `0.25` for budget > $200 + `0.25` for time < 48h

---

## Running with Docker

```bash
# Build
docker build -t a-to-b-travel .

# Run (provide your HuggingFace token)
docker run --rm \
  -e HF_TOKEN=your_token_here \
  a-to-b-travel
```

To use a different model or API:

```bash
docker run --rm \
  -e HF_TOKEN=your_token_here \
  -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  a-to-b-travel
```
