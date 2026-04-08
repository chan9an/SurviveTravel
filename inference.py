import os
import asyncio
import json
from openai import OpenAI
from travel_env import TravelEnv, TravelAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")


# --- Strict STDOUT loggers ---

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# --- Task runner ---

SYSTEM_PROMPT = (
    "You are a travel agent AI. Given the current travel state, decide the next action. "
    "Reply with ONLY a valid JSON object matching this schema — no markdown, no explanation:\n"
    '{"action_type": "take_route" | "wait", "target_route_id": "<id or null>", "wait_hours": <int or null>}'
)

async def run_task(client: OpenAI, task_id: str):
    env = TravelEnv(task_id=task_id)
    log_start(task_id, "TravelEnv", MODEL_NAME)

    await env.reset()

    rewards = []
    done = False
    step = 0

    for step in range(1, 11):
        obs = env.state_data
        user_msg = (
            f"Current city: {obs.current_city}\n"
            f"Destination: {obs.destination_city}\n"
            f"Time elapsed: {obs.current_time_hours}h\n"
            f"Budget remaining: ${obs.remaining_budget:.2f}\n"
            f"Weather: {obs.weather_condition}\n"
            f"Active events: {obs.active_events}\n"
            f"Available routes:\n"
            + "\n".join(
                f"  - id={r.id} mode={r.mode} dest={r.destination} "
                f"cost=${r.cost} duration={r.duration_hours}h status={r.status}"
                for r in obs.available_routes
            )
        )

        error = None
        action_str = "null"
        reward = 0.0

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=128,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            action = TravelAction(**parsed)
            action_str = action.action_type

            result = await env.step(action)
            reward = result.reward
            done   = result.done

        except Exception as exc:
            error = str(exc)
            done  = False

        rewards.append(reward)
        log_step(step, action_str, reward, done, error)

        if done:
            break

    score = await env.grade()
    success = env.state_data.current_city == env.state_data.destination_city
    log_end(success, step, score, rewards)
    await env.close()


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in ("easy-clear-skies", "medium-strike", "hard-storm"):
        await run_task(client, task_id)


if __name__ == "__main__":
    asyncio.run(main())
