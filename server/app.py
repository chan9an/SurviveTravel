from fastapi import FastAPI
from .travel_env import TravelEnv, TravelAction

app = FastAPI()
env = TravelEnv()

@app.get("/")
def read_root():
    return {"status": "OpenEnv Travel Survival is Running!"}

@app.post("/reset")
async def reset():
    # Ensure this returns the observation, reward, done structure
    return await env.reset()

@app.post("/step")
async def step(action: TravelAction):
    return await env.step(action)

@app.get("/state")
async def state():
    return await env.state()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
