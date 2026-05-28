import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.chat import router as chat_router
from api.vapi_webhook import router as vapi_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Solstice Pilates AI Receptionist",
    description="Sol — GPT-4o powered receptionist for Solstice Pilates studio.",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(vapi_router)


@app.get("/health", tags=["Infra"])
async def health() -> dict:
    return {"status": "ok", "service": "solstice-pilates-agent"}


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
