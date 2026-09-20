from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from apps.api.config import settings
from apps.api.db.session import SessionLocal
from apps.api.intelligence import router as intelligence_router


app = FastAPI(
    title="EarningsOS API",
    version="0.2.0",
    description="Market and earnings intelligence platform",
)

allowed_origins = [
    origin.strip()
    for origin in settings.frontend_origins.split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )

app.include_router(intelligence_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "earningsos-api",
        "version": "0.2.0",
    }


@app.get("/health/db")
def database_health():
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT 1")).scalar()
        return {
            "status": "ok",
            "database": "postgresql",
            "result": result,
        }
    finally:
        db.close()
