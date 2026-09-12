from fastapi import FastAPI
from sqlalchemy import text

from apps.api.db.session import SessionLocal


app = FastAPI(
    title="EarningsOS API",
    version="0.1.0",
    description="Market and earnings intelligence platform",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "earningsos-api",
        "version": "0.1.0",
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