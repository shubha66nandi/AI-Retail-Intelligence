from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.database.connection import engine
from app.models import Product, Sale, Inventory

from app.api.products import router as products_router


# Application configuration
app = FastAPI(
    title="AI Retail Intelligence API",
    description="""
    Backend API for the AI Retail Intelligence System.

    Features:
    - Retail data management
    - Inventory management
    - Demand forecasting
    - AI Data Analysis Agent
    - Retail Decision Agent
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(products_router)
from app.database.connection import Base, engine


@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

# CORS configuration
# Allows the Streamlit frontend to communicate with FastAPI.
allowed_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Welcome to AI Retail Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


# Health check endpoint
@app.get("/health", tags=["General"])
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "project": "AI Retail Intelligence System",
        "version": "1.0.0",
    }


# Database health check
@app.get("/health/db", tags=["Database"])
def database_health() -> dict[str, str]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "connected",
            "database": "PostgreSQL",
            "message": "Database connection successful",
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed",
        )