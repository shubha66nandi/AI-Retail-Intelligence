
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.agent.data_analyst.agent import RetailDataAnalystAgent


router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.post("/ask")
def ask_analytics(
    question: str,
    db: Session = Depends(get_db),
):

    agent = RetailDataAnalystAgent(db)

    return agent.analyze(question)
