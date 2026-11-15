
import json

from sqlalchemy.orm import Session

from app.agent.data_analyst.query_engine import (
    get_total_revenue,
    get_top_selling_products,
    get_low_stock_products,
)

from app.agent.data_analyst.prompts import (
    DATA_ANALYST_SYSTEM_PROMPT,
    build_user_prompt,
)

from app.agent.llm.ollama_client import generate_response


class RetailDataAnalystAgent:

    def __init__(self, db: Session):
        self.db = db

    def analyze(self, question: str):

        if not question or not question.strip():
            return {
                "question": question,
                "answer": "Please provide a valid question.",
                "data": None,
            }

        normalized_question = question.lower()

        # Revenue analysis
        if any(word in normalized_question for word in [
            "revenue", "income", "earning"
        ]):

            data = get_total_revenue(self.db)

            analysis_type = "Total Revenue Analysis"

        # Top-selling products analysis
        elif any(word in normalized_question for word in [
            "top", "best-selling", "highest selling"
        ]):

            data = get_top_selling_products(self.db)

            analysis_type = "Top-Selling Products Analysis"

        # Low-stock analysis
        elif any(word in normalized_question for word in [
            "low stock", "stock shortage", "reorder"
        ]):

            data = get_low_stock_products(self.db)

            analysis_type = "Low-Stock Inventory Analysis"

        else:
            return {
                "question": question,
                "answer": (
                    "I can currently analyze total revenue, "
                    "top-selling products, and low-stock inventory."
                ),
                "data": None,
            }

        # Prepare database results for Qwen
        context = {
            "analysis_type": analysis_type,
            "user_question": question,
            "database_results": data,
        }

        user_prompt = build_user_prompt(
            f"""
            User question:
            {question}

            Analysis type:
            {analysis_type}

            Actual database results:
            {json.dumps(context, default=str)}

            Explain these results clearly.
            Use only the supplied database information.
            Do not invent numbers or product details.
            """
        )

        # Generate natural-language business insight
        try:
            answer = generate_response(
                user_prompt=user_prompt,
                system_prompt=DATA_ANALYST_SYSTEM_PROMPT,
            )

        except Exception:
            answer = (
                "Database analysis completed successfully, "
                "but the AI explanation service is currently unavailable."
            )

        return {
            "question": question,
            "answer": answer,
            "data": data,
        }
