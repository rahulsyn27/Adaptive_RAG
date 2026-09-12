"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from app.core.constants import Route
from app.database.models import Base
from app.schemas.chat import GradeResult, RouteDecision


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
        session.commit()
    finally:
        session.close()
        engine.dispose()


class FakeChatModel(Runnable):
    """LangChain-shaped stand-in that never calls Groq."""

    def __init__(
        self,
        *,
        route: Route = Route.GENERAL,
        relevant: bool = True,
        rewrite: str = "transformer efficiency",
        answer: str = "Gradient descent is an optimization algorithm.",
    ) -> None:
        super().__init__()
        self.route = route
        self.relevant = relevant
        self.rewrite = rewrite
        self.answer = answer
        self.invoke_calls = 0

    def with_structured_output(self, schema: type[object]) -> RunnableLambda:
        def _inner(_input: object) -> object:
            if schema is RouteDecision:
                return RouteDecision(route=self.route, reason="Test classification rationale.")
            if schema is GradeResult:
                return GradeResult(
                    relevant=self.relevant,
                    score=0.9 if self.relevant else 0.15,
                    reason="Test grading rationale.",
                )
            raise AssertionError(f"Unexpected schema {schema}")

        return RunnableLambda(_inner)

    def invoke(self, _input: object, config: object | None = None, **kwargs: object) -> AIMessage:
        self.invoke_calls += 1
        if self.invoke_calls == 1 and self.route is Route.INDEX and not self.relevant:
            return AIMessage(content=self.rewrite)
        return AIMessage(content=self.answer)
