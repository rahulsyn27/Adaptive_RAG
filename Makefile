.PHONY: install test lint typecheck api ui eval init-db docker-up docker-down

install:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check app tests evaluation frontend scripts

typecheck:
	uv run mypy app

api:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ui:
	uv run streamlit run frontend/app.py --server.port 8501

eval:
	uv run python -m evaluation.evaluate

init-db:
	uv run python scripts/init_db.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
