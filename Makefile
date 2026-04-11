.PHONY: all
all:
	install lint format typecheck

.PHONY: install
install:
	uv sync --frozen

.PHONY:lint
lint:
	uv run ruff check 

.PHONY:format
format:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY:typecheck
typecheck:
	uv run  ty check .

.PHONY:dev
dev:
	uv run uvicorn main:app --port 8002 --reload