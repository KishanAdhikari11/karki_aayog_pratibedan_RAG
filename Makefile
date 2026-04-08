help:
	@echo 
	@echo "install				-- install backend dependencies"
	@echo "lint 				-- linting"
	@echo "format				-- formatter"
	@echo "type					-- type checker"
	@echo "dev					-- start backend development"


.PHONY: install
install:
	uv sync --frozen

.PHONY:lint
lint:
	uv run ruff check .

.PHONY:format
format:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY:type
type:
	uv run mypy .


.PHONY:dev
dev:
	uv run uvicorn main:app --port 8002 --reload