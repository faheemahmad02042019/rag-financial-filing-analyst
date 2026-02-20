.PHONY: install test lint format run api clean docker-up docker-down

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
run:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8080 --reload

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ app/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ app/
	ruff check --fix src/ tests/ app/

# ---------------------------------------------------------------------------
# Docker (ChromaDB)
# ---------------------------------------------------------------------------
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info
