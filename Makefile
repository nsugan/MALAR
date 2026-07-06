.PHONY: up down logs ps pull-model train infer checkpoint test lint health dev-api

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

pull-model:
	docker compose exec ollama ollama pull gemma3n:e4b
	docker compose exec ollama ollama pull gemma3n:e2b

health:
	python -m malar.health

train:
	python -m malar.training.campaign --domain $(DOMAIN)

infer:
	python -m malar.inference.service --file "$(FILE)" --text "$(TEXT)"

checkpoint:
	python -m malar.training.checkpoint --name $(NAME)

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

dev-api:
	uvicorn malar.api.app:app --reload --port 8000
