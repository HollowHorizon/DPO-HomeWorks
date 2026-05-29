IMAGE_NAME=noterag-base
CONTAINER_NAME=noterag-base

.PHONY: docker-build docker-run docker-up docker-down docker-test docker-evaluate docker-logs

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p 8000:8000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-test:
	docker run --rm $(IMAGE_NAME) pytest -q

docker-evaluate:
	docker run --rm $(IMAGE_NAME) python -m src.models.evaluate

docker-logs:
	docker logs -f $(CONTAINER_NAME)
