# Variables
IMAGE_NAME=flask-app
CONTAINER_NAME=flask-container

# Targets
build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d -p 8001:8001 --name $(CONTAINER_NAME) $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME)

rm:
	docker rm $(CONTAINER_NAME)

logs:
	docker logs -f $(CONTAINER_NAME)

.PHONY: build run stop rm logs
