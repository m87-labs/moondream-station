.PHONY: inference-server
inference-server: docker-build-inference-server docker-run-inference-server

.PHONY: docker-build-inference-server
docker-build-inference-server:
	@docker build -t moondream-station-inference:latest -f dockerfile.inference .

.PHONY: docker-run-inference-server
docker-run-inference-server:
	@docker run -p 20200:20200 moondream-station-inference:latest

.PHONY: docker-setup
docker-setup:
	@chmod +x setup_docker.sh && \
	./setup_docker.sh

.PHONY: modal_setup
modal_setup:
	@chmod +x modal_setup.sh && \
	./modal_setup.sh

