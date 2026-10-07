PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

test:
	pytest -q

generate:
	@echo "Generate step is not implemented in this phase."

train:
	@echo "Train step is not implemented in this phase."

evaluate:
	@echo "Evaluate step is not implemented in this phase."

report:
	@echo "Report step is not implemented in this phase."

api:
	@echo "API step is not implemented in this phase."

app:
	@echo "App step is not implemented in this phase."

all: install test

fast: test
