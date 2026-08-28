.PHONY: bootstrap check python-test python-lint swift-test swift-sample

PYTHON ?= python3

bootstrap:
	$(PYTHON) -m pip install -e . -r requirements-dev.lock

python-test:
	$(PYTHON) -m unittest discover -s tests

python-lint:
	$(PYTHON) -m compileall -q vllm_apple tests
	$(PYTHON) -m ruff check vllm_apple tests

swift-test:
	swift test --package-path sdk/swift

swift-sample:
	swift build --package-path samples/VLLMAppleChat

check: python-test python-lint swift-test swift-sample
