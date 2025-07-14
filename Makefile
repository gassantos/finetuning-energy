# Makefile para automação de testes e desenvolvimento
# ===================================================

.PHONY: help test test-unit test-integration test-slow test-coverage clean lint format install install-dev docs

# Variáveis
PYTHON = python
UV = uv
PYTEST = pytest
COV_REPORT = htmlcov
TEST_MARKER ?= ""

# Help
help: ## Mostra esta mensagem de ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Testes
test: ## Executa todos os testes (exceto lentos)
	$(UV) run $(PYTEST) -v -m "not slow and not integration"

test-unit: ## Executa apenas testes unitários
	$(UV) run $(PYTEST) -v tests/test_*.py -m "not slow and not integration and not gpu"

test-integration: ## Executa testes de integração
	$(UV) run $(PYTEST) -v -m integration

test-slow: ## Executa testes lentos
	$(UV) run $(PYTEST) -v -m slow

test-gpu: ## Executa testes que requerem GPU
	$(UV) run $(PYTEST) -v -m gpu

test-all: ## Executa todos os testes (incluindo lentos)
	$(UV) run $(PYTEST) -v

test-coverage: ## Executa testes com cobertura
	$(UV) run $(PYTEST) --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

test-parallel: ## Executa testes em paralelo
	$(UV) run $(PYTEST) -v -n auto

test-failed: ## Re-executa apenas testes que falharam
	$(UV) run $(PYTEST) --lf -v

test-watch: ## Executa testes em modo watch (requer pytest-watch)
	$(UV) run ptw tests/ src/

# Análise de código
lint: ## Executa linting
	$(UV) run flake8 src tests
	$(UV) run mypy src

format: ## Formata código
	$(UV) run black src tests
	$(UV) run isort src tests

format-check: ## Verifica formatação sem modificar
	$(UV) run black --check src tests
	$(UV) run isort --check-only src tests

# Limpeza
clean: ## Remove arquivos temporários e cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf $(COV_REPORT)/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf wandb/
	rm -rf test_output/

# Relatórios
coverage-report: test-coverage ## Gera e abre relatório de cobertura
	@echo "Abrindo relatório de cobertura..."
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(COV_REPORT)/index.html; \
	elif command -v open >/dev/null 2>&1; then \
		open $(COV_REPORT)/index.html; \
	else \
		echo "Relatório disponível em: $(COV_REPORT)/index.html"; \
	fi

coverage-stats: ## Mostra estatísticas de cobertura
	$(UV) run coverage report --show-missing

# Verificação completa
check: format-check lint test-coverage ## Executa todas as verificações


# Testes por categoria
test-finetuning: ## Testa módulo de fine-tuning
	$(UV) run $(PYTEST) -v tests/test_finetuning.py

test-monitor: ## Testa módulo de monitoramento
	$(UV) run $(PYTEST) -v tests/test_monitor.py

test-config: ## Testa módulo de configuração
	$(UV) run $(PYTEST) -v tests/test_config.py

test-training-setup: ## Testa configuração de treinamento
	$(UV) run $(PYTEST) -v tests/test_training_setup.py

test-early-stopping: ## Testa early stopping
	$(UV) run $(PYTEST) -v tests/test_early_stopping.py

# Testes com configurações especiais
test-verbose: ## Executa testes com saída verbosa
	$(UV) run $(PYTEST) -vvv --tb=long

test-quiet: ## Executa testes silenciosamente
	$(UV) run $(PYTEST) -q

test-debug: ## Executa testes em modo debug
	$(UV) run $(PYTEST) -v --pdb --tb=short

# Benchmarks e performance
test-performance: ## Executa testes de performance
	$(UV) run $(PYTEST) -v -m "performance"

benchmark: ## Executa benchmarks
	@echo "Executando benchmarks básicos..."
	$(UV) run $(PYTHON) -c "import time; from src.monitor import RobustGPUMonitor; m = RobustGPUMonitor(); print('Monitor criado em:', time.time())"

# CI/CD
ci-test: ## Testes para CI/CD
	$(UV) run $(PYTEST) -v --tb=short --cov=src --cov-report=xml -m "not slow and not gpu"

ci-full: format-check lint ci-test ## Verificação completa para CI/CD

# Documentação (placeholder para futura expansão)
docs: ## Gera documentação (placeholder)
	@echo "Documentação será implementada futuramente"

# Informações do sistema
info: ## Mostra informações do ambiente
	@echo "=== Informações do Ambiente ==="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "UV version: $$($(UV) --version)"
	@echo "PyTest version: $$($(UV) run $(PYTEST) --version)"
	@echo "Working directory: $$(pwd)"
	@echo "Virtual environment: $$($(UV) run $(PYTHON) -c 'import sys; print(sys.prefix)')"
	@echo ""
	@echo "=== Pacotes Instalados ==="
	@$(UV) pip list | head -20
	@echo ""
	@echo "=== Estrutura de Testes ==="
	@find tests/ -name "*.py" | head -10

# Regras especiais
.DEFAULT_GOAL := help