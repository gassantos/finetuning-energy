# Makefile para automação de testes e desenvolvimento
# ===================================================

.PHONY: help test test-unit test-integration test-slow test-coverage clean lint format install install-dev docs setup check-config

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

test-ci: ## Executa testes para CI/CD
	$(UV) run $(PYTEST) -v --tb=no -m "not network"

# test-ci-full: format-check lint ci-test ## Verificação completa para CI/CD

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
	$(UV) run $(PYTEST) --tb=no -v -m "not network"

ci-full: format-check lint ci-test ## Verificação completa para CI/CD

# Documentação (placeholder para futura expansão)
docs: ## Gera documentação (placeholder)
	@echo "Documentação será implementada futuramente"

# Informações do sistema
info: ## Mostra informações do ambiente
	@echo "=== Informações do Ambiente ==="
	@echo "Python version: $$($(UV) run $(PYTHON) --version)"
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

# Setup e configuração inicial
setup: ## Configura ambiente inicial (cria .secrets.toml se necessário)
	@echo "=== Configuração Inicial ==="
	@echo "Verificando arquivo de configuração de secrets..."
	@if [ ! -f config/.secrets.toml ]; then \
		echo "Criando config/.secrets.toml..."; \
		echo "# Arquivo de configuração de secrets" > config/.secrets.toml; \
		echo "# Este arquivo contém informações sensíveis e não deve ser commitado" >> config/.secrets.toml; \
		echo "" >> config/.secrets.toml; \
		echo "[global]" >> config/.secrets.toml; \
		echo "# Token para Weights & Biases (https://wandb.ai/settings)" >> config/.secrets.toml; \
		echo "WANDB_KEY = \"sua_wandb_key_aqui\"" >> config/.secrets.toml; \
		echo "" >> config/.secrets.toml; \
		echo "# Token para Hugging Face (https://huggingface.co/settings/tokens)" >> config/.secrets.toml; \
		echo "HF_TOKEN = \"seu_hf_token_aqui\"" >> config/.secrets.toml; \
		echo "" >> config/.secrets.toml; \
		echo "[development]" >> config/.secrets.toml; \
		echo "# Tokens específicos para desenvolvimento (opcional)" >> config/.secrets.toml; \
		echo "" >> config/.secrets.toml; \
		echo "[production]" >> config/.secrets.toml; \
		echo "# Tokens específicos para produção (opcional)" >> config/.secrets.toml; \
		echo ""; \
		echo "✅ Arquivo config/.secrets.toml criado com sucesso!"; \
		echo ""; \
		echo "IMPORTANTE: Edite o arquivo config/.secrets.toml e adicione seus tokens:"; \
		echo "  - WANDB_KEY: Token do Weights & Biases"; \
		echo "  - HF_TOKEN: Token do Hugging Face"; \
		echo ""; \
		echo "Os tokens podem ser obtidos em:"; \
		echo "  - WANDB: https://wandb.ai/settings"; \
		echo "  - Hugging Face: https://huggingface.co/settings/tokens"; \
	else \
		echo "✅ Arquivo config/.secrets.toml já existe"; \
	fi
	@echo "=== Configuração Concluída ==="

check-config: ## Verifica se as configurações estão válidas
	@echo "=== Verificação de Configuração ==="
	@if [ ! -f config/.secrets.toml ]; then \
		echo "❌ Arquivo config/.secrets.toml não encontrado"; \
		echo "Execute 'make setup' para criar o arquivo"; \
		exit 1; \
	fi
	@echo "✅ Arquivo config/.secrets.toml encontrado"
	@$(UV) run $(PYTHON) -c "from config.config import settings; \
		print('✅ Configurações carregadas com sucesso'); \
		print(f'  - WANDB_ENTITY: {settings.WANDB_ENTITY}'); \
		print(f'  - WANDB_PROJECT: {settings.WANDB_PROJECT}'); \
		print(f'  - MODEL_ID: {settings.MODEL_ID}'); \
		try: \
			wandb_key = str(settings.WANDB_KEY); \
			hf_token = str(settings.HF_TOKEN); \
			if 'sua_wandb_key_aqui' in wandb_key or 'seu_hf_token_aqui' in hf_token: \
				print('⚠️  ATENÇÃO: Tokens ainda não foram configurados em .secrets.toml'); \
			else: \
				print('✅ Tokens configurados'); \
		except Exception as e: \
			print(f'❌ Erro ao carregar tokens: {e}');"
	@echo "=== Verificação Concluída ==="

# Pré-processamento Avançado de Texto
demo-preprocessing-advanced: ## Demonstra o módulo de pré-processamento avançado
	$(UV) run $(PYTHON) example_text_preprocessing.py

test-preprocessing-advanced: ## Testa módulo de pré-processamento avançado
	$(UV) run $(PYTEST) -v tests/test_text_preprocessing_advanced.py

process-dataset-advanced: ## Processa dataset Excel para formato estruturado
	@echo "Processando dataset para formato estruturado..."
	$(UV) run $(PYTHON) -c "from src.text_preprocessing_advanced import process_excel_to_dataset; \
		result = process_excel_to_dataset('data/dataset.xlsx'); \
		print('✅ Processamento concluído!' if result['success'] else '❌ Erro no processamento'); \
		print(f'Arquivos: {result[\"saved_files\"]}'); \
		info = result['dataset_info']; \
		print(f'Total: {info[\"total_examples\"]} exemplos'); \
		for split, data in info['splits'].items(): \
			print(f'  {split}: {data[\"num_examples\"]} exemplos');"

validate-advanced-dataset: ## Valida dataset estruturado processado
	@echo "Validando dataset estruturado..."
	@if [ -d "data/processed" ]; then \
		echo "✅ Diretório data/processed encontrado"; \
		ls -la data/processed/; \
		if [ -f "data/processed/dataset_simple.json" ]; then \
			echo "✅ Arquivo JSON encontrado"; \
			$(UV) run $(PYTHON) -c "import json; \
				with open('data/processed/dataset_simple.json') as f: \
					data = json.load(f); \
				print(f'Registros: {len(data[\"text\"])}'); \
				print(f'Features: {list(data.keys())}');" ; \
		fi; \
	else \
		echo "❌ Execute 'make process-dataset-advanced' primeiro"; \
	fi

advanced-stats: ## Mostra estatísticas do dataset estruturado
	$(UV) run $(PYTHON) -c "from src.text_preprocessing_advanced import AdvancedTextProcessor; \
		processor = AdvancedTextProcessor(); \
		processor.load_excel('data/dataset.xlsx'); \
		processor.preprocess_data(); \
		processor.create_dataset_splits(); \
		info = processor.get_dataset_info(); \
		import json; \
		print(json.dumps(info, indent=2, ensure_ascii=False))"

# Regras especiais
.DEFAULT_GOAL := help