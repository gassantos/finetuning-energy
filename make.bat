@echo off
REM Batch file alternative to Makefile for Windows
REM Usage: make.bat <command>

if "%1"=="" (
    echo Uso: make.bat ^<command^>
    echo Execute "make.bat help" para ver comandos disponíveis
    goto :EOF
)

if "%1"=="help" goto :help
if "%1"=="test" goto :test
if "%1"=="test-unit" goto :test-unit
if "%1"=="test-config" goto :test-config
if "%1"=="test-monitor" goto :test-monitor
if "%1"=="test-offline" goto :test-offline
if "%1"=="test-coverage" goto :test-coverage
if "%1"=="format" goto :format
if "%1"=="lint" goto :lint
if "%1"=="clean" goto :clean
if "%1"=="info" goto :info

echo Comando desconhecido: %1
echo Execute "make.bat help" para ver comandos disponíveis
goto :EOF

:help
echo Comandos disponíveis:
echo   help                 - Mostra esta mensagem de ajuda
echo   test                 - Executa todos os testes (exceto lentos)
echo   test-unit            - Executa apenas testes unitários
echo   test-config          - Testa módulo de configuração
echo   test-monitor         - Testa módulo de monitoramento
echo   test-offline         - Executa testes offline (sem rede)
echo   test-coverage        - Executa testes com cobertura
echo   format               - Formata código
echo   lint                 - Executa linting
echo   clean                - Remove arquivos temporários
echo   info                 - Mostra informações do ambiente
goto :EOF

:test
echo Executando todos os testes (exceto lentos)...
uv run pytest -v -m "not slow and not integration"
goto :EOF

:test-unit
echo Executando testes unitários...
uv run pytest -v tests/test_*.py -m "not slow and not integration and not gpu"
goto :EOF

:test-config
echo Testando módulo de configuração...
uv run pytest -v tests/test_config.py
goto :EOF

:test-monitor
echo Testando módulo de monitoramento...
uv run pytest -v tests/test_monitor.py
goto :EOF

:test-offline
echo Executando testes offline (sem conectividade de rede)...
uv run pytest -v -m "not network" --ignore-glob="**/test_integration.py" -k "not test_load_and_prepare_dataset and not test_run_complete_pipeline and not test_complete_training_pipeline_mock"
goto :EOF

:test-coverage
echo Executando testes com cobertura...
uv run pytest --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
goto :EOF

:format
echo Formatando código...
uv run black src tests
uv run isort src tests
goto :EOF

:lint
echo Executando linting...
uv run flake8 src tests
uv run mypy src
goto :EOF

:clean
echo Removendo arquivos temporários e cache...
for /r %%i in (*.pyc) do del "%%i" 2>nul
for /d /r %%i in (__pycache__) do rd /s /q "%%i" 2>nul
for /d /r %%i in (*.egg-info) do rd /s /q "%%i" 2>nul
if exist build rd /s /q build 2>nul
if exist dist rd /s /q dist 2>nul
if exist htmlcov rd /s /q htmlcov 2>nul
if exist .coverage del .coverage 2>nul
if exist coverage.xml del coverage.xml 2>nul
if exist .pytest_cache rd /s /q .pytest_cache 2>nul
if exist .mypy_cache rd /s /q .mypy_cache 2>nul
if exist wandb rd /s /q wandb 2>nul
if exist test_output rd /s /q test_output 2>nul
echo Limpeza concluída.
goto :EOF

:info
echo === Informações do Ambiente ===
echo Python version: 
uv run python --version
echo UV version: 
uv --version
echo PyTest version: 
uv run pytest --version
echo Working directory: %CD%
echo Virtual environment: 
uv run python -c "import sys; print(sys.prefix)"
echo.
echo === Pacotes Instalados (Top 20) ===
uv pip list | findstr /n ".*" | findstr "^[1-9]:" | findstr "^[1-9]:\|^1[0-9]:\|^20:"
echo.
echo === Estrutura de Testes ===
dir tests\*.py /b
goto :EOF
