# finetuning-energy

Sistema de fine-tuning para modelos de linguagem com foco em eficiência energética para modelos LLMs.

## 🚀 Pipeline Integrado

O sistema agora oferece um **pipeline completo automatizado** que inclui:

1. **Pré-processamento de Texto**: Processa dados para formato de sumarização
2. **Fine-tuning com LoRA**: Ajuste fino eficiente do LLaMA 3.2 3B
3. **Monitoramento Energético**: Rastreamento completo de consumo durante treino

### Execução Simples

```bash
# Pipeline completo em um comando
uv run python main.py
```

## Módulos Principais

### 📊 Pré-processamento de Texto Avançado

O sistema inclui um módulo robusto de pré-processamento de texto que:

- Processa arquivos (excel) para datasets estruturados de sumarização
- Oferece limpeza e normalização de texto
- Cria splits de treino, validação e teste automaticamente
- Valida formato e qualidade dos dados
- Suporta múltiplos formatos de saída (JSON, Parquet, Arrow)
- Compatível com HuggingFace Datasets

### ⚡ Fine-tuning Integrado

O módulo de fine-tuning foi atualizado para:

- Aceitar datasets processados localmente
- Manter compatibilidade com datasets HuggingFace
- Detectar automaticamente o formato dos dados
- Aplicar LoRA (Low-Rank Adaptation) para eficiência
- Monitorar consumo energético durante o treino

### 📈 Monitoramento Energético

Sistema robusto de monitoramento energético sincronizado que rastreia:

- **Consumo de GPU** (NVIDIA-SMI, PyNVML, nvitop)
- **Métricas sincronizadas** (a cada 10s + por step de treinamento)
- **Baseline energético** (consumo diferencial preciso)
- **Eficiência energética** (Wh por step, scores de performance)
- **Pegada de carbono** (estimativa de CO2)
- **Logging estruturado** (Wandb + arquivos locais)

#### Funcionalidades Avançadas:
- ✅ **Sincronização dupla**: Intervalos fixos + steps de treinamento
- ✅ **Alta precisão**: Baseline automático para cálculos diferenciais  
- ✅ **Callback nativo**: Integração perfeita com Transformers
- ✅ **Métricas de eficiência**: Análise automatizada de performance energética
- ✅ **Relatórios detalhados**: Histórico completo por step e intervalo
- ✅ **Estimativa de CO2**: Consciência ambiental no desenvolvimento


## Pré-processamento de Dados

### Formato de Entrada (Dado Estruturado)
```
| Texto                     | Resumo                   |
| ------------------------- | ------------------------ |
| Texto para sumarização... | Resumo correspondente... |
```

### Formato de Saída (HuggingFace)
```json
{
    "text": "Texto original para sumarização...",
    "summary": "Resumo correspondente..."
}
```

## 🧪 Testes

Sistema completo de validação com **14 testes abrangentes**:

```bash
# Executar todos os testes
uv run pytest

# Testes específicos de pré-processamento
uv run pytest tests/test_text_preprocessing_validation.py

# Testes de integração do pipeline
uv run pytest test_pipeline.py

# Executar com verbose
uv run pytest -v
```

### Cobertura de Testes

- ✅ Processamento direto de arquivos Excel
- ✅ Pipeline manual de pré-processamento
- ✅ Validação de formatos e tipos
- ✅ Tratamento de erros e casos extremos
- ✅ Compatibilidade HuggingFace
- ✅ Integração completa do pipeline
- ✅ Validação de configuração
- ✅ Carregamento de datasets processados

## ⚙️ Configuração

### Arquivos de Configuração

O sistema usa `config/settings.toml` para configurações:

```toml
[global]
MODEL_ID = "meta-llama/Llama-3.2-3B"
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 16
DATASET = "billsum"
TASK = "summarization"
QUANTIZATION = "4-bit"
```
OBS.: Para informações sensíveis, recomenda-se o uso de secrets  (`config/.secrets.toml`), conforme Dynaconf.

### Secrets de Ambiente

```bash
# Tokens obrigatórios para fine-tuning
WANDB_KEY="your_wandb_key"
HF_TOKEN="your_hf_token"
```

## 📂 Estrutura do Projeto

```
finetuning-energy/
├── main.py                    # Execução principal
├── src/
│   ├── finetuning.py          # Fine-tuning com LoRA
│   ├── monitor.py             # Monitoramento energético
│   └── text_preprocessing.py  # Pré-processamento
├── tests/
├── config/
│   ├── config.py              # Configurações
│   └── settings.toml          # Parâmetros
└── data/                      # Datasets (local)
```

## 📊 Exemplo Prático

```bash
# 1. Executar pipeline com monitoramento avançado
uv run python main.py

# 2. Monitorar progresso (Weights & Biases)
# Acesse wandb.ai para ver métricas em tempo real:
# - energy_interval/*: Métricas a cada 10s
# - energy_step/*: Métricas por step de treinamento
# - final_energy/*: Relatório final com eficiência

# 3. Executar exemplos de monitoramento
uv run python examples/energy_monitoring_examples.py
```

### Resultado Esperado
- Dataset processado: `data/processed`
- Modelo fine-tuned: `results/llama_finetuned/`
- **Logs energéticos sincronizados**: 
  - `results/energy_step_history_*.json`
  - `results/energy_interval_history_*.json`
  - `results/energy_monitoring_summary_*.json`
- **Dashboard W&B com métricas avançadas**:
  - Consumo por step e intervalo
  - Scores de eficiência energética
  - Correlação performance x energia
  - Estimativa de pegada de carbono

### Executar Testes

```bash
# Todos os testes
uv run pytest

# Apenas testes de validação do pré-processamento
uv run pytest tests/test_text_preprocessing_validation.py -v

# Testes com cobertura
uv run pytest --cov=src
```

## Desenvolvimento

### Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (gerenciador de pacotes)


```bash
# Clonar repositório
git clone <repo-url>
cd finetuning-energy

# Instalar dependências
uv sync

# Executar aplicação
uv run python main.py