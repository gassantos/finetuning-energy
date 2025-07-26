# Módulo Avançado de Pré-processamento de Texto

Este documento descreve o módulo de pré-processamento de texto desenvolvido para o projeto `finetuning-energy`, seguindo as boas práticas de engenharia de software para projetos de IA com PyTorch e HuggingFace.

## 📋 Visão Geral

O módulo `text_preprocessing_advanced.py` converte datasets em formato Excel para um formato estruturado, compatível com a biblioteca HuggingFace Datasets e adequado para tarefas de sumarização automática.

### 🎯 Características Principais

- **Formato Estruturado**: Compatível com bibliotecas de processamento de texto
- **Integração HuggingFace**: Datasets prontos para usar com transformers
- **Limpeza Avançada**: Correção automática de problemas de encoding
- **Splits Automáticos**: Divisão em treino, validação e teste
- **Múltiplos Formatos**: Saída em JSON, Parquet e CSV
- **Validação Robusta**: Verificação de qualidade dos dados
- **Configuração Flexível**: Parâmetros ajustáveis para diferentes cenários

## 🚀 Uso Rápido

### Método Simples (Recomendado)

```python
from src.text_preprocessing_advanced import process_excel_to_dataset

# Processamento completo com configuração padrão
result = process_excel_to_dataset("data/dataset.xlsx")

if result['success']:
    print(f"✅ Dataset processado: {result['dataset_info']['total_examples']} exemplos")
    print(f"Arquivos salvos: {result['saved_files']}")
```

### Método Avançado (Controle Total)

```python
from src.text_preprocessing_advanced import AdvancedTextProcessor, TextPreprocessingConfig

# Configuração customizada
config = TextPreprocessingConfig(
    text_column="Texto",
    summary_column="Resumo", 
    title_column="Processo",
    min_text_length=100,
    max_text_length=4000,
    test_size=0.15,
    validation_size=0.1,
    additional_columns=["Legislacao", "Pareceres"],
    save_formats=['json', 'parquet']
)

# Processamento passo a passo
processor = AdvancedTextProcessor(config)
processor.load_excel("data/dataset.xlsx")
processed_data = processor.preprocess_data()
dataset_dict = processor.create_dataset_splits()
validation = processor.validate_dataset_format()
saved_files = processor.save_dataset()
```

## 📁 Estrutura de Arquivos

```
```
src/
├── text_preprocessing_advanced.py    # Módulo principal
└── pipeline_base.py                 # Base para pipelines (se existir)

tests/
└── test_text_preprocessing_advanced.py  # Testes unitários

examples/
└── example_text_preprocessing.py    # Demonstração completa
```

data/
├── dataset.xlsx                     # Dataset original
└── processed/                       # Dados processados
    ├── dataset_simple.json         # Formato JSON simples
    ├── dataset_train.parquet       # Split de treino
    ├── dataset_validation.parquet  # Split de validação
    └── dataset_test.parquet        # Split de teste
```

## ⚙️ Configuração

### TextPreprocessingConfig

```python
@dataclass
class TextPreprocessingConfig:
    # Mapeamento de colunas
    text_column: str = "Texto"           # Campo principal
    summary_column: str = "Resumo"       # Campo de resumo
    title_column: str = "Processo"       # Campo de título
    
    # Colunas adicionais
    additional_columns: List[str] = []   # Campos extras
    
    # Limpeza de texto
    clean_text: bool = True
    normalize_unicode: bool = True
    remove_extra_whitespace: bool = True
    fix_encoding_issues: bool = True
    
    # Filtros de qualidade
    min_text_length: int = 500
    max_text_length: int = 50000
    min_summary_length: int = 100
    max_summary_length: int = 5000
    
    # Divisão dos dados
    test_size: float = 0.2              # 20% para teste
    validation_size: float = 0.1        # 10% para validação
    random_state: int = 42
    
    # Saída
    output_dir: str = "data/processed"
    save_formats: List[str] = ['json', 'parquet']
```

## 🔧 Comandos Make

```bash
# Demonstração completa
make demo-preprocessing-advanced

# Processamento do dataset
make process-dataset-advanced

# Validação dos dados processados
make validate-advanced-dataset

# Estatísticas do dataset
make advanced-stats

# Testes unitários
make test-preprocessing-advanced
```

## 📊 Formato de Saída

### Formato Estruturado Compatível

```json
{
  "text": [
    "Texto completo do documento 1...",
    "Texto completo do documento 2..."
  ],
  "summary": [
    "Resumo do documento 1...",
    "Resumo do documento 2..."
  ],
  "title": [
    "Título/ID do documento 1",
    "Título/ID do documento 2"
  ]
}
```

### Splits Criados

- **train**: 70% dos dados (após remover teste)
- **validation**: 10% dos dados de treino
- **test**: 20% dos dados totais

## 🧪 Testes e Validação

### Testes Unitários

```bash
# Executar todos os testes
pytest tests/test_text_preprocessing_advanced.py -v

# Executar testes específicos
pytest tests/test_text_preprocessing_advanced.py::TestTextCleaner -v
```

### Validação Automática

O módulo inclui validação automática que verifica:

- ✅ Presença das features obrigatórias (`text`, `summary`, `title`)
- ✅ Tipos de dados corretos
- ✅ Qualidade dos dados (textos vazios, comprimentos)
- ✅ Integridade dos splits

## 🔍 Limpeza de Texto

### Problemas Corrigidos Automaticamente

- **Encoding Issues**: `_x000D_`, `_x000A_`, `_x0009_`
- **Unicode**: Normalização NFKC
- **Espaços**: Remoção de espaços extras e quebras múltiplas
- **Caracteres Especiais**: Limpeza de caracteres mal codificados

### Exemplo de Limpeza

```python
# Antes
texto_sujo = "PLENÁRIO _x000D_\n_x000D_\nTEXTO    com   espaços"

# Depois  
texto_limpo = "PLENÁRIO\n\nTEXTO com espaços"
```

## 🔗 Integração com HuggingFace

### Carregamento Direto

```python
from datasets import load_from_disk

# Carregar dataset processado
dataset = load_from_disk("data/processed/dataset_structured_format")

# Verificar estrutura
print("Splits:", list(dataset.keys()))
print("Features:", list(dataset['train'].features.keys()))
print("Exemplo:", dataset['train'][0])
```

### Uso com Transformers

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Carregar modelo para sumarização
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

# Tokenizar dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding=True)
    labels = tokenizer(examples['summary'], truncation=True, padding=True)
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

## 📈 Estatísticas de Exemplo

Para o dataset atual (510 registros):

```
Total de exemplos: 507 (3 removidos por filtros)

Splits:
├── train: 364 exemplos (72%)
├── validation: 41 exemplos (8%)
└── test: 102 exemplos (20%)

Estatísticas por campo:
├── Texto: 500-50000 caracteres (média: ~5200)
├── Resumo: 100-5000 caracteres (média: ~1400)
└── Título: ID do processo (ex: "112962-5/2024")
```

## 🚨 Resolução de Problemas

### Erro: "Colunas obrigatórias não encontradas"

```python
# Verificar colunas disponíveis
df = pd.read_excel("data/dataset.xlsx")
print("Colunas:", list(df.columns))

# Ajustar configuração
config = TextPreprocessingConfig(
    text_column="sua_coluna_texto",
    summary_column="sua_coluna_resumo", 
    title_column="sua_coluna_titulo"
)
```

### Erro: "Todos os registros foram filtrados"

```python
# Ajustar filtros de comprimento
config = TextPreprocessingConfig(
    min_text_length=50,      # Reduzir mínimo
    max_text_length=100000,  # Aumentar máximo
    min_summary_length=10,   # Reduzir mínimo
    max_summary_length=10000 # Aumentar máximo
)
```

### Problema de Memória

```python
# Processar em lotes menores (implementação futura)
# Usar formatos mais eficientes
config = TextPreprocessingConfig(
    save_formats=['parquet']  # Apenas Parquet (mais eficiente)
)
```

## 🔄 Integração com Pipeline Principal

Para integrar com o sistema de fine-tuning existente:

```python
from src.text_preprocessing_advanced import process_excel_to_dataset
from src.finetuning import LlamaFineTuner

# 1. Processar dados
result = process_excel_to_dataset("data/dataset.xlsx")

# 2. Usar no fine-tuning
if result['success']:
    # Carregar dataset processado
    dataset = load_from_disk("data/processed/dataset_structured_format")
    
    # Configurar fine-tuning
    finetuner = LlamaFineTuner()
    finetuner.load_dataset_from_dict(dataset)
    finetuner.train()
```

## 📝 Próximos Passos

1. **Otimização de Performance**: Processamento em paralelo para datasets grandes
2. **Mais Formatos**: Suporte para CSV, TSV, JSON nativo
3. **Validação Avançada**: Detecção de duplicatas e qualidade semântica
4. **Metrics**: Estatísticas mais detalhadas (perplexidade, diversidade)
5. **Configurações Predefinidas**: Templates para diferentes tipos de tarefa

## 📄 Licença

Este módulo faz parte do projeto `finetuning-energy` e segue a mesma licença do projeto principal.
