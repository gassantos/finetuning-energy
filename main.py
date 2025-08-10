from pathlib import Path
from src.finetuning import LlamaFineTuner
# from src.text_preprocessing_advanced import process_excel_to_dataset
from config.config import settings
import logging

from src.text_preprocessing import (
    PreprocessingConfig,
    process_text_data
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# def preprocess_dataset():
#     """Pré-processa o dataset Excel para formato de sumarização"""
#     logger.info("Iniciando pré-processamento do dataset...")
    
#     # Configurar caminhos
#     excel_file = Path("data/dataset.xlsx")
#     output_dir = Path("data/processed")
    
#     # Verificar se arquivo existe
#     if not excel_file.exists():
#         raise FileNotFoundError(f"Dataset não encontrado: {excel_file}")

#     df  = pd.read_excel(excel_file)
#     process_texto_resumo_tokens(df, dataset_path=str(excel_file))

#     # Exibir informações de tokens
#     df_tokens = pd.read_csv("data/dataset_tokens.csv")
#     print(df_tokens)

#     # Configurações de pré-processamento para sumarização
#     config_params = {
#         "text_column": "Texto",
#         "summary_column": "Resumo", 
#         "title_column": "Processo",
#         "min_text_length": 100,                           # Textos muito curtos não são úteis para sumarização
#         "max_text_length": df_tokens["tokens_texto"].max(),      # Limite para modelos de linguagem
#         "min_summary_length": 100,                        # Resumos muito curtos não são informativos
#         "max_summary_length": df_tokens["tokens_resumo"].max(),  # Resumos muito longos perdem o propósito
#         "test_size": 0.15,             # 15% para teste
#         "validation_size": 0.10,       # 10% para validação 
#         "clean_text": True,            # Ativar limpeza de texto
#         "save_formats": ["json", "parquet"]  # Formatos compatíveis com HF
#     }
    
#     try:
#         # Executar pré-processamento
#         result = process_excel_to_dataset(
#             excel_file=str(excel_file),
#             output_dir=str(output_dir),
#             **config_params
#         )
        
#         # Verificar resultado
#         if result["success"] and result["validation"]["valid"]:
#             logger.info("✅ Pré-processamento concluído com sucesso!")
#             logger.info(f"Total de exemplos: {result['dataset_info']['total_examples']}")
#             logger.info(f"Splits criados: {list(result['dataset_info']['splits'].keys())}")
#             logger.info(f"Arquivos salvos: {result['saved_files']}")
            
#             # Retornar caminho do dataset processado
#             dataset_path = output_dir / "dataset_structured_format"
#             return str(dataset_path)
#         else:
#             raise RuntimeError("Falha na validação do dataset processado")
            
#     except Exception as e:
#         logger.error(f"Erro no pré-processamento: {e}")
#         raise


# Exemplo de uso do novo sistema
config = PreprocessingConfig(
    text_column="Texto",
    summary_column="Resumo", 
    id_column="Processo",
    output_format="jsonl",
    instruction_template="[INST] Sumarize o seguinte texto: {text} [/INST] {summary}"
)

def main():
    """Função principal para executar o pré-processamento e fine-tuning com monitoramento energético"""
    logger.info("=== Iniciando Pipeline Completo: Pré-processamento + Fine-tuning ===")
    
    try:
        # Etapa 1: Pré-processamento do dataset
        logger.info("🔄 Etapa 1: Pré-processamento do dataset")
        output_dir = Path("data/processed") #preprocess_dataset()
        dataset_path = output_dir / "dataset_structured_format"

        result = process_text_data("data/dataset.xlsx", config)
        print(f"Resultado: {result['message']}")
        logger.info(f"Resultado: {result['message']}")
        if result['success']:
            print(f"Estatísticas: {result['statistics']}")
            logger.info(f"Estatísticas: {result['statistics']}")
            print(f"Arquivos salvos: {result['saved_files']}")
            logger.info(f"Arquivos salvos: {result['saved_files']}")

        # Etapa 2: Configuração do fine-tuning
        logger.info("🔄 Etapa 2: Configuração do fine-tuning")
        fine_tuner = LlamaFineTuner(str(settings.WANDB_KEY), str(settings.HF_TOKEN))
        
        # Etapa 3: Executar fine-tuning
        logger.info("🔄 Etapa 3: Executando fine-tuning com monitoramento energético")
        trainer = fine_tuner.run_complete_pipeline(dataset_path=dataset_path)
        
        logger.info("✅ Pipeline completo executado com sucesso!")
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Erro no pipeline: {e}")
        raise


if __name__ == "__main__":
    trainer = main()
