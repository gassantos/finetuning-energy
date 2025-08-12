from src.finetuning import LlamaFineTuner
from config.config import settings
from src.logging_config import setup_project_logging, get_pipeline_logger

from src.text_preprocessing import (
    PreprocessingConfig,
    process_text_data
)

# Configurar logging estruturado
setup_project_logging()
logger = get_pipeline_logger()

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
    logger.info("Iniciando Pipeline Completo: Pré-processamento + Fine-tuning")
    
    try:
        # Etapa 1: Pré-processamento do dataset
        logger.info("Etapa 1: Pré-processamento do dataset", step=1, stage="preprocessing")
        
        result = process_text_data("data/dataset.xlsx", config)
        print(f"Resultado: {result['message']}")
        logger.info("Resultado do pré-processamento", message=result['message'])
        
        if not result['success']:
            error_msg = result.get('error', 'Erro desconhecido')
            logger.error("Falha no pré-processamento", error=error_msg)
            raise RuntimeError(f"Pré-processamento falhou: {error_msg}")
            
        print(f"Estatísticas: {result['statistics']}")
        logger.info("Estatísticas do dataset processado", statistics=result['statistics'])
        print(f"Arquivos salvos: {result['saved_files']}")
        logger.info("Arquivos do dataset salvos", saved_files=result['saved_files'])
        
        # Obter o caminho do arquivo de treino dos resultados
        dataset_train_path = result['saved_files']['train']
        logger.info("Arquivo de treino selecionado", dataset_path=dataset_train_path)

        # Etapa 2: Configuração do fine-tuning
        logger.info("Etapa 2: Configuração do fine-tuning", step=2, stage="setup")
        fine_tuner = LlamaFineTuner(str(settings.WANDB_KEY), str(settings.HF_TOKEN))
        
        # Etapa 3: Executar fine-tuning
        logger.info("Etapa 3: Executando fine-tuning com monitoramento energético", 
                   step=3, stage="training")
        trainer = fine_tuner.run_complete_pipeline(dataset_path=dataset_train_path)

        logger.info("Pipeline completo executado com sucesso!", status="success")
        return trainer
        
    except Exception as e:
        logger.error("Erro no pipeline", error=str(e), exception_type=type(e).__name__)
        raise


if __name__ == "__main__":
    trainer = main()
