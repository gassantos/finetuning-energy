from src.finetuning import LlamaFineTuner
from config.config import settings


def main():
    """Função principal para executar o fine-tuning com monitoramento energético"""
    # Configurar suas chaves aqui
    WANDB_KEY = str(settings.WANDB_KEY)
    HF_TOKEN = str(settings.HF_TOKEN)

    # Criar instância do fine-tuner
    fine_tuner = LlamaFineTuner(WANDB_KEY, HF_TOKEN)

    # Executar pipeline completo
    trainer = fine_tuner.run_complete_pipeline()

    return trainer


if __name__ == "__main__":
    trainer = main()
