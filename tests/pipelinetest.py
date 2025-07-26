#!/usr/bin/env python3
"""
Script de teste para validar o pipeline completo de pré-processamento + fine-tuning

Este script testa todas as etapas sem executar o fine-tuning real.
"""

import sys
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.text_preprocessing_advanced import process_excel_to_dataset
from src.finetuning import LlamaFineTuner
from config.config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_preprocessing():
    """Testa o pré-processamento do dataset"""
    logger.info("=== Teste: Pré-processamento do Dataset ===")
    
    excel_file = Path("data/dataset.xlsx")
    output_dir = Path("data/processed")
    
    if not excel_file.exists():
        logger.error(f"❌ Dataset não encontrado: {excel_file}")
        return False
    
    try:
        result = process_excel_to_dataset(
            excel_file=str(excel_file),
            output_dir=str(output_dir),
            min_text_length=100,
            max_text_length=8000,
            min_summary_length=20,
            max_summary_length=1000,
            test_size=0.15,
            validation_size=0.10,
            clean_text=True,
            save_formats=["json", "parquet"]
        )
        
        if result["success"] and result["validation"]["valid"]:
            logger.info("✅ Pré-processamento: SUCESSO")
            logger.info(f"   Total de exemplos: {result['dataset_info']['total_examples']}")
            logger.info(f"   Splits: {list(result['dataset_info']['splits'].keys())}")
            return True
        else:
            logger.error("❌ Pré-processamento: FALHOU na validação")
            return False
            
    except Exception as e:
        logger.error(f"❌ Pré-processamento: ERRO - {e}")
        return False


def test_finetuner_initialization():
    """Testa a inicialização do fine-tuner"""
    logger.info("=== Teste: Inicialização do Fine-tuner ===")
    
    try:
        # Valores de teste (não precisam ser reais para teste de inicialização)
        fine_tuner = LlamaFineTuner("test-key", "test-token")
        logger.info("✅ Fine-tuner: INICIALIZADO")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fine-tuner: ERRO na inicialização - {e}")
        return False


def test_dataset_loading():
    """Testa o carregamento do dataset processado"""
    logger.info("=== Teste: Carregamento do Dataset Processado ===")
    
    dataset_path = "data/processed/dataset_structured_format"
    
    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset processado não encontrado: {dataset_path}")
        return False
    
    try:
        from datasets import load_from_disk
        
        dataset_dict = load_from_disk(dataset_path)
        
        if hasattr(dataset_dict, "keys") and "train" in dataset_dict:
            train_dataset = dataset_dict["train"]
            logger.info("✅ Dataset: CARREGADO")
            logger.info(f"   Split 'train': {len(train_dataset)} exemplos")
            logger.info(f"   Features: {list(train_dataset.features.keys())}")
            
            # Verificar se tem as colunas necessárias
            required_features = {"text", "summary"}
            available_features = set(train_dataset.features.keys())
            
            if required_features.issubset(available_features):
                logger.info("✅ Dataset: FORMATO VÁLIDO")
                return True
            else:
                missing = required_features - available_features
                logger.error(f"❌ Dataset: FEATURES FALTANDO - {missing}")
                return False
        else:
            logger.error("❌ Dataset: ESTRUTURA INVÁLIDA")
            return False
            
    except Exception as e:
        logger.error(f"❌ Dataset: ERRO no carregamento - {e}")
        return False


def test_configuration():
    """Testa as configurações do projeto"""
    logger.info("=== Teste: Configurações do Projeto ===")
    
    try:
        # Verificar configurações principais
        config_items = [
            ("MODEL_ID", settings.MODEL_ID),
            ("TASK", settings.TASK),
            ("BATCH_SIZE", settings.BATCH_SIZE),
            ("LEARNING_RATE", settings.LEARNING_RATE),
            ("EPOCHS", settings.EPOCHS),
        ]
        
        for name, value in config_items:
            logger.info(f"   {name}: {value}")
        
        logger.info("✅ Configurações: CARREGADAS")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configurações: ERRO - {e}")
        return False


def main():
    """Executa todos os testes do pipeline"""
    logger.info("🧪 === TESTE DO PIPELINE COMPLETO ===")
    
    tests = [
        ("Configurações", test_configuration),
        ("Pré-processamento", test_preprocessing),
        ("Carregamento do Dataset", test_dataset_loading),
        ("Inicialização do Fine-tuner", test_finetuner_initialization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔄 Executando: {test_name}")
        results[test_name] = test_func()
    
    # Resumo dos resultados
    logger.info("\n📊 === RESUMO DOS TESTES ===")
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSOU" if result else "❌ FALHOU"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Resultado Final: {passed}/{total} testes passaram")
    
    if passed == total:
        logger.info("🎉 Todos os testes passaram! Pipeline está pronto.")
        logger.info("💡 Para executar o fine-tuning real, execute: python main.py")
        return True
    else:
        logger.error("⚠️ Alguns testes falharam. Verifique os logs acima.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
