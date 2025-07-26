#!/usr/bin/env python3
"""
Pipeline simples para teste sem quantização
"""
import logging
from pathlib import Path
from src.text_preprocessing_advanced import process_excel_to_dataset

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_preprocessing_only():
    """Testa apenas o pré-processamento sem fine-tuning"""
    logger.info("=== Teste de Pré-processamento ===")
    
    # Verificar se existe um arquivo de dados
    data_files = [
        "data/dataset.xlsx",
        "data/dados.xlsx", 
        "data/exemplo.xlsx"
    ]
    
    excel_file = None
    for file_path in data_files:
        if Path(file_path).exists():
            excel_file = file_path
            break
    
    if not excel_file:
        logger.info("Nenhum arquivo Excel encontrado. Criando arquivo de exemplo...")
        
        # Criar diretório data se não existir
        Path("data").mkdir(exist_ok=True)
        
        # Criar arquivo Excel de exemplo
        import pandas as pd
        
        # Dados de exemplo para sumarização
        exemplo_data = {
            "Texto": [
                "Este é um texto muito longo que precisa ser resumido. Contém várias informações importantes sobre um tópico específico. O texto deve ter pelo menos 100 caracteres para ser válido no pipeline de processamento. Aqui temos informações detalhadas sobre um processo complexo que envolve múltiplas etapas e considerações técnicas importantes.",
                "Outro documento extenso com conteúdo relevante para sumarização. Este texto também deve atender aos critérios mínimos de comprimento para ser processado adequadamente. Contém informações técnicas e científicas que são importantes para o treinamento do modelo de linguagem. A qualidade do resumo depende da qualidade do texto original.",
                "Terceiro exemplo de texto longo para demonstrar o funcionamento do sistema. Este documento contém informações variadas e serve como base para testar a capacidade de sumarização automática. É importante que o texto tenha contexto suficiente para gerar resumos meaningful e informativos."
            ],
            "Resumo": [
                "Resumo do primeiro texto com informações principais sobre o processo complexo.",
                "Resumo do segundo documento destacando aspectos técnicos e científicos importantes.",
                "Resumo conciso do terceiro exemplo mostrando as funcionalidades de sumarização."
            ],
            "Processo": [
                "Processo A",
                "Processo B", 
                "Processo C"
            ]
        }
        
        df = pd.DataFrame(exemplo_data)
        excel_file = "data/exemplo.xlsx"
        df.to_excel(excel_file, index=False)
        logger.info(f"Arquivo de exemplo criado: {excel_file}")
    
    # Executar pré-processamento
    try:
        logger.info(f"Processando arquivo: {excel_file}")
        
        result = process_excel_to_dataset(
            excel_file=excel_file,
            output_dir="data/processed",
            text_column="Texto",
            summary_column="Resumo",
            title_column="Processo",
            min_text_length=100,
            max_text_length=8000,
            min_summary_length=20,
            max_summary_length=1000,
            test_size=0.15,
            validation_size=0.10,
            clean_text=True,
            save_formats=["json", "parquet"]
        )
        
        if result["success"]:
            logger.info("✅ Pré-processamento concluído com sucesso!")
            logger.info(f"Total de exemplos: {result['dataset_info']['total_examples']}")
            logger.info(f"Splits criados: {list(result['dataset_info']['splits'].keys())}")
            logger.info(f"Arquivos salvos: {result['saved_files']}")
            
            # Mostrar estatísticas se disponíveis
            if 'statistics' in result['dataset_info']:
                stats = result['dataset_info']['statistics']
                logger.info("📊 Estatísticas do dataset:")
                logger.info(f"  - Comprimento médio do texto: {stats['text_length']['mean']:.1f}")
                logger.info(f"  - Comprimento médio do resumo: {stats['summary_length']['mean']:.1f}")
            
            return True
        else:
            logger.error("❌ Falha no pré-processamento")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro no pré-processamento: {e}")
        return False


def main():
    """Função principal para teste simples"""
    success = test_preprocessing_only()
    
    if success:
        logger.info("🎉 Pipeline de pré-processamento funcionando corretamente!")
        logger.info("Para executar o fine-tuning, configure os tokens necessários e execute main.py")
    else:
        logger.error("💥 Erro no pipeline de pré-processamento")


if __name__ == "__main__":
    main()
