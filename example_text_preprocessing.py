#!/usr/bin/env python3
"""
Exemplo de uso do módulo avançado de pré-processamento de texto

Este script demonstra como usar o módulo text_preprocessing_advanced
para processar arquivos Excel e gerar datasets estruturados para sumarização.
"""

import sys
from pathlib import Path
import json

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from text_preprocessing_advanced import (
    AdvancedTextProcessor,
    TextPreprocessingConfig,
    process_excel_to_dataset
)


def main():
    """Função principal de demonstração"""
    print("=== Demonstração do Módulo Avançado de Pré-processamento ===\n")
    
    # 1. Método direto - função de conveniência
    print("1. Processamento direto (método recomendado):")
    try:
        result = process_excel_to_dataset(
            excel_file="data/dataset.xlsx",
            output_dir="data/processed"
        )
        
        print("✅ Processamento concluído com sucesso!")
        print(f"Arquivos salvos: {result['saved_files']}")
        print(f"Validação: {'✅' if result['validation']['valid'] else '❌'}")
        
        # Mostrar estatísticas
        info = result['dataset_info']
        print(f"\nTotal de exemplos: {info['total_examples']}")
        print("Splits criados:")
        for split_name, split_info in info['splits'].items():
            print(f"  • {split_name}: {split_info['num_examples']} exemplos")
        
    except Exception as e:
        print(f"❌ Erro no processamento direto: {e}")
        return
    
    print("\n" + "="*50 + "\n")
    
    # 2. Método manual - controle total
    print("2. Processamento manual (controle avançado):")
    try:
        # Configuração customizada
        config = TextPreprocessingConfig(
            text_column="Texto",
            summary_column="Resumo",
            title_column="Processo",
            min_text_length=100,  # Filtro mais restritivo
            max_text_length=4000,
            test_size=0.15,       # 15% para teste
            validation_size=0.1,  # 10% para validação
            additional_columns=["Legislacao", "Pareceres"],  # Campos extras
            save_formats=['json', 'parquet']
        )
        
        # Criar processador
        processor = AdvancedTextProcessor(config)
        
        # Pipeline passo a passo
        print("  • Carregando dados...")
        processor.load_excel("data/dataset.xlsx")
        
        print("  • Pré-processando texto...")
        processed_data = processor.preprocess_data()
        print(f"    Registros válidos: {len(processed_data['text'])}")
        
        print("  • Criando splits...")
        dataset_dict = processor.create_dataset_splits()
        
        print("  • Validando formato...")
        validation = processor.validate_dataset_format()
        
        print("  • Salvando dataset...")
        saved_files = processor.save_dataset()
        
        print("✅ Processamento manual concluído!")
        print(f"Validação: {'✅' if validation['valid'] else '❌'}")
        print(f"Arquivos: {saved_files}")
        
        # Mostrar exemplo de dados processados
        print("\nExemplo de registro processado:")
        sample = dataset_dict['train'][0]
        print(f"  Título: {sample['title'][:60]}...")
        print(f"  Texto: {sample['text'][:100]}...")
        print(f"  Resumo: {sample['summary'][:100]}...")
        
    except Exception as e:
        print(f"❌ Erro no processamento manual: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50 + "\n")
    
    # 3. Verificar compatibilidade com HuggingFace
    print("3. Teste de compatibilidade com HuggingFace:")
    try:
        from datasets import load_from_disk
        
        # Tentar carregar dataset salvo
        dataset_path = Path("data/processed/dataset_structured_format")
        if dataset_path.exists():
            dataset = load_from_disk(str(dataset_path))
            print("✅ Dataset carregado com datasets.load_from_disk()")
            print(f"Splits disponíveis: {list(dataset.keys())}")
            print(f"Features: {list(dataset['train'].features.keys())}")
            
            # Verificar compatibilidade com formato de sumarização
            required_features = {'text', 'summary', 'title'}
            available_features = set(dataset['train'].features.keys())
            
            if required_features.issubset(available_features):
                print("✅ Compatível com formato de sumarização")
            else:
                missing = required_features - available_features
                print(f"❌ Features faltando para sumarização: {missing}")
        else:
            print("❌ Dataset não encontrado (execute o processamento primeiro)")
            
    except ImportError:
        print("❌ Biblioteca 'datasets' não disponível")
    except Exception as e:
        print(f"❌ Erro na verificação: {e}")
    
    print("\n=== Demonstração concluída ===")


if __name__ == "__main__":
    main()
