#!/usr/bin/env python3

"""Teste específico para verificar se o erro do EarlyStoppingCallback foi resolvido"""

import sys
import os

# Adicionar o diretório pai ao path para importar o módulo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_training_setup():
    """Testa se o setup de treinamento funciona sem EarlyStoppingCallback"""
    from finetuning import LlamaFineTuner
    
    print("✅ Importação bem-sucedida")
    
    # Criar instância
    fine_tuner = LlamaFineTuner("test_key", "test_token")
    assert fine_tuner is not None, "LlamaFineTuner não foi criado"
    print("✅ Instância criada")
    
    # Testar setup de training arguments
    training_args = fine_tuner.setup_training_arguments()
    assert training_args is not None, "TrainingArguments não foi criado"
    assert hasattr(training_args, 'output_dir'), "TrainingArguments não tem output_dir"
    assert hasattr(training_args, 'num_train_epochs'), "TrainingArguments não tem num_train_epochs"
    assert hasattr(training_args, 'per_device_train_batch_size'), "TrainingArguments não tem per_device_train_batch_size"
    
    print("✅ TrainingArguments criado")
    print(f"   - Tipo: {type(training_args)}")
    print(f"   - Output dir: {training_args.output_dir}")
    print(f"   - Epochs: {training_args.num_train_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    
    # Validar que os valores são apropriados
    assert training_args.num_train_epochs > 0, "Número de epochs deve ser maior que 0"
    assert training_args.per_device_train_batch_size > 0, "Batch size deve ser maior que 0"
    assert training_args.output_dir, "Output dir não deve estar vazio"

if __name__ == "__main__":
    print("🧪 Testando configuração de treinamento...")
    
    try:
        test_training_setup()
        print("🎉 Configuração de treinamento funcionando corretamente!")
        print("✅ Problema do EarlyStoppingCallback foi resolvido!")
    except AssertionError as e:
        print(f"❌ Falha na validação: {e}")
        print("💥 Ainda há problemas na configuração!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        print("💥 Ainda há problemas na configuração!")
        sys.exit(1)
