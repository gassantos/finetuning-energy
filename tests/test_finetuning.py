#!/usr/bin/env python3

"""Teste simples para verificar se o módulo finetuning está funcionando"""

import sys
import os

# Adicionar o diretório pai ao path para importar o módulo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#!/usr/bin/env python3

"""Teste simples para verificar se o módulo finetuning está funcionando"""

import sys
import os

# Adicionar o diretório pai ao path para importar o módulo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import():
    """Testa a importação do módulo"""
    import finetuning
    from finetuning import LlamaFineTuner
    assert LlamaFineTuner is not None, "LlamaFineTuner não foi importado corretamente"
    print("✅ Importação do LlamaFineTuner bem-sucedida")

def test_initialization():
    """Testa a inicialização da classe"""
    from finetuning import LlamaFineTuner
    fine_tuner = LlamaFineTuner("test_key", "test_token")
    assert fine_tuner is not None, "LlamaFineTuner não foi inicializado"
    assert hasattr(fine_tuner, 'wandb_key'), "LlamaFineTuner não tem wandb_key"
    assert hasattr(fine_tuner, 'hf_token'), "LlamaFineTuner não tem hf_token"
    assert hasattr(fine_tuner, 'model_id'), "LlamaFineTuner não tem model_id"
    assert hasattr(fine_tuner, 'output_dir'), "LlamaFineTuner não tem output_dir"
    print("✅ Inicialização bem-sucedida")

def test_config_methods():
    """Testa os métodos de configuração"""
    from finetuning import LlamaFineTuner
    fine_tuner = LlamaFineTuner("test_key", "test_token")
    
    # Teste quantization config
    quant_config = fine_tuner.setup_quantization_config()
    assert quant_config is not None, "Configuração de quantização não foi criada"
    print(f"✅ Configuração de quantização: {type(quant_config)}")
    
    # Teste training arguments
    training_args = fine_tuner.setup_training_arguments()
    assert training_args is not None, "Argumentos de treinamento não foram criados"
    assert hasattr(training_args, 'output_dir'), "TrainingArguments não tem output_dir"
    assert hasattr(training_args, 'num_train_epochs'), "TrainingArguments não tem num_train_epochs"
    print(f"✅ Argumentos de treinamento: {type(training_args)}")

if __name__ == "__main__":
    print("🧪 Iniciando testes do módulo finetuning...")
    
    tests = [
        ("Importação", test_import),
        ("Inicialização", test_initialization),
        ("Métodos de Configuração", test_config_methods),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testando: {test_name}")
        try:
            test_func()
            passed += 1
            print(f"✅ {test_name} passou")
        except AssertionError as e:
            print(f"❌ Falha no teste {test_name}: {e}")
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram!")
        sys.exit(0)
    else:
        print("💥 Alguns testes falharam!")
        sys.exit(1)
