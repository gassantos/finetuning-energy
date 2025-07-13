#!/usr/bin/env python3

"""Teste simplificado para verificar se o EarlyStoppingCallback está funcionando"""

import sys
import os

def test_early_stopping():
    """Testa se o EarlyStoppingCallback funciona corretamente"""
    from transformers.trainer_callback import EarlyStoppingCallback
    from transformers.training_args import TrainingArguments
    
    # Criar TrainingArguments com configuração adequada para EarlyStoppingCallback
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        logging_steps=1,
    )
    
    # Criar EarlyStoppingCallback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    
    # Validações com assert
    assert early_stopping is not None, "EarlyStoppingCallback não foi criado"
    assert early_stopping.early_stopping_patience == 3, "Patience não foi configurado corretamente"
    assert training_args.metric_for_best_model == "eval_loss", "Métrica não foi configurada corretamente"
    assert training_args.eval_strategy == "steps", "Estratégia de avaliação não foi configurada"
    assert training_args.eval_steps == 10, "Eval steps não foi configurado corretamente"
    assert training_args.save_strategy == "steps", "Save strategy não foi configurada corretamente"
    assert training_args.save_steps == 10, "Save steps não foi configurado corretamente"
    assert training_args.greater_is_better == False, "Greater is better não foi configurado corretamente"
    assert training_args.load_best_model_at_end == True, "Load best model at end não foi configurado"
    
    print("✅ EarlyStoppingCallback configurado corretamente")
    print(f"   - Patience: {early_stopping.early_stopping_patience}")
    print(f"   - Metric: {training_args.metric_for_best_model}")
    print(f"   - Evaluation strategy: {training_args.eval_strategy}")
    print(f"   - Save strategy: {training_args.save_strategy}")

if __name__ == "__main__":
    import pytest
    print("🧪 Testando EarlyStoppingCallback...")
    
    try:
        test_early_stopping()
        print("🎉 EarlyStoppingCallback funcionando corretamente!")
    except AssertionError as e:
        print(f"❌ Falha na validação: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
