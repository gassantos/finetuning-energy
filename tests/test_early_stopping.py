"""
Testes específicos para Early Stopping functionality.

Este módulo testa a funcionalidade de early stopping e sua integração
com o sistema de treinamento.
"""

import tempfile


# Imports do transformers
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments


class TestEarlyStoppingCallback:
    """Testa funcionalidade do EarlyStoppingCallback."""

    def test_early_stopping_import(self):
        """Testa se EarlyStoppingCallback pode ser importado."""
        from transformers.trainer_callback import EarlyStoppingCallback

        assert EarlyStoppingCallback is not None

    def test_early_stopping_initialization(self):
        """Testa inicialização do EarlyStoppingCallback."""
        callback = EarlyStoppingCallback(early_stopping_patience=3)

        assert callback is not None
        assert hasattr(callback, "early_stopping_patience")
        assert callback.early_stopping_patience == 3

    def test_early_stopping_with_different_patience(self):
        """Testa EarlyStoppingCallback com diferentes valores de patience."""
        patience_values = [1, 3, 5, 10]

        for patience in patience_values:
            callback = EarlyStoppingCallback(early_stopping_patience=patience)
            assert callback.early_stopping_patience == patience

    def test_early_stopping_with_threshold(self):
        """Testa EarlyStoppingCallback com threshold."""
        callback = EarlyStoppingCallback(
            early_stopping_patience=3, early_stopping_threshold=0.01
        )

        assert callback.early_stopping_patience == 3
        assert callback.early_stopping_threshold == 0.01


class TestTrainingArgumentsWithEarlyStopping:
    """Testa TrainingArguments configurados para early stopping."""

    def test_training_args_for_early_stopping(self):
        """Testa configuração adequada de TrainingArguments para early stopping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            training_args = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=10,
                per_device_train_batch_size=2,
                eval_strategy="steps",
                eval_steps=50,
                save_strategy="steps",
                save_steps=50,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=True,
                logging_steps=10,
            )

            # Verificar configurações necessárias para early stopping
            assert training_args.eval_strategy != "no"
            assert training_args.metric_for_best_model is not None
            assert training_args.load_best_model_at_end is True
            assert training_args.eval_steps is not None


# Função legacy mantida para compatibilidade
def test_early_stopping():
    """Função legacy para compatibilidade com testes anteriores."""
    from transformers.trainer_callback import EarlyStoppingCallback
    from transformers.training_args import TrainingArguments

    with tempfile.TemporaryDirectory() as temp_dir:
        # Criar TrainingArguments com configuração adequada para EarlyStoppingCallback
        training_args = TrainingArguments(
            output_dir=temp_dir,
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
        assert (
            early_stopping.early_stopping_patience == 3
        ), "Patience não foi configurado corretamente"
        assert (
            training_args.metric_for_best_model == "eval_loss"
        ), "Métrica não foi configurada corretamente"
        assert (
            training_args.eval_strategy == "steps"
        ), "Estratégia de avaliação não foi configurada"
        assert (
            training_args.eval_steps == 10
        ), "Eval steps não foi configurado corretamente"
        assert (
            training_args.save_strategy == "steps"
        ), "Save strategy não foi configurada corretamente"
        assert (
            training_args.save_steps == 10
        ), "Save steps não foi configurado corretamente"
        assert (
            not training_args.greater_is_better
        ), "Greater is better não foi configurado corretamente"
        assert (
            training_args.load_best_model_at_end
        ), "Load best model at end não foi configurado"


if __name__ == "__main__":
    print("🧪 Testando EarlyStoppingCallback...")

    try:
        test_early_stopping()
        print("🎉 EarlyStoppingCallback funcionando corretamente!")
    except AssertionError as e:
        print(f"❌ Falha na validação: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
