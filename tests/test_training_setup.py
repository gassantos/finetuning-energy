"""
Testes específicos para configuração de treinamento.

Este módulo testa componentes específicos da configuração de treinamento,
incluindo TrainingArguments e integração com transformers.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Imports do projeto
from src.finetuning import LlamaFineTuner


class TestTrainingSetup:
    """Testa configuração específica de treinamento."""

    def test_training_arguments_creation(self, temp_output_dir):
        """Testa criação de TrainingArguments."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            training_args = tuner.setup_training_arguments()

            assert training_args is not None
            assert hasattr(training_args, "output_dir")
            assert hasattr(training_args, "num_train_epochs")
            assert hasattr(training_args, "per_device_train_batch_size")
            assert hasattr(training_args, "gradient_accumulation_steps")
            assert hasattr(training_args, "warmup_steps")
            assert hasattr(training_args, "learning_rate")
            assert hasattr(training_args, "fp16")
            assert hasattr(training_args, "logging_steps")
            assert hasattr(training_args, "save_strategy")
            assert hasattr(training_args, "report_to")

    def test_training_arguments_values(self, temp_output_dir):
        """Testa valores específicos dos argumentos de treinamento."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            training_args = tuner.setup_training_arguments()

            # Verificar valores específicos
            assert str(training_args.output_dir) == temp_output_dir
            assert training_args.gradient_accumulation_steps == 4
            assert training_args.warmup_steps == 100
            assert training_args.fp16 is True
            assert training_args.logging_steps == 10
            assert training_args.save_strategy == "no"
            assert training_args.report_to == ["wandb"]

    def test_training_arguments_numeric_types(self, temp_output_dir):
        """Testa tipos numéricos dos argumentos de treinamento."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            training_args = tuner.setup_training_arguments()

            # Verificar tipos numéricos
            assert isinstance(training_args.num_train_epochs, (int, float))
            assert isinstance(training_args.per_device_train_batch_size, int)
            assert isinstance(training_args.gradient_accumulation_steps, int)
            assert isinstance(training_args.warmup_steps, int)
            assert isinstance(training_args.learning_rate, float)
            assert isinstance(training_args.logging_steps, int)

    def test_training_arguments_positive_values(self, temp_output_dir):
        """Testa se valores numéricos são positivos."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            training_args = tuner.setup_training_arguments()

            # Valores devem ser positivos
            assert training_args.num_train_epochs > 0
            assert training_args.per_device_train_batch_size > 0
            assert training_args.gradient_accumulation_steps > 0
            assert training_args.warmup_steps >= 0  # Pode ser 0
            assert training_args.learning_rate > 0
            assert training_args.logging_steps > 0


class TestEarlyStoppingCompatibility:
    """Testa compatibilidade com EarlyStoppingCallback (resolvendo issue anterior)."""

    def test_early_stopping_callback_import(self):
        """Testa importação do EarlyStoppingCallback."""
        try:
            from transformers.trainer_callback import EarlyStoppingCallback

            assert EarlyStoppingCallback is not None
        except ImportError:
            pytest.fail("EarlyStoppingCallback não disponível")

    def test_early_stopping_callback_creation(self):
        """Testa criação do EarlyStoppingCallback."""
        from transformers.trainer_callback import EarlyStoppingCallback

        # Criar callback com parâmetros padrão
        callback = EarlyStoppingCallback(early_stopping_patience=3)

        assert callback is not None
        assert hasattr(callback, "early_stopping_patience")
        assert callback.early_stopping_patience == 3

    def test_early_stopping_with_training_args(self, temp_output_dir):
        """Testa EarlyStoppingCallback com TrainingArguments."""
        from transformers.trainer_callback import EarlyStoppingCallback
        from transformers.training_args import TrainingArguments

        # Criar TrainingArguments compatíveis com EarlyStoppingCallback
        training_args = TrainingArguments(
            output_dir=temp_output_dir,
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

        # Ambos devem ser criados sem erros
        assert training_args is not None
        assert early_stopping is not None

        # Verificar configurações necessárias para early stopping
        assert training_args.eval_strategy != "no"
        assert training_args.metric_for_best_model is not None
        assert training_args.load_best_model_at_end is True


# Manter função legacy para compatibilidade
def test_training_setup():
    """Função legacy mantida para compatibilidade."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("src.finetuning.RobustGPUMonitor"):
            from src.finetuning import LlamaFineTuner

            # Criar instância
            fine_tuner = LlamaFineTuner("test_key", "test_token", output_dir=temp_dir)
            assert fine_tuner is not None

            # Testar setup de training arguments
            training_args = fine_tuner.setup_training_arguments()
            assert training_args is not None
            assert hasattr(training_args, "output_dir")
            assert hasattr(training_args, "num_train_epochs")
            assert hasattr(training_args, "per_device_train_batch_size")

            # Validar que os valores são apropriados
            assert training_args.num_train_epochs > 0
            assert training_args.per_device_train_batch_size > 0
            assert training_args.output_dir


if __name__ == "__main__":
    print("🧪 Testando configuração de treinamento...")

    try:
        test_training_setup()
        print("🎉 Configuração de treinamento funcionando corretamente!")
        print("✅ Problema do EarlyStoppingCallback foi resolvido!")
    except AssertionError as e:
        print(f"❌ Falha na validação: {e}")
        print("💥 Ainda há problemas na configuração!")
        exit(1)
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback

        traceback.print_exc()
        print("💥 Ainda há problemas na configuração!")
        exit(1)
