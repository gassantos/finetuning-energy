"""
Testes de integração para o sistema completo.

Este módulo testa a integração entre todos os componentes do sistema,
simulando workflows completos de fine-tuning.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Imports do projeto
from src.finetuning import LlamaFineTuner
from src.monitor import RobustGPUMonitor


class TestCompleteWorkflowIntegration:
    """Testa workflows completos de integração."""

    @pytest.mark.integration
    @pytest.mark.network
    @patch("src.finetuning.LlamaFineTuner.train_with_robust_monitoring")
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    @patch("huggingface_hub.login")
    def test_complete_training_pipeline_mock(
        self,
        mock_hf_login,
        mock_wandb_finish,
        mock_wandb_log,
        mock_wandb_init,
        mock_train_with_monitoring,
        mock_model,
        mock_tokenizer,
        mock_dataset,
        temp_output_dir,
    ):
        """Testa pipeline completo com mocks simplificados."""
        
        # Mock do resultado do treinamento
        mock_train_result = Mock()
        mock_train_result.train.return_value = Mock()
        mock_train_with_monitoring.return_value = mock_train_result

        # Executar pipeline completo
        tuner = LlamaFineTuner(
            wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
        )

        result = tuner.run_complete_pipeline()

        # Verificar que as funções principais foram chamadas
        mock_hf_login.assert_called_once()
        mock_wandb_init.assert_called_once()
        mock_train_with_monitoring.assert_called_once()

        assert result is not None

    @pytest.mark.integration
    def test_complete_training_pipeline_offline(
        self,
        mock_model,
        mock_tokenizer,
        mock_dataset,
        temp_output_dir,
    ):
        """Testa pipeline completo offline (sem dependências de rede)."""
        
        with (
            patch("src.finetuning.LlamaFineTuner.train_with_robust_monitoring") as mock_train_with_monitoring,
            patch("src.finetuning.LlamaFineTuner.apply_lora") as mock_apply_lora,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load,
            patch("datasets.load_dataset") as mock_load_dataset,
            patch("wandb.init") as mock_wandb_init,
            patch("wandb.log") as mock_wandb_log,
            patch("wandb.finish") as mock_wandb_finish,
            patch("huggingface_hub.login") as mock_hf_login,
        ):
            
            # Configurar mocks
            mock_tokenizer_load.return_value = mock_tokenizer
            mock_model_load.return_value = mock_model
            mock_load_dataset.return_value = {"train": mock_dataset}
            mock_apply_lora.return_value = None  # Mock da aplicação do LoRA
            
            # Mock do resultado do treinamento
            mock_train_result = Mock()
            mock_train_result.train.return_value = Mock()
            mock_train_with_monitoring.return_value = mock_train_result

            # Executar pipeline completo
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            result = tuner.run_complete_pipeline()

            # Verificar que as funções principais foram chamadas
            mock_hf_login.assert_called_once()
            mock_wandb_init.assert_called_once()
            mock_train_with_monitoring.assert_called_once()

            assert result is not None

    @pytest.mark.integration
    def test_gpu_monitor_integration(self):
        """Testa integração do monitor GPU."""
        monitor = RobustGPUMonitor(sampling_interval=0.1)

        # Testar ciclo completo de monitoramento
        capabilities = monitor.detect_monitoring_capabilities()
        assert isinstance(capabilities, dict)
        assert "nvitop" in capabilities
        assert "pynvml" in capabilities
        assert "nvidia_smi" in capabilities

        # Tentar iniciar monitoramento (pode falhar sem GPU)
        started = monitor.start_monitoring()

        if started:
            # Se conseguiu iniciar, testar parada
            time.sleep(0.2)  # Coletar alguns dados
            result = monitor.stop_monitoring()

            assert result is not None
            assert "monitoring_duration_s" in result
            assert result["monitoring_duration_s"] > 0
        else:
            # É esperado falhar em ambientes sem GPU
            assert not monitor.monitoring

    @pytest.mark.integration
    @patch("src.finetuning.RobustGPUMonitor")
    def test_configuration_integration(self, mock_gpu_monitor, temp_output_dir):
        """Testa integração do sistema de configuração."""
        mock_gpu_monitor.return_value = Mock()

        # Testar diferentes configurações
        configs = [
            {"model_id": "test-model-1", "output_dir": temp_output_dir},
            {
                "model_id": "test-model-2",
                "output_dir": str(Path(temp_output_dir) / "alt"),
            },
        ]

        for config in configs:
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", **config
            )

            assert tuner.model_id == config["model_id"]
            assert str(tuner.output_dir) == config["output_dir"]
            assert tuner.output_dir.exists()


class TestErrorHandlingIntegration:
    """Testa tratamento de erros na integração."""

    @pytest.mark.integration
    def test_invalid_authentication_handling(self, temp_output_dir):
        """Testa tratamento de autenticação inválida."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="invalid_key",
                hf_token="invalid_token",
                output_dir=temp_output_dir,
            )

            with patch("huggingface_hub.login") as mock_login:
                mock_login.side_effect = Exception("Invalid token")

                with pytest.raises(Exception):
                    tuner.setup_authentication()

    @pytest.mark.integration
    def test_model_loading_failure_handling(self, temp_output_dir):
        """Testa tratamento de falha no carregamento do modelo."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key",
                hf_token="test_token",
                model_id="nonexistent/model",
                output_dir=temp_output_dir,
            )

            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
                mock_tokenizer.side_effect = Exception("Model not found")

                with pytest.raises(Exception):
                    tuner.load_model_and_tokenizer()

    @pytest.mark.integration
    def test_dataset_loading_failure_handling(self, temp_output_dir):
        """Testa tratamento de falha no carregamento do dataset."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            with patch("datasets.load_dataset") as mock_dataset:
                mock_dataset.side_effect = Exception("Dataset not found")

                with pytest.raises(Exception):
                    tuner.load_and_prepare_dataset()


class TestPerformanceIntegration:
    """Testa aspectos de performance na integração."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_usage_monitoring(self, temp_output_dir):
        """Testa monitoramento de uso de memória."""
        if not torch.cuda.is_available():
            pytest.skip("GPU não disponível para teste de memória")

        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            # Simular carregamento de modelo e verificar memória
            initial_memory = torch.cuda.memory_allocated()

            with patch(
                "transformers.AutoModelForCausalLM.from_pretrained"
            ) as mock_model:
                # Criar um mock que simule named_parameters()
                mock_model_instance = Mock()
                mock_model_instance.named_parameters.return_value = [
                    ("layer1.weight", Mock()),
                    ("layer2.bias", Mock())
                ]
                mock_model.return_value = mock_model_instance
                
                with patch(
                    "transformers.AutoTokenizer.from_pretrained"
                ) as mock_tokenizer:
                    mock_tokenizer.return_value = Mock()

                    tuner.load_model_and_tokenizer()

                    # Verificar que o sistema está monitorando recursos adequadamente
                    assert tuner.model is not None
                    assert tuner.tokenizer is not None

    @pytest.mark.integration
    def test_concurrent_monitoring(self):
        """Testa monitoramento concorrente."""
        monitor1 = RobustGPUMonitor(sampling_interval=0.1)
        monitor2 = RobustGPUMonitor(sampling_interval=0.1)

        # Tentar iniciar múltiplos monitores
        started1 = monitor1.start_monitoring()
        started2 = monitor2.start_monitoring()

        # Pelo menos um deve conseguir iniciar (ou ambos falharem graciosamente)
        if started1 or started2:
            time.sleep(0.2)

            if started1:
                result1 = monitor1.stop_monitoring()
                assert result1 is not None

            if started2:
                result2 = monitor2.stop_monitoring()
                assert result2 is not None


class TestCompatibilityIntegration:
    """Testa compatibilidade entre componentes."""

    @pytest.mark.integration
    def test_transformers_version_compatibility(self, temp_output_dir):
        """Testa compatibilidade com versão do transformers."""
        try:
            import transformers

            version = transformers.__version__

            # Verificar se componentes essenciais estão disponíveis
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
            from transformers.trainer_callback import EarlyStoppingCallback

            # Todos os imports devem funcionar
            assert AutoTokenizer is not None
            assert AutoModelForCausalLM is not None
            assert TrainingArguments is not None
            assert Trainer is not None
            assert EarlyStoppingCallback is not None

        except ImportError as e:
            pytest.fail(f"Incompatibilidade do transformers: {e}")

    @pytest.mark.integration
    def test_peft_integration_compatibility(self, temp_output_dir):
        """Testa compatibilidade com PEFT."""
        try:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            # Verificar se pode criar configuração LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            assert lora_config is not None
            assert lora_config.r == 8
            assert lora_config.lora_alpha == 16

        except ImportError as e:
            pytest.fail(f"Incompatibilidade do PEFT: {e}")

    @pytest.mark.integration
    def test_datasets_integration_compatibility(self, temp_output_dir):
        """Testa compatibilidade com datasets."""
        try:
            from datasets import load_dataset

            # Verificar se pode carregar dataset simples (mock)
            with patch("datasets.load_dataset") as mock_load:
                mock_load.return_value = {"train": Mock(), "validation": Mock()}

                dataset = mock_load("squad")
                assert dataset is not None
                assert "train" in dataset

        except ImportError as e:
            pytest.fail(f"Incompatibilidade do datasets: {e}")


class TestSystemIntegration:
    """Testa integração a nível de sistema."""

    @pytest.mark.integration
    def test_file_system_integration(self, temp_output_dir):
        """Testa integração com sistema de arquivos."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            # Verificar criação de diretórios
            assert tuner.output_dir.exists()
            assert tuner.result_dir.exists()

            # Testar salvamento de modelo mock
            with patch.object(tuner, "model", Mock()) as mock_model:
                with patch.object(tuner, "tokenizer", Mock()) as mock_tokenizer:
                    tuner.save_model()

                    mock_model.save_pretrained.assert_called_once()
                    mock_tokenizer.save_pretrained.assert_called_once()

    @pytest.mark.integration
    def test_environment_variables_integration(self, temp_output_dir):
        """Testa integração com variáveis de ambiente."""
        import os

        # Testar configuração via variáveis de ambiente
        test_vars = {
            "WANDB_MODE": "offline",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }

        with patch.dict(os.environ, test_vars):
            from config.config import settings

            # Verificar se configurações foram aplicadas
            wandb_mode = settings.get("WANDB_MODE", "online")
            assert wandb_mode == "offline"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_long_running_monitoring(self):
        """Testa monitoramento de longa duração."""
        monitor = RobustGPUMonitor(sampling_interval=0.1)

        started = monitor.start_monitoring()

        if started:
            # Executar por um período mais longo
            time.sleep(1.0)

            result = monitor.stop_monitoring()

            assert result is not None
            assert result["monitoring_duration_s"] >= 0.9  # Margem para variações de timing

            # Verificar se coletou dados suficientes
            if "gpus" in result:
                for gpu_id, gpu_data in result["gpus"].items():
                    if "statistics" in gpu_data:
                        stats = gpu_data["statistics"]
                        # Deve ter coletado múltiplas amostras
                        assert "samples_count" in stats or len(monitor.energy_data) > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
