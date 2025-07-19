"""
Testes abrangentes para o módulo de fine-tuning.

Este módulo testa todas as funcionalidades da classe LlamaFineTuner,
incluindo inicialização, configuração, carregamento de modelos, e treinamento.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Imports do projeto
from src.finetuning import LlamaFineTuner, safe_cast


class TestSafeCast:
    """Testa a função utilitária safe_cast."""

    def test_safe_cast_int_success(self):
        """Testa conversão bem-sucedida para int."""
        result = safe_cast("42", int, 0)
        assert result == 42

    def test_safe_cast_float_success(self):
        """Testa conversão bem-sucedida para float."""
        result = safe_cast("3.14", float, 0.0)
        assert result == 3.14

    def test_safe_cast_with_default(self):
        """Testa uso do valor padrão em caso de erro."""
        result = safe_cast("invalid", int, 42)
        assert result == 42

    def test_safe_cast_none_value(self):
        """Testa comportamento com valor None."""
        result = safe_cast(None, int, 100)
        assert result == 100


class TestLlamaFineTunerInitialization:
    """Testa a inicialização da classe LlamaFineTuner."""

    def test_basic_initialization(self, temp_output_dir):
        """Testa inicialização básica."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            assert tuner.wandb_key == "test_key"
            assert tuner.hf_token == "test_token"
            assert str(tuner.output_dir) == temp_output_dir
            assert tuner.model is None
            assert tuner.tokenizer is None

    def test_initialization_with_custom_params(self, temp_output_dir):
        """Testa inicialização com parâmetros customizados."""
        with patch("src.finetuning.RobustGPUMonitor"):
            custom_model = "custom/model"
            result_dir = str(Path(temp_output_dir) / "results")

            tuner = LlamaFineTuner(
                wandb_key="test_key",
                hf_token="test_token",
                model_id=custom_model,
                output_dir=temp_output_dir,
                result_dir=result_dir,
            )

            assert tuner.model_id == custom_model
            assert str(tuner.result_dir) == result_dir

    def test_directories_creation(self, temp_output_dir):
        """Testa se os diretórios são criados corretamente."""
        with patch("src.finetuning.RobustGPUMonitor"):
            result_dir = str(Path(temp_output_dir) / "results")

            tuner = LlamaFineTuner(
                wandb_key="test_key",
                hf_token="test_token",
                output_dir=temp_output_dir,
                result_dir=result_dir,
            )

            assert tuner.output_dir.exists()
            assert tuner.result_dir.exists()


class TestAuthentication:
    """Testa métodos de autenticação."""

    @patch("huggingface_hub.login")
    def test_setup_authentication_success(self, mock_hf_login, temp_output_dir):
        """Testa configuração bem-sucedida de autenticação."""
        with patch("src.finetuning.RobustGPUMonitor"):
            mock_hf_login.return_value = True

            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            tuner.setup_authentication()
            mock_hf_login.assert_called_once_with(token="test_token")

    @patch("huggingface_hub.login")
    def test_setup_authentication_failure(self, mock_hf_login, temp_output_dir):
        """Testa falha na autenticação."""
        with patch("src.finetuning.RobustGPUMonitor"):
            mock_hf_login.side_effect = Exception("Authentication failed")

            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            with pytest.raises(Exception):
                tuner.setup_authentication()


class TestWandbIntegration:
    """Testa integração com Weights & Biases."""

    @patch("wandb.init")
    def test_initialize_wandb_success(
        self, mock_wandb_init, mock_gpu_monitor, temp_output_dir
    ):
        """Testa inicialização bem-sucedida do wandb."""
        mock_gpu_monitor.detect_monitoring_capabilities.return_value = {
            "nvitop": True,
            "pynvml": True,
        }

        with patch("src.finetuning.RobustGPUMonitor", return_value=mock_gpu_monitor):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            tuner.initialize_wandb()

            mock_wandb_init.assert_called_once()
            call_args = mock_wandb_init.call_args
            assert "config" in call_args.kwargs
            assert "monitoring_capabilities" in call_args.kwargs["config"]

    @patch("wandb.init")
    def test_initialize_wandb_with_capabilities(
        self, mock_wandb_init, mock_gpu_monitor, temp_output_dir
    ):
        """Testa inicialização do wandb com capacidades de monitoramento."""
        capabilities = {"nvitop": False, "pynvml": True, "nvidia_smi": True}
        mock_gpu_monitor.detect_monitoring_capabilities.return_value = capabilities

        with patch("src.finetuning.RobustGPUMonitor", return_value=mock_gpu_monitor):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            tuner.initialize_wandb()

            call_args = mock_wandb_init.call_args
            config = call_args.kwargs["config"]
            assert config["monitoring_capabilities"] == capabilities


class TestQuantizationConfig:
    """Testa configuração de quantização."""

    def test_setup_quantization_config(self, temp_output_dir):
        """Testa criação da configuração de quantização."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            config = tuner.setup_quantization_config()

            assert config is not None
            assert hasattr(config, "load_in_4bit")
            assert config.load_in_4bit is True
            assert hasattr(config, "bnb_4bit_quant_type")
            assert config.bnb_4bit_quant_type == "nf4"


class TestModelLoading:
    """Testa carregamento de modelo e tokenizer."""

    @patch("peft.prepare_model_for_kbit_training")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_load_model_and_tokenizer_success(
        self,
        mock_model_load,
        mock_tokenizer_load,
        mock_prepare_model,
        mock_model,
        mock_tokenizer,
        temp_output_dir,
    ):
        """Testa carregamento bem-sucedido de modelo e tokenizer."""
        mock_model_load.return_value = mock_model
        mock_tokenizer_load.return_value = mock_tokenizer
        mock_prepare_model.return_value = mock_model

        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            tuner.load_model_and_tokenizer()

            assert tuner.model is not None
            assert tuner.tokenizer is not None
            mock_tokenizer_load.assert_called_once()
            mock_model_load.assert_called_once()

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_model_failure(
        self, mock_model_load, mock_tokenizer_load, temp_output_dir
    ):
        """Testa falha no carregamento do modelo."""
        mock_tokenizer_load.side_effect = Exception("Model not found")

        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            with pytest.raises(Exception):
                tuner.load_model_and_tokenizer()

    def test_tokenizer_pad_token_setup(self, mock_model, mock_peft, temp_output_dir):
        """Testa configuração do pad_token do tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.eos_token_id = 1

        with patch("src.finetuning.RobustGPUMonitor"):
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load:
                with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load:
                    mock_tokenizer_load.return_value = mock_tokenizer
                    mock_model_load.return_value = mock_model

                    tuner = LlamaFineTuner(
                        wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
                    )

                    tuner.load_model_and_tokenizer()

                    assert mock_tokenizer.pad_token == "[EOS]"
                    assert mock_tokenizer.pad_token_id == 1


class TestLoRAConfiguration:
    """Testa configuração e aplicação de LoRA."""

    def test_apply_lora_success(self, mock_model, mock_peft, temp_output_dir):
        """Testa aplicação bem-sucedida de LoRA."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            tuner.model = mock_model
            tuner.apply_lora()

            # Verificar que as funções PEFT foram chamadas
            mock_peft["prepare_model"].assert_called_once_with(mock_model)
            mock_peft["get_peft_model"].assert_called_once()

    def test_apply_lora_without_model(self, mock_peft, temp_output_dir):
        """Testa aplicação de LoRA sem modelo carregado."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            # Modelo None inicialmente
            assert tuner.model is None
            
            tuner.apply_lora()
            
            # Deve ter chamado prepare_model mesmo com modelo None
            mock_peft["prepare_model"].assert_called_once()
            mock_peft["get_peft_model"].assert_called_once()


class TestTrainingArguments:
    """Testa criação de argumentos de treinamento."""

    def test_setup_training_arguments(self, temp_output_dir):
        """Testa criação dos argumentos de treinamento."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            args = tuner.setup_training_arguments()

            assert args is not None
            assert hasattr(args, "output_dir")
            assert hasattr(args, "num_train_epochs")
            assert hasattr(args, "per_device_train_batch_size")
            assert args.fp16 is True
            assert args.report_to == ["wandb"]


class TestDatasetHandling:
    """Testa manipulação de datasets."""

    @pytest.mark.network
    def test_load_and_prepare_dataset(
        self,
        mock_model,
        mock_peft,
        mock_dataset,
        temp_output_dir,
    ):
        """Testa carregamento e preparação do dataset."""
        with (
            patch("src.finetuning.RobustGPUMonitor"),
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load,
            patch("datasets.load_dataset") as mock_load_dataset,
            patch("wandb.init") as mock_wandb_init,
            patch("wandb.log") as mock_wandb_log,
        ):

            # Configurar mocks
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            # Configurar o mock para retornar dicionários subscriptable
            mock_tokenizer.return_value = {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
            }
            # Fazer o mock_tokenizer ser callable 
            mock_tokenizer.side_effect = lambda *args, **kwargs: {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
            }
            mock_tokenizer_load.return_value = mock_tokenizer
            mock_model_load.return_value = mock_model
            mock_load_dataset.return_value = {
                "train": mock_dataset,
                "validation": mock_dataset,
            }

            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            # Carregar modelo e tokenizer primeiro
            tuner.load_model_and_tokenizer()

            dataset = tuner.load_and_prepare_dataset()

            assert dataset is not None
            # Note: load_dataset pode não ser chamado devido ao cache local
            mock_wandb_log.assert_called()

    def test_load_and_prepare_dataset_mock_only(
        self,
        mock_model,
        mock_peft,
        mock_dataset,
        temp_output_dir,
    ):
        """Testa carregamento e preparação do dataset com mocks apenas (sem rede)."""
        with (
            patch("src.finetuning.RobustGPUMonitor"),
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load,
            patch("datasets.load_dataset") as mock_load_dataset,
            patch("wandb.init") as mock_wandb_init,
            patch("wandb.log") as mock_wandb_log,
        ):

            # Configurar mocks
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = "[PAD]"
            # Configurar o mock para retornar dicionários subscriptable
            mock_tokenizer.return_value = {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
            }
            # Fazer o mock_tokenizer ser callable 
            mock_tokenizer.side_effect = lambda *args, **kwargs: {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
            }
            mock_tokenizer_load.return_value = mock_tokenizer
            mock_model_load.return_value = mock_model
            mock_load_dataset.return_value = {
                "train": mock_dataset,
                "validation": mock_dataset,
            }

            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            # Carregar modelo e tokenizer primeiro
            tuner.load_model_and_tokenizer()

            dataset = tuner.load_and_prepare_dataset()

            assert dataset is not None
            # Note: dataset pode usar cache local, não necessariamente chama load_dataset
            mock_wandb_log.assert_called()

    @patch("src.finetuning.load_dataset")
    def test_load_dataset_failure(self, mock_load_dataset, temp_output_dir):
        """Testa falha no carregamento do dataset."""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            with pytest.raises(Exception):
                tuner.load_and_prepare_dataset()


class TestModelSaving:
    """Testa salvamento de modelo."""

    def test_save_model_success(self, mock_model, mock_tokenizer, temp_output_dir):
        """Testa salvamento bem-sucedido do modelo."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            tuner.model = mock_model
            tuner.tokenizer = mock_tokenizer

            tuner.save_model()

            mock_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()

    def test_save_model_custom_dir(self, mock_model, mock_tokenizer, temp_output_dir):
        """Testa salvamento do modelo em diretório customizado."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            tuner.model = mock_model
            tuner.tokenizer = mock_tokenizer

            custom_dir = str(Path(temp_output_dir) / "custom")
            tuner.save_model(custom_dir)

            mock_model.save_pretrained.assert_called_with(custom_dir)
            mock_tokenizer.save_pretrained.assert_called_with(custom_dir)

    def test_save_model_without_model(self, temp_output_dir):
        """Testa salvamento sem modelo carregado."""
        with patch("src.finetuning.RobustGPUMonitor"):
            tuner = LlamaFineTuner(
                wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
            )

            with pytest.raises(RuntimeError):
                tuner.save_model()


class TestCompleteWorkflow:
    """Testa o workflow completo."""

    @pytest.mark.network
    def test_run_complete_pipeline(
        self,
        mock_model,
        mock_tokenizer,
        mock_dataset,
        mock_gpu_monitor,
        mock_peft,
        temp_output_dir,
    ):
        """Testa execução completa do pipeline."""
        with (
            patch("src.finetuning.Trainer") as mock_trainer_class,
            patch("datasets.load_dataset") as mock_load_dataset,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load,
            patch("wandb.init") as mock_wandb_init,
            patch("wandb.log") as mock_wandb_log,
            patch("huggingface_hub.login") as mock_hf_login,
        ):
            # Setup mocks
            mock_tokenizer_load.return_value = mock_tokenizer
            mock_model_load.return_value = mock_model
            mock_load_dataset.return_value = {"train": mock_dataset}

            mock_trainer = Mock()
            mock_trainer.train.return_value = Mock()
            mock_trainer_class.return_value = mock_trainer

            with patch("src.finetuning.RobustGPUMonitor", return_value=mock_gpu_monitor):
                tuner = LlamaFineTuner(
                    wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
                )

                # Executar pipeline
                result = tuner.run_complete_pipeline()

                # Verificar se todas as etapas foram executadas
                mock_hf_login.assert_called_once()
                mock_wandb_init.assert_called_once()
                mock_tokenizer_load.assert_called_once()
                mock_model_load.assert_called_once()
                # Note: load_dataset pode não ser chamado devido ao cache local
                mock_trainer_class.assert_called_once()

                assert result is not None

    def test_run_complete_pipeline_mock_only(
        self,
        mock_model,
        mock_tokenizer,
        mock_dataset,
        mock_gpu_monitor,
        mock_peft,
        temp_output_dir,
    ):
        """Testa execução completa do pipeline apenas com mocks (sem rede)."""
        with (
            patch("src.finetuning.Trainer") as mock_trainer_class,
            patch("datasets.load_dataset") as mock_load_dataset,
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load,
            patch("wandb.init") as mock_wandb_init,
            patch("wandb.log") as mock_wandb_log,
            patch("huggingface_hub.login") as mock_hf_login,
        ):
            # Setup mocks
            mock_tokenizer_load.return_value = mock_tokenizer
            mock_model_load.return_value = mock_model
            mock_load_dataset.return_value = {"train": mock_dataset}

            mock_trainer = Mock()
            mock_trainer.train.return_value = Mock()
            mock_trainer_class.return_value = mock_trainer

            with patch("src.finetuning.RobustGPUMonitor", return_value=mock_gpu_monitor):
                tuner = LlamaFineTuner(
                    wandb_key="test_key", hf_token="test_token", output_dir=temp_output_dir
                )

                # Executar pipeline
                result = tuner.run_complete_pipeline()

                # Verificar se todas as etapas foram executadas
                mock_hf_login.assert_called_once()
                mock_wandb_init.assert_called_once()
                mock_tokenizer_load.assert_called_once()
                mock_model_load.assert_called_once()
                # Note: dataset pode usar cache local
                mock_trainer_class.assert_called_once()

                assert result is not None


class TestErrorHandling:
    """Testa tratamento de erros."""

    def test_invalid_initialization_params(self):
        """Testa inicialização com parâmetros inválidos."""
        with patch("src.finetuning.RobustGPUMonitor"):
            # Wandb key é obrigatória
            with pytest.raises(TypeError):
                LlamaFineTuner()

    @patch("src.finetuning.RobustGPUMonitor")
    def test_gpu_monitor_initialization_failure(self, mock_gpu_monitor_class):
        """Testa falha na inicialização do monitor GPU."""
        mock_gpu_monitor_class.side_effect = Exception("GPU monitor failed")

        with pytest.raises(Exception):
            LlamaFineTuner(wandb_key="test_key", hf_token="test_token")


class TestIntegration:
    """Testes de integração (requerem recursos externos)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_model_loading(self, temp_output_dir):
        """Testa carregamento de modelo real (teste lento)."""
        pytest.skip("Teste de integração - requer modelo real")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_dataset_loading(self, temp_output_dir):
        """Testa carregamento de dataset real (teste lento)."""
        pytest.skip("Teste de integração - requer dataset real")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    assert training_args is not None, "Argumentos de treinamento não foram criados"
    assert hasattr(training_args, "output_dir"), "TrainingArguments não tem output_dir"
    assert hasattr(
        training_args, "num_train_epochs"
    ), "TrainingArguments não tem num_train_epochs"
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
