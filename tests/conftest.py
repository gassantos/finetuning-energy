"""
Configuração e utilitários para testes.

Este módulo fornece fixtures, mocks e utilitários comuns para os testes.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configurações de teste
TEST_CONFIG = {
    "model_id": "microsoft/DialoGPT-small",  # Modelo pequeno para testes
    "dataset_name": "squad",
    "max_length": 128,
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": 5e-5,
    "output_dir": "./test_output",
    "wandb_project": "test-project",
}


class MockModel:
    """Mock robusto para modelo do transformers compatível com PEFT/LoRA."""

    def __init__(self):
        self.config = Mock()
        self.config.vocab_size = 1000
        self.config.hidden_size = 768
        self.config.num_attention_heads = 12
        self.config.num_hidden_layers = 6
        self.config.model_type = "llama"
        self.config.torch_dtype = torch.float16
        self.pad_token = None
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.device = "cpu"
        self.training = True
        
        # Atributos necessários para PEFT
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        
        # Atributos PEFT (podem ser definidos dinamicamente)
        self.base_model = None
        self.peft_config = None
        
        # Criar estrutura de módulos simulada
        self._setup_mock_modules()

    def _setup_mock_modules(self):
        """Configura estrutura de módulos para compatibilidade com PEFT."""
        from torch import nn
        
        # Simular estrutura típica de um modelo Llama
        self._modules = {
            "model": Mock(),
            "model.embed_tokens": nn.Embedding(1000, 768),
            "model.layers": nn.ModuleList(),
        }
        
        # Adicionar algumas camadas simuladas
        for i in range(6):  # 6 layers
            layer_name = f"model.layers.{i}"
            self._modules[f"{layer_name}.self_attn.q_proj"] = nn.Linear(768, 768)
            self._modules[f"{layer_name}.self_attn.k_proj"] = nn.Linear(768, 768)
            self._modules[f"{layer_name}.self_attn.v_proj"] = nn.Linear(768, 768)
            self._modules[f"{layer_name}.self_attn.o_proj"] = nn.Linear(768, 768)
            self._modules[f"{layer_name}.mlp.gate_proj"] = nn.Linear(768, 2048)
            self._modules[f"{layer_name}.mlp.up_proj"] = nn.Linear(768, 2048)
            self._modules[f"{layer_name}.mlp.down_proj"] = nn.Linear(2048, 768)

    def parameters(self):
        """Mock de parâmetros do modelo."""
        return [torch.randn(100, 100, requires_grad=True) for _ in range(10)]

    def named_parameters(self):
        """Mock para named_parameters compatível com PEFT."""
        for name in [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight", 
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
        ]:
            yield name, torch.zeros(768, 768, requires_grad=True)

    def modules(self):
        """Mock para modules() usado pelo PEFT."""
        return list(self._modules.values())

    def named_modules(self):
        """Mock para named_modules() compatível com PEFT."""
        for name, module in self._modules.items():
            yield name, module
    
    def get_submodule(self, target):
        """Mock para get_submodule() necessário para PEFT."""
        if target in self._modules:
            return self._modules[target]
        
        # Simular navegação hierárquica
        parts = target.split(".")
        current = self
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                # Retornar um mock genérico se não encontrar
                return Mock()
        return current
    
    def get_parameter(self, target):
        """Mock para get_parameter() usado pelo PEFT."""
        if target in self._parameters:
            return self._parameters[target]
        return torch.zeros(768, 768, requires_grad=True)

    def save_pretrained(self, path):
        """Mock de salvamento."""
        Path(path).mkdir(parents=True, exist_ok=True)
        return True

    def to(self, device):
        """Mock de movimento para device."""
        self.device = device
        return self

    def train(self, mode=True):
        """Mock de modo de treinamento."""
        self.training = mode
        return self

    def eval(self):
        """Mock de modo de avaliação."""
        self.training = False
        return self
    
    def state_dict(self):
        """Mock para state_dict()."""
        return {
            "model.embed_tokens.weight": torch.randn(1000, 768),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """Mock para load_state_dict()."""
        return Mock()
    
    def forward(self, *args, **kwargs):
        """Mock para forward pass."""
        batch_size = args[0].shape[0] if args else 1
        seq_len = args[0].shape[1] if args else 10
        return Mock(
            logits=torch.randn(batch_size, seq_len, 1000),
            loss=torch.tensor(0.5)
        )

    def __setattr__(self, name, value):
        """Permite definir atributos dinamicamente."""
        super().__setattr__(name, value)


class MockTokenizer:
    """Mock básico para tokenizer."""

    def __init__(self):
        self.vocab_size = 1000
        self.pad_token = None
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.model_max_length = 2048

    def save_pretrained(self, path):
        """Mock de salvamento."""
        Path(path).mkdir(parents=True, exist_ok=True)
        return True

    def __call__(self, text, **kwargs):
        """Mock de tokenização."""
        # Retornar listas simples em vez de tensors para compatibilidade com .copy()
        input_ids = list(range(10))  # IDs sequenciais para simplicidade
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


class MockDataset:
    """Mock básico para dataset."""

    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (10,)),
            "attention_mask": torch.ones(10),
            "labels": torch.randint(0, 1000, (10,)),
        }

    def map(self, function, **kwargs):
        """Mock de mapeamento."""
        return self

    def filter(self, function, **kwargs):
        """Mock de filtro."""
        return self


class MockGPUMonitor:
    """Mock para monitor de GPU."""

    def __init__(self, sampling_interval=1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.capabilities = {"nvitop": True, "pynvml": True, "nvidia_smi": True}
        # Usar Mock para métodos que precisam de return_value
        self.detect_monitoring_capabilities = Mock(return_value=self.capabilities)

    def start_monitoring(self):
        """Mock de início de monitoramento."""
        self.monitoring = True
        return True

    def stop_monitoring(self):
        """Mock de parada de monitoramento."""
        self.monitoring = False
        return {
            "monitoring_duration_s": 60,
            "total_samples": 100,
            "gpus": {
                "gpu0": {
                    "name": "Mock GPU",
                    "statistics": {
                        "power_avg_w": 150.0,
                        "energy_consumed_kwh": 0.0025,
                        "temperature_avg_c": 65.0,
                        "utilization_avg_percent": 80.0,
                    }
                }
            },
            "monitoring_method": "mock",
        }

    def get_current_stats(self):
        """Mock de estatísticas atuais."""
        return {
            "gpus": {
                "gpu0": {
                    "power_w": 150.0,
                    "temperature_c": 65.0,
                    "utilization_percent": 80.0,
                    "memory_used_mb": 4096,
                    "memory_total_mb": 8192,
                }
            },
            "system": {"total_power_w": 150.0},
        }


@pytest.fixture
def mock_model():
    """Fixture para modelo mock."""
    model = MockModel()
    model.save_pretrained = Mock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Fixture para tokenizer mock."""
    tokenizer = MockTokenizer()
    tokenizer.save_pretrained = Mock()
    return tokenizer


@pytest.fixture
def mock_dataset():
    """Fixture para dataset mock."""
    return MockDataset()


@pytest.fixture
def mock_gpu_monitor():
    """Fixture para monitor de GPU mock."""
    return MockGPUMonitor()


@pytest.fixture
def test_config():
    """Fixture para configuração de teste."""
    return TEST_CONFIG.copy()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture para diretório temporário."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)


@pytest.fixture
def mock_wandb():
    """Fixture para mock do wandb."""
    with (
        patch("wandb.init") as mock_init,
        patch("wandb.log") as mock_log,
        patch("wandb.finish") as mock_finish,
    ):

        mock_init.return_value = Mock()
        yield {"init": mock_init, "log": mock_log, "finish": mock_finish}


@pytest.fixture
def mock_huggingface():
    """Fixture para mock das funcionalidades do Hugging Face."""
    with patch("huggingface_hub.login") as mock_login:
        mock_login.return_value = True
        yield mock_login


@pytest.fixture
def mock_transformers():
    """Fixture para mock dos componentes do transformers."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model,
        patch("transformers.TrainingArguments") as mock_args,
        patch("transformers.Trainer") as mock_trainer,
    ):

        mock_tokenizer.return_value = MockTokenizer()
        mock_model.return_value = MockModel()
        mock_args.return_value = Mock()
        mock_trainer.return_value = Mock()

        yield {
            "tokenizer": mock_tokenizer,
            "model": mock_model,
            "args": mock_args,
            "trainer": mock_trainer,
        }


@pytest.fixture
def mock_peft():
    """Fixture para mock dos componentes do PEFT/LoRA."""
    with (
        patch("src.finetuning.prepare_model_for_kbit_training") as mock_prepare,
        patch("src.finetuning.get_peft_model") as mock_get_peft,
        patch("src.finetuning.LoraConfig") as mock_lora_config,
        patch("src.finetuning.TaskType") as mock_task_type,
    ):
        
        # Configurar retornos adequados
        def prepare_model_side_effect(model):
            """Side effect que retorna o modelo modificado."""
            # Se o modelo for None, cria um modelo mock (simula carregamento automático)
            if model is None:
                model = MockModel()
            
            prepared_model = MockModel()
            # Copiar atributos importantes se existirem
            if hasattr(model, 'config'):
                prepared_model.config = model.config
            if hasattr(model, 'device'):
                prepared_model.device = model.device
            return prepared_model
        
        def get_peft_model_side_effect(model, config):
            """Side effect que retorna um modelo com PEFT aplicado."""
            # Se o modelo for None, cria um modelo mock
            if model is None:
                model = MockModel()
                
            peft_model = MockModel()
            if hasattr(model, 'config'):
                peft_model.config = model.config
            if hasattr(model, 'device'):
                peft_model.device = model.device
            peft_model.base_model = model
            peft_model.peft_config = config
            return peft_model
        
        mock_prepare.side_effect = prepare_model_side_effect
        mock_get_peft.side_effect = get_peft_model_side_effect
        mock_lora_config.return_value = Mock(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"

        yield {
            "prepare_model": mock_prepare,
            "get_peft_model": mock_get_peft,
            "lora_config": mock_lora_config,
            "task_type": mock_task_type,
        }


@pytest.fixture
def mock_datasets():
    """Fixture para mock do datasets."""
    with patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = {
            "train": MockDataset(100),
            "validation": MockDataset(20),
            "test": MockDataset(20),
        }
        yield mock_load


@pytest.fixture
def mock_torch():
    """Fixture para mock das funcionalidades do torch."""
    with (
        patch("torch.cuda.is_available") as mock_cuda_available,
        patch("torch.cuda.device_count") as mock_device_count,
    ):

        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1

        yield {"cuda_available": mock_cuda_available, "device_count": mock_device_count}


def skip_if_no_gpu():
    """Decorator para pular testes que requerem GPU."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU não disponível"
    )


def skip_if_no_internet():
    """Decorator para pular testes que requerem internet."""
    import urllib.request

    try:
        urllib.request.urlopen("http://www.google.com", timeout=5)
        has_internet = True
    except:
        has_internet = False

    return pytest.mark.skipif(
        not has_internet, reason="Conexão com internet não disponível"
    )


class TestEnvironment:
    """Classe para configurar ambiente de teste."""

    @staticmethod
    def setup_test_environment():
        """Configura variáveis de ambiente para teste."""
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    @staticmethod
    def cleanup_test_environment():
        """Limpa arquivos temporários de teste."""
        import shutil

        test_dirs = ["./test_output", "./wandb", "./__pycache__", "./tests/__pycache__"]

        for dir_path in test_dirs:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except:
                    pass


# Configuração global para pytest
def pytest_configure(config):
    """Configuração global do pytest."""
    TestEnvironment.setup_test_environment()


def pytest_unconfigure(config):
    """Limpeza global do pytest."""
    TestEnvironment.cleanup_test_environment()
