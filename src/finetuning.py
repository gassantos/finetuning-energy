import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import huggingface_hub
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.utils.quantization_config import BitsAndBytesConfig

from config.config import settings
from src.monitor import RobustGPUMonitor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_cast(value, cast_func, default):
    """Helper para fazer cast seguro de valores do dynaconf"""
    try:
        return cast_func(value)
    except (ValueError, TypeError):
        return default


class LlamaFineTuner:
    """Classe para fine-tuning do LLaMA 3.2 3B com monitoramento energético

    Implementa as melhores práticas para fine-tuning de LLMs:
    - Quantização 4-bit com BitsAndBytes
    - LoRA (Low-Rank Adaptation)
    - Monitoramento energético robusto
    - Early stopping
    - Logging estruturado
    """

    def __init__(
        self,
        wandb_key: str,
        hf_token: str,
        model_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        result_dir: Optional[str] = None,
    ):
        self.model_id = model_id or str(settings.MODEL_ID)
        self.wandb_key = wandb_key
        self.hf_token = hf_token
        self.output_dir = Path(output_dir or str(settings.OUTPUT_DIR))
        self.result_dir = Path(result_dir or str(settings.RESULT_DIR))
        self.model = None
        self.tokenizer = None
        self.gpu_monitor = RobustGPUMonitor(
            sampling_interval=safe_cast(settings.MONITORING_INTERVAL, float, 1.0)
        )

        # Criar diretório de saída se não existir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Criar diretório para resultados se não existir
        self.result_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Inicializando LlamaFineTuner com modelo: {self.model_id}")

    def setup_authentication(self):
        """Configura autenticação para Wandb e Hugging Face"""
        try:
            wandb.login(key=self.wandb_key)
            huggingface_hub.login(token=self.hf_token)
            logger.info("Autenticação configurada com sucesso")
        except Exception as e:
            logger.error(f"Erro na autenticação: {e}")
            raise

    def initialize_wandb(self):
        """Inicializa o projeto no Wandb"""
        capabilities = self.gpu_monitor.detect_monitoring_capabilities()

        wandb.init(
            entity=str(settings.WANDB_ENTITY),
            project=str(settings.WANDB_PROJECT),
            name=str(settings.WANDB_RUN_NAME),
            config={
                "model": str(settings.MODEL_NAME),
                "batch_size": safe_cast(settings.BATCH_SIZE, int, 2),
                "learning_rate": safe_cast(settings.LEARNING_RATE, float, 2e-4),
                "epochs": safe_cast(settings.EPOCHS, int, 3),
                "lora_r": safe_cast(settings.LORA_R, int, 8),
                "lora_alpha": safe_cast(settings.LORA_ALPHA, int, 16),
                "dataset": str(settings.DATASET),
                "task": str(settings.TASK),
                "quantization": str(settings.QUANTIZATION),
                "monitoring_capabilities": capabilities,
                "monitoring_interval_s": self.gpu_monitor.sampling_interval,
            },
        )
        # Configurar modo do wandb
        wandb_mode = str(settings.WANDB_MODE)
        if wandb.run and hasattr(wandb.run, "settings"):
            if wandb_mode == "offline":
                wandb.run.settings.mode = "offline"
            elif wandb_mode == "online":
                wandb.run.settings.mode = "online"
            else:
                wandb.run.settings.mode = "offline"

    def setup_quantization_config(self):
        """Configura quantização 4-bit com BitsAndBytes"""

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    def load_model_and_tokenizer(self):
        """Carrega o modelo e tokenizer com quantização"""
        logger.info(f"Carregando modelo: {self.model_id}")
        bnb_config = self.setup_quantization_config()

        try:
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True, use_fast=True
            )

            # Configurar pad_token se não existir
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Carregar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            # Preparar modelo para treinamento k-bit
            self.model = prepare_model_for_kbit_training(self.model)

            logger.info("Modelo e tokenizer carregados com sucesso")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def apply_lora(self):
        """Aplica LoRA (Low-Rank Adaptation) ao modelo"""
        lora_config = LoraConfig(
            r=safe_cast(settings.LORA_R, int, 8),
            lora_alpha=safe_cast(settings.LORA_ALPHA, int, 16),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

    def load_and_prepare_dataset(self, num_samples: int = 10):
        """Carrega e prepara o dataset para sumarização"""
        dataset = load_dataset(str(settings.DATASET), split=f"train[:{num_samples}]")

        # Log de exemplo do dataset
        sample_data = []
        for i, example in enumerate(dataset):
            if i >= 5:  # Apenas 5 exemplos para evitar logs excessivos
                break
            sample_data.append(
                [example["text"][:100] + "...", example["summary"][:100] + "..."]
            )

        wandb.log(
            {
                "dataset_sample": wandb.Table(
                    data=sample_data, columns=["text_preview", "summary_preview"]
                )
            }
        )

        def preprocess(example):
            prompt = f"Summarize:\n{example['text']}\nSummary:"
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer não foi inicializado")

            model_input = self.tokenizer(
                prompt, truncation=True, padding="max_length", max_length=512
            )
            model_input["labels"] = model_input["input_ids"].copy()
            return model_input

        # Obter nomes das colunas originais
        original_columns = (
            list(dataset.column_names)
            if hasattr(dataset, "column_names") and dataset.column_names
            else []
        )
        tokenized_dataset = dataset.map(preprocess, remove_columns=original_columns)
        return tokenized_dataset

    def setup_training_arguments(self):
        """Configura argumentos de treinamento"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=safe_cast(settings.EPOCHS, int, 3),
            per_device_train_batch_size=safe_cast(settings.BATCH_SIZE, int, 2),
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=safe_cast(settings.LEARNING_RATE, float, 2e-4),
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="wandb",
        )

    def train_with_robust_monitoring(self, tokenized_dataset):
        """Executa treinamento com monitoramento"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Modelo e tokenizer devem ser carregados antes do treinamento"
            )

        setup_training_args = self.setup_training_arguments()

        # Não usar EarlyStoppingCallback para evitar problemas de compatibilidade
        trainer = Trainer(
            model=self.model,
            args=setup_training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        # Iniciar monitoramento
        print("🔋 Iniciando monitoramento ...")
        monitoring_started = self.gpu_monitor.start_monitoring()

        try:
            trainer.train()
        finally:
            if monitoring_started:
                energy_data = self.gpu_monitor.stop_monitoring()
                self._log_energy_data_to_wandb(energy_data)
                self._save_energy_data(energy_data)

        return trainer

    def _log_energy_data_to_wandb(self, energy_data: Dict):
        """Registra dados energéticos no Wandb"""
        if "error" in energy_data:
            wandb.log(
                {
                    "monitoring_error": energy_data["error"],
                    "monitoring_method": energy_data.get("method", "unknown"),
                }
            )
            return

        wandb.log(
            {
                "energy/monitoring_method": energy_data["monitoring_method"],
                "energy/monitoring_duration_s": energy_data["monitoring_duration_s"],
                "energy/total_samples": energy_data["total_samples"],
            }
        )

        for gpu_key, gpu_data in energy_data["gpus"].items():
            if "statistics" in gpu_data:
                stats = gpu_data["statistics"]
                wandb.log(
                    {
                        f"energy/{gpu_key}/power_avg_w": stats["power_avg_w"],
                        f"energy/{gpu_key}/energy_consumed_kwh": stats[
                            "energy_consumed_kwh"
                        ],
                        f"energy/{gpu_key}/gpu_name": gpu_data["name"],
                    }
                )

    def _save_energy_data(self, energy_data: Dict):
        """Salva dados energéticos em arquivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.result_dir / f"robust_gpu_energy_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(energy_data, f, indent=2, default=str)

        print(f"💾 Dados energéticos salvos em: {filename}")

    def save_model(self, output_dir: Optional[str] = None):
        """Salva o modelo fine-tuned"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modelo e tokenizer devem estar carregados para salvar")

        save_dir = output_dir or str(self.output_dir)

        try:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"Modelo salvo em: {save_dir}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise

    def run_complete_pipeline(self, num_samples: Optional[int] = None):
        """Executa pipeline completo com monitoramento robusto"""
        start_time = time.time()

        # Usar configuração se não especificado
        if num_samples is None:
            num_samples = safe_cast(settings.DATASET_NUM_SAMPLES, int, 10)

        print("🚀 Iniciando fine-tuning com monitoramento robusto...")

        self.setup_authentication()
        self.initialize_wandb()

        print("📥 Carregando modelo e tokenizer...")
        self.load_model_and_tokenizer()

        print("🔧 Aplicando LoRA...")
        self.apply_lora()

        print("📊 Preparando dataset...")
        tokenized_dataset = self.load_and_prepare_dataset(num_samples)

        print("🎯 Iniciando treinamento...")
        trainer = self.train_with_robust_monitoring(tokenized_dataset)

        print("💾 Salvando modelo...")
        self.save_model()

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"✅ Treinamento concluído em {elapsed_time:.2f} segundos")
        wandb.finish()

        return trainer
