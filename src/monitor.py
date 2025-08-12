import subprocess
import threading
import time
from typing import Dict, List, Optional

import nvitop
import pynvml

from config.config import settings
from src.logging_config import get_monitor_logger
from src.utils.common import safe_cast
from src.utils.gpu_strategies import GPUStrategyFactory, GPUDataCollectorStrategy

# Configurar logging estruturado para monitoramento
monitor_logger = get_monitor_logger()

# Constantes de configuração
DEFAULT_BASELINE_DURATION_S = 5
DEFAULT_THREAD_JOIN_TIMEOUT_S = 5.0
DEFAULT_NVIDIA_SMI_TIMEOUT_S = 10


# Importações robustas para monitoramento GPU
try:
    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False
    print("⚠️ nvitop não disponível")

try:
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️ pynvml não disponível")

try:
    import subprocess

    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False


class RobustGPUMonitor:
    """Classe robusta para monitoramento GPU com múltiplas estratégias de fallback"""

    def __init__(self, sampling_interval: float = 1.0, enable_high_precision: bool = True):
        self.sampling_interval = sampling_interval
        self.enable_high_precision = enable_high_precision
        self.monitoring = False
        self.energy_data = []
        self.monitor_thread = None
        self.monitoring_method = None
        self.devices = []
        self.sync_markers = []  # Para sincronização com eventos externos
        self.baseline_power = None  # Para cálculo de consumo diferencial
        self._gpu_strategy: Optional[GPUDataCollectorStrategy] = None  # Strategy Pattern

    def detect_monitoring_capabilities(self) -> Dict[str, bool]:
        """Detecta capacidades de monitoramento disponíveis usando Strategy Pattern"""
        # Manter compatibilidade com testes existentes
        capabilities = {
            "nvitop": NVITOP_AVAILABLE,
            "pynvml": PYNVML_AVAILABLE,
            "nvidia_smi": self._test_nvidia_smi_capability(),
            "system_tools": SYSTEM_MONITORING_AVAILABLE,
        }
        
        # Testar funcionalidade do pynvml se disponível
        if PYNVML_AVAILABLE:
            pynvml_status = self._test_pynvml_capability()
            capabilities.update(pynvml_status)
        
        # Usar o factory para configurar estratégia baseada nas capacidades
        available_strategies = GPUStrategyFactory.get_available_strategies(monitor_logger)
        
        # Configurar a melhor estratégia disponível
        if available_strategies:
            self._gpu_strategy = available_strategies[0]
            self.monitoring_method = self._gpu_strategy.get_strategy_name()
            monitor_logger.info(
                "Estratégia GPU configurada", 
                strategy=self.monitoring_method,
                total_available=len(available_strategies)
            )
        
        return capabilities

    def _test_nvidia_smi_capability(self) -> bool:
        """Testa se nvidia-smi está disponível e funcional"""
        try:
            timeout = safe_cast(settings.MONITORING_NVIDIA_SMI_TIMEOUT, int, DEFAULT_NVIDIA_SMI_TIMEOUT_S)
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _test_pynvml_capability(self) -> Dict[str, bool]:
        """Testa se pynvml está funcional"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            return {
                "pynvml_functional": device_count > 0
            }
        except Exception:
            return {
                "pynvml_functional": False,
                "pynvml_error": False
            }

    def _calculate_gpu_statistics(self, power_samples: List[float], temp_samples: List[float], 
                                 util_samples: List[float], memory_samples: List[float], 
                                 gpu_name: str, duration_s: float) -> Dict:
        """Calcula estatísticas para uma GPU baseado nas amostras coletadas"""
        if not power_samples:
            return {}
            
        # Calcular estatísticas básicas de potência
        power_avg = sum(power_samples) / len(power_samples)
        power_max = max(power_samples)
        power_min = min(power_samples)
        
        # Calcular consumo diferencial (subtraindo baseline)
        baseline_adjusted_power = max(0, power_avg - (self.baseline_power or 0))
        duration_hours = duration_s / 3600
        
        statistics = {
            "power_avg_w": power_avg,
            "power_max_w": power_max,
            "power_min_w": power_min,
            "power_baseline_adjusted_w": baseline_adjusted_power,
            "energy_consumed_wh": baseline_adjusted_power * duration_hours,
            "energy_consumed_kwh": (baseline_adjusted_power * duration_hours) / 1000,
            "name": gpu_name,
            "samples_count": len(power_samples)
        }
        
        # Adicionar estatísticas de temperatura se disponíveis
        if temp_samples:
            statistics["temperature_avg_c"] = sum(temp_samples) / len(temp_samples)
            statistics["temperature_max_c"] = max(temp_samples)
        else:
            statistics["temperature_avg_c"] = 0
            statistics["temperature_max_c"] = 0
        
        # Adicionar estatísticas de utilização se disponíveis
        if util_samples:
            statistics["utilization_avg_percent"] = sum(util_samples) / len(util_samples)
        else:
            statistics["utilization_avg_percent"] = 0
        
        # Adicionar estatísticas de memória se disponíveis
        if memory_samples:
            statistics["memory_used_avg_mb"] = sum(memory_samples) / len(memory_samples)
        else:
            statistics["memory_used_avg_mb"] = 0
            
        return statistics

    def initialize_nvitop_monitoring(self) -> bool:
        """Inicializa monitoramento usando nvitop"""
        if not NVITOP_AVAILABLE:
            return False

        try:
            self.devices = nvitop.Device.all()
            if not self.devices:
                return False

            # Testar se conseguimos obter dados básicos
            for device in self.devices:
                device.name()  # Teste básico

            self.monitoring_method = "nvitop"
            return True
        except Exception as e:
            print(f"❌ Erro ao inicializar nvitop: {e}")
            return False

    def initialize_pynvml_monitoring(self) -> bool:
        """Inicializa monitoramento usando pynvml com fallbacks"""
        if not PYNVML_AVAILABLE:
            return False

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                return False

            # Testar funções disponíveis
            self.pynvml_functions = self._test_pynvml_functions()
            self.monitoring_method = "pynvml"
            return True

        except Exception as e:
            print(f"❌ Erro ao inicializar pynvml: {e}")
            return False

    def _test_pynvml_functions(self) -> Dict[str, bool]:
        """Testa quais funções pynvml estão disponíveis"""
        functions = {
            "power_draw": False,
            "temperature": False,
            "utilization": False,
            "memory_info": False,
            "processes_v2": False,
            "processes_v1": False,
        }

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Testar power draw
            try:
                pynvml.nvmlDeviceGetPowerUsage(handle)
                functions["power_draw"] = True
            except Exception:
                pass

            # Testar temperatura
            try:
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                functions["temperature"] = True
            except Exception:
                pass

            # Testar utilização
            try:
                pynvml.nvmlDeviceGetUtilizationRates(handle)
                functions["utilization"] = True
            except Exception:
                pass

            # Testar informações de memória
            try:
                pynvml.nvmlDeviceGetMemoryInfo(handle)
                functions["memory_info"] = True
            except Exception:
                pass

            # Testar processos v2
            try:
                pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
                functions["processes_v2"] = True
            except Exception:
                pass

            # Testar processos v1 (fallback)
            try:
                pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                functions["processes_v1"] = True
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ Erro ao testar funções pynvml: {e}")

        return functions

    def initialize_nvidia_smi_monitoring(self) -> bool:
        """Inicializa monitoramento usando nvidia-smi"""
        try:
            timeout = safe_cast(settings.MONITORING_NVIDIA_SMI_TIMEOUT, int, 10)
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                self.monitoring_method = "nvidia_smi"
                return True
        except Exception as e:
            print(f"❌ Erro ao testar nvidia-smi: {e}")

        return False

    def initialize_monitoring(self) -> bool:
        """Inicializa monitoramento usando Strategy Pattern"""
        if self._gpu_strategy is None:
            # Se não há estratégia configurada, detectar capacidades
            self.detect_monitoring_capabilities()
        
        if self._gpu_strategy is None:
            monitor_logger.error("Nenhuma estratégia GPU disponível")
            return False
        
        strategy_name = self._gpu_strategy.get_strategy_name()
        monitor_logger.info("Inicializando monitoramento", strategy=strategy_name)
        print(f"✅ Monitoramento inicializado com {strategy_name}")
        return True

    def _collect_nvitop_data(self) -> List[Dict]:
        """Coleta dados usando nvitop"""
        gpu_metrics = []

        try:
            for i, device in enumerate(self.devices):
                try:
                    metrics = {
                        "gpu_id": i,
                        "name": device.name(),
                        "power_draw_w": self._safe_get_attribute(
                            device, "power_draw", 0
                        ),
                        "temperature_c": self._safe_get_attribute(
                            device, "temperature", 0
                        ),
                        "utilization_gpu_percent": self._safe_get_attribute(
                            device, "gpu_utilization", 0
                        ),
                        "utilization_memory_percent": self._safe_get_attribute(
                            device, "memory_utilization", 0
                        ),
                        "memory_used_mb": self._safe_get_attribute(
                            device, "memory_used", 0
                        )
                        // (1024 * 1024),
                        "memory_total_mb": self._safe_get_attribute(
                            device, "memory_total", 0
                        )
                        // (1024 * 1024),
                    }

                    gpu_metrics.append(metrics)

                except Exception as e:
                    print(f"⚠️ Erro ao coletar dados da GPU {i}: {e}")
                    gpu_metrics.append({"gpu_id": i, "error": str(e)})

        except Exception as e:
            print(f"❌ Erro geral na coleta nvitop: {e}")

        return gpu_metrics

    def _safe_get_attribute(self, device, attr_name, default_value):
        """Obtém atributo de forma segura com fallback"""
        try:
            return getattr(device, attr_name)()
        except Exception:
            return default_value

    def _collect_pynvml_data(self) -> List[Dict]:
        """Coleta dados usando pynvml com fallbacks"""
        gpu_metrics = []

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                metrics = {
                    "gpu_id": i,
                    "name": "Unknown",
                    "power_draw_w": 0,
                    "temperature_c": 0,
                    "utilization_gpu_percent": 0,
                    "utilization_memory_percent": 0,
                    "memory_used_mb": 0,
                    "memory_total_mb": 0,
                }

                # Nome da GPU
                try:
                    metrics["name"] = pynvml.nvmlDeviceGetName(handle).decode()
                except Exception:
                    pass

                # Potência (se disponível)
                if self.pynvml_functions.get("power_draw", False):
                    try:
                        metrics["power_draw_w"] = (
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )
                    except Exception:
                        pass

                # Temperatura
                if self.pynvml_functions.get("temperature", False):
                    try:
                        metrics["temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    except Exception:
                        pass

                # Utilização
                if self.pynvml_functions.get("utilization", False):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics["utilization_gpu_percent"] = util.gpu
                        metrics["utilization_memory_percent"] = util.memory
                    except Exception:
                        pass

                # Memória
                if self.pynvml_functions.get("memory_info", False):
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics["memory_used_mb"] = int(mem_info.used) // (1024 * 1024)
                        metrics["memory_total_mb"] = int(mem_info.total) // (
                            1024 * 1024
                        )
                    except Exception:
                        pass

                gpu_metrics.append(metrics)

        except Exception as e:
            print(f"❌ Erro na coleta pynvml: {e}")

        return gpu_metrics

    def _collect_nvidia_smi_data(self) -> List[Dict]:
        """Coleta dados usando nvidia-smi"""
        gpu_metrics = []

        try:
            timeout = safe_cast(settings.MONITORING_NVIDIA_SMI_TIMEOUT, int, 10)
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 7:
                            try:
                                metrics = {
                                    "gpu_id": int(parts[0]),
                                    "name": parts[1],
                                    "power_draw_w": (
                                        float(parts[2])
                                        if parts[2] != "[Not Supported]"
                                        else 0
                                    ),
                                    "temperature_c": (
                                        float(parts[3])
                                        if parts[3] != "[Not Supported]"
                                        else 0
                                    ),
                                    "utilization_gpu_percent": (
                                        float(parts[4])
                                        if parts[4] != "[Not Supported]"
                                        else 0
                                    ),
                                    "memory_used_mb": (
                                        float(parts[5])
                                        if parts[5] != "[Not Supported]"
                                        else 0
                                    ),
                                    "memory_total_mb": (
                                        float(parts[6])
                                        if parts[6] != "[Not Supported]"
                                        else 0
                                    ),
                                    "utilization_memory_percent": 0,  # Não disponível via nvidia-smi básico
                                }
                                gpu_metrics.append(metrics)
                            except ValueError as e:
                                print(
                                    f"⚠️ Erro ao parsear linha nvidia-smi: {line} - {e}"
                                )

        except Exception as e:
            print(f"❌ Erro na coleta nvidia-smi: {e}")

        return gpu_metrics

    def _monitor_loop(self):
        """Loop principal de monitoramento usando Strategy Pattern"""
        while self.monitoring:
            timestamp = time.time()

            # Coletar dados usando a estratégia configurada
            if self._gpu_strategy:
                gpu_metrics = self._gpu_strategy.collect_data()
            else:
                gpu_metrics = []

            if gpu_metrics:
                self.energy_data.append(
                    {
                        "timestamp": timestamp,
                        "method": self.monitoring_method,
                        "gpus": gpu_metrics,
                    }
                )

            time.sleep(self.sampling_interval)

    def start_monitoring(self) -> bool:
        """Inicia monitoramento"""
        if not self.initialize_monitoring():
            return False

        self._reset_monitoring_state()
        self._setup_baseline_if_needed()
        self._start_monitoring_thread()
        self._log_monitoring_status()
        
        return True

    def _reset_monitoring_state(self):
        """Reseta o estado interno do monitoramento"""
        self.monitoring = True
        self.energy_data = []
        self.sync_markers = []

    def _setup_baseline_if_needed(self):
        """Estabelece baseline se o modo de alta precisão estiver ativado"""
        if self.enable_high_precision:
            self._establish_baseline()

    def _start_monitoring_thread(self):
        """Inicia thread de monitoramento em background"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _log_monitoring_status(self):
        """Registra status do monitoramento iniciado"""
        method_name = self.monitoring_method or "unknown"
        print(f"🔋 Monitoramento iniciado usando: {method_name}")
        
        if self.baseline_power:
            print(f"📊 Baseline estabelecido: {self.baseline_power:.2f}W")

    def stop_monitoring(self) -> Dict:
        """Para monitoramento e processa dados"""
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT_S)

        print("🔋 Monitoramento finalizado")
        return self._process_energy_data()

    def add_sync_marker(self, marker_type: str, step: Optional[int] = None, epoch: Optional[float] = None, metadata: Optional[Dict] = None):
        """
        Adiciona marcador de sincronização para correlacionar com eventos do treinamento.
        
        Args:
            marker_type: Tipo do marcador (step, epoch, checkpoint, etc.)
            step: Número do step de treinamento
            epoch: Número da época
            metadata: Dados adicionais para contexto
        """
        marker = {
            "timestamp": time.time(),
            "type": marker_type,
            "step": step,
            "epoch": epoch,
            "metadata": metadata or {}
        }
        self.sync_markers.append(marker)

    def get_metrics_since_timestamp(self, since_timestamp: float) -> Dict:
        """
        Obtém métricas energéticas desde um timestamp específico.
        Útil para cálculos incrementais por step ou intervalo.
        """
        if not self.energy_data:
            return {"error": "Nenhum dado disponível"}

        # Filtrar dados desde o timestamp
        filtered_data = [
            sample for sample in self.energy_data
            if sample["timestamp"] >= since_timestamp
        ]

        if not filtered_data:
            return {"error": "Nenhum dado no período especificado"}

        # Processar dados filtrados
        return self._process_filtered_energy_data(filtered_data, since_timestamp)

    def _establish_baseline(self):
        """Estabelece baseline de consumo energético sem carga de trabalho"""
        print("📊 Estabelecendo baseline energético...")
        
        baseline_samples = []
        baseline_duration = DEFAULT_BASELINE_DURATION_S  # segundos para baseline
        
        start_time = time.time()
        while time.time() - start_time < baseline_duration:
            # Coletar amostra usando estratégia
            if self._gpu_strategy:
                gpu_metrics = self._gpu_strategy.collect_data()
            else:
                break

            if gpu_metrics:
                baseline_samples.append(gpu_metrics)
            
            time.sleep(self.sampling_interval)

        # Calcular baseline médio
        if baseline_samples:
            total_power = 0
            sample_count = 0
            
            for sample in baseline_samples:
                try:
                    # Verificar se sample é iterável
                    if not hasattr(sample, '__iter__'):
                        continue
                        
                    for gpu_data in sample:
                        if "error" not in gpu_data and gpu_data.get("power_w") is not None:
                            total_power += gpu_data.get("power_w", 0)
                            sample_count += 1
                except (TypeError, AttributeError) as e:
                    # Para testes com Mocks que não são iteráveis
                    monitor_logger.debug("Erro processando baseline", error=str(e))
                    continue
            
            if sample_count > 0:
                self.baseline_power = total_power / sample_count
                print(f"✅ Baseline estabelecido: {self.baseline_power:.2f}W")
            else:
                print("⚠️ Não foi possível estabelecer baseline")

    def _process_filtered_energy_data(self, filtered_data: List[Dict], since_timestamp: float) -> Dict:
        """Processa dados energéticos filtrados para um período específico"""
        if not filtered_data:
            return {"error": "Nenhum dado para processar"}

        duration_s = time.time() - since_timestamp
        total_samples = len(filtered_data)

        # Processar por GPU
        gpu_metrics = {}
        gpu_ids = set()
        
        for sample in filtered_data:
            for gpu_data in sample["gpus"]:
                if "error" not in gpu_data:
                    gpu_ids.add(gpu_data["gpu_id"])

        for gpu_id in gpu_ids:
            power_samples = []
            temp_samples = []
            util_samples = []
            memory_samples = []
            gpu_name = "Unknown"

            # Coletar amostras para esta GPU
            for sample in filtered_data:
                for gpu_data in sample["gpus"]:
                    if gpu_data.get("gpu_id") == gpu_id and "error" not in gpu_data:
                        # Suportar ambos formatos: power_w (estratégias) e power_draw_w (métodos legados)
                        power_value = gpu_data.get("power_w") or gpu_data.get("power_draw_w", 0)
                        power_samples.append(power_value)
                        temp_samples.append(gpu_data.get("temperature_c", 0))
                        # Suportar ambos formatos de utilização
                        util_value = gpu_data.get("utilization_percent") or gpu_data.get("utilization_gpu_percent", 0)
                        util_samples.append(util_value)
                        memory_samples.append(gpu_data.get("memory_used_mb", 0))
                        gpu_name = gpu_data.get("name", "Unknown")

            if power_samples:
                gpu_statistics = self._calculate_gpu_statistics(
                    power_samples, temp_samples, util_samples, memory_samples, 
                    gpu_name, duration_s
                )
                gpu_metrics[f"gpu_{gpu_id}"] = gpu_statistics

        return {
            "monitoring_method": self.monitoring_method,
            "duration_s": duration_s,
            "total_samples": total_samples,
            "baseline_power_w": self.baseline_power,
            "gpus": gpu_metrics
        }

    def _process_energy_data(self) -> Dict:
        """Processa dados coletados"""
        if not self.energy_data:
            return self._create_empty_result()

        processed_data = self._create_base_processed_data()
        
        if len(self.energy_data) > 1:
            processed_data["monitoring_duration_s"] = (
                self.energy_data[-1]["timestamp"] - self.energy_data[0]["timestamp"]
            )

        # Processar dados por GPU
        gpu_ids = self._extract_gpu_ids()
        processed_data["gpus"] = self._process_gpu_data(gpu_ids, processed_data["monitoring_duration_s"])

        return processed_data

    def _create_empty_result(self) -> Dict:
        """Cria resultado vazio para quando não há dados"""
        return {
            "error": "Nenhum dado coletado", 
            "method": self.monitoring_method,
            "monitoring_duration_s": 0,
            "total_samples": 0,
            "sampling_interval_s": self.sampling_interval,
            "baseline_power_w": self.baseline_power,
            "sync_markers": self.sync_markers,
            "high_precision_enabled": self.enable_high_precision,
            "gpus": {},
        }

    def _create_base_processed_data(self) -> Dict:
        """Cria estrutura base dos dados processados"""
        return {
            "monitoring_method": self.monitoring_method,
            "monitoring_duration_s": 0,
            "total_samples": len(self.energy_data),
            "sampling_interval_s": self.sampling_interval,
            "baseline_power_w": self.baseline_power,
            "sync_markers": self.sync_markers,
            "high_precision_enabled": self.enable_high_precision,
            "gpus": {},
        }

    def _extract_gpu_ids(self) -> set:
        """Extrai IDs únicos das GPUs dos dados coletados"""
        gpu_ids = set()
        for sample in self.energy_data:
            for gpu_data in sample["gpus"]:
                if "error" not in gpu_data:
                    gpu_ids.add(gpu_data["gpu_id"])
        return gpu_ids

    def _process_gpu_data(self, gpu_ids: set, monitoring_duration_s: float) -> Dict:
        """Processa dados para todas as GPUs"""
        gpu_results = {}
        
        for gpu_id in gpu_ids:
            gpu_samples = self._collect_gpu_samples(gpu_id)
            
            if gpu_samples["power_samples_w"]:
                gpu_samples["statistics"] = self._calculate_comprehensive_gpu_statistics(
                    gpu_samples, monitoring_duration_s
                )
            
            gpu_results[f"gpu_{gpu_id}"] = gpu_samples
        
        return gpu_results

    def _collect_gpu_samples(self, gpu_id: int) -> Dict:
        """Coleta amostras para uma GPU específica"""
        gpu_samples = {
            "power_samples_w": [],
            "temperature_samples_c": [],
            "utilization_gpu_samples": [],
            "memory_used_samples_mb": [],
            "name": "Unknown",
        }

        # Coletar amostras de todos os timestamps
        for sample in self.energy_data:
            for gpu_data in sample["gpus"]:
                if gpu_data.get("gpu_id") == gpu_id and "error" not in gpu_data:
                    # Suportar ambos formatos
                    power_value = gpu_data.get("power_w") or gpu_data.get("power_draw_w", 0)
                    gpu_samples["power_samples_w"].append(power_value)
                    gpu_samples["temperature_samples_c"].append(
                        gpu_data.get("temperature_c", 0)
                    )
                    # Suportar ambos formatos de utilização  
                    util_value = gpu_data.get("utilization_percent") or gpu_data.get("utilization_gpu_percent", 0)
                    gpu_samples["utilization_gpu_samples"].append(util_value)
                    gpu_samples["memory_used_samples_mb"].append(
                        gpu_data.get("memory_used_mb", 0)
                    )
                    gpu_samples["name"] = gpu_data.get("name", "Unknown")
        
        return gpu_samples

    def _calculate_comprehensive_gpu_statistics(self, gpu_samples: Dict, duration_s: float) -> Dict:
        """Calcula estatísticas abrangentes para uma GPU"""
        power_samples = gpu_samples["power_samples_w"]
        temp_samples = gpu_samples["temperature_samples_c"] 
        util_samples = gpu_samples["utilization_gpu_samples"]
        memory_samples = gpu_samples["memory_used_samples_mb"]
        gpu_name = gpu_samples["name"]

        # Usar método helper básico
        statistics = self._calculate_gpu_statistics(
            power_samples, temp_samples, util_samples, memory_samples,
            gpu_name, duration_s
        )
        
        # Adicionar campos específicos do processamento completo
        duration_hours = duration_s / 3600
        power_avg = sum(power_samples) / len(power_samples)
        baseline_adjusted_avg = max(0, power_avg - (self.baseline_power or 0))
        
        statistics.update({
            "energy_consumed_wh": power_avg * duration_hours,
            "energy_consumed_baseline_adjusted_wh": baseline_adjusted_avg * duration_hours,
            "energy_consumed_baseline_adjusted_kwh": (baseline_adjusted_avg * duration_hours) / 1000,
        })
        
        # Adicionar estatísticas máximas específicas
        if util_samples:
            statistics["utilization_max_percent"] = max(util_samples)
        if memory_samples:
            statistics["memory_used_max_mb"] = max(memory_samples)
        
        return statistics
