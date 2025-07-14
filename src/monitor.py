import time
import threading
from typing import Dict, List
import subprocess
import pynvml
import nvitop

from config.config import settings


def safe_cast(value, cast_func, default):
    """Helper para fazer cast seguro de valores do dynaconf"""
    try:
        return cast_func(value)
    except (ValueError, TypeError):
        return default


# Importações robustas para monitoramento GPU
try:
    import nvitop
    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False
    print("⚠️ nvitop não disponível")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️ pynvml não disponível")

try:
    import subprocess
    import psutil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False


class RobustGPUMonitor:
    """Classe robusta para monitoramento GPU com múltiplas estratégias de fallback"""

    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.energy_data = []
        self.monitor_thread = None
        self.monitoring_method = None
        self.devices = []

    def detect_monitoring_capabilities(self) -> Dict[str, bool]:
        """Detecta capacidades de monitoramento disponíveis"""
        capabilities = {
            "nvitop": NVITOP_AVAILABLE,
            "pynvml": PYNVML_AVAILABLE,
            "nvidia_smi": False,
            "system_tools": SYSTEM_MONITORING_AVAILABLE
        }

        # Testar nvidia-smi
        try:
            timeout = safe_cast(settings.MONITORING_NVIDIA_SMI_TIMEOUT, int, 10)
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=timeout)
            capabilities["nvidia_smi"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Testar pynvml se disponível
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                capabilities["pynvml_functional"] = device_count > 0
            except Exception as e:
                capabilities["pynvml_functional"] = False
                capabilities["pynvml_error"] = False

        return capabilities

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
            "processes_v1": False
        }

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Testar power draw
            try:
                pynvml.nvmlDeviceGetPowerUsage(handle)
                functions["power_draw"] = True
            except:
                pass

            # Testar temperatura
            try:
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                functions["temperature"] = True
            except:
                pass

            # Testar utilização
            try:
                pynvml.nvmlDeviceGetUtilizationRates(handle)
                functions["utilization"] = True
            except:
                pass

            # Testar informações de memória
            try:
                pynvml.nvmlDeviceGetMemoryInfo(handle)
                functions["memory_info"] = True
            except:
                pass

            # Testar processos v2
            try:
                pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
                functions["processes_v2"] = True
            except:
                pass

            # Testar processos v1 (fallback)
            try:
                pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                functions["processes_v1"] = True
            except:
                pass

        except Exception as e:
            print(f"⚠️ Erro ao testar funções pynvml: {e}")

        return functions

    def initialize_nvidia_smi_monitoring(self) -> bool:
        """Inicializa monitoramento usando nvidia-smi"""
        try:
            timeout = safe_cast(settings.MONITORING_NVIDIA_SMI_TIMEOUT, int, 10)
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=index,name,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                self.monitoring_method = "nvidia_smi"
                return True
        except Exception as e:
            print(f"❌ Erro ao testar nvidia-smi: {e}")

        return False

    def initialize_monitoring(self) -> bool:
        """Inicializa monitoramento com estratégia de fallback"""
        capabilities = self.detect_monitoring_capabilities()
        print(f"🔍 Capacidades detectadas: {capabilities}")

        # Tentar nvitop primeiro (mais confiável)
        if self.initialize_nvitop_monitoring():
            print("✅ Monitoramento inicializado com nvitop")
            return True

        # Fallback para pynvml
        if self.initialize_pynvml_monitoring():
            print("✅ Monitoramento inicializado com pynvml")
            return True

        # Fallback para nvidia-smi
        if self.initialize_nvidia_smi_monitoring():
            print("✅ Monitoramento inicializado com nvidia-smi")
            return True

        print("❌ Nenhum método de monitoramento disponível")
        return False

    def _collect_nvitop_data(self) -> List[Dict]:
        """Coleta dados usando nvitop"""
        gpu_metrics = []

        try:
            for i, device in enumerate(self.devices):
                try:
                    metrics = {
                        "gpu_id": i,
                        "name": device.name(),
                        "power_draw_w": self._safe_get_attribute(device, 'power_draw', 0),
                        "temperature_c": self._safe_get_attribute(device, 'temperature', 0),
                        "utilization_gpu_percent": self._safe_get_attribute(device, 'gpu_utilization', 0),
                        "utilization_memory_percent": self._safe_get_attribute(device, 'memory_utilization', 0),
                        "memory_used_mb": self._safe_get_attribute(device, 'memory_used', 0) // (1024 * 1024),
                        "memory_total_mb": self._safe_get_attribute(device, 'memory_total', 0) // (1024 * 1024),
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
        except:
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
                except:
                    pass

                # Potência (se disponível)
                if self.pynvml_functions.get("power_draw", False):
                    try:
                        metrics["power_draw_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except:
                        pass

                # Temperatura
                if self.pynvml_functions.get("temperature", False):
                    try:
                        metrics["temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        pass

                # Utilização
                if self.pynvml_functions.get("utilization", False):
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics["utilization_gpu_percent"] = util.gpu
                        metrics["utilization_memory_percent"] = util.memory
                    except:
                        pass

                # Memória
                if self.pynvml_functions.get("memory_info", False):
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics["memory_used_mb"] = int(mem_info.used) // (1024 * 1024)
                        metrics["memory_total_mb"] = int(mem_info.total) // (1024 * 1024)
                    except:
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
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=index,name,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            try:
                                metrics = {
                                    "gpu_id": int(parts[0]),
                                    "name": parts[1],
                                    "power_draw_w": float(parts[2]) if parts[2] != '[Not Supported]' else 0,
                                    "temperature_c": float(parts[3]) if parts[3] != '[Not Supported]' else 0,
                                    "utilization_gpu_percent": float(parts[4]) if parts[4] != '[Not Supported]' else 0,
                                    "memory_used_mb": float(parts[5]) if parts[5] != '[Not Supported]' else 0,
                                    "memory_total_mb": float(parts[6]) if parts[6] != '[Not Supported]' else 0,
                                    "utilization_memory_percent": 0  # Não disponível via nvidia-smi básico
                                }
                                gpu_metrics.append(metrics)
                            except ValueError as e:
                                print(f"⚠️ Erro ao parsear linha nvidia-smi: {line} - {e}")

        except Exception as e:
            print(f"❌ Erro na coleta nvidia-smi: {e}")

        return gpu_metrics

    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring:
            timestamp = time.time()

            # Coletar dados baseado no método disponível
            if self.monitoring_method == "nvitop":
                gpu_metrics = self._collect_nvitop_data()
            elif self.monitoring_method == "pynvml":
                gpu_metrics = self._collect_pynvml_data()
            elif self.monitoring_method == "nvidia_smi":
                gpu_metrics = self._collect_nvidia_smi_data()
            else:
                gpu_metrics = []

            if gpu_metrics:
                self.energy_data.append({
                    "timestamp": timestamp,
                    "method": self.monitoring_method,
                    "gpus": gpu_metrics
                })

            time.sleep(self.sampling_interval)

    def start_monitoring(self) -> bool:
        """Inicia monitoramento"""
        if not self.initialize_monitoring():
            return False

        self.monitoring = True
        self.energy_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        print(f"🔋 Monitoramento iniciado usando: {self.monitoring_method}")
        return True

    def stop_monitoring(self) -> Dict:
        """Para monitoramento e processa dados"""
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        print("🔋 Monitoramento finalizado")
        return self._process_energy_data()

    def _process_energy_data(self) -> Dict:
        """Processa dados coletados"""
        if not self.energy_data:
            return {"error": "Nenhum dado coletado", "method": self.monitoring_method}

        processed_data = {
            "monitoring_method": self.monitoring_method,
            "monitoring_duration_s": 0,
            "total_samples": len(self.energy_data),
            "sampling_interval_s": self.sampling_interval,
            "gpus": {}
        }

        if len(self.energy_data) > 1:
            processed_data["monitoring_duration_s"] = (
                self.energy_data[-1]["timestamp"] - self.energy_data[0]["timestamp"]
            )

        # Processar por GPU
        gpu_ids = set()
        for sample in self.energy_data:
            for gpu_data in sample["gpus"]:
                if "error" not in gpu_data:
                    gpu_ids.add(gpu_data["gpu_id"])

        for gpu_id in gpu_ids:
            gpu_samples = {
                "power_samples_w": [],
                "temperature_samples_c": [],
                "utilization_gpu_samples": [],
                "memory_used_samples_mb": [],
                "name": "Unknown"
            }

            # Coletar amostras
            for sample in self.energy_data:
                for gpu_data in sample["gpus"]:
                    if gpu_data.get("gpu_id") == gpu_id and "error" not in gpu_data:
                        gpu_samples["power_samples_w"].append(gpu_data.get("power_draw_w", 0))
                        gpu_samples["temperature_samples_c"].append(gpu_data.get("temperature_c", 0))
                        gpu_samples["utilization_gpu_samples"].append(gpu_data.get("utilization_gpu_percent", 0))
                        gpu_samples["memory_used_samples_mb"].append(gpu_data.get("memory_used_mb", 0))
                        gpu_samples["name"] = gpu_data.get("name", "Unknown")

            # Calcular estatísticas
            if gpu_samples["power_samples_w"]:
                power_samples = gpu_samples["power_samples_w"]
                duration_hours = processed_data["monitoring_duration_s"] / 3600

                gpu_samples["statistics"] = {
                    "power_avg_w": sum(power_samples) / len(power_samples),
                    "power_max_w": max(power_samples),
                    "power_min_w": min(power_samples),
                    "energy_consumed_wh": (sum(power_samples) / len(power_samples)) * duration_hours,
                    "energy_consumed_kwh": ((sum(power_samples) / len(power_samples)) * duration_hours) / 1000
                }

                if gpu_samples["temperature_samples_c"]:
                    gpu_samples["statistics"]["temperature_avg_c"] = sum(gpu_samples["temperature_samples_c"]) / len(gpu_samples["temperature_samples_c"])
                    gpu_samples["statistics"]["temperature_max_c"] = max(gpu_samples["temperature_samples_c"])

            processed_data["gpus"][f"gpu_{gpu_id}"] = gpu_samples

        return processed_data
