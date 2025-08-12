"""
Estratégias para coleta de dados GPU usando o padrão Strategy.

Este módulo implementa diferentes estratégias para coleta de métricas GPU,
permitindo maior flexibilidade e testabilidade do código de monitoramento.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import subprocess

# Importações condicionais com TYPE_CHECKING
if TYPE_CHECKING:
    import pynvml
    from nvitop import Device

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

try:
    from nvitop import Device
    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False
    Device = None


class GPUDataCollectorStrategy(ABC):
    """Interface base para estratégias de coleta de dados GPU."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verifica se a estratégia está disponível no sistema."""
        pass
    
    @abstractmethod
    def collect_data(self) -> List[Dict[str, Any]]:
        """Coleta dados de todas as GPUs disponíveis."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Retorna o nome da estratégia."""
        pass


class NvitopStrategy(GPUDataCollectorStrategy):
    """Estratégia usando a biblioteca nvitop."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self._devices = None
        
    def is_available(self) -> bool:
        """Verifica se nvitop está disponível."""
        return NVITOP_AVAILABLE
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """Coleta dados usando nvitop."""
        if not NVITOP_AVAILABLE or Device is None:
            return []
        
        try:
            if self._devices is None:
                self._devices = Device.all()
            
            gpus = []
            for i, device in enumerate(self._devices):
                try:
                    device.update()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Erro ao atualizar device {i}: {e}")
                    continue
                
                # Coletar dados básicos
                gpu_data = {
                    "gpu_id": i,
                    "name": device.name() if hasattr(device, 'name') else f"GPU_{i}",
                    "power_w": None,
                    "temperature_c": None,
                    "utilization_percent": None,
                    "memory_used_mb": None,
                    "memory_total_mb": None,
                }
                
                # Coletar power draw se disponível
                try:
                    if hasattr(device, 'power_draw'):
                        gpu_data["power_w"] = device.power_draw()
                except Exception:
                    pass
                
                # Coletar temperatura se disponível
                try:
                    if hasattr(device, 'temperature'):
                        gpu_data["temperature_c"] = device.temperature()
                except Exception:
                    pass
                
                # Coletar utilização se disponível
                try:
                    if hasattr(device, 'gpu_utilization'):
                        gpu_data["utilization_percent"] = device.gpu_utilization()
                except Exception:
                    pass
                
                # Coletar dados de memória
                try:
                    memory_used_bytes = device.memory_used()
                    memory_total_bytes = device.memory_total()
                    gpu_data["memory_used_mb"] = memory_used_bytes / (1024 * 1024) if memory_used_bytes else None
                    gpu_data["memory_total_mb"] = memory_total_bytes / (1024 * 1024) if memory_total_bytes else None
                except Exception:
                    pass
                
                gpus.append(gpu_data)
            return gpus
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Erro ao coletar dados nvitop: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "nvitop"


class PyNVMLStrategy(GPUDataCollectorStrategy):
    """Estratégia usando a biblioteca pynvml."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self._initialized = False
        
    def is_available(self) -> bool:
        """Verifica se pynvml está disponível."""
        return PYNVML_AVAILABLE
    
    def _ensure_initialized(self) -> bool:
        """Inicializa pynvml se necessário."""
        if not PYNVML_AVAILABLE or pynvml is None:
            return False
            
        if not self._initialized:
            try:
                pynvml.nvmlInit()
                self._initialized = True
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Erro ao inicializar pynvml: {e}")
                return False
        return True
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """Coleta dados usando pynvml."""
        if not self._ensure_initialized() or pynvml is None:
            return []
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                
                # Coleta métricas disponíveis
                gpu_data = {
                    "gpu_id": i,
                    "name": name,
                    "power_w": None,
                    "temperature_c": None,
                    "utilization_percent": None,
                    "memory_used_mb": None,
                    "memory_total_mb": None,
                }
                
                try:
                    power_info = pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_data["power_w"] = power_info / 1000  # mW para W
                except Exception:
                    pass
                
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_data["temperature_c"] = temp
                except Exception:
                    pass
                
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_data["utilization_percent"] = util.gpu
                except Exception:
                    pass
                
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_data["memory_used_mb"] = int(mem_info.used) / (1024 * 1024)
                    gpu_data["memory_total_mb"] = int(mem_info.total) / (1024 * 1024)
                except Exception:
                    pass
                
                gpus.append(gpu_data)
            
            return gpus
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Erro ao coletar dados pynvml: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "pynvml"


class NvidiaSMIStrategy(GPUDataCollectorStrategy):
    """Estratégia usando nvidia-smi via subprocess."""
    
    def __init__(self, logger=None):
        self.logger = logger
        
    def is_available(self) -> bool:
        """Verifica se nvidia-smi está disponível."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """Coleta dados usando nvidia-smi."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                if self.logger:
                    self.logger.warning(f"nvidia-smi falhou: {result.stderr}")
                return []
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 7:
                    try:
                        gpus.append({
                            "gpu_id": int(parts[0]),
                            "name": parts[1],
                            "power_w": float(parts[2]) if parts[2] != "[N/A]" else None,
                            "temperature_c": float(parts[3]) if parts[3] != "[N/A]" else None,
                            "utilization_percent": float(parts[4]) if parts[4] != "[N/A]" else None,
                            "memory_used_mb": float(parts[5]) if parts[5] != "[N/A]" else None,
                            "memory_total_mb": float(parts[6]) if parts[6] != "[N/A]" else None,
                        })
                    except (ValueError, IndexError):
                        continue
            
            return gpus
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Erro ao coletar dados nvidia-smi: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "nvidia_smi"


class GPUStrategyFactory:
    """Factory para criar estratégias GPU baseadas na disponibilidade."""
    
    @staticmethod
    def create_strategy(strategy_name: str, logger=None) -> Optional[GPUDataCollectorStrategy]:
        """Cria uma estratégia específica."""
        strategies = {
            "nvitop": NvitopStrategy,
            "pynvml": PyNVMLStrategy,
            "nvidia_smi": NvidiaSMIStrategy,
        }
        
        strategy_class = strategies.get(strategy_name)
        if strategy_class:
            return strategy_class(logger)
        return None
    
    @staticmethod
    def get_available_strategies(logger=None) -> List[GPUDataCollectorStrategy]:
        """Retorna lista de estratégias disponíveis ordenadas por prioridade."""
        strategies = [
            NvidiaSMIStrategy(logger),  # Primeira opção - mais estável
            PyNVMLStrategy(logger),
            NvitopStrategy(logger),
        ]
        
        return [strategy for strategy in strategies if strategy.is_available()]
    
    @staticmethod
    def get_best_available_strategy(logger=None) -> Optional[GPUDataCollectorStrategy]:
        """Retorna a melhor estratégia disponível."""
        available_strategies = GPUStrategyFactory.get_available_strategies(logger)
        return available_strategies[0] if available_strategies else None
