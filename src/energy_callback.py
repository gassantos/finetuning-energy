"""
Callback personalizado para monitoramento energético sincronizado com o treinamento.
Implementa melhores práticas para medição de consumo energético em LLMs.
"""

import time
from typing import Optional
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments
import wandb
from dataclasses import dataclass
from src.logging_config import get_energy_logger

# Configurar logging estruturado para energia
energy_logger = get_energy_logger()

@dataclass
class EnergyMetrics:
    """Estrutura para armazenar métricas energéticas de um período"""
    timestamp: float
    step: int
    epoch: float
    power_avg_w: float
    power_max_w: float
    power_min_w: float
    energy_consumed_wh: float
    temperature_avg_c: float
    temperature_max_c: float
    utilization_avg_percent: float
    memory_used_avg_mb: float
    gpu_name: str
    duration_s: float
    samples_count: int


class EnergyTrackingCallback(TrainerCallback):
    """
    Callback para monitoramento energético sincronizado com o treinamento.
    
    Implementa as melhores práticas:
    - Medição contínua em background
    - Sincronização com steps de treinamento 
    - Logging estruturado no Wandb
    - Cálculos precisos de energia consumida
    - Correlação com métricas de performance
    """
    
    def __init__(self, gpu_monitor, sync_interval_s: float = 10.0):
        self.gpu_monitor = gpu_monitor
        self.sync_interval_s = sync_interval_s
        self.last_sync_time = None
        self.last_sync_step = 0
        self.training_start_time = None
        self.step_energy_history = []
        self.interval_energy_history = []
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Inicia monitoramento quando o treinamento começa"""
        energy_logger.info("Iniciando monitoramento energético sincronizado", 
                          sync_interval_s=self.sync_interval_s)
        
        self.training_start_time = time.time()
        self.last_sync_time = self.training_start_time
        self.last_sync_step = 0
        
        # Iniciar monitoramento contínuo
        if not self.gpu_monitor.start_monitoring():
            energy_logger.error("Falha ao iniciar monitoramento GPU")
            return
            
        energy_logger.info("Monitoramento iniciado", sync_interval_s=self.sync_interval_s)
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Processa métricas energéticas a cada step e intervalo de tempo"""
        current_time = time.time()
        current_step = state.global_step
        
        # Verificar se é hora de sincronizar por intervalo de tempo
        time_since_last_sync = current_time - self.last_sync_time
        should_sync_by_time = time_since_last_sync >= self.sync_interval_s
        
        # Sempre processar por step (mas só logar se configurado)
        should_log_step = current_step % args.logging_steps == 0
        
        if should_sync_by_time or should_log_step:
            energy_metrics = self._calculate_energy_metrics_since_last_sync(
                current_time, current_step, state.epoch
            )
            
            if energy_metrics:
                # Log por intervalo de tempo
                if should_sync_by_time:
                    self._log_interval_metrics(energy_metrics)
                    self.interval_energy_history.append(energy_metrics)
                    self.last_sync_time = current_time
                
                # Log por step (se habilitado)
                if should_log_step:
                    self._log_step_metrics(energy_metrics)
                    self.step_energy_history.append(energy_metrics)
                
                self.last_sync_step = current_step
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finaliza monitoramento e gera relatório final"""
        print("🔋 Finalizando monitoramento energético...")
        
        # Parar monitoramento e obter dados finais
        final_energy_data = self.gpu_monitor.stop_monitoring()
        
        # Calcular métricas finais do treinamento
        final_metrics = self._calculate_final_training_metrics(final_energy_data, state)
        
        # Log das métricas finais
        self._log_final_metrics(final_metrics)
        
        print("✅ Monitoramento energético finalizado")
        
        return final_energy_data
    
    def _calculate_energy_metrics_since_last_sync(self, current_time: float, current_step: int, current_epoch: float) -> Optional[EnergyMetrics]:
        """Calcula métricas energéticas desde a última sincronização"""
        if not self.gpu_monitor.energy_data:
            return None
            
        # Filtrar dados desde a última sincronização
        since_last_sync = [
            sample for sample in self.gpu_monitor.energy_data
            if sample["timestamp"] >= self.last_sync_time
        ]
        
        if not since_last_sync:
            return None
            
        # Calcular métricas agregadas para todas as GPUs
        return self._aggregate_gpu_metrics(
            since_last_sync, current_time, current_step, current_epoch
        )
    
    def _aggregate_gpu_metrics(self, samples: list, current_time: float, current_step: int, current_epoch: float) -> EnergyMetrics:
        """Agrega métricas de todas as GPUs em um período"""
        total_power_samples = []
        total_temp_samples = []
        total_util_samples = []
        total_memory_samples = []
        gpu_names = set()
        
        for sample in samples:
            for gpu_data in sample["gpus"]:
                if "error" not in gpu_data:
                    total_power_samples.append(gpu_data.get("power_draw_w", 0))
                    total_temp_samples.append(gpu_data.get("temperature_c", 0))
                    total_util_samples.append(gpu_data.get("utilization_gpu_percent", 0))
                    total_memory_samples.append(gpu_data.get("memory_used_mb", 0))
                    gpu_names.add(gpu_data.get("name", "Unknown"))
        
        if not total_power_samples:
            return None
            
        duration_s = current_time - self.last_sync_time
        duration_h = duration_s / 3600
        
        # Calcular estatísticas
        power_avg = sum(total_power_samples) / len(total_power_samples)
        energy_wh = power_avg * duration_h
        
        return EnergyMetrics(
            timestamp=current_time,
            step=current_step,
            epoch=current_epoch,
            power_avg_w=power_avg,
            power_max_w=max(total_power_samples) if total_power_samples else 0,
            power_min_w=min(total_power_samples) if total_power_samples else 0,
            energy_consumed_wh=energy_wh,
            temperature_avg_c=sum(total_temp_samples) / len(total_temp_samples) if total_temp_samples else 0,
            temperature_max_c=max(total_temp_samples) if total_temp_samples else 0,
            utilization_avg_percent=sum(total_util_samples) / len(total_util_samples) if total_util_samples else 0,
            memory_used_avg_mb=sum(total_memory_samples) / len(total_memory_samples) if total_memory_samples else 0,
            gpu_name=", ".join(gpu_names),
            duration_s=duration_s,
            samples_count=len(samples)
        )
    
    def _log_interval_metrics(self, metrics: EnergyMetrics):
        """Log métricas por intervalo de tempo (10s)"""
        wandb.log({
            "energy_interval/timestamp": metrics.timestamp,
            "energy_interval/step": metrics.step,
            "energy_interval/epoch": metrics.epoch,
            "energy_interval/power_avg_w": metrics.power_avg_w,
            "energy_interval/power_max_w": metrics.power_max_w,
            "energy_interval/power_min_w": metrics.power_min_w,
            "energy_interval/energy_consumed_wh": metrics.energy_consumed_wh,
            "energy_interval/temperature_avg_c": metrics.temperature_avg_c,
            "energy_interval/temperature_max_c": metrics.temperature_max_c,
            "energy_interval/gpu_utilization_avg_percent": metrics.utilization_avg_percent,
            "energy_interval/memory_used_avg_mb": metrics.memory_used_avg_mb,
            "energy_interval/duration_s": metrics.duration_s,
            "energy_interval/samples_count": metrics.samples_count,
            "energy_interval/efficiency_wh_per_step": metrics.energy_consumed_wh / max(1, metrics.step - self.last_sync_step),
        })
    
    def _log_step_metrics(self, metrics: EnergyMetrics):
        """Log métricas por step de treinamento"""
        steps_since_last = metrics.step - self.last_sync_step
        
        wandb.log({
            "energy_step/step": metrics.step,
            "energy_step/epoch": metrics.epoch,
            "energy_step/power_avg_w": metrics.power_avg_w,
            "energy_step/energy_per_step_wh": metrics.energy_consumed_wh / max(1, steps_since_last),
            "energy_step/temperature_avg_c": metrics.temperature_avg_c,
            "energy_step/gpu_utilization_avg_percent": metrics.utilization_avg_percent,
            "energy_step/memory_used_avg_mb": metrics.memory_used_avg_mb,
            "energy_step/energy_efficiency": self._calculate_energy_efficiency(metrics),
        }, step=metrics.step)
    
    def _calculate_energy_efficiency(self, metrics: EnergyMetrics) -> float:
        """
        Calcula eficiência energética como uma métrica combinada.
        Menor é melhor (menos energia por unidade de progresso).
        """
        if metrics.step <= self.last_sync_step or metrics.utilization_avg_percent <= 0:
            return float('inf')
            
        steps_processed = metrics.step - self.last_sync_step
        energy_per_step = metrics.energy_consumed_wh / steps_processed
        utilization_factor = metrics.utilization_avg_percent / 100.0
        
        # Eficiência = energia por step ajustada pela utilização
        return energy_per_step / max(0.01, utilization_factor)
    
    def _calculate_final_training_metrics(self, final_energy_data: dict, state: TrainerState) -> dict:
        """Calcula métricas finais do treinamento completo"""
        if "error" in final_energy_data:
            return {"error": final_energy_data["error"]}
            
        total_duration_s = time.time() - self.training_start_time
        total_steps = state.global_step
        total_epochs = state.epoch
        
        # Agregar dados de todas as GPUs
        total_energy_kwh = 0
        avg_power_w = 0
        max_temperature_c = 0
        avg_utilization = 0
        
        for gpu_key, gpu_data in final_energy_data.get("gpus", {}).items():
            if "statistics" in gpu_data:
                stats = gpu_data["statistics"]
                total_energy_kwh += stats.get("energy_consumed_kwh", 0)
                avg_power_w += stats.get("power_avg_w", 0)
                max_temperature_c = max(max_temperature_c, stats.get("temperature_max_c", 0))
        
        # Calcular eficiência final
        energy_per_step_wh = (total_energy_kwh * 1000) / max(1, total_steps)
        energy_per_epoch_kwh = total_energy_kwh / max(1, total_epochs)
        
        return {
            "total_duration_s": total_duration_s,
            "total_steps": total_steps,
            "total_epochs": total_epochs,
            "total_energy_consumed_kwh": total_energy_kwh,
            "avg_power_w": avg_power_w,
            "max_temperature_c": max_temperature_c,
            "energy_per_step_wh": energy_per_step_wh,
            "energy_per_epoch_kwh": energy_per_epoch_kwh,
            "energy_efficiency_score": self._calculate_final_efficiency_score(total_energy_kwh, total_steps, total_duration_s),
            "carbon_footprint_estimate_kg": self._estimate_carbon_footprint(total_energy_kwh),
        }
    
    def _calculate_final_efficiency_score(self, total_energy_kwh: float, total_steps: int, duration_s: float) -> float:
        """
        Calcula um score de eficiência energética final.
        Considera energia por step e tempo de execução.
        """
        if total_steps == 0 or duration_s == 0:
            return 0
            
        energy_per_step = (total_energy_kwh * 1000) / total_steps  # Wh por step
        steps_per_hour = total_steps / (duration_s / 3600)
        
        # Score baseado na combinação de energia por step e velocidade
        # Valores menores são melhores
        efficiency_score = energy_per_step / max(1, steps_per_hour / 100)
        return round(efficiency_score, 4)
    
    def _estimate_carbon_footprint(self, total_energy_kwh: float) -> float:
        """
        Estima pegada de carbono baseada no consumo energético.
        Usa fator médio global de emissão de CO2.
        """
        # Fator médio global: ~0.5 kg CO2 por kWh (varia por região)
        carbon_factor_kg_per_kwh = 0.5
        return round(total_energy_kwh * carbon_factor_kg_per_kwh, 6)
    
    def _log_final_metrics(self, final_metrics: dict):
        """Log das métricas finais do treinamento"""
        if "error" in final_metrics:
            wandb.log({
                "final_energy/error": final_metrics["error"]
            })
            return
            
        wandb.log({
            "final_energy/total_duration_s": final_metrics["total_duration_s"],
            "final_energy/total_steps": final_metrics["total_steps"],
            "final_energy/total_epochs": final_metrics["total_epochs"],
            "final_energy/total_energy_consumed_kwh": final_metrics["total_energy_consumed_kwh"],
            "final_energy/avg_power_w": final_metrics["avg_power_w"],
            "final_energy/max_temperature_c": final_metrics["max_temperature_c"],
            "final_energy/energy_per_step_wh": final_metrics["energy_per_step_wh"],
            "final_energy/energy_per_epoch_kwh": final_metrics["energy_per_epoch_kwh"],
            "final_energy/energy_efficiency_score": final_metrics["energy_efficiency_score"],
            "final_energy/carbon_footprint_estimate_kg": final_metrics["carbon_footprint_estimate_kg"],
        })
        
        # Criar resumo textual
        summary = f"""
            🔋 RELATÓRIO ENERGÉTICO FINAL:
            • Duração: {final_metrics['total_duration_s']:.0f}s ({final_metrics['total_duration_s']/3600:.2f}h)
            • Steps: {final_metrics['total_steps']}
            • Épocas: {final_metrics['total_epochs']:.2f}
            • Energia total: {final_metrics['total_energy_consumed_kwh']:.4f} kWh
            • Potência média: {final_metrics['avg_power_w']:.1f}W
            • Energia por step: {final_metrics['energy_per_step_wh']:.2f}Wh
            • Energia por época: {final_metrics['energy_per_epoch_kwh']:.4f}kWh
            • Score de eficiência: {final_metrics['energy_efficiency_score']:.4f}
            • Pegada de carbono estimada: {final_metrics['carbon_footprint_estimate_kg']:.6f}kg CO2
        """
        
        print(summary)
        wandb.log({"final_energy/summary": summary})
    
    def get_energy_history(self) -> dict:
        """Retorna histórico completo de métricas energéticas"""
        return {
            "step_history": self.step_energy_history,
            "interval_history": self.interval_energy_history,
            "total_intervals": len(self.interval_energy_history),
            "total_logged_steps": len(self.step_energy_history)
        }
