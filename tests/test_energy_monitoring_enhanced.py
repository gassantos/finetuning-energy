"""
Testes para o sistema de monitoramento energético sincronizado.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from src.monitor import RobustGPUMonitor
from src.energy_callback import EnergyTrackingCallback, EnergyMetrics


class TestRobustGPUMonitorEnhanced:
    """Testes para as melhorias do RobustGPUMonitor"""

    def test_initialization_with_high_precision(self):
        """Testa inicialização com modo de alta precisão"""
        monitor = RobustGPUMonitor(sampling_interval=0.5, enable_high_precision=True)
        
        assert monitor.sampling_interval == 0.5
        assert monitor.enable_high_precision is True
        assert monitor.baseline_power is None
        assert monitor.sync_markers == []

    def test_sync_marker_addition(self):
        """Testa adição de marcadores de sincronização"""
        monitor = RobustGPUMonitor()
        
        # Adicionar marcador de step
        monitor.add_sync_marker("step", step=100, epoch=1.5, metadata={"loss": 0.5})
        
        assert len(monitor.sync_markers) == 1
        marker = monitor.sync_markers[0]
        assert marker["type"] == "step"
        assert marker["step"] == 100
        assert marker["epoch"] == 1.5
        assert marker["metadata"]["loss"] == 0.5
        assert "timestamp" in marker

    @patch('src.monitor.time.time')
    def test_get_metrics_since_timestamp(self, mock_time):
        """Testa obtenção de métricas desde timestamp específico"""
        monitor = RobustGPUMonitor()
        
        # Simular dados de energia
        base_time = 1000.0
        mock_time.return_value = base_time + 10
        
        monitor.energy_data = [
            {
                "timestamp": base_time + 1,
                "gpus": [{"gpu_id": 0, "power_draw_w": 100, "name": "GPU0"}]
            },
            {
                "timestamp": base_time + 5,
                "gpus": [{"gpu_id": 0, "power_draw_w": 150, "name": "GPU0"}]
            },
            {
                "timestamp": base_time + 8,
                "gpus": [{"gpu_id": 0, "power_draw_w": 120, "name": "GPU0"}]
            }
        ]
        
        # Obter métricas desde base_time + 3
        result = monitor.get_metrics_since_timestamp(base_time + 3)
        
        assert "error" not in result
        assert result["duration_s"] == 7.0  # base_time + 10 - (base_time + 3)
        assert "gpu_0" in result["gpus"]

    def test_baseline_establishment_simulation(self):
        """Testa estabelecimento de baseline (simulado)"""
        monitor = RobustGPUMonitor(enable_high_precision=True)
        
        # Simular método de coleta de dados
        def mock_collect_data():
            return [{"gpu_id": 0, "power_draw_w": 50, "name": "GPU0"}]
        
        monitor._collect_nvitop_data = mock_collect_data
        monitor.monitoring_method = "nvitop"
        
        # Simular estabelecimento de baseline
        with patch('src.monitor.time.time', side_effect=[1000, 1001, 1002, 1003, 1004, 1005, 1006]):
            with patch('src.monitor.time.sleep'):
                monitor._establish_baseline()
        
        assert monitor.baseline_power == 50.0


class TestEnergyTrackingCallback:
    """Testes para o callback de tracking energético"""

    def test_callback_initialization(self):
        """Testa inicialização do callback"""
        mock_monitor = Mock()
        callback = EnergyTrackingCallback(mock_monitor, sync_interval_s=5.0)
        
        assert callback.gpu_monitor == mock_monitor
        assert callback.sync_interval_s == 5.0
        assert callback.last_sync_time is None
        assert callback.step_energy_history == []
        assert callback.interval_energy_history == []

    def test_energy_metrics_creation(self):
        """Testa criação de objeto EnergyMetrics"""
        metrics = EnergyMetrics(
            timestamp=1000.0,
            step=100,
            epoch=1.5,
            power_avg_w=200.0,
            power_max_w=250.0,
            power_min_w=150.0,
            energy_consumed_wh=5.0,
            temperature_avg_c=75.0,
            temperature_max_c=80.0,
            utilization_avg_percent=85.0,
            memory_used_avg_mb=8192.0,
            gpu_name="RTX 4090",
            duration_s=10.0,
            samples_count=10
        )
        
        assert metrics.timestamp == 1000.0
        assert metrics.step == 100
        assert metrics.power_avg_w == 200.0
        assert metrics.gpu_name == "RTX 4090"

    @patch('src.energy_callback.wandb')
    def test_interval_metrics_logging(self, mock_wandb):
        """Testa logging de métricas por intervalo"""
        mock_monitor = Mock()
        callback = EnergyTrackingCallback(mock_monitor)
        
        metrics = EnergyMetrics(
            timestamp=1000.0, step=100, epoch=1.5, power_avg_w=200.0,
            power_max_w=250.0, power_min_w=150.0, energy_consumed_wh=5.0,
            temperature_avg_c=75.0, temperature_max_c=80.0,
            utilization_avg_percent=85.0, memory_used_avg_mb=8192.0,
            gpu_name="RTX 4090", duration_s=10.0, samples_count=10
        )
        
        callback.last_sync_step = 90
        callback._log_interval_metrics(metrics)
        
        # Verificar se wandb.log foi chamado
        mock_wandb.log.assert_called_once()
        logged_data = mock_wandb.log.call_args[0][0]
        
        assert "energy_interval/power_avg_w" in logged_data
        assert "energy_interval/efficiency_wh_per_step" in logged_data
        assert logged_data["energy_interval/power_avg_w"] == 200.0

    @patch('src.energy_callback.wandb')
    def test_step_metrics_logging(self, mock_wandb):
        """Testa logging de métricas por step"""
        mock_monitor = Mock()
        callback = EnergyTrackingCallback(mock_monitor)
        
        metrics = EnergyMetrics(
            timestamp=1000.0, step=100, epoch=1.5, power_avg_w=200.0,
            power_max_w=250.0, power_min_w=150.0, energy_consumed_wh=5.0,
            temperature_avg_c=75.0, temperature_max_c=80.0,
            utilization_avg_percent=85.0, memory_used_avg_mb=8192.0,
            gpu_name="RTX 4090", duration_s=10.0, samples_count=10
        )
        
        callback.last_sync_step = 90
        callback._log_step_metrics(metrics)
        
        # Verificar se wandb.log foi chamado com step
        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args
        logged_data = call_args[0][0]
        step_param = call_args[1]
        
        assert "energy_step/step" in logged_data
        assert "energy_step/energy_per_step_wh" in logged_data
        assert step_param["step"] == 100

    def test_energy_efficiency_calculation(self):
        """Testa cálculo de eficiência energética"""
        mock_monitor = Mock()
        callback = EnergyTrackingCallback(mock_monitor)
        callback.last_sync_step = 90
        
        metrics = EnergyMetrics(
            timestamp=1000.0, step=100, epoch=1.5, power_avg_w=200.0,
            power_max_w=250.0, power_min_w=150.0, energy_consumed_wh=10.0,
            temperature_avg_c=75.0, temperature_max_c=80.0,
            utilization_avg_percent=80.0, memory_used_avg_mb=8192.0,
            gpu_name="RTX 4090", duration_s=10.0, samples_count=10
        )
        
        efficiency = callback._calculate_energy_efficiency(metrics)
        
        # Cálculo esperado: (10.0 / 10 steps) / (80.0 / 100) = 1.0 / 0.8 = 1.25
        expected_efficiency = 1.25
        assert abs(efficiency - expected_efficiency) < 0.01

    def test_carbon_footprint_estimation(self):
        """Testa estimativa de pegada de carbono"""
        mock_monitor = Mock()
        callback = EnergyTrackingCallback(mock_monitor)
        
        # 1 kWh deve resultar em 0.5 kg CO2
        carbon_footprint = callback._estimate_carbon_footprint(1.0)
        assert carbon_footprint == 0.5
        
        # 0.5 kWh deve resultar em 0.25 kg CO2
        carbon_footprint = callback._estimate_carbon_footprint(0.5)
        assert carbon_footprint == 0.25

    def test_final_efficiency_score_calculation(self):
        """Testa cálculo do score final de eficiência"""
        mock_monitor = Mock()
        callback = EnergyTrackingCallback(mock_monitor)
        
        # Teste com valores típicos
        score = callback._calculate_final_efficiency_score(
            total_energy_kwh=0.5,  # 0.5 kWh
            total_steps=1000,      # 1000 steps
            duration_s=3600        # 1 hora
        )
        
        # Cálculo esperado:
        # energy_per_step = (0.5 * 1000) / 1000 = 0.5 Wh por step
        # steps_per_hour = 1000 / 1 = 1000 steps/hora
        # efficiency_score = 0.5 / (1000 / 100) = 0.5 / 10 = 0.05
        expected_score = 0.05
        assert abs(score - expected_score) < 0.01

    @patch('src.energy_callback.time.time')
    def test_aggregate_gpu_metrics(self, mock_time):
        """Testa agregação de métricas de GPU"""
        mock_monitor = Mock()
        callback = EnergyTrackingCallback(mock_monitor)
        callback.last_sync_time = 1000.0
        
        mock_time.return_value = 1010.0  # 10 segundos depois
        
        # Simular amostras de múltiplas GPUs
        samples = [
            {
                "timestamp": 1005.0,
                "gpus": [
                    {"gpu_id": 0, "power_draw_w": 200, "temperature_c": 70, 
                     "utilization_gpu_percent": 80, "memory_used_mb": 8000, "name": "GPU0"},
                    {"gpu_id": 1, "power_draw_w": 180, "temperature_c": 65,
                     "utilization_gpu_percent": 75, "memory_used_mb": 7500, "name": "GPU1"}
                ]
            },
            {
                "timestamp": 1008.0,
                "gpus": [
                    {"gpu_id": 0, "power_draw_w": 220, "temperature_c": 75,
                     "utilization_gpu_percent": 85, "memory_used_mb": 8200, "name": "GPU0"},
                    {"gpu_id": 1, "power_draw_w": 190, "temperature_c": 68,
                     "utilization_gpu_percent": 78, "memory_used_mb": 7800, "name": "GPU1"}
                ]
            }
        ]
        
        metrics = callback._aggregate_gpu_metrics(samples, 1010.0, 100, 1.5)
        
        assert metrics is not None
        assert metrics.step == 100
        assert metrics.epoch == 1.5
        assert metrics.duration_s == 10.0
        # Média de potência: (200 + 180 + 220 + 190) / 4 = 197.5
        assert abs(metrics.power_avg_w - 197.5) < 0.1


class TestIntegrationScenarios:
    """Testes de cenários de integração"""

    def test_complete_monitoring_cycle_simulation(self):
        """Simula um ciclo completo de monitoramento"""
        # Criar monitor com alta precisão
        monitor = RobustGPUMonitor(sampling_interval=0.1, enable_high_precision=True)
        
        # Simular métodos de coleta
        def mock_collect():
            return [{"gpu_id": 0, "power_draw_w": 150, "temperature_c": 70, 
                    "utilization_gpu_percent": 80, "memory_used_mb": 8000, "name": "TestGPU"}]
        
        monitor._collect_nvitop_data = mock_collect
        monitor.monitoring_method = "nvitop"
        
        # Simular estabelecimento de baseline
        monitor.baseline_power = 50.0
        
        # Adicionar alguns dados de energia simulados
        base_time = time.time()
        monitor.energy_data = [
            {
                "timestamp": base_time + i * 0.1,
                "method": "nvitop",
                "gpus": [{"gpu_id": 0, "power_draw_w": 150 + i * 5, "temperature_c": 70 + i,
                         "utilization_gpu_percent": 80, "memory_used_mb": 8000, "name": "TestGPU"}]
            }
            for i in range(10)
        ]
        
        # Processar dados
        result = monitor._process_energy_data()
        
        assert "error" not in result
        assert result["baseline_power_w"] == 50.0
        assert result["high_precision_enabled"] is True
        assert "gpu_0" in result["gpus"]
        
        gpu_stats = result["gpus"]["gpu_0"]["statistics"]
        assert "power_baseline_adjusted_w" in gpu_stats
        assert "energy_consumed_baseline_adjusted_kwh" in gpu_stats

    @patch('src.energy_callback.wandb')
    def test_callback_training_simulation(self, mock_wandb):
        """Simula uso do callback durante treinamento"""
        mock_monitor = Mock()
        mock_monitor.start_monitoring.return_value = True
        mock_monitor.energy_data = []
        
        callback = EnergyTrackingCallback(mock_monitor, sync_interval_s=1.0)
        
        # Simular estados de treinamento
        mock_args = Mock()
        mock_args.logging_steps = 5
        
        mock_state = Mock()
        mock_state.global_step = 0
        mock_state.epoch = 0.0
        
        mock_control = Mock()
        
        # Simular início do treinamento
        callback.on_train_begin(mock_args, mock_state, mock_control)
        
        assert callback.training_start_time is not None
        assert callback.last_sync_time is not None
        mock_monitor.start_monitoring.assert_called_once()
        
        # Simular alguns steps
        with patch('src.energy_callback.time.time', side_effect=[1000, 1002, 1005, 1010]):
            with patch.object(callback, '_calculate_energy_metrics_since_last_sync') as mock_calc:
                mock_calc.return_value = EnergyMetrics(
                    timestamp=1005.0, step=5, epoch=0.1, power_avg_w=200.0,
                    power_max_w=250.0, power_min_w=150.0, energy_consumed_wh=2.0,
                    temperature_avg_c=70.0, temperature_max_c=75.0,
                    utilization_avg_percent=80.0, memory_used_avg_mb=8000.0,
                    gpu_name="TestGPU", duration_s=5.0, samples_count=5
                )
                
                # Step que deveria logar (múltiplo de logging_steps)
                mock_state.global_step = 5
                mock_state.epoch = 0.1
                callback.on_step_end(mock_args, mock_state, mock_control)
                
                # Verificar se métricas foram calculadas e logadas
                mock_calc.assert_called()

    def test_error_handling_scenarios(self):
        """Testa cenários de tratamento de erros"""
        monitor = RobustGPUMonitor()
        
        # Teste com dados vazios
        result = monitor.get_metrics_since_timestamp(time.time())
        assert "error" in result
        assert "Nenhum dado disponível" in result["error"]
        
        # Teste com timestamp futuro
        monitor.energy_data = [{"timestamp": time.time() - 100, "gpus": []}]
        result = monitor.get_metrics_since_timestamp(time.time() + 100)
        assert "error" in result
        assert "Nenhum dado no período" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
