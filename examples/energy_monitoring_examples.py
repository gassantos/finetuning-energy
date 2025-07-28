#!/usr/bin/env python3
"""
Exemplo prático de uso do sistema de monitoramento energético sincronizado.
Demonstra como utilizar as novas funcionalidades para análise detalhada.
"""

import time
import json
from pathlib import Path
from src.monitor import RobustGPUMonitor
from src.energy_callback import EnergyTrackingCallback


def exemplo_monitoramento_basico():
    """Exemplo básico de monitoramento com alta precisão"""
    print("🔋 === EXEMPLO: Monitoramento Básico com Alta Precisão ===")
    
    # Criar monitor com alta precisão
    monitor = RobustGPUMonitor(
        sampling_interval=0.5,  # Amostragem a cada 0.5s
        enable_high_precision=True  # Ativar baseline automático
    )
    
    # Verificar capacidades disponíveis
    capabilities = monitor.detect_monitoring_capabilities()
    print(f"📊 Capacidades detectadas: {capabilities}")
    
    # Iniciar monitoramento
    if monitor.start_monitoring():
        print("✅ Monitoramento iniciado com sucesso!")
        
        # Simular algum trabalho
        print("⏳ Simulando carga de trabalho por 10 segundos...")
        time.sleep(10)
        
        # Obter métricas incrementais
        print("📈 Obtendo métricas dos últimos 5 segundos...")
        recent_metrics = monitor.get_metrics_since_timestamp(time.time() - 5)
        
        if "error" not in recent_metrics:
            print(f"⚡ Consumo recente: {recent_metrics}")
        
        # Parar monitoramento
        final_data = monitor.stop_monitoring()
        print(f"📊 Dados finais: {json.dumps(final_data, indent=2, default=str)}")
        
        return final_data
    else:
        print("❌ Falha ao iniciar monitoramento")
        return None


def exemplo_marcadores_sincronizacao():
    """Exemplo de uso de marcadores de sincronização"""
    print("\n🔋 === EXEMPLO: Marcadores de Sincronização ===")
    
    monitor = RobustGPUMonitor(enable_high_precision=True)
    
    if monitor.start_monitoring():
        print("✅ Monitoramento com marcadores iniciado!")
        
        # Simular steps de treinamento
        for step in range(1, 6):
            print(f"🔄 Simulando step {step}...")
            
            # Adicionar marcador para este step
            monitor.add_sync_marker(
                marker_type="training_step",
                step=step,
                epoch=step * 0.2,
                metadata={
                    "learning_rate": 0.001,
                    "batch_size": 16,
                    "simulated_loss": 1.0 / step  # Loss decrescente simulada
                }
            )
            
            time.sleep(2)  # Simular tempo de processamento
        
        # Parar e analisar
        final_data = monitor.stop_monitoring()
        
        print(f"📍 Marcadores coletados: {len(final_data['sync_markers'])}")
        for i, marker in enumerate(final_data['sync_markers']):
            print(f"  {i+1}. Step {marker['step']}: Loss={marker['metadata']['simulated_loss']:.3f}")
        
        return final_data
    else:
        print("❌ Falha ao iniciar monitoramento com marcadores")
        return None


def exemplo_callback_simulado():
    """Exemplo simulado do callback energético"""
    print("\n🔋 === EXEMPLO: Callback Energético (Simulado) ===")
    
    # Criar monitor
    monitor = RobustGPUMonitor(sampling_interval=0.2, enable_high_precision=True)
    
    # Criar callback
    callback = EnergyTrackingCallback(
        gpu_monitor=monitor,
        sync_interval_s=3.0  # Sincronizar a cada 3 segundos
    )
    
    # Simular dados de treinamento
    class MockTrainingArgs:
        logging_steps = 2
    
    class MockTrainerState:
        def __init__(self):
            self.global_step = 0
            self.epoch = 0.0
    
    class MockTrainerControl:
        pass
    
    args = MockTrainingArgs()
    state = MockTrainerState()
    control = MockTrainerControl()
    
    print("🚀 Simulando início do treinamento...")
    callback.on_train_begin(args, state, control)
    
    # Simular steps de treinamento
    for step in range(1, 11):
        state.global_step = step
        state.epoch = step * 0.1
        
        print(f"📈 Processando step {step}...")
        
        # Simular processamento do step
        time.sleep(1)
        
        # Chamar callback de fim de step
        callback.on_step_end(args, state, control)
    
    print("🏁 Simulando fim do treinamento...")
    final_energy_data = callback.on_train_end(args, state, control)
    
    # Obter histórico
    energy_history = callback.get_energy_history()
    print(f"📊 Histórico coletado:")
    print(f"  - Steps logados: {energy_history['total_logged_steps']}")
    print(f"  - Intervalos: {energy_history['total_intervals']}")
    
    return energy_history


def exemplo_analise_eficiencia():
    """Exemplo de análise de eficiência energética"""
    print("\n🔋 === EXEMPLO: Análise de Eficiência ===")
    
    monitor = RobustGPUMonitor(enable_high_precision=True)
    
    if not monitor.start_monitoring():
        print("❌ Falha ao iniciar monitoramento para análise de eficiência")
        return
    
    print("📊 Coletando dados para análise de eficiência...")
    
    # Simular diferentes cargas de trabalho
    workloads = [
        {"name": "Carga Leve", "duration": 3, "description": "Processamento mínimo"},
        {"name": "Carga Média", "duration": 4, "description": "Processamento moderado"},
        {"name": "Carga Pesada", "duration": 5, "description": "Processamento intensivo"}
    ]
    
    efficiency_data = []
    
    for workload in workloads:
        print(f"⚡ Testando: {workload['name']} ({workload['description']})")
        
        start_time = time.time()
        
        # Simular carga (em uma aplicação real, seria o processamento atual)
        time.sleep(workload['duration'])
        
        # Obter métricas para este período
        period_metrics = monitor.get_metrics_since_timestamp(start_time)
        
        if "error" not in period_metrics:
            # Calcular eficiência simples
            total_energy = sum(
                gpu_data.get("energy_consumed_wh", 0)
                for gpu_data in period_metrics["gpus"].values()
            )
            
            efficiency_score = total_energy / workload['duration'] if workload['duration'] > 0 else 0
            
            efficiency_data.append({
                "workload": workload['name'],
                "duration_s": workload['duration'],
                "energy_wh": total_energy,
                "efficiency_wh_per_s": efficiency_score
            })
            
            print(f"  📈 Energia: {total_energy:.2f}Wh, Eficiência: {efficiency_score:.2f}Wh/s")
    
    # Parar monitoramento
    final_data = monitor.stop_monitoring()
    
    # Análise comparativa
    print("\n📊 Análise Comparativa de Eficiência:")
    for data in efficiency_data:
        print(f"  {data['workload']}: {data['efficiency_wh_per_s']:.2f}Wh/s")
    
    # Encontrar mais eficiente
    if efficiency_data:
        most_efficient = min(efficiency_data, key=lambda x: x['efficiency_wh_per_s'])
        print(f"🏆 Mais eficiente: {most_efficient['workload']}")
    
    return efficiency_data


def exemplo_salvar_relatorio():
    """Exemplo de como salvar relatório detalhado"""
    print("\n🔋 === EXEMPLO: Relatório Detalhado ===")
    
    # Executar monitoramento
    monitor = RobustGPUMonitor(enable_high_precision=True)
    
    if monitor.start_monitoring():
        print("📊 Coletando dados para relatório...")
        time.sleep(5)
        
        energy_data = monitor.stop_monitoring()
        
        # Criar relatório
        report = {
            "timestamp": time.time(),
            "monitoring_config": {
                "sampling_interval_s": monitor.sampling_interval,
                "high_precision": monitor.enable_high_precision,
                "method": energy_data.get("monitoring_method", "unknown")
            },
            "summary": {
                "duration_s": energy_data.get("monitoring_duration_s", 0),
                "total_samples": energy_data.get("total_samples", 0),
                "baseline_power_w": energy_data.get("baseline_power_w", 0)
            },
            "gpu_details": energy_data.get("gpus", {}),
            "recommendations": []
        }
        
        # Adicionar recomendações baseadas nos dados
        for gpu_key, gpu_data in report["gpu_details"].items():
            if "statistics" in gpu_data:
                stats = gpu_data["statistics"]
                avg_power = stats.get("power_avg_w", 0)
                max_temp = stats.get("temperature_max_c", 0)
                
                if avg_power > 300:
                    report["recommendations"].append(
                        f"{gpu_key}: Alto consumo ({avg_power:.1f}W) - considere otimização"
                    )
                
                if max_temp > 80:
                    report["recommendations"].append(
                        f"{gpu_key}: Temperatura alta ({max_temp:.1f}°C) - verificar resfriamento"
                    )
        
        # Salvar relatório
        report_path = Path("results") / f"energy_report_{int(time.time())}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Relatório salvo em: {report_path}")
        print(f"📋 Recomendações: {len(report['recommendations'])}")
        for rec in report["recommendations"]:
            print(f"  ⚠️ {rec}")
        
        return report
    else:
        print("❌ Falha ao gerar relatório")
        return None


def main():
    """Executa todos os exemplos"""
    print("🔋 SISTEMA DE MONITORAMENTO ENERGÉTICO SINCRONIZADO")
    print("=" * 60)
    
    try:
        # Exemplo 1: Monitoramento básico
        exemplo_monitoramento_basico()
        
        # Exemplo 2: Marcadores de sincronização
        exemplo_marcadores_sincronizacao()
        
        # Exemplo 3: Callback simulado
        exemplo_callback_simulado()
        
        # Exemplo 4: Análise de eficiência
        exemplo_analise_eficiencia()
        
        # Exemplo 5: Relatório detalhado
        exemplo_salvar_relatorio()
        
        print("\n✅ Todos os exemplos executados com sucesso!")
        print("\n📖 Para usar em produção:")
        print("  1. Integre o EnergyTrackingCallback no seu Trainer")
        print("  2. Configure sync_interval_s baseado nas suas necessidades")
        print("  3. Analise as métricas no Wandb dashboard")
        print("  4. Use os relatórios locais para análise offline")
        
    except Exception as e:
        print(f"❌ Erro durante execução dos exemplos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
