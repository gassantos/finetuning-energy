#!/usr/bin/env python3
"""Teste manual das funcionalidades que estavam falhando"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.monitor import RobustGPUMonitor

def test_manual():
    print("=== Teste Manual das Funcionalidades ===")
    
    # 1. Testar detecção de capacidades
    print("\n1. Testando detecção de capacidades...")
    monitor = RobustGPUMonitor()
    caps = monitor.detect_monitoring_capabilities()
    print(f"Capacidades: {caps}")
    
    # Verificar se nvidia_smi está presente (era um dos erros)
    if 'nvidia_smi' in caps:
        print("✅ nvidia_smi presente nas capacidades")
    else:
        print("❌ nvidia_smi ausente nas capacidades")
    
    # 2. Testar processamento sem dados
    print("\n2. Testando processamento sem dados...")
    result = monitor._process_energy_data()
    print(f"Resultado: {result}")
    
    # Verificar se monitoring_duration_s está presente (era outro erro)
    if 'monitoring_duration_s' in result:
        print("✅ monitoring_duration_s presente no resultado")
    else:
        print("❌ monitoring_duration_s ausente no resultado")
    
    # 3. Testar monitoramento rápido
    print("\n3. Testando monitoramento rápido...")
    monitor.sampling_interval = 0.1
    success = monitor.start_monitoring()
    print(f"Monitoramento iniciado: {success}")
    
    if success:
        import time
        time.sleep(0.3)  # Coletar algumas amostras
        monitor.stop_monitoring()
        
        result = monitor._process_energy_data()
        print(f"Resumo: {result}")
        
        # Verificar estrutura do resultado
        if 'monitoring_duration_s' in result:
            print("✅ monitoring_duration_s presente no resumo")
        else:
            print("❌ monitoring_duration_s ausente no resumo")
    
    print("\n=== Teste Manual Concluído ===")

if __name__ == "__main__":
    test_manual()
