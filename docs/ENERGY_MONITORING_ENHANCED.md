# Monitoramento Energético Sincronizado - Implementação e Melhores Práticas

## 🎯 Visão Geral

Este documento descreve as melhorias implementadas no sistema de monitoramento energético para fine-tuning de LLMs, com foco em sincronização precisa e melhores práticas para medição de consumo energético.

## 🔧 Principais Melhorias Implementadas

### 1. **Sistema de Callback Sincronizado (`EnergyTrackingCallback`)**

#### Funcionalidades:
- **Sincronização dupla**: Métricas coletadas tanto por intervalos de tempo (10s) quanto por steps de treinamento
- **Baseline energético**: Estabelece consumo base antes do treinamento para cálculos diferenciais
- **Callback nativo do Transformers**: Integração perfeita com o ciclo de vida do treinamento
- **Métricas de eficiência**: Cálculos automatizados de eficiência energética por step

#### Estrutura de Dados:
```python
@dataclass
class EnergyMetrics:
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
```

### 2. **Monitor GPU Aprimorado (`RobustGPUMonitor`)**

#### Novas Funcionalidades:
- **Alta precisão**: Modo de alta precisão com baseline automático
- **Sincronização por timestamp**: Método `get_metrics_since_timestamp()` para cálculos incrementais
- **Marcadores de sincronização**: Sistema para correlacionar eventos do treinamento
- **Consumo diferencial**: Subtração do baseline para medições mais precisas

#### Método de Baseline:
```python
def _establish_baseline(self):
    """
    Coleta 5 segundos de dados antes do treinamento começar
    para estabelecer consumo base do sistema
    """
```

### 3. **Logging Estruturado no Wandb**

#### Categorias de Métricas:

**Por Intervalo de Tempo (10s):**
- `energy_interval/power_avg_w`: Potência média
- `energy_interval/energy_consumed_wh`: Energia consumida no intervalo
- `energy_interval/efficiency_wh_per_step`: Eficiência energética
- `energy_interval/temperature_avg_c`: Temperatura média
- `energy_interval/gpu_utilization_avg_percent`: Utilização média

**Por Step de Treinamento:**
- `energy_step/power_avg_w`: Potência no step
- `energy_step/energy_per_step_wh`: Energia por step
- `energy_step/energy_efficiency`: Score de eficiência calculado
- `energy_step/temperature_avg_c`: Temperatura no step

**Métricas Finais:**
- `final_energy/total_energy_consumed_kwh`: Consumo total
- `final_energy/energy_per_step_wh`: Energia média por step
- `final_energy/energy_per_epoch_kwh`: Energia média por época
- `final_energy/energy_efficiency_score`: Score final de eficiência
- `final_energy/carbon_footprint_estimate_kg`: Pegada de carbono estimada

## 📊 Melhores Práticas Implementadas

### 1. **Medição Precisa de Energia**

#### Baseline Energético:
- Coleta 5 segundos de dados antes do treinamento
- Subtrai consumo base do sistema operacional e drivers
- Isola o consumo específico do treinamento

#### Cálculo Diferencial:
```python
baseline_adjusted_power = max(0, current_power - baseline_power)
energy_consumed = baseline_adjusted_power * duration_hours
```

### 2. **Sincronização Temporal**

#### Duas Estratégias Complementares:
1. **Intervalo fixo (10s)**: Garante medições consistentes independente da velocidade do treinamento
2. **Por step**: Permite correlação direta com progresso do modelo

#### Vantagens:
- Dados consistentes para comparação entre experimentos
- Correlação precisa entre performance e consumo energético
- Detecção de anomalias ou picos de consumo

### 3. **Métricas de Eficiência**

#### Score de Eficiência:
```python
def _calculate_energy_efficiency(self, metrics):
    """
    Calcula eficiência como energia por step ajustada pela utilização.
    Valores menores = maior eficiência
    """
    energy_per_step = metrics.energy_consumed_wh / steps_processed
    utilization_factor = metrics.utilization_avg_percent / 100.0
    return energy_per_step / max(0.01, utilization_factor)
```

#### Pegada de Carbono:
```python
def _estimate_carbon_footprint(self, total_energy_kwh):
    """
    Estima CO2 usando fator médio global de 0.5 kg CO2/kWh
    """
    return total_energy_kwh * 0.5
```

### 4. **Armazenamento e Análise**

#### Múltiplos Formatos:
- **Wandb**: Visualização em tempo real e análise online
- **JSON local**: Backup e análise offline
- **Históricos separados**: Por step e por intervalo para diferentes análises

#### Estrutura de Arquivos:
```
results/
├── energy_step_history_20250728_143022.json
├── energy_interval_history_20250728_143022.json
├── energy_monitoring_summary_20250728_143022.json
└── robust_gpu_energy_20250728_143022.json
```

## 🚀 Como Usar

### 1. **Configuração Automática**
O sistema é ativado automaticamente no `LlamaFineTuner`:

```python
fine_tuner = LlamaFineTuner(wandb_key, hf_token)
trainer = fine_tuner.run_complete_pipeline(dataset_path)
```

### 2. **Configurações Personalizáveis**

#### Intervalo de Sincronização:
```python
energy_callback = EnergyTrackingCallback(
    gpu_monitor=self.gpu_monitor,
    sync_interval_s=5.0  # Sincronizar a cada 5s
)
```

#### Modo de Alta Precisão:
```python
gpu_monitor = RobustGPUMonitor(
    sampling_interval=0.5,  # Amostragem mais frequente
    enable_high_precision=True  # Baseline automático
)
```

### 3. **Análise no Wandb**

#### Dashboard Recomendado:
1. **Gráfico de linha**: `energy_interval/power_avg_w` vs tempo
2. **Scatter plot**: `energy_step/energy_per_step_wh` vs step
3. **Heatmap**: Correlação entre temperatura e utilização
4. **Barras**: Comparação de eficiência entre épocas

## 📈 Benefícios da Implementação

### 1. **Precisão Melhorada**
- Baseline elimina ruído do sistema operacional
- Sincronização precisa com events do treinamento
- Cálculos diferenciais mais confiáveis

### 2. **Análise Detalhada**
- Correlação direta entre performance e consumo
- Identificação de ineficiências por step
- Tracking de tendências ao longo do treinamento

### 3. **Comparabilidade**
- Métricas padronizadas entre experimentos
- Intervalos de tempo consistentes
- Scores de eficiência normalizados

### 4. **Sustentabilidade**
- Estimativa de pegada de carbono
- Otimização energética baseada em dados
- Consciência ambiental no desenvolvimento de AI

## 🔮 Extensões Futuras

1. **Previsão de Consumo**: Modelos preditivos baseados no histórico
2. **Otimização Automática**: Ajuste de hiperparâmetros para eficiência
3. **Alertas Inteligentes**: Notificações para consumo anômalo
4. **Comparação de Modelos**: Benchmarks energéticos automáticos
5. **Integração com Carbon Tracker**: APIs de fatores de emissão regionais

## 📝 Exemplo de Saída

```
🔋 RELATÓRIO ENERGÉTICO FINAL:
• Duração: 3600s (1.00h)
• Steps: 1000
• Épocas: 3.00
• Energia total: 0.5500 kWh
• Potência média: 550.0W
• Energia por step: 1.98Wh
• Energia por época: 0.1833kWh
• Score de eficiência: 2.1543
• Pegada de carbono estimada: 0.000275kg CO2
```

Este sistema implementa as melhores práticas atuais para monitoramento energético em treinamento de LLMs, proporcionando visibilidade completa sobre o consumo e eficiência energética do processo.
