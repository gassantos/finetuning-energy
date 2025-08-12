"""
Configuração centralizada de logging estruturado com structlog
Implementa logging com emojis, cores e estrutura consistente
"""

import sys
import structlog
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
import colorama
from colorama import Fore, Style

# Inicializar colorama para Windows
colorama.init(autoreset=True)


class ColoredFormatter:
    """Formatter customizado com cores e emojis"""
    
    # Mapeamento de níveis para cores e emojis
    LEVEL_COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT
    }
    
    LEVEL_EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '📋',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    # Emojis para contextos específicos
    CONTEXT_EMOJIS = {
        'preprocessing': '🔄',
        'finetuning': '🎯',
        'energy': '⚡',
        'model': '🤖',
        'dataset': '📊',
        'training': '🚀',
        'monitoring': '📈',
        'gpu': '🎮',
        'memory': '💾',
        'file': '📁',
        'network': '🌐',
        'auth': '🔐',
        'config': '⚙️',
        'pipeline': '🏭',
        'success': '✅',
        'failure': '❌',
        'start': '🚀',
        'stop': '🛑',
        'save': '💾',
        'load': '📥',
        'process': '⚙️'
    }


def add_emojis_and_colors(logger, method_name, event_dict):
    """Processador que adiciona emojis e cores aos logs"""
    
    # Obter nível do log
    level = event_dict.get('level', '').upper()
    
    # Obter contexto se disponível
    context = event_dict.get('context', '')
    
    # Adicionar emoji baseado no nível
    level_emoji = ColoredFormatter.LEVEL_EMOJIS.get(level, '📝')
    
    # Adicionar emoji baseado no contexto
    context_emoji = ''
    if context:
        context_emoji = ColoredFormatter.CONTEXT_EMOJIS.get(context.lower(), '')
    
    # Construir prefixo com emojis
    emoji_prefix = f"{level_emoji} {context_emoji}".strip()
    if emoji_prefix:
        event_dict['emoji'] = emoji_prefix
    
    # Adicionar cor ao nível
    if level in ColoredFormatter.LEVEL_COLORS:
        color = ColoredFormatter.LEVEL_COLORS[level]
        event_dict['level_colored'] = f"{color}{level}{Style.RESET_ALL}"
    else:
        event_dict['level_colored'] = level
    
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Processador que adiciona timestamp formatado"""
    event_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return event_dict


def add_module_info(logger, method_name, event_dict):
    """Processador que adiciona informações do módulo"""
    # Obter nome do módulo atual
    import inspect
    frame = inspect.currentframe()
    try:
        # Subir na pilha para encontrar o módulo chamador
        for _ in range(10):  # Limite para evitar loop infinito
            frame = frame.f_back
            if frame is None:
                break
            
            module_name = frame.f_globals.get('__name__')
            if module_name and not module_name.startswith('structlog') and module_name != '__main__':
                # Extrair apenas o nome do módulo sem o caminho completo
                if '.' in module_name:
                    module_name = module_name.split('.')[-1]
                event_dict['module'] = module_name
                break
    finally:
        del frame
    
    return event_dict


def format_event(logger, method_name, event_dict):
    """Formatter final que monta a mensagem de log"""
    
    # Componentes da mensagem
    timestamp = event_dict.get('timestamp', '')
    level_colored = event_dict.get('level_colored', event_dict.get('level', ''))
    emoji = event_dict.get('emoji', '')
    module = event_dict.get('module', '')
    event = event_dict.get('event', '')
    
    # Montar prefixo
    prefix_parts = []
    if timestamp:
        prefix_parts.append(f"{Fore.CYAN}{timestamp}{Style.RESET_ALL}")
    if level_colored:
        prefix_parts.append(f"[{level_colored}]")
    if module:
        prefix_parts.append(f"{Fore.BLUE}{module}{Style.RESET_ALL}")
    
    prefix = " ".join(prefix_parts)
    
    # Montar mensagem principal
    message_parts = []
    if emoji:
        message_parts.append(emoji)
    if event:
        message_parts.append(event)
    
    main_message = " ".join(message_parts)
    
    # Combinar tudo
    if prefix:
        formatted_message = f"{prefix} - {main_message}"
    else:
        formatted_message = main_message
    
    # Adicionar campos extras se existirem
    extra_fields = []
    for key, value in event_dict.items():
        if key not in ['timestamp', 'level', 'level_colored', 'emoji', 'module', 'event', 'context']:
            if isinstance(value, (dict, list)):
                extra_fields.append(f"{key}={value}")
            else:
                extra_fields.append(f"{key}={value}")
    
    if extra_fields:
        extra_str = " | ".join(extra_fields)
        formatted_message += f" | {Fore.MAGENTA}{extra_str}{Style.RESET_ALL}"
    
    return formatted_message


def configure_structlog(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json_logs: bool = False
):
    """
    Configura o structlog para todo o projeto
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho opcional para arquivo de log
        enable_json_logs: Se deve habilitar logs em formato JSON
    """
    
    # Configurar nível de logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Processadores para console (com cores e emojis)
    console_processors = [
        add_timestamp,
        add_module_info,
        add_emojis_and_colors,
        format_event,
    ]
    
    # Processadores para arquivo (estruturados)
    file_processors = [
        add_timestamp,
        add_module_info,
        structlog.processors.JSONRenderer() if enable_json_logs else structlog.dev.ConsoleRenderer()
    ]
    
    # Configurar handlers
    handlers = []
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)
    
    # Handler para arquivo se especificado
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    # Configurar logging básico
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format='%(message)s'  # structlog vai formatar
    )
    
    # Configurar structlog
    structlog.configure(
        processors=console_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Silenciar logs excessivos de bibliotecas externas
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None, context: Optional[str] = None) -> structlog.BoundLogger:
    """
    Obter logger estruturado
    
    Args:
        name: Nome do logger (opcional)
        context: Contexto do logger para emojis apropriados
    
    Returns:
        Logger estruturado configurado
    """
    logger = structlog.get_logger(name)
    
    # Adicionar contexto se fornecido
    if context:
        logger = logger.bind(context=context)
    
    return logger


# Funções de conveniência com contextos predefinidos
def get_preprocessing_logger() -> structlog.BoundLogger:
    """Logger para pré-processamento"""
    return get_logger(context="preprocessing")


def get_finetuning_logger() -> structlog.BoundLogger:
    """Logger para fine-tuning"""
    return get_logger(context="finetuning")


def get_energy_logger() -> structlog.BoundLogger:
    """Logger para monitoramento energético"""
    return get_logger(context="energy")


def get_model_logger() -> structlog.BoundLogger:
    """Logger para operações do modelo"""
    return get_logger(context="model")


def get_dataset_logger() -> structlog.BoundLogger:
    """Logger para operações de dataset"""
    return get_logger(context="dataset")


def get_training_logger() -> structlog.BoundLogger:
    """Logger para treinamento"""
    return get_logger(context="training")


def get_pipeline_logger() -> structlog.BoundLogger:
    """Logger para pipeline principal"""
    return get_logger(context="pipeline")


def get_monitor_logger() -> structlog.BoundLogger:
    """Logger para monitoramento de GPU e energia"""
    return get_logger(context="monitor")


# Configuração padrão do projeto
def setup_project_logging():
    """Configura logging padrão para o projeto finetuning-energy"""
    
    # Determinar nível baseado em ambiente
    import os
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Criar diretório de logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configurar com arquivo de log
    log_file = log_dir / f"finetuning_energy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    configure_structlog(
        log_level=log_level,
        log_file=str(log_file),
        enable_json_logs=False  # Usar formato legível por padrão
    )
    
    # Logger inicial
    logger = get_pipeline_logger()
    logger.info("Sistema de logging estruturado configurado", 
                log_level=log_level, log_file=str(log_file))

