#!/usr/bin/env python3
"""
Configuração centralizada para filtros de warnings.
Suprime warnings conhecidos que poluem o logging.
"""

import warnings
from typing import List


def configure_warning_filters() -> None:
    """
    Configura filtros de warnings para suprimir mensagens desnecessárias.
    
    Esta função deve ser chamada no início da aplicação para configurar
    todos os filtros de warnings em um local centralizado.
    """
    
    # Lista de warnings específicos para suprimir
    warning_filters = [
        # --- PyTorch DataLoader Warnings ---
        {
            "action": "ignore",
            "message": r".*pin_memory.*argument is set as true but no accelerator is found.*",
            "category": UserWarning,
            "module": r".*dataloader.*"
        },
        
        # --- PyTorch Dynamo/Checkpoint Warnings ---
        {
            "action": "ignore", 
            "message": r".*torch\.utils\.checkpoint.*use_reentrant parameter should be passed explicitly.*",
            "category": UserWarning,
            "module": r".*torch.*dynamo.*eval_frame.*"
        },
        
        # --- Transformers Warnings Comuns ---
        {
            "action": "ignore",
            "message": r".*was not found in model config.*",
            "category": UserWarning,
            "module": r".*transformers.*"
        },
        
        # --- HuggingFace Hub Warnings ---
        {
            "action": "ignore",
            "message": r".*You are using the default legacy behaviour.*",
            "category": UserWarning,
            "module": r".*transformers.*tokenization_utils_base.*"
        },
        
        # --- Accelerate Warnings ---
        {
            "action": "ignore",
            "message": r".*was not found in config\..*",
            "category": UserWarning,
            "module": r".*accelerate.*"
        },
        
        # --- PEFT Warnings ---
        {
            "action": "ignore",
            "message": r".*target_modules.*",
            "category": UserWarning,
            "module": r".*peft.*"
        },
        
        # --- General Warnings Conhecidos ---
        {
            "action": "ignore",
            "message": r".*FutureWarning.*",
            "category": FutureWarning,
            "module": r".*"
        },
        
        # --- Warnings de Deprecation que não podemos controlar ---
        {
            "action": "ignore",
            "message": r".*deprecated.*",
            "category": DeprecationWarning,
            "module": r".*"
        },
        
        # --- Warnings específicos do ambiente de desenvolvimento ---
        {
            "action": "ignore",
            "message": r".*Setuptools is replacing distutils.*",
            "category": UserWarning,
            "module": r".*setuptools.*"
        }
    ]
    
    # Aplicar todos os filtros
    for filter_config in warning_filters:
        warnings.filterwarnings(**filter_config)
    
    print("🔇 Filtros de warnings configurados com sucesso!")


def configure_specific_torch_warnings() -> None:
    """
    Configuração específica para warnings do PyTorch que são muito verbosos.
    """
    
    # Warnings específicos do PyTorch
    torch_warnings = [
        # DataLoader pin_memory warning
        r".*pin_memory.*argument is set as true but no accelerator is found.*",
        
        # Checkpoint reentrant warning  
        r".*torch\.utils\.checkpoint.*use_reentrant parameter should be passed explicitly.*",
        
        # Compilation warnings
        r".*torch\.jit\..*",
        
        # CUDA warnings desnecessários
        r".*CUDA.*initialization.*",
    ]
    
    for warning_pattern in torch_warnings:
        warnings.filterwarnings(
            "ignore",
            message=warning_pattern,
            category=UserWarning
        )
    
    print("⚡ Filtros específicos do PyTorch configurados!")


def configure_transformers_warnings() -> None:
    """
    Configuração específica para warnings do Transformers/HuggingFace.
    """
    
    transformers_warnings = [
        # Tokenizer warnings
        ".*You are using the default legacy behaviour.*",
        
        # Model config warnings
        ".*was not found in model config.*",
        
        # Loading warnings
        ".*Some weights.*were not initialized.*",
        
        # Attention warnings
        ".*The attention mask and the pad token.*",
    ]
    
    for warning_pattern in transformers_warnings:
        warnings.filterwarnings(
            "ignore", 
            message=warning_pattern,
            category=UserWarning
        )
    
    print("🤗 Filtros específicos do Transformers configurados!")


def get_current_warning_filters() -> List[str]:
    """
    Retorna lista dos filtros de warnings atualmente configurados.
    
    Returns:
        Lista de strings descrevendo os filtros ativos.
    """
    
    active_filters = []
    
    for filter_item in warnings.filters:
        action, message, category, module, lineno = filter_item
        filter_desc = f"{action}: {category.__name__ if category else 'Any'}"
        
        if message:
            filter_desc += f" - {message.pattern if hasattr(message, 'pattern') else message}"
        
        if module:
            filter_desc += f" (module: {module.pattern if hasattr(module, 'pattern') else module})"
            
        active_filters.append(filter_desc)
    
    return active_filters


def configure_all_warning_filters() -> None:
    """
    Configuração completa de todos os filtros de warnings.
    
    Esta é a função principal que deve ser chamada para configurar
    todos os filtros de warnings do projeto.
    """
    
    print("🔧 Configurando filtros de warnings...")
    
    # Configurar filtros gerais
    configure_warning_filters()
    
    # Configurar filtros específicos
    configure_specific_torch_warnings()
    configure_transformers_warnings()
    
    # Mostrar resumo
    active_filters = get_current_warning_filters()
    print(f"✅ Total de {len(active_filters)} filtros de warnings configurados")
    
    # Log dos filtros mais importantes (primeiros 5)
    print("📋 Principais filtros ativos:")
    for i, filter_desc in enumerate(active_filters[:5], 1):
        print(f"  {i}. {filter_desc}")
    
    if len(active_filters) > 5:
        print(f"  ... e mais {len(active_filters) - 5} filtros")


if __name__ == "__main__":
    # Teste da configuração
    configure_all_warning_filters()
    
    # Mostrar todos os filtros se executado diretamente
    print("\n🔍 TODOS OS FILTROS CONFIGURADOS:")
    for i, filter_desc in enumerate(get_current_warning_filters(), 1):
        print(f"{i:2d}. {filter_desc}")
