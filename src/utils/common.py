"""
Utilitários comuns compartilhados entre módulos.
Contém funções auxiliares reutilizáveis para evitar duplicação de código.
"""

from typing import Any, Callable, TypeVar

T = TypeVar('T')


def safe_cast(value: Any, cast_func: Callable[[Any], T], default: T) -> T:
    """
    Helper para fazer cast seguro de valores com fallback.
    
    Args:
        value: Valor a ser convertido
        cast_func: Função de conversão (int, float, str, etc.)
        default: Valor padrão caso a conversão falhe
        
    Returns:
        Valor convertido ou valor padrão
        
    Example:
        >>> safe_cast("123", int, 0)
        123
        >>> safe_cast("invalid", int, 0)
        0
        >>> safe_cast(None, float, 1.0)
        1.0
    """
    try:
        return cast_func(value)
    except (ValueError, TypeError):
        return default
