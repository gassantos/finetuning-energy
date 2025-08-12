import tiktoken
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from src.logging_config import get_logger

# Configurar logging estruturado
logger = get_logger(context="utils")


class TokenAnalyzer:
    """Classe para análise e contabilização de tokens em DataFrames."""
    
    def __init__(self, encoding: str = "cl100k_base"):
        """
        Inicializa o analisador de tokens.
        
        Args:
            encoding (str): Nome da codificação do tokenizer
        """
        self.tokenizer = tiktoken.get_encoding(encoding)
    
    def _calculate_tokens_for_column(self, series: pd.Series) -> pd.Series:
        """
        Calcula tokens para uma série de dados.
        
        Args:
            series (pd.Series): Série de dados para calcular tokens
            
        Returns:
            pd.Series: Série com contagem de tokens
        """
        return series.fillna('').apply(lambda x: len(self.tokenizer.encode(str(x))))
    
    def _get_token_statistics(self, tokens: pd.Series, column_name: str) -> Dict:
        """
        Calcula estatísticas de tokens para uma coluna.
        
        Args:
            tokens (pd.Series): Série com contagem de tokens
            column_name (str): Nome da coluna original
            
        Returns:
            Dict: Estatísticas de tokens
        """
        stats = {
            'min_tokens': tokens.min(),
            'max_tokens': tokens.max(),
            'mean_tokens': tokens.mean(),
            'total_tokens': tokens.sum(),
            'rows_with_content': tokens.count()
        }

        print(f"Estatísticas de tokens - {column_name}: "
              f"min={stats['min_tokens']}, max={stats['max_tokens']}, "
              f"média={stats['mean_tokens']:.1f}"
        )

        return stats
    
    def identify_text_columns(
        self, 
        df: pd.DataFrame, 
        min_avg_length: int = 50, 
        text_threshold: float = 0.8
    ) -> List[str]:
        """
        Identifica colunas de texto longo no DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame para análise
            min_avg_length (int): Comprimento médio mínimo para considerar texto longo
            text_threshold (float): Proporção mínima de valores não-nulos
            
        Returns:
            List[str]: Lista de nomes das colunas de texto identificadas
        """
        text_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Verificar proporção de valores não-nulos
                non_null_ratio = df[col].notna().sum() / len(df)
                if non_null_ratio >= text_threshold:
                    # Calcular comprimento médio
                    avg_length = df[col].fillna('').astype(str).str.len().mean()
                    if avg_length >= min_avg_length:
                        text_columns.append(col)
        
        return text_columns
    
    def get_info_tokens(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        min_avg_length: int = 50, 
        text_threshold: float = 0.8,
        add_token_columns: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Analisa tokens em colunas especificadas ou identificadas automaticamente.
        
        Args:
            df (pd.DataFrame): DataFrame para análise
            columns (List[str], optional): Colunas específicas para analisar
            min_avg_length (int): Comprimento médio mínimo para identificação automática
            text_threshold (float): Proporção mínima de valores não-nulos
            add_token_columns (bool): Se deve adicionar colunas de tokens ao DataFrame
            
        Returns:
            Tuple[pd.DataFrame, Dict]: DataFrame modificado e informações sobre tokens
        """
        # Criar cópia do DataFrame
        df_result = df.copy()
        
        # Determinar colunas para análise
        if columns is None:
            columns = self.identify_text_columns(df, min_avg_length, text_threshold)
            print(f"Colunas de texto identificadas: {columns}")
        else:
            # Validar se as colunas existem
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Colunas não encontradas: {missing_columns}")
        
        # Calcular tokens para cada coluna
        token_info = {}
        for col in columns:
            print(f"Calculando tokens para coluna '{col}'...")

            # Calcular tokens
            tokens = self._calculate_tokens_for_column(df[col])
            
            # Adicionar coluna de tokens se solicitado
            if add_token_columns:
                token_column_name = f'tokens_{col.lower()}'
                df_result[token_column_name] = tokens
            
            # Coletar estatísticas
            token_info[col] = self._get_token_statistics(tokens, col)
            token_info[col]['rows_with_content'] = df[col].notna().sum()
        
        return df_result, token_info


def process_texto_resumo_tokens(
    df: pd.DataFrame, 
    dataset_path: Optional[str] = None,
    encoding: str = "cl100k_base"
) -> pd.DataFrame:
    """
    Contabiliza tokens nas colunas 'Texto' e 'Resumo' e exporta o dataset.
    
    Args:
        df (pd.DataFrame): DataFrame com colunas 'Texto' e 'Resumo'
        dataset_path (str, optional): Caminho do dataset original
        encoding (str): Codificação do tokenizer
        
    Returns:
        pd.DataFrame: DataFrame com colunas de tokens adicionadas
    """
    # Verificar se as colunas obrigatórias existem
    required_columns = ['Texto', 'Resumo']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas não encontradas: {missing_columns}")
    
    # Usar o TokenAnalyzer para processar as colunas
    analyzer = TokenAnalyzer(encoding)
    df_tokens, token_info = analyzer.get_info_tokens(
        df, 
        columns=required_columns,
        add_token_columns=True
    )
    
    # Determinar caminho de saída
    output_path = _determine_output_path(dataset_path)

    # Exportar dataset
    df_tokens.to_csv(output_path, index=False)
    print(f"Dataset com tokens exportado para: {output_path}")

    return df_tokens


def _determine_output_path(dataset_path: Optional[str]) -> str:
    """
    Determina o caminho de saída baseado no caminho do dataset original.
    
    Args:
        dataset_path (str, optional): Caminho do dataset original
        
    Returns:
        str: Caminho para o arquivo de saída
    """
    if dataset_path:
        dir_path = os.path.dirname(dataset_path)
        filename = os.path.splitext(os.path.basename(dataset_path))[0]
        return os.path.join(dir_path, f"{filename}_tokens.csv")
    else:
        return "dataset_tokens.csv"


# Função de conveniência para manter compatibilidade
def get_info_tokens(df, min_avg_length=50, text_threshold=0.8):
    """
    Função de conveniência que mantém a interface original.
    
    Args:
        df (pd.DataFrame): DataFrame para análise
        min_avg_length (int): Comprimento médio mínimo
        text_threshold (float): Proporção mínima de valores não-nulos
        
    Returns:
        dict: Informações sobre tokens por coluna
    """
    analyzer = TokenAnalyzer()
    _, token_info = analyzer.get_info_tokens(
        df, 
        min_avg_length=min_avg_length, 
        text_threshold=text_threshold
    )
    return token_info