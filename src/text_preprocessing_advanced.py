"""
Módulo avançado de pré-processamento de texto para datasets de sumarização

Este módulo processa arquivos Excel para criar datasets estruturados,
seguindo boas práticas de engenharia de software para projetos de IA.

Compatível com:
- PyTorch
- HuggingFace Transformers
- Datasets library
"""

import pandas as pd
import re
import unicodedata
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import warnings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TextPreprocessingConfig:
    """Configuração para pré-processamento avançado de texto"""
    
    # Mapeamento de colunas do Excel para dataset estruturado
    text_column: str = "Texto"           # Campo principal do documento
    summary_column: str = "Resumo"       # Campo de resumo/sumarização
    title_column: str = "Processo"       # Campo de título/identificação
    
    # Colunas adicionais opcionais
    additional_columns: List[str] = field(default_factory=list)
    
    # Configurações de limpeza de texto
    clean_text: bool = True
    normalize_unicode: bool = True
    remove_extra_whitespace: bool = True
    fix_encoding_issues: bool = True
    
    # Configurações de filtros
    min_text_length: int = 500         # Ajustado para o dataset
    max_text_length: int = 50000       # Ajustado para textos longos
    min_summary_length: int = 100      # Ajustado para resumos
    max_summary_length: int = 5000     # Ajustado para resumos longos
    
    # Configurações de split
    test_size: float = 0.2
    validation_size: float = 0.1       # Do conjunto de treino
    random_state: int = 42
    
    # Configurações de saída
    output_dir: str = "data/processed"
    save_formats: List[str] = field(default_factory=lambda: ['json', 'parquet'])
    
    def __post_init__(self):
        """Pós-processamento de inicialização"""
        pass


class TextCleaner:
    """Classe para limpeza e normalização de texto"""
    
    @staticmethod
    def clean_text(text: str, config: TextPreprocessingConfig) -> str:
        """
        Limpa e normaliza texto
        
        Args:
            text: Texto a ser limpo
            config: Configuração de limpeza
            
        Returns:
            Texto limpo
        """
        if not isinstance(text, str):
            return ""
        
        # Corrigir problemas de encoding comuns
        if config.fix_encoding_issues:
            text = TextCleaner._fix_encoding_issues(text)
        
        # Normalizar unicode
        if config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remover espaços extras
        if config.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    @staticmethod
    def _fix_encoding_issues(text: str) -> str:
        """Corrige problemas comuns de encoding"""
        # Substituições comuns para caracteres mal codificados
        replacements = {
            '_x000D_': '\n',
            '_x000A_': '\n',
            '_x0009_': '\t',
            'â€™': "'",
            'â€œ': '"',
            'â€\u009d': '"',
            'â€"': '—',
            'â€"': '–',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Limpar quebras de linha múltiplas
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text


class AdvancedTextProcessor:
    """Processador principal para conversão de dados textuais estruturados"""
    
    def __init__(self, config: Optional[TextPreprocessingConfig] = None):
        """
        Inicializa o processador
        
        Args:
            config: Configuração de processamento
        """
        self.config = config or TextPreprocessingConfig()
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[Dict[str, List]] = None
        self.dataset_dict: Optional[DatasetDict] = None
        
        # Criar diretório de saída
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_excel(self, file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """
        Carrega dados do arquivo Excel
        
        Args:
            file_path: Caminho para o arquivo Excel
            sheet_name: Nome ou índice da planilha
            
        Returns:
            DataFrame carregado
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
            logger.info(f"Carregando arquivo Excel: {file_path}")
            self.raw_data = pd.read_excel(file_path, sheet_name=sheet_name)
            
            logger.info(f"Dados carregados: {len(self.raw_data)} linhas, {len(self.raw_data.columns)} colunas")
            logger.info(f"Colunas disponíveis: {list(self.raw_data.columns)}")
            
            # Verificar se as colunas necessárias existem
            self._validate_columns()
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo Excel: {e}")
            raise
    
    def _validate_columns(self):
        """Valida se as colunas necessárias existem no dataset"""
        if self.raw_data is None:
            raise ValueError("Dados não carregados")
            
        required_columns = [
            self.config.text_column,
            self.config.summary_column,
            self.config.title_column
        ]
        
        missing_columns = []
        for col in required_columns:
            if col not in self.raw_data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            available_cols = list(self.raw_data.columns)
            raise ValueError(
                f"Colunas obrigatórias não encontradas: {missing_columns}\n"
                f"Colunas disponíveis: {available_cols}\n"
                f"Configure as colunas corretas em TextPreprocessingConfig"
            )
    
    def preprocess_data(self) -> Dict[str, List]:
        """
        Processa os dados para formato estruturado de sumarização
        
        Returns:
            Dados processados no formato estruturado
        """
        if self.raw_data is None:
            raise ValueError("Nenhum dado carregado. Use load_excel() primeiro.")
        
        logger.info("Iniciando pré-processamento dos dados...")
        
        processed_records = []
        skipped_records = 0
        
        for idx, row in self.raw_data.iterrows():
            try:
                # Extrair e limpar campos principais
                text = TextCleaner.clean_text(str(row[self.config.text_column]), self.config)
                summary = TextCleaner.clean_text(str(row[self.config.summary_column]), self.config)
                title = TextCleaner.clean_text(str(row[self.config.title_column]), self.config)
                
                # Validar comprimentos
                if not self._validate_lengths(text, summary):
                    skipped_records += 1
                    continue
                
                # Criar registro no formato estruturado
                record = {
                    'text': text,
                    'summary': summary,
                    'title': title
                }
                
                # Adicionar campos extras se configurado
                for col in self.config.additional_columns:
                    if col in self.raw_data.columns:
                        value = TextCleaner.clean_text(str(row[col]), self.config)
                        record[col.lower().replace(' ', '_')] = value
                
                processed_records.append(record)
                
            except Exception as e:
                logger.warning(f"Erro ao processar linha {idx}: {e}")
                skipped_records += 1
                continue
        
        logger.info(f"Processamento concluído: {len(processed_records)} registros válidos, {skipped_records} ignorados")
        
        # Converter para formato de listas
        self.processed_data = {
            key: [record[key] for record in processed_records]
            for key in processed_records[0].keys() if processed_records
        }
        
        return self.processed_data
    
    def _validate_lengths(self, text: str, summary: str) -> bool:
        """Valida comprimentos dos campos de texto"""
        text_len = len(text)
        summary_len = len(summary)
        
        if text_len < self.config.min_text_length or text_len > self.config.max_text_length:
            return False
        
        if summary_len < self.config.min_summary_length or summary_len > self.config.max_summary_length:
            return False
        
        return True
    
    def create_dataset_splits(self) -> DatasetDict:
        """
        Cria splits de treino, validação e teste
        
        Returns:
            DatasetDict com os splits
        """
        if self.processed_data is None:
            raise ValueError("Dados não processados. Use preprocess_data() primeiro.")
        
        logger.info("Criando splits do dataset...")
        
        # Criar dataset inicial
        full_dataset = Dataset.from_dict(self.processed_data)
        
        # Split inicial: treino+validação vs teste
        train_val_data, test_data = train_test_split(
            list(range(len(full_dataset))),
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        train_val_dataset = full_dataset.select(train_val_data)
        test_dataset = full_dataset.select(test_data)
        
        # Split do conjunto de treino: treino vs validação
        if self.config.validation_size > 0:
            train_indices, val_indices = train_test_split(
                list(range(len(train_val_dataset))),
                test_size=self.config.validation_size,
                random_state=self.config.random_state
            )
            
            train_dataset = train_val_dataset.select(train_indices)
            val_dataset = train_val_dataset.select(val_indices)
            
            self.dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            })
        else:
            self.dataset_dict = DatasetDict({
                'train': train_val_dataset,
                'test': test_dataset
            })
        
        # Log dos tamanhos
        for split_name, dataset in self.dataset_dict.items():
            logger.info(f"Split '{split_name}': {len(dataset)} exemplos")
        
        return self.dataset_dict
    
    def save_dataset(self, format_types: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Salva o dataset nos formatos especificados
        
        Args:
            format_types: Lista de formatos para salvar ['json', 'parquet', 'csv']
            
        Returns:
            Dicionário com caminhos dos arquivos salvos
        """
        if self.dataset_dict is None:
            raise ValueError("Dataset não criado. Use create_dataset_splits() primeiro.")
        
        if format_types is None:
            format_types = self.config.save_formats
        
        saved_files = {}
        output_path = Path(self.config.output_dir)
        
        for format_type in format_types:
            logger.info(f"Salvando dataset no formato {format_type}...")
            
            if format_type == 'json':
                file_path = output_path / "dataset_structured_format.json"
                self.dataset_dict.save_to_disk(str(file_path.with_suffix('')))
                # Também salvar como JSON simples para compatibilidade
                simple_json_path = output_path / "dataset_simple.json"
                with open(simple_json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
                saved_files['json'] = str(simple_json_path)
                
            elif format_type == 'parquet':
                file_path = output_path / "dataset_structured_format.parquet"
                # Salvar cada split separadamente
                for split_name, dataset in self.dataset_dict.items():
                    split_path = output_path / f"dataset_{split_name}.parquet"
                    dataset.to_parquet(str(split_path))
                saved_files['parquet'] = str(output_path / "dataset_*.parquet")
                
            elif format_type == 'csv':
                # Salvar cada split como CSV
                for split_name, dataset in self.dataset_dict.items():
                    csv_path = output_path / f"dataset_{split_name}.csv"
                    dataset.to_csv(str(csv_path))
                saved_files['csv'] = str(output_path / "dataset_*.csv")
        
        logger.info(f"Dataset salvo em: {saved_files}")
        return saved_files
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o dataset processado
        
        Returns:
            Dicionário com estatísticas do dataset
        """
        if self.dataset_dict is None:
            return {"error": "Dataset não processado"}
        
        info = {
            "config": asdict(self.config),
            "splits": {},
            "features": list(self.dataset_dict['train'].features.keys()),
            "total_examples": sum(len(split) for split in self.dataset_dict.values()),
            "created_at": datetime.now().isoformat()
        }
        
        # Estatísticas por split
        for split_name, dataset in self.dataset_dict.items():
            split_info = {
                "num_examples": len(dataset),
                "avg_text_length": 0,
                "avg_summary_length": 0,
                "avg_title_length": 0,
            }
            
            # Calcular estatísticas básicas com amostra pequena
            try:
                sample_size = min(10, len(dataset))
                if sample_size > 0:
                    text_lengths = []
                    summary_lengths = []
                    title_lengths = []
                    
                    for i in range(sample_size):
                        item = dataset[i]
                        text_lengths.append(len(item['text']))
                        summary_lengths.append(len(item['summary']))
                        title_lengths.append(len(item['title']))
                    
                    split_info["avg_text_length"] = sum(text_lengths) // len(text_lengths)
                    split_info["avg_summary_length"] = sum(summary_lengths) // len(summary_lengths)
                    split_info["avg_title_length"] = sum(title_lengths) // len(title_lengths)
            except Exception:
                pass  # Usar valores padrão
            
            info["splits"][split_name] = split_info
        
        return info
    
    def validate_dataset_format(self) -> Dict[str, Any]:
        """
        Valida se o dataset está no formato correto para sumarização
        
        Returns:
            Resultado da validação
        """
        if self.dataset_dict is None:
            return {"valid": False, "error": "Dataset não processado"}
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Verificar features obrigatórias
        required_features = {'text', 'summary', 'title'}
        available_features = set(self.dataset_dict['train'].features.keys())
        
        missing_features = required_features - available_features
        if missing_features:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Features obrigatórias faltando: {missing_features}")
        
        # Verificar tipos de dados
        for feature_name in required_features:
            if feature_name in available_features:
                feature_type = self.dataset_dict['train'].features[feature_name]
                if not hasattr(feature_type, 'dtype') or feature_type.dtype != 'string':
                    validation_result["warnings"].append(f"Feature '{feature_name}' deveria ser string")
        
        # Estatísticas de validação
        for split_name, dataset in self.dataset_dict.items():
            split_stats = {
                "empty_texts": 0,
                "empty_summaries": 0,
                "empty_titles": 0,
            }
            
            # Verificar uma amostra para estatísticas
            try:
                sample_size = min(10, len(dataset))
                for i in range(sample_size):
                    item = dataset[i]
                    if len(item['text'].strip()) == 0:
                        split_stats["empty_texts"] += 1
                    if len(item['summary'].strip()) == 0:
                        split_stats["empty_summaries"] += 1
                    if len(item['title'].strip()) == 0:
                        split_stats["empty_titles"] += 1
            except Exception:
                pass  # Usar valores padrão
                
            validation_result["stats"][split_name] = split_stats
        
        return validation_result


def create_text_processor(
    excel_file: Union[str, Path],
    text_column: str = "Texto",
    summary_column: str = "Resumo", 
    title_column: str = "Processo",
    output_dir: str = "data/processed",
    **kwargs
) -> AdvancedTextProcessor:
    """
    Factory function para criar processador de texto avançado
    
    Args:
        excel_file: Caminho para arquivo Excel
        text_column: Nome da coluna com texto principal
        summary_column: Nome da coluna com resumo
        title_column: Nome da coluna com título
        output_dir: Diretório de saída
        **kwargs: Argumentos adicionais para TextPreprocessingConfig
        
    Returns:
        Processador configurado e carregado
    """
    config = TextPreprocessingConfig(
        text_column=text_column,
        summary_column=summary_column,
        title_column=title_column,
        output_dir=output_dir,
        **kwargs
    )
    
    processor = AdvancedTextProcessor(config)
    processor.load_excel(excel_file)
    
    return processor


# Função de conveniência para processamento completo
def process_excel_to_dataset(
    excel_file: Union[str, Path],
    output_dir: str = "data/processed",
    **config_kwargs
) -> Dict[str, Any]:
    """
    Processa arquivo Excel completo para formato estruturado de sumarização
    
    Args:
        excel_file: Caminho para arquivo Excel
        output_dir: Diretório de saída
        **config_kwargs: Argumentos para configuração
        
    Returns:
        Informações sobre o processamento
    """
    logger.info(f"Iniciando processamento completo: {excel_file}")
    
    # Criar processador
    processor = create_text_processor(excel_file, output_dir=output_dir, **config_kwargs)
    
    # Pipeline completo
    processor.preprocess_data()
    processor.create_dataset_splits()
    saved_files = processor.save_dataset()
    
    # Validação
    validation = processor.validate_dataset_format()
    dataset_info = processor.get_dataset_info()
    
    result = {
        "success": True,
        "saved_files": saved_files,
        "validation": validation,
        "dataset_info": dataset_info,
        "processor": processor
    }
    
    logger.info("Processamento completo finalizado com sucesso!")
    return result


if __name__ == "__main__":
    # Exemplo de uso
    result = process_excel_to_dataset("data/dataset.xlsx")
    print(json.dumps(result["dataset_info"], indent=2, ensure_ascii=False))
