"""
Módulo de pré-processamento de texto para projetos de NLP
Suporta múltiplos formatos de entrada (Excel, CSV, JSON, TXT) e saída
Segue as melhores práticas de engenharia de software para IA
"""

import pandas as pd
import numpy as np
import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuração para pré-processamento de texto"""
    
    # Configurações de colunas (mapeamento flexível)
    text_column: str = "Texto"
    summary_column: str = "Resumo"
    id_column: Optional[str] = "Processo"
    
    # Configurações de limpeza
    remove_encoding_artifacts: bool = True
    normalize_unicode: bool = True
    remove_extra_whitespace: bool = True
    lowercase: bool = False
    remove_punctuation: bool = False
    
    # Configurações de filtros
    min_text_length: int = 50
    max_text_length: int = 50000
    min_summary_length: int = 10
    max_summary_length: int = 5000
    
    # Configurações de divisão dos dados
    test_size: float = 0.2
    val_size: float = 0.1  # Em relação ao conjunto de treino
    random_state: int = 42
    stratify: bool = False
    
    # Configurações de saída
    output_dir: str = "data/processed"
    output_format: str = "jsonl"  # jsonl, json, csv, parquet
    instruction_template: str = "[INST] Sumarize o seguinte texto: {text} [/INST] {summary}"
    
    # Configurações de validação
    validate_data_quality: bool = True
    sample_for_validation: int = 100
    
    def __post_init__(self):
        """Validação após inicialização"""
        if not 0 < self.test_size < 1:
            raise ValueError("test_size deve estar entre 0 e 1")
        
        if not 0 < self.val_size < 1:
            raise ValueError("val_size deve estar entre 0 e 1")
        
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size deve ser menor que 1.0")


class DataLoader(ABC):
    """Classe abstrata para carregamento de diferentes formatos de dados"""
    
    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Carrega dados do arquivo especificado"""
        pass


class ExcelLoader(DataLoader):
    """Carregador para arquivos Excel (.xlsx, .xls)"""
    
    def load(self, file_path: Union[str, Path], sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        logger.info(f"Carregando arquivo Excel: {file_path}")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Excel carregado: {len(df)} linhas, {len(df.columns)} colunas")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar Excel: {e}")
            raise


class CSVLoader(DataLoader):
    """Carregador para arquivos CSV"""
    
    def load(self, file_path: Union[str, Path], encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
        logger.info(f"Carregando arquivo CSV: {file_path}")
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            logger.info(f"CSV carregado: {len(df)} linhas, {len(df.columns)} colunas")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar CSV: {e}")
            raise


class JSONLoader(DataLoader):
    """Carregador para arquivos JSON"""
    
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        logger.info(f"Carregando arquivo JSON: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Formato JSON não suportado")
            
            logger.info(f"JSON carregado: {len(df)} linhas, {len(df.columns)} colunas")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar JSON: {e}")
            raise


class TextLoader(DataLoader):
    """Carregador para arquivos de texto simples"""
    
    def load(self, file_path: Union[str, Path], 
             text_column: str = "text", 
             summary_column: Optional[str] = None,
             separator: str = "\n\n", **kwargs) -> pd.DataFrame:
        logger.info(f"Carregando arquivo de texto: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir por separador se especificado
            if separator and summary_column is None:
                texts = [text.strip() for text in content.split(separator) if text.strip()]
                df = pd.DataFrame({text_column: texts})
            else:
                # Tratar como texto único
                df = pd.DataFrame({text_column: [content]})
            
            logger.info(f"Texto carregado: {len(df)} registros")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar texto: {e}")
            raise


class TextCleaner:
    """Classe para limpeza e normalização de texto"""
    
    @staticmethod
    def clean_text(text: Any, config: PreprocessingConfig) -> str:
        """Limpa e normaliza texto baseado na configuração"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        cleaned_text = text
        
        # Remover artefatos de encoding
        if config.remove_encoding_artifacts:
            cleaned_text = TextCleaner._remove_encoding_artifacts(cleaned_text)
        
        # Normalizar unicode
        if config.normalize_unicode:
            cleaned_text = unicodedata.normalize('NFKC', cleaned_text)
        
        # Remover espaços extras
        if config.remove_extra_whitespace:
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Converter para minúsculas
        if config.lowercase:
            cleaned_text = cleaned_text.lower()
        
        # Remover pontuação (opcional para alguns casos)
        if config.remove_punctuation:
            cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    @staticmethod
    def _remove_encoding_artifacts(text: str) -> str:
        """Remove artefatos comuns de encoding"""
        # Dicionário de substituições comuns
        artifacts = {
            '_x000D_': '\n',
            '_x000A_': '\n', 
            '_x0009_': '\t',
            'â€™': "'",
            'â€œ': '"',
            'â€\u009d': '"',
            'â€"': '—'
        }
        
        for artifact, replacement in artifacts.items():
            text = text.replace(artifact, replacement)
        
        # Normalizar quebras de linha
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        return text


class DataValidator:
    """Classe para validação da qualidade dos dados"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, config: PreprocessingConfig) -> Dict[str, Any]:
        """Valida a qualidade do DataFrame"""
        validation_results = {
            'valid': True,
            'issues': [],
            'statistics': {},
            'sample_data': {}
        }
        
        # Verificar colunas obrigatórias
        required_columns = [config.text_column, config.summary_column]
        if config.id_column:
            required_columns.append(config.id_column)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Colunas faltando: {missing_columns}")
        
        # Estatísticas básicas
        validation_results['statistics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_text': df[config.text_column].isna().sum() if config.text_column in df else 0,
            'missing_summary': df[config.summary_column].isna().sum() if config.summary_column in df else 0,
            'empty_text': 0,
            'empty_summary': 0
        }
        
        # Verificar textos vazios
        if config.text_column in df:
            validation_results['statistics']['empty_text'] = (df[config.text_column].str.strip() == '').sum()
        if config.summary_column in df:
            validation_results['statistics']['empty_summary'] = (df[config.summary_column].str.strip() == '').sum()
        
        # Amostra de dados para análise
        if len(df) > 0 and config.sample_for_validation > 0:
            sample_size = min(config.sample_for_validation, len(df))
            sample = df.head(sample_size)
            
            if config.text_column in sample and config.summary_column in sample:
                validation_results['sample_data'] = {
                    'text_lengths': sample[config.text_column].str.len().describe().to_dict(),
                    'summary_lengths': sample[config.summary_column].str.len().describe().to_dict()
                }
        
        return validation_results


class UniversalTextPreprocessor:
    """Preprocessador universal para diferentes formatos de dados textuais"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.loaders = {
            '.xlsx': ExcelLoader(),
            '.xls': ExcelLoader(), 
            '.csv': CSVLoader(),
            '.json': JSONLoader(),
            '.txt': TextLoader(),
            '.text': TextLoader()
        }
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[Dict[str, List]] = None
        
        # Criar diretório de saída
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, file_path: Union[str, Path], **loader_kwargs) -> pd.DataFrame:
        """Carrega dados automaticamente baseado na extensão do arquivo"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.loaders:
            raise ValueError(f"Formato não suportado: {extension}")
        
        loader = self.loaders[extension]
        self.raw_data = loader.load(file_path, **loader_kwargs)
        
        # Ensure we have valid data
        if self.raw_data is None:
            raise ValueError("Erro ao carregar dados: resultado None")
        
        # Validar dados carregados
        if self.config.validate_data_quality:
            validation = DataValidator.validate_dataframe(self.raw_data, self.config)
            if not validation['valid']:
                logger.warning(f"Problemas encontrados nos dados: {validation['issues']}")
            logger.info(f"Estatísticas dos dados: {validation['statistics']}")
        
        return self.raw_data
    
    def preprocess_data(self) -> Dict[str, List]:
        """Processa os dados carregados aplicando limpeza e filtros"""
        if self.raw_data is None:
            raise ValueError("Nenhum dado carregado. Use load_data() primeiro.")
        
        logger.info("Iniciando pré-processamento dos dados...")
        
        # Fazer cópia para não alterar dados originais
        df = self.raw_data.copy()
        
        # Remover linhas com valores nulos nas colunas essenciais
        initial_count = len(df)
        df.dropna(subset=[self.config.text_column, self.config.summary_column], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        removed_null = initial_count - len(df)
        if removed_null > 0:
            logger.info(f"Removidas {removed_null} linhas com valores nulos")
        
        # Aplicar limpeza de texto
        logger.info("Aplicando limpeza de texto...")
        df['text_clean'] = df[self.config.text_column].apply(
            lambda x: TextCleaner.clean_text(x, self.config)
        )
        df['summary_clean'] = df[self.config.summary_column].apply(
            lambda x: TextCleaner.clean_text(x, self.config)
        )
        
        # Aplicar filtros de comprimento
        initial_count = len(df)
        df = self._apply_length_filters(df)
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Removidas {filtered_count} linhas por filtros de comprimento")
        
        # Preparar dados finais
        processed_records = []
        for idx, row in df.iterrows():
            record = {
                'text': row['text_clean'],
                'summary': row['summary_clean']
            }
            
            # Adicionar ID se disponível
            if self.config.id_column and self.config.id_column in df.columns:
                record['id'] = str(row[self.config.id_column])
            else:
                record['id'] = str(idx)
            
            processed_records.append(record)
        
        # Converter para formato de listas para compatibilidade
        self.processed_data = {
            'text': [r['text'] for r in processed_records],
            'summary': [r['summary'] for r in processed_records],
            'id': [r['id'] for r in processed_records]
        }
        
        logger.info(f"Pré-processamento concluído: {len(processed_records)} registros válidos")
        return self.processed_data
    
    def _apply_length_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica filtros de comprimento de texto"""
        # Filtros de comprimento mínimo
        df = df[df['text_clean'].str.len() >= self.config.min_text_length]
        df = df[df['summary_clean'].str.len() >= self.config.min_summary_length]
        
        # Filtros de comprimento máximo
        df = df[df['text_clean'].str.len() <= self.config.max_text_length]
        df = df[df['summary_clean'].str.len() <= self.config.max_summary_length]
        
        return df.reset_index(drop=True)
    
    def create_splits(self) -> Dict[str, Dict[str, List]]:
        """Cria divisões de treino, validação e teste"""
        if self.processed_data is None:
            raise ValueError("Dados não processados. Use preprocess_data() primeiro.")
        
        logger.info("Criando divisões dos dados...")
        
        # Preparar dados para divisão
        indices = list(range(len(self.processed_data['text'])))
        
        # Divisão inicial: treino+validação vs teste
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Divisão do conjunto de treino: treino vs validação
        if self.config.val_size > 0:
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=self.config.val_size,
                random_state=self.config.random_state
            )
        else:
            train_indices = train_val_indices
            val_indices = []
        
        # Criar splits
        splits = {}
        
        # Treino
        splits['train'] = {
            'text': [self.processed_data['text'][i] for i in train_indices],
            'summary': [self.processed_data['summary'][i] for i in train_indices],
            'id': [self.processed_data['id'][i] for i in train_indices]
        }
        
        # Validação (se configurada)
        if val_indices:
            splits['validation'] = {
                'text': [self.processed_data['text'][i] for i in val_indices],
                'summary': [self.processed_data['summary'][i] for i in val_indices],
                'id': [self.processed_data['id'][i] for i in val_indices]
            }
        
        # Teste
        splits['test'] = {
            'text': [self.processed_data['text'][i] for i in test_indices],
            'summary': [self.processed_data['summary'][i] for i in test_indices],
            'id': [self.processed_data['id'][i] for i in test_indices]
        }
        
        # Log dos tamanhos
        for split_name, split_data in splits.items():
            logger.info(f"Split '{split_name}': {len(split_data['text'])} exemplos")
        
        return splits
    
    def save_data(self, splits: Dict[str, Dict[str, List]], 
                  format_type: Optional[str] = None) -> Dict[str, str]:
        """Salva os dados nos formatos especificados"""
        format_type = format_type or self.config.output_format
        saved_files = {}
        
        for split_name, split_data in splits.items():
            if format_type == 'jsonl':
                saved_files[split_name] = self._save_jsonl(split_data, split_name)
            elif format_type == 'json':
                saved_files[split_name] = self._save_json(split_data, split_name)
            elif format_type == 'csv':
                saved_files[split_name] = self._save_csv(split_data, split_name)
            elif format_type == 'parquet':
                saved_files[split_name] = self._save_parquet(split_data, split_name)
            else:
                raise ValueError(f"Formato não suportado: {format_type}")
        
        logger.info(f"Dados salvos: {saved_files}")
        return saved_files
    
    def _save_jsonl(self, split_data: Dict[str, List], split_name: str) -> str:
        """Salva dados no formato JSONL com template de instrução"""
        output_path = Path(self.config.output_dir) / f"dataset_{split_name}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(len(split_data['text'])):
                # Usar template de instrução se configurado
                if self.config.instruction_template:
                    formatted_text = self.config.instruction_template.format(
                        text=split_data['text'][i],
                        summary=split_data['summary'][i]
                    )
                    record = {'text': formatted_text}
                else:
                    record = {
                        'text': split_data['text'][i],
                        'summary': split_data['summary'][i],
                        'id': split_data['id'][i]
                    }
                
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Dados JSONL salvos: {output_path}")
        return str(output_path)
    
    def _save_json(self, split_data: Dict[str, List], split_name: str) -> str:
        """Salva dados no formato JSON"""
        output_path = Path(self.config.output_dir) / f"dataset_{split_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dados JSON salvos: {output_path}")
        return str(output_path)
    
    def _save_csv(self, split_data: Dict[str, List], split_name: str) -> str:
        """Salva dados no formato CSV"""
        output_path = Path(self.config.output_dir) / f"dataset_{split_name}.csv"
        
        df = pd.DataFrame(split_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Dados CSV salvos: {output_path}")
        return str(output_path)
    
    def _save_parquet(self, split_data: Dict[str, List], split_name: str) -> str:
        """Salva dados no formato Parquet"""
        output_path = Path(self.config.output_dir) / f"dataset_{split_name}.parquet"
        
        df = pd.DataFrame(split_data)
        df.to_parquet(output_path, index=False)
        
        logger.info(f"Dados Parquet salvos: {output_path}")
        return str(output_path)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos dados processados"""
        if self.processed_data is None:
            return {"error": "Dados não processados"}
        
        text_lengths = [len(text) for text in self.processed_data['text']]
        summary_lengths = [len(summary) for summary in self.processed_data['summary']]
        
        return {
            'total_samples': len(self.processed_data['text']),
            'text_stats': {
                'mean_length': np.mean(text_lengths),
                'median_length': np.median(text_lengths),
                'min_length': np.min(text_lengths),
                'max_length': np.max(text_lengths)
            },
            'summary_stats': {
                'mean_length': np.mean(summary_lengths),
                'median_length': np.median(summary_lengths),
                'min_length': np.min(summary_lengths),
                'max_length': np.max(summary_lengths)
            },
            'config': {
                'text_column': self.config.text_column,
                'summary_column': self.config.summary_column,
                'output_format': self.config.output_format,
                'instruction_template': self.config.instruction_template
            }
        }


def process_text_data(input_file: Union[str, Path], 
                     config: Optional[PreprocessingConfig] = None,
                     **loader_kwargs) -> Dict[str, Any]:
    """
    Função de conveniência para processamento completo de dados textuais
    
    Args:
        input_file: Caminho para arquivo de entrada
        config: Configuração de pré-processamento
        **loader_kwargs: Argumentos adicionais para o carregador
    
    Returns:
        Dicionário com resultados do processamento
    """
    # Usar configuração padrão se não fornecida
    if config is None:
        config = PreprocessingConfig()
    
    # Criar processador
    processor = UniversalTextPreprocessor(config)
    
    try:
        # Pipeline completo
        processor.load_data(input_file, **loader_kwargs)
        processor.preprocess_data()
        splits = processor.create_splits()
        saved_files = processor.save_data(splits)
        
        # Estatísticas
        stats = processor.get_data_statistics()
        
        return {
            'success': True,
            'saved_files': saved_files,
            'statistics': stats,
            'message': 'Processamento concluído com sucesso!'
        }
        
    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Erro durante o processamento'
        }


# Função principal para compatibilidade com o código anterior
def preprocess_and_save(input_file_path: Union[str, Path], 
                       output_jsonl_paths: Optional[Dict[str, str]] = None,
                       **config_kwargs):
    """
    Função de compatibilidade com a interface anterior
    Mantida para não quebrar código existente
    """
    # Configurar saídas padrão se não especificadas
    if output_jsonl_paths is None:
        output_jsonl_paths = {
            'train': "dataset_train.jsonl",
            'validation': "dataset_val.jsonl", 
            'test': "dataset_test.jsonl"
        }
    
    # Criar configuração
    config = PreprocessingConfig(
        output_format='jsonl',
        **config_kwargs
    )
    
    # Processar dados
    result = process_text_data(input_file_path, config)
    
    if result['success']:
        logger.info("Processamento legado concluído com sucesso!")
        for split_name in output_jsonl_paths.keys():
            if split_name in result['saved_files']:
                print(f"Dados de {split_name} processados salvos em: {result['saved_files'][split_name]}")
    else:
        logger.error(f"Erro no processamento legado: {result['error']}")


# if __name__ == "__main__":
#     # Exemplo de uso do novo sistema
#     config = PreprocessingConfig(
#         text_column="Texto",
#         summary_column="Resumo", 
#         id_column="Processo",
#         output_format="jsonl",
#         instruction_template="[INST] Sumarize o seguinte texto: {text} [/INST] {summary}"
#     )
    
#     result = process_text_data("data/dataset.xlsx", config)
#     print(f"Resultado: {result['message']}")
#     if result['success']:
#         print(f"Estatísticas: {result['statistics']}")
#         print(f"Arquivos salvos: {result['saved_files']}")