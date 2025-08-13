"""
Módulo de pré-processamento de texto distribuído para projetos de NLP
Suporta múltiplos formatos de entrada (Excel, CSV, JSON, TXT) e saída
Usa Dask para processamento distribuído e paralelizado
Segue as melhores práticas de engenharia de software para IA
"""

import dask.dataframe as dd
import dask.bag as db  
import dask.config
from dask.distributed import Client
import pandas as pd
import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from src.logging_config import get_preprocessing_logger
import dask

# Configurar logging
logger = get_preprocessing_logger()


@dataclass
class PreprocessingConfig:
    """Configuração para pré-processamento de texto distribuído"""
    
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
    
    # Configurações de Dask
    n_partitions: Optional[int] = None  # Auto-detectar baseado no número de workers
    chunk_size: str = "100MB"
    scheduler: str = "threads"  # "threads", "processes", "distributed"
    client_address: Optional[str] = None  # Para cluster distribuído
    
    def __post_init__(self):
        """Validação após inicialização"""
        if not 0 < self.test_size < 1:
            raise ValueError("test_size deve estar entre 0 e 1")
        
        if not 0 < self.val_size < 1:
            raise ValueError("val_size deve estar entre 0 e 1")
        
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size deve ser menor que 1.0")
        
        if self.scheduler not in ["threads", "processes", "distributed"]:
            raise ValueError("scheduler deve ser 'threads', 'processes', ou 'distributed'")


class DaskDataLoader(ABC):
    """Classe abstrata para carregamento distribuído de diferentes formatos de dados"""
    
    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> dd.DataFrame:
        """Carrega dados do arquivo especificado como Dask DataFrame"""
        pass


class DaskExcelLoader(DaskDataLoader):
    """Carregador distribuído para arquivos Excel (.xlsx, .xls)"""
    
    def load(self, file_path: Union[str, Path], sheet_name: Union[str, int] = 0, **kwargs) -> dd.DataFrame:
        logger.info("Carregando arquivo Excel com Dask", file_path=str(file_path), sheet_name=sheet_name)
        try:
            # Excel não tem suporte nativo no Dask, então carregamos com pandas e convertemos
            pdf = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            
            # Converter para Dask DataFrame
            ddf = dd.from_pandas(pdf, npartitions=self._calculate_partitions(len(pdf)))
            
            logger.info("Excel carregado com sucesso", rows=len(pdf), columns=len(pdf.columns))
            return ddf
        except Exception as e:
            logger.error("Erro ao carregar Excel", error=str(e), file_path=str(file_path))
            raise
    
    def _calculate_partitions(self, n_rows: int, rows_per_partition: int = 10000) -> int:
        """Calcula número ideal de partições baseado no tamanho dos dados"""
        return max(1, n_rows // rows_per_partition)


class DaskCSVLoader(DaskDataLoader):
    """Carregador distribuído para arquivos CSV"""
    
    def load(self, file_path: Union[str, Path], encoding: str = 'utf-8', **kwargs) -> dd.DataFrame:
        logger.info(f"Carregando arquivo CSV com Dask: {file_path}")
        try:
            # Filtrar argumentos específicos do Dask read_csv
            csv_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['blocksize', 'scheduler']}
            
            # Usar Dask para carregar CSV de forma distribuída
            ddf = dd.read_csv(
                file_path, 
                encoding=encoding,
                blocksize=kwargs.get('blocksize', '100MB'),
                **csv_kwargs
            )
            
            logger.info(f"CSV carregado com Dask: {ddf.npartitions} partições")
            return ddf
        except Exception as e:
            logger.error(f"Erro ao carregar CSV com Dask: {e}")
            raise


class DaskJSONLoader(DaskDataLoader):
    """Carregador distribuído para arquivos JSON"""
    
    def load(self, file_path: Union[str, Path], **kwargs) -> dd.DataFrame:
        logger.info(f"Carregando arquivo JSON com Dask: {file_path}")
        try:
            # Para JSON, usamos Dask Bag primeiro e depois convertemos
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Criar Dask Bag e converter para DataFrame
                bag = db.from_sequence(data, npartitions=max(1, len(data) // 1000))
                ddf = bag.to_dataframe()
            elif isinstance(data, dict):
                pdf = pd.DataFrame([data])
                ddf = dd.from_pandas(pdf, npartitions=1)
            else:
                raise ValueError("Formato JSON não suportado")
            
            logger.info(f"JSON carregado com Dask: {ddf.npartitions} partições")
            return ddf
        except Exception as e:
            logger.error(f"Erro ao carregar JSON com Dask: {e}")
            raise


class DaskTextLoader(DaskDataLoader):
    """Carregador distribuído para arquivos de texto simples"""
    
    def load(self, file_path: Union[str, Path], 
             text_column: str = "text", 
             summary_column: Optional[str] = None,
             separator: str = "\n\n", **kwargs) -> dd.DataFrame:
        logger.info(f"Carregando arquivo de texto com Dask: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir por separador se especificado
            if separator and summary_column is None:
                texts = [text.strip() for text in content.split(separator) if text.strip()]
                data_dict = {text_column: texts}
            else:
                # Tratar como texto único
                data_dict = {text_column: [content]}
            
            # Converter para Dask DataFrame
            pdf = pd.DataFrame(data_dict)
            ddf = dd.from_pandas(pdf, npartitions=max(1, len(pdf) // 1000))
            
            logger.info(f"Texto carregado com Dask: {len(pdf)} registros, {ddf.npartitions} partições")
            return ddf
        except Exception as e:
            logger.error(f"Erro ao carregar texto com Dask: {e}")
            raise


class DaskTextCleaner:
    """Classe para limpeza e normalização distribuída de texto"""
    
    @staticmethod
    def clean_text(text: Any, config: PreprocessingConfig) -> str:
        """Limpa e normaliza texto baseado na configuração"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        cleaned_text = text
        
        # Remover artefatos de encoding
        if config.remove_encoding_artifacts:
            cleaned_text = DaskTextCleaner._remove_encoding_artifacts(cleaned_text)
        
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
    
    @staticmethod
    def apply_cleaning_distributed(ddf: dd.DataFrame, column: str, config: PreprocessingConfig) -> dd.DataFrame:
        """Aplica limpeza de texto de forma distribuída"""
        logger.info(f"Aplicando limpeza distribuída na coluna: {column}")
        
        # Usar map_partitions para aplicar limpeza de forma distribuída
        def clean_partition(partition_df):
            partition_df = partition_df.copy()
            partition_df[f'{column}_clean'] = partition_df[column].apply(
                lambda x: DaskTextCleaner.clean_text(x, config)
            )
            return partition_df
        
        # Criar metadata com a nova coluna
        new_meta = ddf._meta.copy()
        new_meta[f'{column}_clean'] = 'str'
        
        return ddf.map_partitions(clean_partition, meta=new_meta)


class DaskDataValidator:
    """Classe para validação distribuída da qualidade dos dados"""
    
    @staticmethod
    def validate_dataframe_distributed(ddf: dd.DataFrame, config: PreprocessingConfig) -> Dict[str, Any]:
        """Valida a qualidade do DataFrame de forma distribuída"""
        logger.info("Iniciando validação distribuída dos dados")
        
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
        
        missing_columns = [col for col in required_columns if col not in ddf.columns]
        if missing_columns:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Colunas faltando: {missing_columns}")
        
        # Estatísticas básicas computadas de forma distribuída
        logger.info("Computando estatísticas distribuídas")
        
        total_rows = len(ddf)
        total_columns = len(ddf.columns)
        
        # Computar valores faltantes de forma distribuída
        missing_text = 0
        missing_summary = 0
        empty_text = 0
        empty_summary = 0
        
        if config.text_column in ddf.columns:
            missing_text = ddf[config.text_column].isna().sum().compute()
            empty_text = (ddf[config.text_column].str.strip() == '').sum().compute()
        
        if config.summary_column in ddf.columns:
            missing_summary = ddf[config.summary_column].isna().sum().compute()
            empty_summary = (ddf[config.summary_column].str.strip() == '').sum().compute()
        
        validation_results['statistics'] = {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'missing_text': missing_text,
            'missing_summary': missing_summary,
            'empty_text': empty_text,
            'empty_summary': empty_summary
        }
        
        # Amostra de dados para análise
        if total_rows > 0 and config.sample_for_validation > 0:
            sample_size = min(config.sample_for_validation, total_rows)
            sample = ddf.head(sample_size)
            
            if config.text_column in sample and config.summary_column in sample:
                text_lengths = sample[config.text_column].str.len()
                summary_lengths = sample[config.summary_column].str.len()
                
                validation_results['sample_data'] = {
                    'text_lengths': {
                        'mean': text_lengths.mean(),
                        'median': text_lengths.quantile(0.5),
                        'min': text_lengths.min(),
                        'max': text_lengths.max()
                    },
                    'summary_lengths': {
                        'mean': summary_lengths.mean(),
                        'median': summary_lengths.quantile(0.5),
                        'min': summary_lengths.min(),
                        'max': summary_lengths.max()
                    }
                }
        
        logger.info("Validação distribuída concluída")
        return validation_results


class DistributedTextPreprocessor:
    """Preprocessador distribuído para diferentes formatos de dados textuais usando Dask"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.client: Optional[Client] = None
        
        # Configurar cliente Dask
        self._setup_dask_client()
        
        self.loaders = {
            '.xlsx': DaskExcelLoader(),
            '.xls': DaskExcelLoader(), 
            '.csv': DaskCSVLoader(),
            '.json': DaskJSONLoader(),
            '.txt': DaskTextLoader(),
            '.text': DaskTextLoader()
        }
        self.raw_data: Optional[dd.DataFrame] = None
        self.processed_data: Optional[dd.DataFrame] = None
        
        # Criar diretório de saída
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_dask_client(self):
        """Configura o cliente Dask baseado na configuração"""
        try:
            if self.config.scheduler == "distributed" and self.config.client_address:
                logger.info(f"Conectando ao cluster Dask: {self.config.client_address}")
                self.client = Client(self.config.client_address)
            elif self.config.scheduler == "distributed":
                logger.info("Iniciando cluster Dask local")
                self.client = Client()
            else:
                logger.info(f"Usando scheduler {self.config.scheduler}")
                dask.config.set(scheduler=self.config.scheduler)
            
            if self.client:
                logger.info(f"Cliente Dask configurado: {self.client.dashboard_link}")
                
        except Exception as e:
            logger.warning(f"Erro ao configurar cliente Dask: {e}. Usando configuração padrão.")
            dask.config.set(scheduler='threads')
    
    def load_data(self, file_path: Union[str, Path], **loader_kwargs) -> dd.DataFrame:
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
        
        # Reparticionar se necessário
        if self.config.n_partitions:
            self.raw_data = self.raw_data.repartition(npartitions=self.config.n_partitions)
            logger.info(f"Dados reparticionados para {self.config.n_partitions} partições")
        
        # Validar dados carregados
        if self.config.validate_data_quality:
            validation = DaskDataValidator.validate_dataframe_distributed(self.raw_data, self.config)
            if not validation['valid']:
                logger.warning(f"Problemas encontrados nos dados: {validation['issues']}")
            logger.info(f"Estatísticas dos dados: {validation['statistics']}")
        
        return self.raw_data
    
    def preprocess_data(self) -> dd.DataFrame:
        """Processa os dados carregados aplicando limpeza e filtros de forma distribuída"""
        if self.raw_data is None:
            raise ValueError("Nenhum dado carregado. Use load_data() primeiro.")
        
        logger.info("Iniciando pré-processamento distribuído dos dados")
        
        # Fazer cópia para não alterar dados originais
        ddf = self.raw_data.copy()
        
        # Remover linhas com valores nulos nas colunas essenciais
        initial_count = len(ddf)
        ddf = ddf.dropna(subset=[self.config.text_column, self.config.summary_column])
        
        removed_null = initial_count - len(ddf)
        if removed_null > 0:
            logger.info("Removidas linhas com valores nulos", removed_count=removed_null)
        
        # Aplicar limpeza de texto de forma distribuída
        logger.info("Aplicando limpeza distribuída de texto")
        ddf = DaskTextCleaner.apply_cleaning_distributed(ddf, self.config.text_column, self.config)
        ddf = DaskTextCleaner.apply_cleaning_distributed(ddf, self.config.summary_column, self.config)
        
        # Aplicar filtros de comprimento de forma distribuída
        logger.info("Aplicando filtros de comprimento")
        initial_count = len(ddf)
        ddf = self._apply_length_filters_distributed(ddf)
        filtered_count = initial_count - len(ddf)
        
        if filtered_count > 0:
            logger.info("Removidas linhas por filtros de comprimento", filtered_count=filtered_count)
        
        # Adicionar coluna de ID se não existir
        if self.config.id_column not in ddf.columns:
            logger.info("Adicionando IDs sequenciais")
            ddf = self._add_sequential_ids(ddf)
        
        self.processed_data = ddf
        
        total_records = len(ddf)
        logger.info(f"Pré-processamento distribuído concluído: {total_records} registros válidos")
        return self.processed_data
    
    def _apply_length_filters_distributed(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Aplica filtros de comprimento de texto de forma distribuída"""
        logger.info("Aplicando filtros de comprimento distribuídos")
        
        def filter_partition(partition_df):
            # Filtros de comprimento mínimo
            mask = (
                (partition_df[f'{self.config.text_column}_clean'].str.len() >= self.config.min_text_length) &
                (partition_df[f'{self.config.summary_column}_clean'].str.len() >= self.config.min_summary_length) &
                # Filtros de comprimento máximo
                (partition_df[f'{self.config.text_column}_clean'].str.len() <= self.config.max_text_length) &
                (partition_df[f'{self.config.summary_column}_clean'].str.len() <= self.config.max_summary_length)
            )
            return partition_df[mask]
        
        return ddf.map_partitions(filter_partition, meta=ddf._meta)
    
    def _add_sequential_ids(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Adiciona IDs sequenciais de forma distribuída"""
        def add_ids_to_partition(partition_df):
            partition_df = partition_df.copy()
            # Gerar IDs simples baseado no índice da partição
            partition_df['id'] = range(len(partition_df))
            return partition_df
        
        # Criar meta com a nova coluna
        new_meta = ddf._meta.copy()
        new_meta['id'] = 'int64'
        
        # Usar map_partitions
        return ddf.map_partitions(add_ids_to_partition, meta=new_meta)
    
    def create_splits_distributed(self) -> Dict[str, dd.DataFrame]:
        """Cria divisões de treino, validação e teste de forma distribuída"""
        if self.processed_data is None:
            raise ValueError("Dados não processados. Use preprocess_data() primeiro.")
        
        logger.info("Criando divisões distribuídas dos dados...")
        
        # Converter para pandas temporariamente para fazer split estratificado
        # Para datasets muito grandes, isso pode ser otimizado usando sampling
        pdf = self.processed_data.compute()
        
        # Preparar dados para divisão
        indices = list(range(len(pdf)))
        
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
        
        # Criar splits como Dask DataFrames
        splits = {}
        
        # Treino
        train_pdf = pdf.iloc[train_indices].reset_index(drop=True)
        splits['train'] = dd.from_pandas(train_pdf, npartitions=max(1, len(train_pdf) // 10000))
        
        # Validação (se configurada)
        if val_indices:
            val_pdf = pdf.iloc[val_indices].reset_index(drop=True)
            splits['validation'] = dd.from_pandas(val_pdf, npartitions=max(1, len(val_pdf) // 10000))
        
        # Teste
        test_pdf = pdf.iloc[test_indices].reset_index(drop=True)
        splits['test'] = dd.from_pandas(test_pdf, npartitions=max(1, len(test_pdf) // 10000))
        
        # Log dos tamanhos
        for split_name, split_data in splits.items():
            logger.info(f"Split distribuído '{split_name}': {len(split_data)} exemplos, {split_data.npartitions} partições")
        
        return splits
    
    def save_data_distributed(self, splits: Dict[str, dd.DataFrame], 
                             format_type: Optional[str] = None) -> Dict[str, str]:
        """Salva os dados nos formatos especificados de forma distribuída"""
        format_type = format_type or self.config.output_format
        saved_files = {}
        
        logger.info(f"Salvando dados distribuídos no formato: {format_type}")
        
        for split_name, split_data in splits.items():
            if format_type == 'jsonl':
                saved_files[split_name] = self._save_jsonl_distributed(split_data, split_name)
            elif format_type == 'json':
                saved_files[split_name] = self._save_json_distributed(split_data, split_name)
            elif format_type == 'csv':
                saved_files[split_name] = self._save_csv_distributed(split_data, split_name)
            elif format_type == 'parquet':
                saved_files[split_name] = self._save_parquet_distributed(split_data, split_name)
            else:
                raise ValueError(f"Formato não suportado: {format_type}")
        
        logger.info(f"Dados salvos distribuídos: {saved_files}")
        return saved_files
    
    def _save_jsonl_distributed(self, split_data: dd.DataFrame, split_name: str) -> str:
        """Salva dados no formato JSONL com template de instrução de forma distribuída"""
        output_path = Path(self.config.output_dir) / f"dataset_{split_name}.jsonl"
        
        def format_partition(partition_df):
            records = []
            for _, row in partition_df.iterrows():
                # Usar template de instrução se configurado
                if self.config.instruction_template:
                    text_clean = row[f'{self.config.text_column}_clean']
                    summary_clean = row[f'{self.config.summary_column}_clean']
                    formatted_text = self.config.instruction_template.format(
                        text=text_clean,
                        summary=summary_clean
                    )
                    record = {'text': formatted_text}
                else:
                    record = {
                        'text': row[f'{self.config.text_column}_clean'],
                        'summary': row[f'{self.config.summary_column}_clean'],
                        'id': str(row.get('id', ''))
                    }
                records.append(json.dumps(record, ensure_ascii=False))
            return '\n'.join(records)
        
        # Processar cada partição e salvar
        formatted_partitions = split_data.map_partitions(format_partition, meta=('x', 'object'))
        
        # Computar e salvar
        results = formatted_partitions.compute()
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                if result:  # Verificar se a partição não está vazia
                    f.write(result + '\n')
        
        logger.info(f"Dados JSONL salvos distribuídos: {output_path}")
        return str(output_path)
    
    def _save_json_distributed(self, split_data: dd.DataFrame, split_name: str) -> str:
        """Salva dados no formato JSON de forma distribuída"""
        output_path = Path(self.config.output_dir) / f"dataset_{split_name}.json"
        
        # Computar e converter para dicionário
        pdf = split_data.compute()
        data_dict = {
            'text': pdf[f'{self.config.text_column}_clean'].tolist(),
            'summary': pdf[f'{self.config.summary_column}_clean'].tolist(),
            'id': pdf.get('id', range(len(pdf))).astype(str).tolist()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dados JSON salvos distribuídos: {output_path}")
        return str(output_path)
    
    def _save_csv_distributed(self, split_data: dd.DataFrame, split_name: str) -> str:
        """Salva dados no formato CSV de forma distribuída"""
        output_path = Path(self.config.output_dir) / f"dataset_{split_name}.csv"
        
        # Usar Dask para salvar CSV distribuído
        split_data.to_csv(str(output_path), index=False, single_file=True)
        
        logger.info(f"Dados CSV salvos distribuídos: {output_path}")
        return str(output_path)
    
    def _save_parquet_distributed(self, split_data: dd.DataFrame, split_name: str) -> str:
        """Salva dados no formato Parquet de forma distribuída"""
        output_dir = Path(self.config.output_dir) / f"dataset_{split_name}.parquet"
        
        # Usar Dask para salvar Parquet distribuído
        split_data.to_parquet(str(output_dir))
        
        logger.info(f"Dados Parquet salvos distribuídos: {output_dir}")
        return str(output_dir)
    
    def get_data_statistics_distributed(self) -> Dict[str, Any]:
        """Retorna estatísticas dos dados processados de forma distribuída"""
        if self.processed_data is None:
            return {"error": "Dados não processados"}
        
        logger.info("Computando estatísticas distribuídas")
        
        # Computar estatísticas de forma distribuída
        text_col = f'{self.config.text_column}_clean'
        summary_col = f'{self.config.summary_column}_clean'
        
        # Usar Dask para computar estatísticas
        text_lengths = self.processed_data[text_col].str.len()
        summary_lengths = self.processed_data[summary_col].str.len()
        
        # Computar estatísticas
        text_stats = {
            'mean_length': text_lengths.mean().compute(),
            'median_length': text_lengths.quantile(0.5).compute(),
            'min_length': text_lengths.min().compute(),
            'max_length': text_lengths.max().compute()
        }
        
        summary_stats = {
            'mean_length': summary_lengths.mean().compute(),
            'median_length': summary_lengths.quantile(0.5).compute(),
            'min_length': summary_lengths.min().compute(),
            'max_length': summary_lengths.max().compute()
        }
        
        return {
            'total_samples': len(self.processed_data),
            'partitions': self.processed_data.npartitions,
            'text_stats': text_stats,
            'summary_stats': summary_stats,
            'config': {
                'text_column': self.config.text_column,
                'summary_column': self.config.summary_column,
                'output_format': self.config.output_format,
                'instruction_template': self.config.instruction_template,
                'scheduler': self.config.scheduler
            }
        }
    
    def close(self):
        """Fecha o cliente Dask"""
        if self.client:
            logger.info("Fechando cliente Dask")
            self.client.close()


def process_text_data_distributed(input_file: Union[str, Path], 
                                 config: Optional[PreprocessingConfig] = None,
                                 **loader_kwargs) -> Dict[str, Any]:
    """
    Função de conveniência para processamento completo distribuído de dados textuais
    
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
    
    # Criar processador distribuído
    processor = DistributedTextPreprocessor(config)
    
    try:
        # Pipeline completo distribuído
        processor.load_data(input_file, **loader_kwargs)
        processor.preprocess_data()
        splits = processor.create_splits_distributed()
        saved_files = processor.save_data_distributed(splits)
        
        # Estatísticas
        stats = processor.get_data_statistics_distributed()
        
        return {
            'success': True,
            'saved_files': saved_files,
            'statistics': stats,
            'message': 'Processamento distribuído concluído com sucesso!'
        }
        
    except Exception as e:
        logger.error(f"Erro no processamento distribuído: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Erro durante o processamento distribuído'
        }
    finally:
        processor.close()


# Função principal para compatibilidade com o código anterior
def preprocess_and_save_distributed(input_file_path: Union[str, Path], 
                                   output_jsonl_paths: Optional[Dict[str, str]] = None,
                                   **config_kwargs) -> Dict[str, Any]:
    """
    Função de compatibilidade com a interface anterior para processamento distribuído
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
    result = process_text_data_distributed(input_file_path, config)
    
    if result['success']:
        logger.info("Processamento distribuído legado concluído com sucesso!")
        for split_name in output_jsonl_paths.keys():
            if split_name in result['saved_files']:
                print(f"Dados de {split_name} processados salvos em: {result['saved_files'][split_name]}")
        
        # Exibir estatísticas do cluster se disponível
        stats = result['statistics']
        print(f"Processamento distribuído - Partições: {stats.get('partitions', 'N/A')}")
        print(f"Scheduler usado: {stats['config']['scheduler']}")
    else:
        logger.error(f"Erro no processamento distribuído legado: {result['error']}")
    
    return result
