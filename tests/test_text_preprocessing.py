"""
Testes para o módulo de pré-processamento de texto distribuído com Dask.

Este módulo testa as funcionalidades do sistema distribuído de pré-processamento
que suporta múltiplos formatos de entrada (Excel, CSV, JSON, TXT) e usa Dask
para processamento paralelo e distribuído.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import dask.dataframe as dd
import pandas as pd
import pytest

from src.text_preprocessing import (
    PreprocessingConfig,
    DaskTextCleaner,
    DaskDataValidator,
    DaskExcelLoader,
    DaskCSVLoader, 
    DaskJSONLoader,
    DaskTextLoader,
    DistributedTextPreprocessor,
    process_text_data_distributed,
    preprocess_and_save_distributed
)


class TestPreprocessingConfig:
    """Testes para a classe PreprocessingConfig."""

    def test_default_config(self):
        """Testa configuração padrão."""
        config = PreprocessingConfig()
        
        assert config.text_column == "Texto"
        assert config.summary_column == "Resumo"
        assert config.output_format == "jsonl"
        assert config.test_size == 0.2
        assert config.val_size == 0.1
        assert config.min_text_length == 50
        assert config.max_text_length == 50000
        assert config.scheduler == "threads"

    def test_custom_config(self):
        """Testa configuração personalizada."""
        config = PreprocessingConfig(
            text_column="content",
            summary_column="abstract",
            min_text_length=100,
            output_format="json",
            scheduler="processes"
        )
        
        assert config.text_column == "content"
        assert config.summary_column == "abstract"
        assert config.min_text_length == 100
        assert config.output_format == "json"
        assert config.scheduler == "processes"

    def test_invalid_test_size(self):
        """Testa validação de test_size inválido."""
        with pytest.raises(ValueError, match="test_size deve estar entre 0 e 1"):
            PreprocessingConfig(test_size=1.5)
        
        with pytest.raises(ValueError, match="test_size deve estar entre 0 e 1"):
            PreprocessingConfig(test_size=-0.1)

    def test_invalid_split_sizes(self):
        """Testa validação de tamanhos de split inválidos."""
        with pytest.raises(ValueError, match="deve ser menor que 1.0"):
            PreprocessingConfig(test_size=0.8, val_size=0.5)

    def test_invalid_scheduler(self):
        """Testa validação de scheduler inválido."""
        with pytest.raises(ValueError, match="scheduler deve ser"):
            PreprocessingConfig(scheduler="invalid_scheduler")


class TestDaskTextCleaner:
    """Testes para a classe DaskTextCleaner distribuída."""

    def test_clean_text_basic(self):
        """Testa limpeza básica de texto."""
        config = PreprocessingConfig()
        
        text = "Texto    com    espaços   extras"
        result = DaskTextCleaner.clean_text(text, config)
        
        assert result == "Texto com espaços extras"

    def test_clean_text_encoding_artifacts(self):
        """Testa remoção de artefatos de encoding."""
        config = PreprocessingConfig(remove_encoding_artifacts=True)
        
        text = "Texto_x000D_com_x000A_artefatos"
        result = DaskTextCleaner.clean_text(text, config)
        
        assert "_x000D_" not in result
        assert "_x000A_" not in result
        assert "com" in result and "artefatos" in result

    def test_clean_text_non_string(self):
        """Testa comportamento com entrada não-string."""
        config = PreprocessingConfig()
        
        result = DaskTextCleaner.clean_text(None, config)
        assert result == ""
        
        result = DaskTextCleaner.clean_text(123, config)
        assert result == ""

    def test_clean_text_lowercase(self):
        """Testa conversão para minúsculas."""
        config = PreprocessingConfig(lowercase=True)
        
        text = "TEXTO EM MAIÚSCULAS"
        result = DaskTextCleaner.clean_text(text, config)
        
        assert result == "texto em maiúsculas"

    def test_clean_text_remove_punctuation(self):
        """Testa remoção de pontuação."""
        config = PreprocessingConfig(remove_punctuation=True)
        
        text = "Texto! Com? Pontuação..."
        result = DaskTextCleaner.clean_text(text, config)
        
        assert "!" not in result
        assert "?" not in result
        assert "." not in result
        assert "Texto Com Pontuação" in result

    def test_apply_cleaning_distributed(self):
        """Testa aplicação distribuída de limpeza."""
        config = PreprocessingConfig(remove_extra_whitespace=True)
        
        # Criar DataFrame de teste
        df = pd.DataFrame({
            'texto': ['Texto   com   espaços', 'Outro    texto  longo']
        })
        ddf = dd.from_pandas(df, npartitions=1)
        
        result_ddf = DaskTextCleaner.apply_cleaning_distributed(ddf, 'texto', config)
        result_df = result_ddf.compute()
        
        assert 'texto_clean' in result_df.columns
        assert 'Texto com espaços' == result_df.iloc[0]['texto_clean']
        assert 'Outro texto longo' == result_df.iloc[1]['texto_clean']


class TestDaskDataValidator:
    """Testes para a classe DaskDataValidator distribuída."""

    def test_validate_dataframe_valid(self):
        """Testa validação de DataFrame distribuído válido."""
        config = PreprocessingConfig()
        df = pd.DataFrame({
            "Texto": ["Texto de exemplo 1", "Texto de exemplo 2"],
            "Resumo": ["Resumo 1", "Resumo 2"]
        })
        ddf = dd.from_pandas(df, npartitions=1)
        
        result = DaskDataValidator.validate_dataframe_distributed(ddf, config)
        
        assert result['statistics']['total_rows'] == 2
        assert result['statistics']['missing_text'] == 0
        assert result['statistics']['missing_summary'] == 0

    def test_validate_dataframe_missing_columns(self):
        """Testa validação com colunas faltando."""
        config = PreprocessingConfig()
        df = pd.DataFrame({
            "content": ["Texto de exemplo"],
            "abstract": ["Resumo"]
        })
        ddf = dd.from_pandas(df, npartitions=1)
        
        result = DaskDataValidator.validate_dataframe_distributed(ddf, config)
        
        assert not result['valid']
        assert any("Colunas faltando" in issue for issue in result['issues'])

    def test_validate_dataframe_missing_values(self):
        """Testa detecção de valores faltantes."""
        config = PreprocessingConfig()
        df = pd.DataFrame({
            "Texto": ["Texto válido", None],
            "Resumo": ["Resumo válido", "Resumo 2"]
        })
        ddf = dd.from_pandas(df, npartitions=1)
        
        result = DaskDataValidator.validate_dataframe_distributed(ddf, config)
        
        assert result['statistics']['missing_text'] == 1
        assert result['statistics']['missing_summary'] == 0


class TestDaskDataLoaders:
    """Testes para os carregadores de dados distribuídos."""

    def test_csv_loader(self):
        """Testa carregamento de CSV distribuído."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("Texto,Resumo\n")
            f.write("Texto de teste,Resumo de teste\n")
            csv_path = f.name

        try:
            loader = DaskCSVLoader()
            ddf = loader.load(csv_path)
            df = ddf.compute()
            
            assert len(df) == 1
            assert "Texto" in df.columns
            assert "Resumo" in df.columns
            assert df.iloc[0]["Texto"] == "Texto de teste"
        finally:
            Path(csv_path).unlink()

    def test_json_loader(self):
        """Testa carregamento de JSON distribuído."""
        data = [
            {"Texto": "Texto de teste", "Resumo": "Resumo de teste"},
            {"Texto": "Segundo texto", "Resumo": "Segundo resumo"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            json_path = f.name

        try:
            loader = DaskJSONLoader()
            ddf = loader.load(json_path)
            df = ddf.compute()
            
            assert len(df) == 2
            assert "Texto" in df.columns
            assert "Resumo" in df.columns
        finally:
            Path(json_path).unlink()

    def test_text_loader(self):
        """Testa carregamento de arquivo de texto distribuído."""
        content = "Este é um texto de exemplo\npara teste do carregador."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            txt_path = f.name

        try:
            loader = DaskTextLoader()
            ddf = loader.load(txt_path)
            df = ddf.compute()
            
            assert len(df) == 1
            assert "text" in df.columns
            assert content in df.iloc[0]["text"]
        finally:
            Path(txt_path).unlink()

    def test_excel_loader(self):
        """Testa carregamento de Excel distribuído."""
        data = {
            "Texto": ["Texto de teste 1", "Texto de teste 2"],
            "Resumo": ["Resumo 1", "Resumo 2"]
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df.to_excel(f.name, index=False)
            excel_path = f.name

        try:
            loader = DaskExcelLoader()
            ddf = loader.load(excel_path)
            loaded_df = ddf.compute()
            
            assert len(loaded_df) == 2
            assert "Texto" in loaded_df.columns
            assert "Resumo" in loaded_df.columns
        finally:
            Path(excel_path).unlink()


class TestDistributedTextPreprocessor:
    """Testes para o processador distribuído."""

    def create_test_excel(self):
        """Cria arquivo Excel para testes."""
        data = {
            "Processo": ["123-2024", "456-2024", "789-2024"],
            "Texto": [
                "Este é um texto longo o suficiente para passar nos filtros de comprimento mínimo e ser processado corretamente.",
                "Outro texto_x000D_com artefatos de encoding que devem ser removidos durante o processamento.",
                "Terceiro texto com conteúdo válido e tamanho adequado para os testes unitários do sistema."
            ],
            "Resumo": [
                "Resumo do primeiro texto com informações essenciais.",
                "Resumo do segundo texto limpo e processado.",
                "Resumo do terceiro texto para validação."
            ]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df.to_excel(f.name, index=False)
            return f.name

    def test_load_data_excel(self):
        """Testa carregamento de dados Excel."""
        excel_path = self.create_test_excel()
        
        try:
            config = PreprocessingConfig(validate_data_quality=False, scheduler='threads')
            processor = DistributedTextPreprocessor(config)
            
            ddf = processor.load_data(excel_path)
            df = ddf.compute()
            
            assert len(df) == 3
            assert "Texto" in df.columns
            assert "Resumo" in df.columns
            assert "Processo" in df.columns
        finally:
            Path(excel_path).unlink()
            processor.close()

    def test_preprocess_data(self):
        """Testa pré-processamento de dados distribuído."""
        excel_path = self.create_test_excel()
        
        try:
            config = PreprocessingConfig(
                min_text_length=30,
                min_summary_length=20,
                validate_data_quality=False,
                scheduler='threads'
            )
            processor = DistributedTextPreprocessor(config)
            
            processor.load_data(excel_path)
            processed_ddf = processor.preprocess_data()
            processed_df = processed_ddf.compute()
            
            assert 'Texto_clean' in processed_df.columns
            assert 'Resumo_clean' in processed_df.columns
            # A coluna ID padrão é "Processo" (definida em config.id_column)
            assert config.id_column in processed_df.columns
            assert len(processed_df) > 0
            
            # Verificar se limpeza foi aplicada
            for text in processed_df['Texto_clean']:
                assert '_x000D_' not in text
                assert len(text.strip()) > 0
        finally:
            Path(excel_path).unlink()
            processor.close()

    def test_create_splits_distributed(self):
        """Testa criação de splits distribuídos."""
        excel_path = self.create_test_excel()
        
        try:
            config = PreprocessingConfig(
                min_text_length=30,
                min_summary_length=20,
                test_size=0.3,
                val_size=0.2,
                validate_data_quality=False,
                scheduler='threads'
            )
            processor = DistributedTextPreprocessor(config)
            
            processor.load_data(excel_path)
            processor.preprocess_data()
            splits = processor.create_splits_distributed()
            
            assert 'train' in splits
            assert 'validation' in splits
            assert 'test' in splits
            
            # Verificar que todos os splits têm dados
            total_samples = sum(len(split.compute()) for split in splits.values())
            assert total_samples > 0
        finally:
            Path(excel_path).unlink()
            processor.close()


class TestFunctionsAPI:
    """Testes para as funções de conveniência da API distribuída."""

    def create_test_csv(self):
        """Cria arquivo CSV para testes."""
        data = {
            "Processo": [f"PROC-{i:03d}" for i in range(1, 6)],
            "Texto": [
                f"Texto de exemplo número {i} longo o suficiente para ser processado pelo sistema de pré-processamento de texto distribuído com Dask."
                for i in range(1, 6)
            ],
            "Resumo": [
                f"Resumo do texto de exemplo número {i} para testes distribuídos."
                for i in range(1, 6)
            ]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name

    def test_process_text_data_distributed_function(self):
        """Testa função process_text_data_distributed."""
        csv_path = self.create_test_csv()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = PreprocessingConfig(
                    output_dir=temp_dir,
                    min_text_length=30,
                    min_summary_length=15,
                    validate_data_quality=False,
                    scheduler='threads'
                )
                
                result = process_text_data_distributed(csv_path, config)
                
                assert result['success']
                assert 'statistics' in result
                assert 'saved_files' in result
                assert 'message' in result
                
                assert result['statistics']['total_samples'] > 0
        finally:
            Path(csv_path).unlink()

    def test_preprocess_and_save_distributed_legacy(self):
        """Testa função legacy preprocess_and_save_distributed."""
        csv_path = self.create_test_csv()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_paths = {
                    'train': f"{temp_dir}/train.jsonl",
                    'validation': f"{temp_dir}/val.jsonl",
                    'test': f"{temp_dir}/test.jsonl"
                }
                
                result = preprocess_and_save_distributed(
                    csv_path, 
                    output_paths,
                    min_text_length=30,
                    min_summary_length=15,
                    output_dir=temp_dir,
                    scheduler='threads'
                )
                
                # Verificar sucesso
                assert result['success']
                
                # Verificar que pelo menos um arquivo foi criado
                created_files = [f for f in Path(temp_dir).glob("*.jsonl")]
                assert len(created_files) > 0
        finally:
            Path(csv_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
