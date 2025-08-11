"""
Testes para o módulo de pré-processamento de texto refatorado.

Este módulo testa as funcionalidades do sistema universal de pré-processamento
que suporta múltiplos formatos de entrada (Excel, CSV, JSON, TXT).
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.text_preprocessing import (
    PreprocessingConfig,
    UniversalTextPreprocessor,
    DataValidator,
    TextCleaner,
    CSVLoader,
    JSONLoader,
    TextLoader,
    process_text_data,
    preprocess_and_save
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

    def test_custom_config(self):
        """Testa configuração personalizada."""
        config = PreprocessingConfig(
            text_column="content",
            summary_column="abstract",
            min_text_length=100,
            output_format="json"
        )
        
        assert config.text_column == "content"
        assert config.summary_column == "abstract"
        assert config.min_text_length == 100
        assert config.output_format == "json"

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


class TestTextCleaner:
    """Testes para a classe TextCleaner."""

    def test_clean_text_basic(self):
        """Testa limpeza básica de texto."""
        config = PreprocessingConfig()
        
        text = "Texto    com    espaços   extras"
        result = TextCleaner.clean_text(text, config)
        
        assert result == "Texto com espaços extras"

    def test_clean_text_encoding_artifacts(self):
        """Testa remoção de artefatos de encoding."""
        config = PreprocessingConfig(remove_encoding_artifacts=True)
        
        text = "Texto_x000D_com_x000A_artefatos"
        result = TextCleaner.clean_text(text, config)
        
        assert "_x000D_" not in result
        assert "_x000A_" not in result
        # Após limpeza de espaços extras, \n vira espaço
        assert "Texto com artefatos" == result

    def test_clean_text_non_string(self):
        """Testa comportamento com entrada não-string."""
        config = PreprocessingConfig()
        
        result = TextCleaner.clean_text(None, config)
        assert result == ""
        
        result = TextCleaner.clean_text(123, config)
        assert result == ""

    def test_clean_text_lowercase(self):
        """Testa conversão para minúsculas."""
        config = PreprocessingConfig(lowercase=True)
        
        text = "TEXTO EM MAIÚSCULAS"
        result = TextCleaner.clean_text(text, config)
        
        assert result == "texto em maiúsculas"

    def test_clean_text_remove_punctuation(self):
        """Testa remoção de pontuação."""
        config = PreprocessingConfig(remove_punctuation=True)
        
        text = "Texto! Com? Pontuação..."
        result = TextCleaner.clean_text(text, config)
        
        assert "!" not in result
        assert "?" not in result
        assert "." not in result
        assert "Texto Com Pontuação" in result


class TestDataValidator:
    """Testes para a classe DataValidator."""

    def test_validate_dataframe_valid(self):
        """Testa validação de DataFrame válido."""
        config = PreprocessingConfig()
        df = pd.DataFrame({
            "Texto": ["Texto de exemplo 1", "Texto de exemplo 2"],
            "Resumo": ["Resumo 1", "Resumo 2"]
        })
        
        result = DataValidator.validate_dataframe(df, config)
        
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
        
        result = DataValidator.validate_dataframe(df, config)
        
        assert not result['valid']
        assert "Colunas faltando" in result['issues'][0]

    def test_validate_dataframe_missing_values(self):
        """Testa detecção de valores faltantes."""
        config = PreprocessingConfig()
        df = pd.DataFrame({
            "Texto": ["Texto válido", None],
            "Resumo": ["Resumo válido", "Resumo 2"]
        })
        
        result = DataValidator.validate_dataframe(df, config)
        
        assert result['statistics']['missing_text'] == 1
        assert result['statistics']['missing_summary'] == 0


class TestDataLoaders:
    """Testes para os carregadores de dados."""

    def test_csv_loader(self):
        """Testa carregamento de CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("Texto,Resumo\n")
            f.write("Texto de teste,Resumo de teste\n")
            csv_path = f.name

        try:
            loader = CSVLoader()
            df = loader.load(csv_path)
            
            assert len(df) == 1
            assert "Texto" in df.columns
            assert "Resumo" in df.columns
            assert df.iloc[0]["Texto"] == "Texto de teste"
        finally:
            Path(csv_path).unlink()

    def test_json_loader(self):
        """Testa carregamento de JSON."""
        data = [
            {"Texto": "Texto de teste", "Resumo": "Resumo de teste"},
            {"Texto": "Segundo texto", "Resumo": "Segundo resumo"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            json_path = f.name

        try:
            loader = JSONLoader()
            df = loader.load(json_path)
            
            assert len(df) == 2
            assert "Texto" in df.columns
            assert "Resumo" in df.columns
        finally:
            Path(json_path).unlink()

    def test_text_loader(self):
        """Testa carregamento de arquivo de texto."""
        content = "Este é um texto de exemplo\npara teste do carregador."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            txt_path = f.name

        try:
            loader = TextLoader()
            df = loader.load(txt_path)
            
            assert len(df) == 1
            assert "text" in df.columns
            assert content in df.iloc[0]["text"]
        finally:
            Path(txt_path).unlink()


class TestUniversalTextPreprocessor:
    """Testes para o processador universal."""

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
            config = PreprocessingConfig(validate_data_quality=False)
            processor = UniversalTextPreprocessor(config)
            
            df = processor.load_data(excel_path)
            
            assert len(df) == 3
            assert "Texto" in df.columns
            assert "Resumo" in df.columns
            assert "Processo" in df.columns
        finally:
            Path(excel_path).unlink()

    def test_preprocess_data(self):
        """Testa pré-processamento de dados."""
        excel_path = self.create_test_excel()
        
        try:
            config = PreprocessingConfig(
                min_text_length=30,
                min_summary_length=20,
                validate_data_quality=False
            )
            processor = UniversalTextPreprocessor(config)
            
            processor.load_data(excel_path)
            processed_data = processor.preprocess_data()
            
            assert 'text' in processed_data
            assert 'summary' in processed_data
            assert 'id' in processed_data
            assert len(processed_data['text']) > 0
            
            # Verificar se limpeza foi aplicada
            for text in processed_data['text']:
                assert '_x000D_' not in text
                assert len(text.strip()) > 0
        finally:
            Path(excel_path).unlink()

    def test_create_splits(self):
        """Testa criação de splits de dados."""
        excel_path = self.create_test_excel()
        
        try:
            config = PreprocessingConfig(
                min_text_length=30,
                min_summary_length=20,
                test_size=0.3,
                val_size=0.2,
                validate_data_quality=False
            )
            processor = UniversalTextPreprocessor(config)
            
            processor.load_data(excel_path)
            processor.preprocess_data()
            splits = processor.create_splits()
            
            assert 'train' in splits
            assert 'validation' in splits
            assert 'test' in splits
            
            # Verificar que todos os splits têm dados
            total_samples = sum(len(split['text']) for split in splits.values())
            assert total_samples > 0
        finally:
            Path(excel_path).unlink()

    def test_save_data_jsonl(self):
        """Testa salvamento em formato JSONL."""
        excel_path = self.create_test_excel()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = PreprocessingConfig(
                    output_dir=temp_dir,
                    output_format="jsonl",
                    min_text_length=30,
                    min_summary_length=20,
                    validate_data_quality=False
                )
                processor = UniversalTextPreprocessor(config)
                
                processor.load_data(excel_path)
                processor.preprocess_data()
                splits = processor.create_splits()
                saved_files = processor.save_data(splits)
                
                # Verificar que arquivos foram criados
                for split_name, file_path in saved_files.items():
                    assert Path(file_path).exists()
                    
                    # Verificar conteúdo JSONL
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        assert len(lines) > 0
                        
                        # Verificar se cada linha é JSON válido
                        for line in lines:
                            data = json.loads(line.strip())
                            assert 'text' in data
        finally:
            Path(excel_path).unlink()

    def test_get_data_statistics(self):
        """Testa geração de estatísticas."""
        excel_path = self.create_test_excel()
        
        try:
            config = PreprocessingConfig(
                min_text_length=30,
                min_summary_length=20,
                validate_data_quality=False
            )
            processor = UniversalTextPreprocessor(config)
            
            processor.load_data(excel_path)
            processor.preprocess_data()
            stats = processor.get_data_statistics()
            
            assert 'total_samples' in stats
            assert 'text_stats' in stats
            assert 'summary_stats' in stats
            assert 'config' in stats
            
            assert stats['total_samples'] > 0
            assert stats['text_stats']['mean_length'] > 0
            assert stats['summary_stats']['mean_length'] > 0
        finally:
            Path(excel_path).unlink()


class TestFunctionsAPI:
    """Testes para as funções de conveniência da API."""

    def create_test_csv(self):
        """Cria arquivo CSV para testes."""
        # Criar dataset maior para permitir splits adequados
        data = {
            "Processo": [f"PROC-{i:03d}" for i in range(1, 11)],  # 10 registros
            "Texto": [
                f"Texto de exemplo número {i} longo o suficiente para ser processado pelo sistema de pré-processamento de texto."
                for i in range(1, 11)
            ],
            "Resumo": [
                f"Resumo do texto de exemplo número {i}."
                for i in range(1, 11)
            ]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name

    def test_process_text_data_function(self):
        """Testa função process_text_data."""
        csv_path = self.create_test_csv()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = PreprocessingConfig(
                    output_dir=temp_dir,
                    min_text_length=30,
                    min_summary_length=15,
                    validate_data_quality=False
                )
                
                result = process_text_data(csv_path, config)
                
                assert result['success']
                assert 'statistics' in result
                assert 'saved_files' in result
                assert 'message' in result
                
                assert result['statistics']['total_samples'] > 0
        finally:
            Path(csv_path).unlink()

    def test_preprocess_and_save_legacy(self):
        """Testa função legacy preprocess_and_save."""
        csv_path = self.create_test_csv()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_paths = {
                    'train': f"{temp_dir}/train.jsonl",
                    'validation': f"{temp_dir}/val.jsonl",
                    'test': f"{temp_dir}/test.jsonl"
                }
                
                # Função deve executar sem erro
                preprocess_and_save(
                    csv_path, 
                    output_paths,
                    min_text_length=30,
                    min_summary_length=15,
                    output_dir=temp_dir
                )
                
                # Verificar que pelo menos um arquivo foi criado
                created_files = [f for f in Path(temp_dir).glob("*.jsonl")]
                assert len(created_files) > 0
        finally:
            Path(csv_path).unlink()


class TestErrorHandling:
    """Testes para tratamento de erros."""

    def test_file_not_found(self):
        """Testa tratamento de arquivo não encontrado."""
        processor = UniversalTextPreprocessor()
        
        with pytest.raises(FileNotFoundError):
            processor.load_data("arquivo_inexistente.xlsx")

    def test_unsupported_format(self):
        """Testa formato de arquivo não suportado."""
        with tempfile.NamedTemporaryFile(suffix='.doc') as f:
            processor = UniversalTextPreprocessor()
            
            with pytest.raises(ValueError, match="Formato não suportado"):
                processor.load_data(f.name)

    def test_process_without_data(self):
        """Testa processamento sem dados carregados."""
        processor = UniversalTextPreprocessor()
        
        with pytest.raises(ValueError, match="Nenhum dado carregado"):
            processor.preprocess_data()

    def test_splits_without_processed_data(self):
        """Testa criação de splits sem dados processados."""
        processor = UniversalTextPreprocessor()
        
        with pytest.raises(ValueError, match="Dados não processados"):
            processor.create_splits()


class TestIntegration:
    """Testes de integração completos."""

    def create_comprehensive_test_data(self):
        """Cria dados abrangentes para teste de integração."""
        data = {
            "Processo": [f"PROC-{i:03d}" for i in range(1, 11)],
            "Texto": [
                f"Este é o texto número {i} que contém informações detalhadas e deve ter comprimento suficiente para ser processado pelo sistema de pré-processamento. Inclui_x000D_artefatos de encoding que devem ser removidos durante a limpeza." 
                for i in range(1, 11)
            ],
            "Resumo": [
                f"Resumo {i} com conteúdo essencial e informações condensadas do texto principal."
                for i in range(1, 11)
            ],
            "Categoria": [f"Categoria-{i%3}" for i in range(1, 11)]
        }
        
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            df.to_excel(f.name, index=False)
            return f.name

    def test_full_pipeline(self):
        """Testa pipeline completo de pré-processamento."""
        excel_path = self.create_comprehensive_test_data()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = PreprocessingConfig(
                    text_column="Texto",
                    summary_column="Resumo",
                    id_column="Processo",
                    output_dir=temp_dir,
                    output_format="jsonl",
                    min_text_length=50,
                    max_text_length=1000,
                    min_summary_length=30,
                    max_summary_length=200,
                    test_size=0.2,
                    val_size=0.1,
                    instruction_template="[INST] Sumarize: {text} [/INST] {summary}",
                    validate_data_quality=True
                )
                
                result = process_text_data(excel_path, config)
                
                # Verificar sucesso
                assert result['success']
                
                # Verificar estatísticas
                stats = result['statistics']
                assert stats['total_samples'] > 0
                assert stats['text_stats']['mean_length'] > config.min_text_length
                assert stats['summary_stats']['mean_length'] > config.min_summary_length
                
                # Verificar arquivos salvos
                saved_files = result['saved_files']
                assert len(saved_files) >= 2  # Pelo menos train e test
                
                # Verificar conteúdo dos arquivos
                for split_name, file_path in saved_files.items():
                    assert Path(file_path).exists()
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        assert len(lines) > 0
                        
                        # Verificar formato JSONL com template
                        first_line = json.loads(lines[0].strip())
                        assert 'text' in first_line
                        assert '[INST]' in first_line['text']
                        assert '[/INST]' in first_line['text']
        finally:
            Path(excel_path).unlink()


@pytest.mark.skipif(not Path("data/dataset.xlsx").exists(), reason="Arquivo de dados real não encontrado")
class TestWithRealData:
    """Testes com dados reais do projeto (opcionais)."""

    def test_real_data_processing(self):
        """Testa processamento com dados reais do projeto."""
        config = PreprocessingConfig(
            min_text_length=100,
            max_text_length=40000,
            min_summary_length=50,
            max_summary_length=3000,
            validate_data_quality=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            result = process_text_data("data/dataset.xlsx", config)
            
            # Deve processar sem erros
            assert result['success']
            assert result['statistics']['total_samples'] > 0
