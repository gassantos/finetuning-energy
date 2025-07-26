"""
Testes unitários para o módulo avançado de pré-processamento de texto

Este módulo testa todas as funcionalidades do text_preprocessing_advanced,
garantindo que o processamento funcione corretamente.
"""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.text_preprocessing_advanced import (
    TextPreprocessingConfig,
    AdvancedTextProcessor,
    TextCleaner,
    create_text_processor,
    process_excel_to_dataset
)


class TestTextPreprocessingConfig:
    """Testes para a configuração de pré-processamento"""
    
    def test_default_config(self):
        """Testa configuração padrão"""
        config = TextPreprocessingConfig()
        
        assert config.text_column == "Texto"
        assert config.summary_column == "Resumo"
        assert config.title_column == "Processo"
        assert config.min_text_length == 500
        assert config.test_size == 0.2
        assert config.save_formats == ['json', 'parquet']
    
    def test_custom_config(self):
        """Testa configuração customizada"""
        config = TextPreprocessingConfig(
            text_column="MeuTexto",
            summary_column="MeuResumo",
            title_column="MeuTitulo",
            min_text_length=200,
            max_text_length=10000,
            min_summary_length=50,
            max_summary_length=1000,
            test_size=0.3,
            validation_size=0.15,
            output_dir="custom_output",
            save_formats=['json']
        )
        
        assert config.text_column == "MeuTexto"
        assert config.summary_column == "MeuResumo"
        assert config.title_column == "MeuTitulo"
        assert config.min_text_length == 200
        assert config.max_text_length == 10000
        assert config.min_summary_length == 50
        assert config.max_summary_length == 1000
        assert config.test_size == 0.3
        assert config.validation_size == 0.15
        assert config.output_dir == "custom_output"
        assert config.save_formats == ['json']


class TestTextCleaner:
    """Testes para limpeza de texto"""
    
    def test_fix_encoding_issues(self):
        """Testa correção de problemas de encoding"""
        config = TextPreprocessingConfig()
        
        text_with_issues = "Texto com problemas de encoding"
        clean_text = TextCleaner.clean_text(text_with_issues, config)
        
        assert isinstance(clean_text, str)
        assert len(clean_text) > 0

    def test_remove_xml_tags(self):
        """Testa remoção de tags XML"""
        config = TextPreprocessingConfig()
        
        dirty_text = "PLENÁRIO _x000D_\n_x000D_\nTEXTO _x000A_ aqui"
        clean_text = TextCleaner.clean_text(dirty_text, config)
        
        assert "_x000D_" not in clean_text
        assert "_x000A_" not in clean_text
        assert "PLENÁRIO" in clean_text
    
    def test_normalize_unicode(self):
        """Testa normalização unicode"""
        config = TextPreprocessingConfig(normalize_unicode=True)
        
        text_with_unicode = "Çafé com açúcar"
        cleaned = TextCleaner.clean_text(text_with_unicode, config)
        
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
    
    def test_remove_extra_whitespace(self):
        """Testa remoção de espaços extras"""
        config = TextPreprocessingConfig(remove_extra_whitespace=True)
        
        messy_text = "Texto   com    espaços      extras\n\n\n"
        cleaned = TextCleaner.clean_text(messy_text, config)
        
        assert "   " not in cleaned
        assert cleaned.strip() == cleaned
    
    def test_handle_non_string(self):
        """Testa tratamento de entrada não-string"""
        config = TextPreprocessingConfig()
        
        assert TextCleaner.clean_text(None, config) == ""
        assert TextCleaner.clean_text(123, config) == ""
        assert TextCleaner.clean_text([], config) == ""


class TestAdvancedTextProcessor:
    """Testes para o processador de dados"""
    
    @pytest.fixture
    def processor(self):
        """Fixture para criar um processador"""
        config = TextPreprocessingConfig()
        return AdvancedTextProcessor(config)


class TestFactoryFunctions:
    """Testes para funções de conveniência"""
    
    @pytest.fixture
    def sample_excel_file(self):
        """Cria arquivo Excel de amostra"""
        data = pd.DataFrame({
            'Processo': ['TEST001'],
            'Texto': ['Texto de teste longo o suficiente para passar nos filtros. ' * 20],
            'Resumo': ['Resumo de teste com conteúdo adequado. ' * 10]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            data.to_excel(tmp.name, index=False)
            yield tmp.name
        Path(tmp.name).unlink(missing_ok=True)
    
    def test_create_text_processor(self, sample_excel_file):
        """Testa função factory para processador"""
        processor = create_text_processor(sample_excel_file)
        
        assert isinstance(processor, AdvancedTextProcessor)
        assert processor.raw_data is not None
        assert len(processor.raw_data) == 1
    
    @patch('src.text_preprocessing_advanced.AdvancedTextProcessor')
    def test_process_excel_to_dataset_mock(self, mock_processor_class):
        """Testa função de processamento completo com mock"""
        # Configurar mock
        mock_processor = MagicMock()
        mock_processor.preprocess_data.return_value = {'text': ['test'], 'summary': ['test'], 'title': ['test']}
        mock_processor.create_dataset_splits.return_value = {'train': MagicMock()}
        mock_processor.save_dataset.return_value = {'json': 'test.json'}
        mock_processor.validate_dataset_format.return_value = {'valid': True}
        mock_processor.get_dataset_info.return_value = {'total_examples': 1}
        mock_processor_class.return_value = mock_processor
        
        result = process_excel_to_dataset("fake_file.xlsx")
        
        assert result['success'] is True
        assert 'saved_files' in result
        assert 'validation' in result
        assert 'dataset_info' in result


class TestIntegration:
    """Testes de integração"""
    
    def test_full_pipeline_small_dataset(self):
        """Testa pipeline completo com dataset pequeno"""
        # Criar dataset pequeno mas válido
        data = pd.DataFrame({
            'Processo': [f'PROC_{i:03d}' for i in range(5)],
            'Texto': [f'Texto de exemplo número {i} com conteúdo suficiente para passar nos filtros de validação. ' * 30 for i in range(5)],
            'Resumo': [f'Resumo do documento {i} com informações relevantes e suficientes. ' * 15 for i in range(5)]
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Salvar arquivo Excel
            excel_path = Path(tmp_dir) / "test_dataset.xlsx"
            data.to_excel(excel_path, index=False)
            
            # Configurar saída
            output_dir = Path(tmp_dir) / "output"
            
            # Executar pipeline
            config = TextPreprocessingConfig(
                min_text_length=100,
                min_summary_length=50,
                output_dir=str(output_dir)
            )
            
            processor = AdvancedTextProcessor(config)
            processor.load_excel(excel_path)
            processor.preprocess_data()
            processor.create_dataset_splits()
            saved_files = processor.save_dataset()
            
            # Verificar resultados
            assert len(saved_files) > 0
            assert output_dir.exists()
            
            # Verificar arquivo JSON
            json_file = output_dir / "dataset_simple.json"
            if json_file.exists():
                with open(json_file) as f:
                    data_json = json.load(f)
                    assert 'text' in data_json
                    assert 'summary' in data_json
                    assert 'title' in data_json
                    assert len(data_json['text']) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
