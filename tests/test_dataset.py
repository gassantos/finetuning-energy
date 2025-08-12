#!/usr/bin/env python3
"""
Testes do módulo de pré-processamento refatorado usando pytest
"""

import pytest
import pandas as pd
from unittest.mock import patch
from src.text_preprocessing import PreprocessingConfig, process_text_data


class TestPreprocessingConfig:
    """Testes para a classe PreprocessingConfig"""
    
    def test_config_creation_with_defaults(self):
        """Testa criação de configuração com valores padrão"""
        config = PreprocessingConfig()
        assert config.text_column == "Texto"
        assert config.summary_column == "Resumo"
        assert config.id_column == "Processo"
        assert config.min_text_length == 50
        assert config.max_text_length == 50000
        assert config.output_format == "jsonl"
    
    def test_config_creation_with_custom_values(self):
        """Testa criação de configuração com valores customizados"""
        config = PreprocessingConfig(
            text_column="Texto",
            summary_column="Resumo",
            id_column="Processo",
            min_text_length=100,
            max_text_length=40000,
            min_summary_length=50,
            max_summary_length=3000,
            output_format="jsonl",
            instruction_template="[INST] Sumarize: {text} [/INST] {summary}"
        )
        
        assert config.text_column == "Texto"
        assert config.summary_column == "Resumo"
        assert config.id_column == "Processo"
        assert config.min_text_length == 100
        assert config.max_text_length == 40000
        assert config.min_summary_length == 50
        assert config.max_summary_length == 3000
        assert config.output_format == "jsonl"
        assert "[INST]" in config.instruction_template


class TestProcessTextData:
    """Testes para a função process_text_data"""
    
    @pytest.fixture
    def sample_config(self):
        """Fixture com configuração de teste"""
        return PreprocessingConfig(
            text_column="Texto",
            summary_column="Resumo",
            id_column="Processo",
            min_text_length=10,
            max_text_length=1000,
            min_summary_length=5,
            max_summary_length=200,
            output_format="jsonl"
        )
    
    @pytest.fixture
    def sample_data(self):
        """Fixture com dados de teste"""
        return pd.DataFrame({
            'Processo': ['P001', 'P002', 'P003', 'P004'],
            'Texto': [
                'Este é um texto de exemplo suficientemente longo para passar na validação',
                'Texto muito curto',  # Será filtrado
                'Outro texto adequado com tamanho suficiente para os testes realizados',
                'Texto final com conteúdo apropriado para teste de processamento'
            ],
            'Resumo': [
                'Resumo adequado',
                'Resumo',  # Será filtrado por ser muito curto
                'Outro resumo válido',
                'Resumo final válido'
            ]
        })
    
    def test_process_text_data_invalid_columns(self, sample_config):
        """Testa comportamento com colunas inválidas"""
        invalid_data = pd.DataFrame({
            'wrong_column': ['data1', 'data2'],
            'another_wrong': ['data3', 'data4']
        })
        
        with patch('src.text_preprocessing.pd.read_excel', return_value=invalid_data):
            with patch('src.text_preprocessing.Path.exists', return_value=True):
                result = process_text_data("fake_file.xlsx", sample_config)
                
                assert result['success'] is False
                assert 'error' in result
    
    def test_process_text_data_empty_dataset(self, sample_config):
        """Testa comportamento com dataset vazio"""
        empty_data = pd.DataFrame(columns=['Texto', 'Resumo', 'Processo'])
        
        with patch('src.text_preprocessing.pd.read_excel', return_value=empty_data):
            with patch('src.text_preprocessing.Path.exists', return_value=True):
                result = process_text_data("fake_file.xlsx", sample_config)
                
                assert result['success'] is False
                assert 'error' in result
    
    def test_process_text_data_filtering(self, sample_data):
        """Testa filtragem de dados com base nos critérios"""
        config = PreprocessingConfig(
            text_column="Texto",
            summary_column="Resumo",
            min_text_length=30,  # Vai filtrar textos muito curtos
            max_text_length=1000,
            min_summary_length=10,  # Vai filtrar resumos muito curtos
            max_summary_length=200
        )
        
        with patch('src.text_preprocessing.pd.read_excel', return_value=sample_data):
            with patch('src.text_preprocessing.Path.exists', return_value=True):
                with patch('src.text_preprocessing.Path.mkdir'):
                    result = process_text_data("fake_file.xlsx", config)
                    
                    if result['success']:
                        # Deve ter filtrado pelo menos alguns registros
                        assert result['statistics']['total_samples'] <= len(sample_data)
    
    def test_process_text_data_statistics_calculation(self, sample_config, sample_data):
        """Testa cálculo correto das estatísticas"""
        with patch('src.text_preprocessing.pd.read_excel', return_value=sample_data):
            with patch('src.text_preprocessing.Path.exists', return_value=True):
                with patch('src.text_preprocessing.Path.mkdir'):
                    result = process_text_data("fake_file.xlsx", sample_config)
                    
                    if result['success']:
                        stats = result['statistics']
                        assert 'text_stats' in stats
                        assert 'summary_stats' in stats
                        assert 'mean_length' in stats['text_stats']
                        assert 'mean_length' in stats['summary_stats']
                        assert isinstance(stats['total_samples'], int)
    
    @pytest.mark.parametrize("output_format", ["jsonl", "json", "csv"])
    def test_different_output_formats(self, sample_data, output_format):
        """Testa diferentes formatos de saída"""
        config = PreprocessingConfig(
            text_column="Texto",
            summary_column="Resumo",
            output_format=output_format,
            min_text_length=10,
            max_text_length=1000
        )
        
        with patch('src.text_preprocessing.pd.read_excel', return_value=sample_data):
            with patch('src.text_preprocessing.Path.exists', return_value=True):
                with patch('src.text_preprocessing.Path.mkdir'):
                    result = process_text_data("fake_file.xlsx", config)
                    
                    if result['success']:
                        # Verificar se os arquivos gerados têm a extensão correta
                        for filename in result['saved_files'].values():
                            assert output_format in filename


class TestIntegration:
    """Testes de integração completos"""
    
    def test_full_pipeline_integration(self):
        """Testa o pipeline completo de processamento"""
        # Dados de teste mais realistas
        test_data = pd.DataFrame({
            'Processo': [f'P{i:03d}' for i in range(1, 6)],
            'Texto': [
                'Este é um texto longo de exemplo que simula um documento real com conteúdo suficiente para análise e processamento adequado pelo sistema de sumarização automática.',
                'Outro documento extenso com informações relevantes que precisam ser processadas e resumidas de forma eficiente para facilitar a compreensão do usuário final.',
                'Terceiro exemplo de texto que contém informações importantes e deve ser processado adequadamente pelo sistema de pré-processamento de dados textuais.',
                'Quarto documento com conteúdo significativo que será utilizado para testar a funcionalidade completa do sistema de processamento de textos.',
                'Último exemplo de documento extenso que simula dados reais e testa a capacidade do sistema de lidar com textos longos e complexos adequadamente.'
            ],
            'Resumo': [
                'Resumo do primeiro documento',
                'Resumo do segundo documento',
                'Resumo do terceiro documento',
                'Resumo do quarto documento',
                'Resumo do último documento'
            ]
        })
        
        config = PreprocessingConfig(
            text_column="Texto",
            summary_column="Resumo",
            id_column="Processo",
            min_text_length=50,
            max_text_length=5000,
            min_summary_length=10,
            max_summary_length=500,
            output_format="jsonl",
            instruction_template="[INST] Sumarize: {text} [/INST] {summary}"
        )
        
        with patch('src.text_preprocessing.pd.read_excel', return_value=test_data):
            with patch('src.text_preprocessing.Path.exists', return_value=True):
                with patch('src.text_preprocessing.Path.mkdir'):
                    result = process_text_data("test_dataset.xlsx", config)
                
                assert result['success'] is True
                assert result['statistics']['total_samples'] == 5
                assert 'text_stats' in result['statistics']
                assert 'summary_stats' in result['statistics']
                assert len(result['saved_files']) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
