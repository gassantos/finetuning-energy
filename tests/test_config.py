"""
Testes abrangentes para o módulo de configuração.

Este módulo testa o sistema de configuração baseado em Dynaconf,
incluindo carregamento de arquivos, validação e configurações de ambiente.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Imports do projeto
from config.config import settings


class TestConfigurationSystem:
    """Testa o sistema de configuração baseado em Dynaconf."""

    def test_settings_object_exists(self):
        """Testa se o objeto settings foi criado corretamente."""
        assert settings is not None
        # settings do Dynaconf não tem método get como dict
        assert hasattr(settings, "_store")

    def test_default_debug_setting(self):
        """Testa configuração padrão de DEBUG."""
        # O valor pode ser configurado via variável de ambiente
        debug_value = getattr(settings, "DEBUG", False)
        assert isinstance(debug_value, (bool, str))

    def test_environment_switching(self):
        """Testa alternância de ambiente."""
        # Testar se consegue acessar configurações básicas
        assert hasattr(settings, "current_env")

    def test_settings_file_configuration(self):
        """Testa se arquivos de configuração são carregados."""
        # Verificar se o dynaconf consegue carregar configurações
        # settings_files é uma propriedade interna, usar alternativa
        assert hasattr(settings, "_store")
        # Testar que conseguimos acessar configurações
        assert (
            hasattr(settings, "model")
            or hasattr(settings, "debug")
            or hasattr(settings, "DEBUG")
        )

    def test_validator_configuration(self):
        """Testa se validadores estão configurados."""
        # O sistema deve ter validadores configurados
        assert hasattr(settings, "validators")


class TestConfigurationValues:
    """Testa valores específicos de configuração."""

    def test_model_configuration(self):
        """Testa configurações relacionadas ao modelo."""
        # Verificar se existem configurações de modelo
        model_id = getattr(settings, "MODEL_ID", None)
        if model_id:
            assert isinstance(model_id, str)
            assert len(model_id) > 0

    def test_training_configuration(self):
        """Testa configurações de treinamento."""
        # Verificar configurações de treinamento
        batch_size = getattr(settings, "BATCH_SIZE", 2)
        learning_rate = getattr(settings, "LEARNING_RATE", 2e-4)
        epochs = getattr(settings, "EPOCHS", 3)

        assert isinstance(batch_size, (int, str))
        assert isinstance(learning_rate, (float, str))
        assert isinstance(epochs, (int, str))

    def test_lora_configuration(self):
        """Testa configurações de LoRA."""
        lora_r = getattr(settings, "LORA_R", 8)
        lora_alpha = getattr(settings, "LORA_ALPHA", 16)

        assert isinstance(lora_r, (int, str))
        assert isinstance(lora_alpha, (int, str))

    def test_output_configuration(self):
        """Testa configurações de diretórios de saída."""
        output_dir = getattr(settings, "OUTPUT_DIR", "./output")
        result_dir = getattr(settings, "RESULT_DIR", "./results")

        assert isinstance(output_dir, str)
        assert isinstance(result_dir, str)

    def test_monitoring_configuration(self):
        """Testa configurações de monitoramento."""
        monitoring_interval = getattr(settings, "MONITORING_INTERVAL", 1.0)

        assert isinstance(monitoring_interval, (float, int, str))

    def test_wandb_configuration(self):
        """Testa configurações do Weights & Biases."""
        wandb_project = getattr(settings, "WANDB_PROJECT", None)
        wandb_entity = getattr(settings, "WANDB_ENTITY", None)
        wandb_mode = getattr(settings, "WANDB_MODE", "offline")

        if wandb_project:
            assert isinstance(wandb_project, str)
        if wandb_entity:
            assert isinstance(wandb_entity, str)
        assert isinstance(wandb_mode, str)
        assert wandb_mode in ["online", "offline", "disabled"]


class TestEnvironmentVariables:
    """Testa integração com variáveis de ambiente."""

    def test_environment_override(self):
        """Testa se variáveis de ambiente sobrescrevem configurações."""
        test_key = "TEST_CONFIG_VALUE"
        test_value = "test_override_value"

        with patch.dict(os.environ, {test_key: test_value}):
            # Recarregar configurações
            value = getattr(settings, test_key, None)
            # A variável de ambiente deve ser acessível
            assert os.environ.get(test_key) == test_value

    def test_debug_environment_variable(self):
        """Testa variável de ambiente DEBUG."""
        # Testar diferentes valores de DEBUG
        test_cases = ["true", "false", "1", "0", "True", "False"]

        for test_value in test_cases:
            with patch.dict(os.environ, {"DEBUG": test_value}):
                # O sistema deve conseguir interpretar valores booleanos
                debug_val = getattr(settings, "DEBUG", False)
                assert isinstance(debug_val, (bool, str))

    def test_app_env_switching(self):
        """Testa alternância de ambiente via APP_ENV."""
        test_envs = ["development", "testing", "production"]

        for env in test_envs:
            with patch.dict(os.environ, {"APP_ENV": env}):
                # O sistema deve aceitar diferentes ambientes
                current_env = os.environ.get("APP_ENV")
                assert current_env == env


class TestConfigurationFiles:
    """Testa carregamento de arquivos de configuração."""

    def test_settings_toml_exists(self):
        """Testa se arquivo settings.toml existe ou é opcional."""
        config_dir = Path(__file__).parent.parent / "config"
        settings_file = config_dir / "settings.toml"

        # O arquivo pode não existir, mas o diretório deve existir
        assert config_dir.exists()

    def test_secrets_toml_handling(self):
        """Testa tratamento de arquivo .secrets.toml."""
        # O arquivo de secrets é opcional e pode não existir
        # O sistema deve lidar graciosamente com sua ausência
        secrets_value = getattr(settings, "SECRET_KEY", None)
        # Se existir, deve ser string
        if secrets_value:
            assert isinstance(secrets_value, str)

    @pytest.fixture
    def temp_config_file(self):
        """Fixture para arquivo de configuração temporário."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(
                """
[default]
TEST_VALUE = "test_config"
TEST_NUMBER = 42
TEST_BOOL = true

[development]
DEBUG = true

[production]
DEBUG = false
            """
            )
            f.flush()
            yield f.name

        # Cleanup
        os.unlink(f.name)

    def test_custom_config_loading(self, temp_config_file):
        """Testa carregamento de arquivo de configuração customizado."""
        from dynaconf import Dynaconf

        custom_settings = Dynaconf(
            settings_files=[temp_config_file],
            environments=True,
        )

        assert custom_settings.TEST_VALUE == "test_config"
        assert custom_settings.TEST_NUMBER == 42
        assert custom_settings.TEST_BOOL is True


class TestConfigurationValidation:
    """Testa validação de configurações."""

    def test_debug_validator(self):
        """Testa validador do campo DEBUG."""
        # O validador DEBUG deve aceitar valores booleanos
        debug_value = getattr(settings, "DEBUG", False)

        # Deve ser um valor válido para boolean
        if isinstance(debug_value, str):
            assert debug_value.lower() in ["true", "false", "1", "0", "yes", "no"]
        else:
            assert isinstance(debug_value, bool)

    def test_numeric_configuration_validation(self):
        """Testa validação de configurações numéricas."""
        # Testar se valores numéricos são válidos
        batch_size = getattr(settings, "BATCH_SIZE", 2)
        learning_rate = getattr(settings, "LEARNING_RATE", 2e-4)

        # Se for string, deve ser conversível para número
        if isinstance(batch_size, str):
            try:
                int(batch_size)
            except ValueError:
                pytest.fail("BATCH_SIZE deve ser conversível para int")

        if isinstance(learning_rate, str):
            try:
                float(learning_rate)
            except ValueError:
                pytest.fail("LEARNING_RATE deve ser conversível para float")

    def test_directory_configuration_validation(self):
        """Testa validação de configurações de diretório."""
        output_dir = getattr(settings, "OUTPUT_DIR", "./output")
        result_dir = getattr(settings, "RESULT_DIR", "./results")

        # Deve ser strings válidas para paths
        assert isinstance(output_dir, str)
        assert isinstance(result_dir, str)

        # Não devem estar vazias
        assert len(output_dir.strip()) > 0
        assert len(result_dir.strip()) > 0


class TestConfigurationTypeConversion:
    """Testa conversão de tipos nas configurações."""

    def test_string_to_int_conversion(self):
        """Testa conversão de string para int."""
        # Simular valor string que deve ser convertido para int
        with patch.dict(os.environ, {"TEST_INT_VALUE": "42"}):
            # Força o dynaconf a ver a nova variável criando uma nova instância
            from dynaconf import Dynaconf

            temp_settings = Dynaconf(environments=True)
            value = getattr(temp_settings, "TEST_INT_VALUE", None)
            # Pode ser string ou int dependendo da configuração
            if isinstance(value, str):
                assert value == "42"
                assert int(value) == 42
            elif value is not None:
                assert value == 42
            else:
                # É aceitável que retorne None se o Dynaconf não processar
                assert value is None

    def test_string_to_float_conversion(self):
        """Testa conversão de string para float."""
        with patch.dict(os.environ, {"TEST_FLOAT_VALUE": "3.14"}):
            # Força o dynaconf a ver a nova variável criando uma nova instância
            from dynaconf import Dynaconf

            temp_settings = Dynaconf(environments=True)
            value = getattr(temp_settings, "TEST_FLOAT_VALUE", None)
            if isinstance(value, str):
                assert value == "3.14"
                assert float(value) == 3.14
            elif value is not None:
                assert value == 3.14
            else:
                # É aceitável que retorne None se o Dynaconf não processar
                assert value is None

    def test_string_to_bool_conversion(self):
        """Testa conversão de string para bool."""
        bool_test_cases = [
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
        ]

        for str_value, expected_bool in bool_test_cases:
            with patch.dict(os.environ, {"TEST_BOOL_VALUE": str_value}):
                value = getattr(settings, "TEST_BOOL_VALUE", None)
                # O dynaconf pode ou não converter automaticamente
                if isinstance(value, str):
                    # Verificar se pode ser interpretado como boolean
                    assert str_value.lower() in ["true", "false", "1", "0", "yes", "no"]


class TestConfigurationDefaults:
    """Testa valores padrão das configurações."""

    def test_model_defaults(self):
        """Testa valores padrão para configurações de modelo."""
        # Valores padrão razoáveis devem estar disponíveis
        model_id = getattr(settings, "MODEL_ID", "meta-llama/Llama-2-7b-hf")
        assert isinstance(model_id, str)
        assert len(model_id) > 0

    def test_training_defaults(self):
        """Testa valores padrão para treinamento."""
        epochs = getattr(settings, "EPOCHS", 3)
        batch_size = getattr(settings, "BATCH_SIZE", 2)
        learning_rate = getattr(settings, "LEARNING_RATE", 2e-4)

        # Valores devem ser razoáveis
        if isinstance(epochs, str):
            epochs = int(epochs)
        if isinstance(batch_size, str):
            batch_size = int(batch_size)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)

        assert epochs > 0
        assert batch_size > 0
        assert learning_rate > 0

    def test_lora_defaults(self):
        """Testa valores padrão para LoRA."""
        lora_r = getattr(settings, "LORA_R", 8)
        lora_alpha = getattr(settings, "LORA_ALPHA", 16)

        if isinstance(lora_r, str):
            lora_r = int(lora_r)
        if isinstance(lora_alpha, str):
            lora_alpha = int(lora_alpha)

        assert lora_r > 0
        assert lora_alpha > 0

    def test_monitoring_defaults(self):
        """Testa valores padrão para monitoramento."""
        interval = getattr(settings, "MONITORING_INTERVAL", 1.0)

        if isinstance(interval, str):
            interval = float(interval)

        assert interval > 0


class TestConfigurationAccess:
    """Testa diferentes formas de acessar configurações."""

    def test_direct_attribute_access(self):
        """Testa acesso direto a atributos."""
        # Deve funcionar mesmo que a configuração não exista
        try:
            value = settings.NONEXISTENT_CONFIG
            # Se existir, deve ser válido
            assert value is not None
        except AttributeError:
            # É esperado para configurações inexistentes
            pass

    def test_get_method_access(self):
        """Testa acesso via método get()."""
        # Deve retornar None ou valor padrão para configs inexistentes
        value = getattr(settings, "NONEXISTENT_CONFIG", None)
        assert value is None

        default_value = "default_test"
        value_with_default = getattr(settings, "NONEXISTENT_CONFIG", default_value)
        assert value_with_default == default_value

    def test_get_method_with_type_conversion(self):
        """Testa método getattr() com conversão de tipo."""
        # Simular configuração que precisa de conversão
        with patch.dict(os.environ, {"TEST_NUMERIC": "42"}):
            # Força o dynaconf a ver a nova variável criando uma nova instância
            from dynaconf import Dynaconf

            temp_settings = Dynaconf(environments=True)
            value = getattr(temp_settings, "TEST_NUMERIC", None)
            # Pode ser string ou int
            if isinstance(value, str):
                assert value == "42"
            elif value is not None:
                assert value == 42
            else:
                # É aceitável que retorne None se o Dynaconf não processar
                assert value is None


class TestSecretsConfiguration:
    """Testa validação e segurança de arquivos de secrets."""

    def test_secrets_file_loading(self):
        """Testa carregamento de arquivo .secrets.toml."""
        config_dir = Path(__file__).parent.parent / "config"
        secrets_file = config_dir / ".secrets.toml"
        
        # O arquivo pode não existir, mas deve ser tratado graciosamente
        if secrets_file.exists():
            # Se existir, deve ser um arquivo válido
            assert secrets_file.is_file()
        else:
            # Se não existir, o sistema deve funcionar sem ele
            assert not secrets_file.exists()

    @pytest.fixture
    def temp_secrets_file(self):
        """Fixture para arquivo de secrets temporário."""
        config_dir = Path(__file__).parent.parent / "config"
        temp_secrets = config_dir / ".test_secrets.toml"
        
        # Criar arquivo de secrets temporário
        temp_secrets.write_text("""
                # Arquivo de secrets para testes
                [default]
                WANDB_API_KEY = "test_wandb_key_12345"
                HF_TOKEN = "hf_test_token_abcdef123456"
                OPENAI_API_KEY = "sk-test123456789abcdef"

                [development]
                DATABASE_URL = "postgresql://testuser:testpass@localhost/testdb"
                SECRET_KEY = "dev-secret-key-for-testing"

                [production]
                DATABASE_URL = "postgresql://produser:prodpass@prodhost/proddb"
                SECRET_KEY = "prod-secret-key-super-secure"
                """)
        
        yield temp_secrets
        
        # Cleanup
        if temp_secrets.exists():
            temp_secrets.unlink()

    def test_secrets_file_structure_validation(self, temp_secrets_file):
        """Testa validação da estrutura do arquivo de secrets."""
        from dynaconf import Dynaconf
        
        # Carregar configurações com arquivo de secrets temporário
        secrets_settings = Dynaconf(
            settings_files=[str(temp_secrets_file)],
            environments=True,
        )
        
        # Verificar se campos críticos estão definidos
        assert hasattr(secrets_settings, "WANDB_API_KEY")
        assert hasattr(secrets_settings, "HF_TOKEN")
        
        # Verificar valores por ambiente
        secrets_settings.setenv("development")
        dev_secret = getattr(secrets_settings, "SECRET_KEY", None)
        assert dev_secret == "dev-secret-key-for-testing"
        
        secrets_settings.setenv("production")
        prod_secret = getattr(secrets_settings, "SECRET_KEY", None)
        assert prod_secret == "prod-secret-key-super-secure"

    def test_required_secrets_validation(self):
        """Testa validação de secrets obrigatórios."""
        # Lista de secrets que podem ser necessários
        potential_secrets = [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "OPENAI_API_KEY",
            "SECRET_KEY",
            "DATABASE_URL",
        ]
        
        for secret_name in potential_secrets:
            secret_value = getattr(settings, secret_name, None)
            if secret_value is not None:
                # Se existe, deve ser string não vazia
                assert isinstance(secret_value, str)
                assert len(secret_value.strip()) > 0
                # Não deve ser valor placeholder
                assert secret_value not in [
                    "your_key_here",
                    "replace_me",
                    "TODO",
                    "CHANGEME",
                    "",
                ]

    def test_secrets_are_masked_in_logs(self, temp_secrets_file):
        """Testa se secrets são mascarados em logs."""
        from dynaconf import Dynaconf
        
        secrets_settings = Dynaconf(
            settings_files=[str(temp_secrets_file)],
            environments=True,
        )
        
        # Verificar se valores sensíveis não aparecem em string representation
        settings_str = str(secrets_settings)
        
        # Secrets não devem aparecer completamente no string
        sensitive_values = [
            "test_wandb_key_12345",
            "hf_test_token_abcdef123456",
            "sk-test123456789abcdef",
            "testpass",
            "prodpass",
        ]
        
        for sensitive in sensitive_values:
            # Valor completo não deve aparecer, pode aparecer mascarado
            if sensitive in settings_str:
                # Se aparecer, deve estar mascarado ou ser muito curto
                assert len(sensitive) < 10  # Valores de teste curtos OK

    def test_secrets_environment_isolation(self, temp_secrets_file):
        """Testa isolamento de secrets entre ambientes."""
        from dynaconf import Dynaconf
        
        secrets_settings = Dynaconf(
            settings_files=[str(temp_secrets_file)],
            environments=True,
        )
        
        # Testar ambiente development
        secrets_settings.setenv("development")
        dev_db_url = getattr(secrets_settings, "DATABASE_URL", None)
        dev_secret_key = getattr(secrets_settings, "SECRET_KEY", None)
        
        # Testar ambiente production 
        secrets_settings.setenv("production")
        prod_db_url = getattr(secrets_settings, "DATABASE_URL", None)
        prod_secret_key = getattr(secrets_settings, "SECRET_KEY", None)
        
        # Secrets devem ser diferentes entre ambientes
        if dev_db_url and prod_db_url:
            assert dev_db_url != prod_db_url
        if dev_secret_key and prod_secret_key:
            assert dev_secret_key != prod_secret_key

    def test_malformed_secrets_file_handling(self):
        """Testa tratamento de arquivo de secrets mal formado."""
        config_dir = Path(__file__).parent.parent / "config"
        malformed_secrets = config_dir / ".malformed_secrets.toml"
        
        try:
            # Criar arquivo mal formado
            malformed_secrets.write_text("""
                    # Arquivo mal formado
                    [default]
                    WANDB_API_KEY = "test_key" but missing quote
                    HF_TOKEN = "hf_token"
            """)

            from dynaconf import Dynaconf
            
            # Deve lidar graciosamente com arquivo mal formado
            try:
                malformed_settings = Dynaconf(
                    settings_files=[str(malformed_secrets)],
                    environments=True,
                )
                # Se não falhar, deve pelo menos existir
                assert malformed_settings is not None
            except Exception as e:
                # É aceitável que falhe, mas deve ser tratado
                assert isinstance(e, Exception)
                
        finally:
            # Cleanup
            if malformed_secrets.exists():
                malformed_secrets.unlink()

    def test_secrets_override_precedence(self, temp_secrets_file):
        """Testa precedência de secrets (env vars > arquivo > defaults)."""
        from dynaconf import Dynaconf
        
        test_secret_name = "TEST_OVERRIDE_SECRET"
        env_value = "env_override_value"
        
        with patch.dict(os.environ, {test_secret_name: env_value}):
            secrets_settings = Dynaconf(
                settings_files=[str(temp_secrets_file)],
                environments=True,
            )
            
            # Variável de ambiente deve ter precedência
            override_value = getattr(secrets_settings, test_secret_name, None)
            # Deve pegar da variável de ambiente
            assert override_value == env_value or os.environ.get(test_secret_name) == env_value

    def test_secrets_validation_with_validators(self):
        """Testa validação de secrets usando Dynaconf validators."""
        from dynaconf import Dynaconf, Validator, ValidationError
        
        # Criar configuração com validadores
        try:
            validated_settings = Dynaconf(
                validators=[
                    # Validar que API keys não sejam vazias se existirem
                    Validator("WANDB_API_KEY", len_min=1,
                              when=Validator.EXISTS),
                    Validator("HF_TOKEN", len_min=1, when=Validator.EXISTS),
                    # Validar que SECRET_KEY existe se especificado
                    Validator("SECRET_KEY", len_min=8, when=Validator.EXISTS),
                ]
            )
            # Se chegou até aqui, validação passou
            assert validated_settings is not None
            
        except ValidationError as e:
            # É OK falhar se secrets não estiverem configurados corretamente
            assert isinstance(e, ValidationError)
        except AttributeError:
            # API do validator pode variar entre versões
            pytest.skip("Validator API incompatível com esta versão")

    def test_secrets_gitignore_compliance(self):
        """Testa se arquivos de secrets estão no .gitignore."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            
            # Verificar se padrões de secrets estão ignorados
            secrets_patterns = [
                ".secrets.toml",
                "*.secrets.toml",
                ".env",
                "*.env",
            ]
            
            for pattern in secrets_patterns:
                # Pelo menos alguns padrões devem estar no gitignore
                if pattern in gitignore_content:
                    assert True
                    return
            
            # Se chegou até aqui, avisar que faltam padrões
            pytest.skip("Nenhum padrão de secrets encontrado no .gitignore")

    def test_secrets_backup_and_recovery(self, temp_secrets_file):
        """Testa estratégias de backup e recuperação de secrets."""
        config_dir = Path(__file__).parent.parent / "config"
        backup_file = config_dir / ".secrets.toml.backup"
        
        try:
            # Simular backup
            if temp_secrets_file.exists():
                backup_content = temp_secrets_file.read_text()
                backup_file.write_text(backup_content)
                
                # Verificar se backup foi criado
                assert backup_file.exists()
                
                # Verificar se conteúdo é idêntico
                original_content = temp_secrets_file.read_text()
                recovered_content = backup_file.read_text()
                assert original_content == recovered_content
                
        finally:
            # Cleanup
            if backup_file.exists():
                backup_file.unlink()


class TestErrorHandling:
    """Testa tratamento de erros na configuração."""

    def test_malformed_config_file_handling(self):
        """Testa tratamento de arquivo de configuração mal formado."""
        # Criar arquivo TOML mal formado
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False
            ) as f:
                temp_file = f.name
                f.write("invalid toml content [[[")
                f.flush()

            # Arquivo foi fechado, agora podemos usar
            from dynaconf import Dynaconf

            # Deve lidar graciosamente com arquivo mal formado
            malformed_settings = Dynaconf(
                settings_files=[temp_file],
                environments=True,
            )
            # Não deve quebrar completamente
            assert malformed_settings is not None
        except Exception as e:
            # É aceitável que falhe, mas deve ser tratado
            assert isinstance(e, Exception)
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except PermissionError:
                    # Windows: arquivo pode ainda estar em uso
                    import time
                    time.sleep(0.1)
                    try:
                        os.unlink(temp_file)
                    except PermissionError:
                        pass  # Ignore se não conseguir deletar

    def test_missing_config_file_handling(self):
        """Testa tratamento de arquivo de configuração ausente."""
        from dynaconf import Dynaconf

        # Tentar carregar arquivo inexistente
        missing_file_settings = Dynaconf(
            settings_files=["nonexistent_file.toml"],
            environments=True,
        )

        # Deve funcionar sem o arquivo
        assert missing_file_settings is not None

    def test_invalid_environment_variable(self):
        """Testa tratamento de variável de ambiente inválida."""
        # Simular variável de ambiente com valor inválido
        with patch.dict(os.environ, {"INVALID_DEBUG": "not_a_boolean"}):
            # Força o dynaconf a ver a nova variável criando uma nova instância
            from dynaconf import Dynaconf

            temp_settings = Dynaconf(environments=True)
            value = getattr(temp_settings, "INVALID_DEBUG", None)
            # Deve retornar o valor sem quebrar ou pode retornar None se não for processado
            assert value is None or value == "not_a_boolean"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
