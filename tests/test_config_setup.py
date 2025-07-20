#!/usr/bin/env python3
"""
Script de teste para verificar se a configuração está funcionando
"""
from config.config import settings

def test_config():
    print("=== Teste de Configuração ===")
    try:
        print(f"✅ WANDB_ENTITY: {settings.WANDB_ENTITY}")
        print(f"✅ WANDB_PROJECT: {settings.WANDB_PROJECT}")
        print(f"✅ MODEL_ID: {settings.MODEL_ID}")
        
        # Teste dos tokens
        try:
            wandb_key = str(settings.WANDB_KEY)
            hf_token = str(settings.HF_TOKEN)
            
            if 'sua_wandb_key_aqui' in wandb_key or 'seu_hf_token_aqui' in hf_token:
                print('⚠️  ATENÇÃO: Tokens ainda não foram configurados em .secrets.toml')
                print('   Edite config/.secrets.toml e adicione seus tokens reais')
            else:
                print('✅ Tokens carregados (primeiros 10 caracteres):')
                print(f'   WANDB_KEY: {wandb_key[:10]}...')
                print(f'   HF_TOKEN: {hf_token[:10]}...')
                
        except Exception as e:
            print(f'❌ Erro ao carregar tokens: {e}')
            
    except Exception as e:
        print(f"❌ Erro ao carregar configurações: {e}")
        
    print("=== Fim do Teste ===")

if __name__ == "__main__":
    test_config()
