import ollama
import os

# --- Configuração ---
# Nome do modelo VLM que você baixou no Ollama.
MODELO_VLM = "gemma3:4b" 

# Caminho para a imagem que você quer analisar.
# IMPORTANTE: Coloque uma imagem na mesma pasta que este script
# e atualize o nome do arquivo aqui.
NOME_ARQUIVO_IMAGEM = "cruzeiro_do_sul2.jpg" 

# A pergunta que você quer fazer sobre a imagem.
PROMPT = "descreva essa imagem do soldado brasileiro da segunda guerrar mundial"

# --- Verificação ---
# Garante que o arquivo de imagem realmente existe antes de continuar.
if not os.path.exists(NOME_ARQUIVO_IMAGEM):
    print(f"Erro: Arquivo de imagem não encontrado em '{NOME_ARQUIVO_IMAGEM}'")
    print("Por favor, coloque uma imagem na pasta e atualize o nome no script.")
    exit()

# --- Execução ---
print(f" Analisando a imagem '{NOME_ARQUIVO_IMAGEM}' com o modelo '{MODELO_VLM}'...")
print(f" Pergunta: {PROMPT}\n")

try:
    # Chama a API do Ollama com o modelo, o prompt e a imagem
    response = ollama.chat(
        model=MODELO_VLM,
        messages=[
            {
                'role': 'user',
                'content': PROMPT,
                'images': [NOME_ARQUIVO_IMAGEM] 
            }
        ]
    )

    # Imprime a resposta do modelo
    print(" Resposta do modelo:")
    print(response['message']['content'])

except Exception as e:
    print(f"Ocorreu um erro ao se comunicar com o Ollama: {e}")
    print("Verifique se o Ollama está rodando e se o modelo foi baixado corretamente.")
