import ollama
import cv2
import time


# --- Configuração ---
MODELO_VLM = "gemma3:4b" 
PROMPT = "Descreva em poucas palavras o que você vê nesta imagem."
INTERVALO_ANALISE_SEGUNDOS = 10 

# --- Setup do Software ---
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

print(f"Câmera Inteligente VLM com '{MODELO_VLM}' - MODO VERBOSE")
print(f"O sistema irá capturar e analisar um frame a cada {INTERVALO_ANALISE_SEGUNDOS} segundos.")
print("Pressione Ctrl+C para sair.")

try:
    while True:
        inicio_ciclo = time.monotonic()

        ret, frame = camera.read()
        if not ret:
            print("Erro ao capturar frame.")
            continue
        
        # As linhas que salvavam a imagem de debug foram removidas daqui.

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("Erro ao codificar o frame.")
            continue
        
        image_bytes = buffer.tobytes()

        print("Analisando com o modelo (modo streaming):")

        try:
            stream = ollama.chat(
                model=MODELO_VLM,
                messages=[{
                    'role': 'user',
                    'content': PROMPT,
                    'images': [image_bytes]
                }],
                stream=True
            )

            resposta_completa = ""
            stats_finais = None
            
            for chunk in stream:
                if 'content' in chunk['message']:
                    texto_parcial = chunk['message']['content']
                    print(texto_parcial, end='', flush=True)
                    resposta_completa += texto_parcial
                
                if chunk['done']:
                    stats_finais = chunk

            print("\n" + "-"*20) 

            if stats_finais:
                total_s = stats_finais['total_duration'] / 1e9
                eval_s = stats_finais['eval_duration'] / 1e9
                tokens_por_segundo = stats_finais['eval_count'] / eval_s if eval_s > 0 else 0
                
                print("--- Estatísticas (Verbose) ---")
                print(f"Tempo Total de Resposta: {total_s:.2f} s")
                print(f"Número de Tokens: {stats_finais['eval_count']}")
                print(f"Velocidade (tokens/s): {tokens_por_segundo:.2f}")
                print("------------------------------")
            
            resposta_limpa = resposta_completa.lower().strip()
            if "pessoa" in resposta_limpa or "homem" in resposta_limpa or "mulher" in resposta_limpa or "person" in resposta_limpa:
                print(">> ALERTA: Pessoa detectada! <<")
            else:
                print(">> Status: Nenhuma pessoa detectada.")

        except Exception as e:
            print(f"\nErro ao comunicar com o Ollama: {e}")
        
        print("-" * 20)
        
        tempo_ciclo = time.monotonic() - inicio_ciclo
        tempo_espera = max(0, INTERVALO_ANALISE_SEGUNDOS - tempo_ciclo)
        time.sleep(tempo_espera)

finally:
    print("\nFechando o programa...")
    camera.release()
