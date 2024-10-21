
# Reconhecimento de Imagens em Tempo Real com TensorFlow Lite e OpenCV

Este projeto utiliza um modelo **TensorFlow Lite** para realizar inferência em tempo real usando a câmera de um dispositivo. Ele identifica a classe com maior probabilidade com base no modelo treinado e exibe essa classe junto com as probabilidades de todas as outras classes.

## Requisitos
- Python 3.x
- TensorFlow Lite Runtime (ou TensorFlow completo)
- OpenCV para capturar imagens da câmera

## Instalação de dependências

### macOS
1. Instale Python 3.x:
   - Verifique a versão:
     ```bash
     python3 --version
     ```
   - Se necessário, instale usando o Homebrew:
     ```bash
     brew install python3
     ```

2. Instale as bibliotecas:
   ```bash
   pip3 install opencv-python tensorflow
   ```

### Windows
1. Instale Python 3.x:
   - Baixe e instale a partir de [python.org](https://www.python.org/downloads/)
   - Certifique-se de marcar a opção "Add Python to PATH" durante a instalação.

2. Instale as bibliotecas:
   ```bash
   pip install opencv-python tensorflow
   ```

### Raspberry Pi
1. Instale Python 3.x (geralmente já vem instalado no Raspberry Pi OS):
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. Instale as bibliotecas:
   - TensorFlow Lite Runtime (preferível para dispositivos com baixo desempenho):
     ```bash
     pip install tflite-runtime opencv-python
     ```

   - Se preferir usar TensorFlow completo (pode ser mais pesado no Raspberry Pi):
     ```bash
     pip install tensorflow opencv-python
     ```

## Uso

1. Conecte a câmera ao seu dispositivo.
2. Coloque o modelo `.tflite` exportado na pasta do projeto e certifique-se de que o caminho para o arquivo está correto.
3. Execute o script Python para capturar imagens em tempo real e classificar objetos.

## Código

```python
import numpy as np
import tensorflow as tf
import cv2

# Carregar o modelo TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Obter detalhes da entrada e saída do modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inicializar a câmera (substitua '0' pelo índice correto da câmera)
cap = cv2.VideoCapture(1)

# Obter o número de classes com base no tamanho da saída do modelo
interpreter.invoke()  # Execute uma inferência para obter a forma de saída
output_data = interpreter.get_tensor(output_details[0]['index'])
num_classes = output_data.shape[1]  # Número de classes previsto pelo modelo

# Gerar nomes de classes genéricos se não houver nomes fornecidos
class_names = [f"Classe {i}" for i in range(num_classes)]

while True:
    # Ler frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem")
        break

    # Redimensionar o frame para o tamanho que o modelo espera (por exemplo, 224x224 pixels)
    input_shape = input_details[0]['shape']
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    
    # Ajustar a imagem para ser usada no modelo (normalização)
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)
    input_data = input_data / 255.0  # Normalizar entre 0 e 1 se o modelo precisar

    # Executar a inferência
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Obter os resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Aplicar softmax para converter logits em probabilidades
    probabilities = tf.nn.softmax(output_data[0]).numpy()

    # Identificar a classe com maior probabilidade
    result = np.argmax(probabilities)
    
    # Exibir a classe com maior probabilidade
    main_class_text = f"Classe Identificada: {class_names[result]} - Probabilidade: {probabilities[result]:.2f}"
    cv2.putText(frame, main_class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Exibir as probabilidades das demais classes
    y_offset = 60  # Posição inicial para exibir as outras classes
    for i, prob in enumerate(probabilities):
        if i == result:
            continue  # Pular a classe com maior probabilidade (já exibida)
        text = f"{class_names[i]}: {prob:.2f}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += 30  # Mover para a próxima linha para a próxima classe

    # Mostrar a imagem com a classe identificada e as probabilidades
    cv2.imshow('Reconhecimento de Imagem', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()
```

## Observações
- Ajuste o índice da câmera (`cv2.VideoCapture(1)`) se necessário. O índice varia conforme o dispositivo.
- Caso tenha os nomes das classes de seu modelo, você pode substituir a lista `class_names` com os nomes reais.

## Licença
Este projeto está licenciado sob os termos da licença MIT.
