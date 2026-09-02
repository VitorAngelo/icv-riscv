import zipfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.colab import files

np.random.seed(42)
tf.random.set_seed(42)

(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_full = np.pad(x_train_full.astype('float32') / 255.0, ((0,0),(2,2),(2,2)), mode='constant')[..., None]
x_test_float = np.pad(x_test.astype('float32') / 255.0, ((0,0),(2,2),(2,2)), mode='constant')[..., None]

x_calib, x_val, x_train = x_train_full[:5000], x_train_full[5000:10000], x_train_full[10000:]
y_train = keras.utils.to_categorical(y_train_full[10000:], 10)
y_val   = keras.utils.to_categorical(y_train_full[5000:10000], 10)

model = keras.Sequential([
    layers.Conv2D(6, (5,5), activation='relu', input_shape=(32,32,1), name='conv1'),
    layers.AveragePooling2D((2,2), strides=2, name='pool1'),
    layers.Conv2D(16, (5,5), activation='relu', name='conv2'),
    layers.AveragePooling2D((2,2), strides=2, name='pool2'),
    layers.Conv2D(120, (5,5), activation='relu', name='conv3'),
    layers.Flatten(),
    layers.Dense(84, activation='relu', name='dense1'),
    layers.Dense(10, activation='softmax', name='output')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val), verbose=1)

def align_up(x, m):
    """Alinha x para o próximo múltiplo superior de m."""
    return ((x + m - 1) // m) * m

def exportar_header_c(arr, nome, tipo_c, elementos_por_linha=16, fmt="{}"):
    """Converte um ndarray NumPy em uma tabela estática em C em arquivo .h"""
    flat = arr.flatten()
    cab = f"#ifndef {nome.upper()}_H\n#define {nome.upper()}_H\n\n#include <stdint.h>\n\n"
    cab += f"#define {nome.upper()}_SIZE {len(flat)}\n"
    cab += f"static const {tipo_c} {nome}[{len(flat)}] = {{\n "
    linhas = [", ".join(fmt.format(v) for v in flat[i:i+elementos_por_linha])
              for i in range(0, len(flat), elementos_por_linha)]
    cab += ",\n ".join(linhas) + "\n};\n\n#endif\n"
    return cab

def quantizar_por_canal(pesos_2d):
    """Quantiza pesos 2D simetricamente por canal de saída."""
    max_abs = np.maximum(np.max(np.abs(pesos_2d), axis=0), 1e-8)
    escala = max_abs / 127.0
    w_int = np.clip(np.round(pesos_2d / escala[None, :]), -127, 127).astype(np.int8)
    return w_int, escala

def preparar_conv_ime(w_khkwcinf, bias, escala_entrada, Cout_pad_mult=4, Kin_pad_mult=8, nome="conv"):
    """Aplica layout Row-Major (Cin, K, K) e Padding exigidos pelo acelerador IME."""
    K, _, Cin, Cout = w_khkwcinf.shape
    w_flat = np.transpose(w_khkwcinf, (2, 0, 1, 3)).reshape(-1, Cout)

    K_real = Cin * K * K
    K_pad = align_up(K_real, Kin_pad_mult)
    Cout_pad = align_up(Cout, Cout_pad_mult)

    w_int8, escala_peso = quantizar_por_canal(w_flat)

    pesos_pad = np.zeros((K_pad, Cout_pad), dtype=np.int8)
    pesos_pad[:K_real, :Cout] = w_int8

    bias_i32 = np.round(bias / (escala_entrada * escala_peso)).astype(np.int32)
    bias_pad = np.zeros((Cout_pad,), dtype=np.int32)
    bias_pad[:Cout] = bias_i32

    return {
        f"int8_{nome}_w_ime.h": exportar_header_c(pesos_pad, f"int8_{nome}_w_ime", "int8_t", 16),
        f"int8_{nome}_b_ime.h": exportar_header_c(bias_pad, f"int8_{nome}_b_ime", "int32_t", 12),
        f"int8_{nome}_escala.h": exportar_header_c(escala_peso, f"int8_{nome}_escala", "float", 8, "{:.10f}f"),
    }

def preparar_conv_int8_std(w_khkwcinf, bias, escala_entrada, nome="conv"):
    """Quantiza Convolução para C PURO (Standard): preserva layout (Kh, Kw, Cin, Cout) e SEM padding."""
    Cout = w_khkwcinf.shape[-1]
    w_flat = w_khkwcinf.reshape(-1, Cout)

    w_int8_flat, escala_peso = quantizar_por_canal(w_flat)

    bias_i32 = np.round(bias / (escala_entrada * escala_peso)).astype(np.int32)

    return {
        f"int8_std_{nome}_w.h": exportar_header_c(w_int8_flat, f"int8_std_{nome}_w", "int8_t", 16),
        f"int8_std_{nome}_b.h": exportar_header_c(bias_i32, f"int8_std_{nome}_b", "int32_t", 12),
        f"int8_std_{nome}_escala.h": exportar_header_c(escala_peso, f"int8_std_{nome}_escala", "float", 8, "{:.10f}f"),
    }

def preparar_dense_int8(w, bias, escala_entrada, nome):
    """Quantiza camadas densas para C puro (sem padding)."""
    w_int8, escala_peso = quantizar_por_canal(w)
    bias_i32 = np.round(bias / (escala_entrada * escala_peso)).astype(np.int32)

    return {
        f"int8_{nome}_w.h": exportar_header_c(w_int8, f"int8_{nome}_w", "int8_t", 16),
        f"int8_{nome}_b.h": exportar_header_c(bias_i32, f"int8_{nome}_b", "int32_t", 12),
        f"int8_{nome}_escala.h": exportar_header_c(escala_peso, f"int8_{nome}_escala", "float", 8, "{:.10f}f"),
    }

def preparar_layer_fp32(w, bias, nome):
    return {
        f"fp32_{nome}_w.h": exportar_header_c(w, f"fp32_{nome}_w", "float", 8, "{:.10f}f"),
        f"fp32_{nome}_b.h": exportar_header_c(bias, f"fp32_{nome}_b", "float", 8, "{:.10f}f")
    }

def preparar_layer_q32(w, bias, scale_bits=16, nome=""):
    scale = 1 << scale_bits
    w_q32 = np.round(w * scale).astype(np.int32)
    b_q32 = np.round(bias * scale).astype(np.int32)
    return {
        f"q32_{nome}_w.h": exportar_header_c(w_q32, f"q32_{nome}_w", "int32_t", 12),
        f"q32_{nome}_b.h": exportar_header_c(b_q32, f"q32_{nome}_b", "int32_t", 12)
    }

nomes_camadas = ['conv1', 'conv2', 'conv3', 'dense1']
saidas = [model.get_layer(n).output for n in nomes_camadas]
modelo_calib = keras.Model(inputs=model.inputs, outputs=saidas)

ativacoes = modelo_calib.predict(x_calib, verbose=0)
escalas_ativ = {}
arquivos_gerados = {}

for nome, ativ in zip(nomes_camadas, ativacoes):
    esc = np.max(np.abs(ativ)) / 255.0
    escalas_ativ[nome] = esc
    header_str = f"#ifndef INT8_{nome.upper()}_ESCALA_ATIV_H\n#define INT8_{nome.upper()}_ESCALA_ATIV_H\n\n#define INT8_{nome.upper()}_ESCALA_ATIV {esc}f\n\n#endif\n"
    arquivos_gerados[f"int8_{nome}_escala_ativ.h"] = header_str

camadas = [
    ('conv1', True),
    ('conv2', True),
    ('conv3', True),
    ('dense1', False),
    ('output', False)
]

escala_in = 1.0 / 255.0
for nome, is_conv in camadas:
    w, b = model.get_layer(nome).get_weights()

    if is_conv:
        arquivos_gerados.update(preparar_conv_ime(w, b, escala_in, nome=nome))
        arquivos_gerados.update(preparar_conv_int8_std(w, b, escala_in, nome=nome))
        w_flat_standard = w.reshape(-1, w.shape[-1])
    else:
        w_flat_standard = w
        arquivos_gerados.update(preparar_dense_int8(w_flat_standard, b, escala_in, nome=nome))

    arquivos_gerados.update(preparar_layer_fp32(w_flat_standard, b, nome=nome))
    arquivos_gerados.update(preparar_layer_q32(w_flat_standard, b, scale_bits=16, nome=nome))

    if nome in escalas_ativ:
        escala_in = escalas_ativ[nome]

imgs_uint8 = np.pad(x_test, ((0,0),(2,2),(2,2)), mode='constant')[..., None]
imgs_q32 = np.round(x_test_float * 65536).astype(np.int32)
rotulos = y_test.astype(np.uint8)

arquivos_gerados["x_test_uint8.h"] = exportar_header_c(imgs_uint8, "x_test_uint8", "uint8_t", 16)
arquivos_gerados["x_test_fp32.h"]  = exportar_header_c(x_test_float, "x_test_fp32", "float", 8, "{:.10f}f")
arquivos_gerados["x_test_q32.h"]   = exportar_header_c(imgs_q32, "x_test_q32", "int32_t", 12)
arquivos_gerados["y_test_uint8.h"] = exportar_header_c(rotulos, "y_test_uint8", "uint8_t", 20)

for nome_file, conteudo in arquivos_gerados.items():
    with open(nome_file, "w") as f:
        f.write(conteudo)

with zipfile.ZipFile('exportacao_completa.zip', 'w') as zipf:
    for arq in arquivos_gerados.keys():
        zipf.write(arq)

files.download('exportacao_completa.zip')
print("Exportação concluída (INT8 puro, INT8 preparado pra IME, INT32 e FP32).")
