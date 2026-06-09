<div align="center">

# Road Surface Segmentation

### Segmentação inteligente de danos em vias com YOLO26

Pipeline completo para **detectar e segmentar buracos (potholes)** em imagens e vídeos POV/dashcam, do pré-processamento de vídeo à inferência com máscaras de instância.

<br>

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO26-Segmentation-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-green?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

<br>

[**Visão Geral**](#visão-geral) ·
[**Modelo**](#modelo) ·
[**Dataset**](#dataset) ·
[**Pré-processamento**](#pré-processamento-de-vídeos) ·
[**Métricas**](#métricas-de-treinamento) ·
[**Avaliação**](#relatório-de-avaliação-em-amostras) ·
[**Quick Start**](#quick-start) ·
[**Estrutura**](#estrutura-do-projeto)

</div>

---

## Visão Geral

Este repositório reúne um fluxo de ponta a ponta para inspeção automatizada de pavimento:

| Etapa | Ferramenta | Saída |
|-------|------------|-------|
| **1. Coleta** | Vídeos POV/dashcam | `videos/` |
| **2. Pré-processamento** | `prepare_road_videos.py` | Frames anotáveis + vídeos desacelerados |
| **3. Treinamento** | Notebook Ultralytics | `models/best.pt` |
| **4. Deploy** | ONNX FP16/FP32 | Inferência em produção |

```mermaid
flowchart LR
    A[📹 Vídeos Dashcam] --> B[prepare_road_videos.py]
    B --> C[🖼️ Frames extraídos]
    B --> D[⏱️ Vídeos slow 1.5x]
    C --> E[Anotação manual]
    E --> F[Dataset YOLO]
    F --> G[🧠 YOLO26s-seg]
    G --> H[best.pt / ONNX]
    H --> I[🔍 Inferência em vídeo]
```

O objetivo é produzir **máscaras de segmentação** de buracos em estrada, úteis para overlays visuais, estimativa de severidade e workflows de inspeção em vídeo.

---

## Modelo

### YOLO26s: Instance Segmentation

Modelo fine-tuned a partir de `yolo26s-seg.pt` (Ultralytics), treinado em 3 estágios:

1. **Baseline:** `imgsz=640`, 100 épocas, validação do pipeline
2. **Fine-tune conservador:** learning rate baixo, augmentação reduzida
3. **Alta resolução:** `imgsz=768`, `lr0=0.001`, melhor qualidade de máscara

### Resultados finais (época 28, melhor checkpoint)

Métricas de validação no conjunto **292 imagens / 560 instâncias**:

| Métrica | Box (B) | Mask (M) |
|---------|---------|----------|
| **Precision** | 76.5% | **78.1%** |
| **Recall** | 67.1% | **68.1%** |
| **mAP@50** | 75.9% | **76.4%** |
| **mAP@50-95** | 47.7% | **48.1%** |

> Early stopping na época 53 (patience=25). Melhor resultado observado na **época 28**.

### Artefatos disponíveis

| Arquivo | Formato | Uso |
|---------|---------|-----|
| `models/best.pt` | PyTorch | Treino, fine-tune, inferência Ultralytics |
| `models/yolo26s_pothole_segmentation_fp32.onnx` | ONNX FP32 | Máxima precisão |
| `models/yolo26s_pothole_segmentation_fp16_compact.onnx` | ONNX FP16 | Deploy otimizado |
| `models/args.yaml` | Config | Hiperparâmetros do último treino |

### Inferência

| Ambiente | Latência |
|----------|----------|
| Tesla T4 (Colab) | ~7.9 ms/imagem |
| Pré-processamento | ~0.3 ms |
| Pós-processamento | ~3.5 ms |

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict("frame.jpg", imgsz=768, conf=0.25)
results[0].show()
```

---

## Dataset

### Dataset de treinamento (Roboflow)

Fonte: [Pothole Segmentation (Roboflow Universe)](https://universe.roboflow.com/emotion-recognition-mcwg0/pothole-segmentation-g6hbh-193vt) · Licença **CC BY 4.0**

O dataset original continha classes duplicadas (`Pothole`, `Potholes`, `pothole`, `Manhole`, `Unmarked Bump`). Para o baseline, as variações foram **unificadas em uma única classe `pothole`**, mantendo o treino limpo e consistente.

| Split | Imagens | Instâncias |
|-------|---------|------------|
| **Train** | 893 | 2.106 |
| **Valid** | 292 | 560 |
| **Test** | 348 | - |

<p align="center">
  <img src="metrics/labels.jpg" alt="Distribuição de labels no dataset" width="720">
  <br>
  <em>Distribuição espacial e dimensional dos labels de segmentação</em>
</p>

### Dados próprios: vídeos dashcam

Vídeos POV gravados em movimento, processados localmente pelo script de pré-processamento. **Não são versionados** (ver `.gitignore`).

| Pasta | Conteúdo | Versionado |
|-------|----------|------------|
| `videos/` | Vídeos brutos `.mp4` | Não |
| `processed_videos/` | Versões desaceleradas 1.5× | Não |
| `frames/` | Frames extraídos para anotação | Não |
| `reports/` | Metadados JSON por vídeo | Não |

---

## Pré-processamento de vídeos

O script [`prepare_road_videos.py`](prepare_road_videos.py) transforma vídeos dashcam em material pronto para anotação YOLO.

### Funcionalidades

- Desaceleração via **ffmpeg** (`setpts`, H.264, CRF 18)
- Extração densa de frames com **amostragem inteligente**
- Pontuação de qualidade na **ROI da via** (parte inferior ~55%)
- TUI interativa (Textual) + CLI completa
- Relatórios JSON com métricas por frame

### Modos de amostragem (`--sampling-mode`)

| Modo | Descrição |
|------|-----------|
| `random` | Timestamps aleatórios distribuídos |
| `uniform` | Espaçamento uniforme no tempo |
| `smart` | Maior score global entre candidatos |
| `hybrid` | **Padrão:** melhor frame por segmento temporal |

### Extração em escala (milhares de frames)

```bash
source .venv/bin/activate
pip install opencv-python numpy rich textual

# ~3.500+ frames dos 5 vídeos atuais (intervalo de 0.05 s)
python prepare_road_videos.py \
  --frame-interval 0.05 \
  --slow-factor 1.5 \
  --sampling-mode hybrid \
  --overwrite
```

Documentação completa: [`README_preprocessing.md`](README_preprocessing.md)

---

## Métricas de treinamento

### Curvas de aprendizado

<p align="center">
  <img src="metrics/results.png" alt="Curvas de treinamento loss, precision, recall e mAP" width="900">
  <br>
  <em>Evolução de loss e métricas ao longo de 53 épocas, com convergência estável após ~20 épocas</em>
</p>

### Curvas de performance

<table>
  <tr>
    <td align="center" width="50%">
      <img src="metrics/MaskPR_curve.png" alt="Mask Precision-Recall curve" width="400"><br>
      <sub><b>Mask PR Curve</b></sub>
    </td>
    <td align="center" width="50%">
      <img src="metrics/MaskF1_curve.png" alt="Mask F1 curve" width="400"><br>
      <sub><b>Mask F1 Curve</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="metrics/BoxPR_curve.png" alt="Box Precision-Recall curve" width="400"><br>
      <sub><b>Box PR Curve</b></sub>
    </td>
    <td align="center">
      <img src="metrics/confusion_matrix_normalized.png" alt="Confusion matrix normalizada" width="400"><br>
      <sub><b>Confusion Matrix (normalizada)</b></sub>
    </td>
  </tr>
</table>

### Predições em validação

Comparação entre **labels** (ground truth) e **predições** do modelo:

<table>
  <tr>
    <td align="center" width="50%">
      <img src="metrics/val_batch0_labels.jpg" alt="Validation batch 0 labels" width="420"><br>
      <sub>Batch 0 Labels</sub>
    </td>
    <td align="center" width="50%">
      <img src="metrics/val_batch0_pred.jpg" alt="Validation batch 0 predictions" width="420"><br>
      <sub>Batch 0 Predições</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="metrics/val_batch1_labels.jpg" alt="Validation batch 1 labels" width="420"><br>
      <sub>Batch 1 Labels</sub>
    </td>
    <td align="center">
      <img src="metrics/val_batch1_pred.jpg" alt="Validation batch 1 predictions" width="420"><br>
      <sub>Batch 1 Predições</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="metrics/val_batch2_labels.jpg" alt="Validation batch 2 labels" width="420"><br>
      <sub>Batch 2 Labels</sub>
    </td>
    <td align="center">
      <img src="metrics/val_batch2_pred.jpg" alt="Validation batch 2 predictions" width="420"><br>
      <sub>Batch 2 Predições</sub>
    </td>
  </tr>
</table>

### Batches de treinamento (augmentação)

<p align="center">
  <img src="metrics/train_batch0.jpg" alt="Training batch 0" width="280">
  <img src="metrics/train_batch1.jpg" alt="Training batch 1" width="280">
  <img src="metrics/train_batch2.jpg" alt="Training batch 2" width="280">
  <br>
  <em>Amostras de batches de treino com augmentação (mosaic, RandAugment, flip)</em>
</p>

---

## Quick Start

### 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python numpy rich textual
```

> **ffmpeg** também é necessário para o pré-processamento de vídeos.

### 2. Extrair frames dos vídeos

```bash
# Coloque os vídeos em videos/ e execute:
python prepare_road_videos.py --frame-interval 0.05 --slow-factor 1.5
```

### 3. Inferência com o modelo treinado

```bash
yolo segment predict model=models/best.pt source=frames/ imgsz=768 conf=0.25
```

### 4. Retreinar / fine-tune

Abra o notebook de baseline:

```bash
jupyter notebook notebooks/01_train_yolo_pothole_segmentation_baseline.ipynb
```

---

## Estrutura do projeto

```
road-surface-segmentation/
├── 📄 README.md                          # Este documento
├── 📄 README_preprocessing.md              # Guia do script de vídeos
├── 🐍 prepare_road_videos.py               # Pré-processamento + TUI
├── 🐍 evaluate_samples.py                  # Avaliação visual em amostras
├── 🐍 test_prepare_videos.py               # Testes do pipeline de vídeo
│
├── 📓 notebooks/
│   └── 01_train_yolo_pothole_segmentation_baseline.ipynb
│
├── 🧠 models/
│   ├── best.pt                             # Checkpoint PyTorch
│   ├── args.yaml                           # Hiperparâmetros
│   ├── yolo26s_pothole_segmentation_fp32.onnx
│   └── yolo26s_pothole_segmentation_fp16_compact.onnx
│
├── 📊 metrics/                             # Gráficos e predições de treino
│   ├── results.png
│   ├── results.csv
│   ├── confusion_matrix_normalized.png
│   ├── MaskPR_curve.png
│   ├── val_batch*_pred.jpg
│   └── evaluation/                         # Relatório de inferência em amostras
│       ├── evaluation_results.json
│       └── valid_*.jpg / test_*.jpg
│
├── 🎬 videos/          (gitignored)        # Vídeos dashcam brutos
├── 🖼️ frames/          (gitignored)        # Frames extraídos
├── ⏱️ processed_videos/ (gitignored)        # Vídeos desacelerados
└── 📋 reports/         (gitignored)        # Relatórios JSON
```

---

## Hiperparâmetros do treino final

Extraídos de `models/args.yaml`:

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | `yolo26s-seg` (fine-tuned) |
| Task | `segment` |
| Image size | **768** |
| Epochs | 120 (early stop @ 53) |
| Batch | 20 |
| Optimizer | AdamW |
| Learning rate | 0.001 → 0.02 (cosine) |
| Augmentação | mosaic 0.2, RandAugment, flip 0.5 |
| Device | NVIDIA Tesla T4 |

---

## Relatório de avaliação em amostras

> Avaliação qualitativa e quantitativa do modelo `best.pt` em **16 imagens aleatórias** do dataset local (`/home/a1rm4x/Downloads/dataset`): 8 de **valid** e 8 de **test**. Comparação visual entre anotações (GT) e predições do modelo.

### Configuração do experimento

| Parâmetro | Valor |
|-----------|-------|
| Modelo | `models/best.pt` (YOLO26s-seg, imgsz=768) |
| Dataset | Roboflow Pothole Segmentation |
| Confiança mínima | 0.25 |
| IoU para match | 0.50 |
| Amostragem | Aleatória (seed=42) |
| Script | `evaluate_samples.py` |

**Legenda nas imagens:** contorno verde = ground truth · overlay vermelho = predição do modelo

### Resumo executivo

<div align="center">

| Métrica | Resultado |
|---------|-----------|
| **Imagens avaliadas** | 16 (8 valid + 8 test) |
| **Taxa de acerto estrito** | **50%** (8/16 sem erro) |
| **Taxa de detecção** | **100%** (todas as imagens com buraco foram detectadas) |
| **Recall por instância** | **91,7%** (22 TP / 24 GT) |
| **Precision por instância** | **53,7%** (22 TP / 41 predições) |
| **Falsos negativos** | 2 instâncias |
| **Falsos positivos** | 19 instâncias |

</div>

```mermaid
pie title Veredito por imagem (n=16)
    "Correto" : 8
    "Parcial" : 8
    "Perdido" : 0
    "Falso positivo" : 0
```

| Veredito | Significado | Quantidade |
|----------|-------------|------------|
| ✅ **Correto** | Todas as instâncias GT detectadas, zero FP/FN | **8** |
| ⚠️ **Parcial** | Detectou buracos reais, mas com FP ou FN extras | **8** |
| ❌ **Perdido** | Buracos no GT não detectados | **0** |
| 🚫 **Falso positivo** | Detecções sem buraco anotado | **0** |

**Conclusão:** o modelo **identifica potholes de forma confiável**; nenhuma imagem ficou sem detecção. O principal ponto de melhoria é a **super-segmentação** (máscaras extras ou fragmentadas), especialmente em cenas com múltiplos buracos próximos.

---

### Resultados: Validation (8 imagens)

| # | Imagem | GT | Pred | TP | FP | FN | IoU médio | Veredito |
|---|--------|----|------|----|----|-----|-----------|----------|
| 1 | `664_png` | 2 | 3 | 2 | 1 | 0 | 0.806 | ⚠️ Parcial |
| 2 | `damaged-asphalt-pavement` | 1 | 2 | 1 | 1 | 0 | 0.581 | ⚠️ Parcial |
| 3 | `44_png` | 2 | 2 | 2 | 0 | 0 | 0.760 | ✅ Correto |
| 4 | `632_png` | 1 | 1 | 1 | 0 | 0 | 0.655 | ✅ Correto |
| 5 | `2goufguo_pothole` | 1 | 1 | 1 | 0 | 0 | 0.884 | ✅ Correto |
| 6 | `image24` | 3 | 7 | 2 | 5 | 1 | 0.635 | ⚠️ Parcial |
| 7 | `768_png` | 4 | 10 | 3 | 7 | 1 | 0.706 | ⚠️ Parcial |
| 8 | `684_png` | 2 | 2 | 2 | 0 | 0 | 0.727 | ✅ Correto |

<p align="center"><strong>Valid | predições corretas</strong></p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="metrics/evaluation/valid_44_png_jpg.rf.kv9q7J6j7A3FxddVY2NT.jpg" width="400"><br>
      <sub>✅ 44_png: 2/2 buracos, IoU 0.76</sub>
    </td>
    <td align="center" width="50%">
      <img src="metrics/evaluation/valid_2goufguo_pothole_625x300_25_September_18_jpg.rf.Q4aAQZDWNk9jCcYWxpDX.jpg" width="400"><br>
      <sub>✅ 2goufguo_pothole: 1/1 buraco, IoU 0.88</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="metrics/evaluation/valid_632_png_jpg.rf.LP79FwuYEa8agst7SCDS.jpg" width="400"><br>
      <sub>✅ 632_png: 1/1 buraco, IoU 0.66</sub>
    </td>
    <td align="center">
      <img src="metrics/evaluation/valid_684_png_jpg.rf.4Y0S2nPYtZCMKv8agExc.jpg" width="400"><br>
      <sub>✅ 684_png: 2/2 buracos, IoU 0.73</sub>
    </td>
  </tr>
</table>

<p align="center"><strong>Valid | predições parciais (detectou, mas com extras)</strong></p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="metrics/evaluation/valid_768_png_jpg.rf.7uqLqz5Aw0Ch7sNlBz2P.jpg" width="400"><br>
      <sub>⚠️ 768_png: 3/4 buracos, 7 FP (múltiplos buracos)</sub>
    </td>
    <td align="center" width="50%">
      <img src="metrics/evaluation/valid_image24_jpeg.rf.1KA070KBBB1zBGkzdTjl.jpg" width="400"><br>
      <sub>⚠️ image24: 2/3 buracos, 5 FP</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="metrics/evaluation/valid_664_png_jpg.rf.dgDQaPYvW4DKdFdXdnJk.jpg" width="400"><br>
      <sub>⚠️ 664_png: 2/2 buracos, 1 FP extra</sub>
    </td>
    <td align="center">
      <img src="metrics/evaluation/valid_damaged-asphalt-pavement-road-potholes-260nw-1134687086_jpg.rf.cowgUL2z2LOORyGYz3p2.jpg" width="400"><br>
      <sub>⚠️ damaged-asphalt: 1/1 buraco, 1 FP</sub>
    </td>
  </tr>
</table>

---

### Resultados: Test (8 imagens)

| # | Imagem | GT | Pred | TP | FP | FN | IoU médio | Veredito |
|---|--------|----|------|----|----|-----|-----------|----------|
| 1 | `891_png` | 1 | 2 | 1 | 1 | 0 | 0.711 | ⚠️ Parcial |
| 2 | `image27` | 1 | 2 | 1 | 1 | 0 | 0.881 | ⚠️ Parcial |
| 3 | `650_png` | 1 | 1 | 1 | 0 | 0 | 0.535 | ✅ Correto |
| 4 | `images105` | 1 | 3 | 1 | 2 | 0 | 0.693 | ⚠️ Parcial |
| 5 | `946_png` | 1 | 1 | 1 | 0 | 0 | 0.845 | ✅ Correto |
| 6 | `978_png` | 1 | 1 | 1 | 0 | 0 | 0.700 | ✅ Correto |
| 7 | `892_png` | 1 | 2 | 1 | 1 | 0 | 0.692 | ⚠️ Parcial |
| 8 | `images125` | 1 | 1 | 1 | 0 | 0 | 0.590 | ✅ Correto |

<p align="center"><strong>Test | predições corretas</strong></p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="metrics/evaluation/test_946_png_jpg.rf.VcnZyyMk5I37IBLZHKXm.jpg" width="400"><br>
      <sub>✅ 946_png: 1/1 buraco, IoU 0.85</sub>
    </td>
    <td align="center" width="50%">
      <img src="metrics/evaluation/test_978_png_jpg.rf.l7s0pebXb8SgWuuCMUoj.jpg" width="400"><br>
      <sub>✅ 978_png: 1/1 buraco, IoU 0.70</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="metrics/evaluation/test_650_png_jpg.rf.0mNBxvaMGh7esqw6m0x9.jpg" width="400"><br>
      <sub>✅ 650_png: 1/1 buraco, IoU 0.54</sub>
    </td>
    <td align="center">
      <img src="metrics/evaluation/test_images125_jpg.rf.4hRdTtW5EXDvGbb5NoNz.jpg" width="400"><br>
      <sub>✅ images125: 1/1 buraco, IoU 0.59</sub>
    </td>
  </tr>
</table>

<p align="center"><strong>Test | predições parciais</strong></p>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="metrics/evaluation/test_image27_jpeg.rf.KQ1zDfjzmo3URYjNO90k.jpg" width="400"><br>
      <sub>⚠️ image27: 1/1 buraco, 1 FP (IoU 0.88 no match)</sub>
    </td>
    <td align="center" width="50%">
      <img src="metrics/evaluation/test_891_png_jpg.rf.c5dzztVnjINQOP5ejGPS.jpg" width="400"><br>
      <sub>⚠️ 891_png: 1/1 buraco, 1 FP</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="metrics/evaluation/test_images105_jpg.rf.dIvL79ge9HDC9zoQqFXL.jpg" width="400"><br>
      <sub>⚠️ images105: 1/1 buraco, 2 FP</sub>
    </td>
    <td align="center">
      <img src="metrics/evaluation/test_892_png_jpg.rf.YUwhxcoeUeWnyTkcpX2G.jpg" width="400"><br>
      <sub>⚠️ 892_png: 1/1 buraco, 1 FP</sub>
    </td>
  </tr>
</table>

---

### Análise dos erros

| Padrão observado | Frequência | Impacto |
|------------------|------------|---------|
| **Super-segmentação:** modelo gera máscaras a mais | 8/16 imagens | FP elevados, recall mantido |
| **Buracos múltiplos próximos:** confunde instâncias (`768_png`, `image24`) | 2 imagens | FN pontuais + muitos FP |
| **Máscara extra adjacente:** detecta o buraco certo + fragmento | 6 imagens | FP=1 ou FP=2, sem FN |
| **Falha total:** nenhum buraco detectado | 0 imagens | - |

### Recomendações

1. **Aumentar `conf`** para inferência (ex: 0.35 a 0.45) e reduzir FP em produção
2. **Pós-processamento morfológico:** unir máscaras sobrepostas da mesma região
3. **NMS de máscaras:** suprimir detecções redundantes antes do overlay
4. **Retreino** com mais exemplos de cenas multi-pothole e hard negative mining

### Reproduzir a avaliação

```bash
source .venv/bin/activate
pip install ultralytics opencv-python numpy
python evaluate_samples.py
```

Resultados salvos em `metrics/evaluation/` (imagens comparativas + `evaluation_results.json`).

---

## Roadmap

- [ ] Expandir classes: trincas, remendos, buracos de bueiro
- [ ] Pipeline de inferência em vídeo com tracking temporal
- [ ] Dashboard de inspeção com mapa de severidade
- [ ] Fine-tune com frames próprios das dashcams
- [ ] Quantização INT8 para edge devices

---

<div align="center">

**Road Surface Segmentation** | Computer Vision aplicada à infraestrutura viária

Desenvolvido com YOLO26 · OpenCV · Ultralytics

</div>
