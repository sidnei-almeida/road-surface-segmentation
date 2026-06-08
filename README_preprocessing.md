# Pré-processamento de vídeos — Road Surface Segmentation

Script para preparar vídeos POV/dashcam antes da anotação manual em um dataset de segmentação de danos em vias (YOLO segmentation).

## Onde colocar os vídeos

Coloque os arquivos de entrada na pasta `videos/` na raiz do projeto:

```
road-surface-segmentation/
├── videos/                  ← seus vídeos aqui (.mp4, .mov, .mkv, .avi)
├── processed_videos/        ← gerado automaticamente
├── frames/                  ← gerado automaticamente
├── reports/                 ← gerado automaticamente
└── prepare_road_videos.py
```

As pastas de saída são criadas automaticamente se não existirem.

## Dependências

### Python

```bash
pip install opencv-python numpy rich textual
```

### Sistema (ffmpeg)

O ffmpeg precisa estar instalado e disponível no `PATH`.

```bash
# Arch / CachyOS
sudo pacman -S ffmpeg

# Debian / Ubuntu
sudo apt install ffmpeg
```

Verifique com:

```bash
ffmpeg -version
```

## Como rodar

### Interface interativa (TUI) — recomendado

Execute sem argumentos para abrir a interface no terminal:

```bash
python prepare_road_videos.py
```

Ou explicitamente:

```bash
python prepare_road_videos.py --tui
```

A TUI oferece quatro modos:

| Atalho | Modo | Descrição |
|--------|------|-----------|
| `1` | Desacelerar | Gera versões mais lentas dos vídeos |
| `2` | Acelerar | Gera versões mais rápidas (útil para pré-visualização) |
| `3` | Extrair frames | Amostra frames com modo `hybrid` (padrão) |
| `4` | Pipeline completo | Desacelera + extrai frames |

Atalhos gerais: `Q` para sair, `Esc` para voltar nas telas de configuração.

### Linha de comando (CLI)

Processamento padrão — desacelerar 1.5x e extrair **~600 frames por vídeo** (milhares no total):

```bash
python prepare_road_videos.py --input-dir videos --slow-factor 1.5 --sampling-mode hybrid
```

Extração densa por intervalo de tempo (~1 frame a cada 0.05 s):

```bash
python prepare_road_videos.py --frame-interval 0.05 --sampling-mode hybrid --overwrite
```

Com sobrescrita de arquivos existentes:

```bash
python prepare_road_videos.py --input-dir videos --slow-factor 2.0 --frames-per-video 100 --overwrite
```

Apenas desacelerar (sem frames):

```bash
python prepare_road_videos.py --no-frames
```

Apenas extrair frames do vídeo original (sem desacelerar):

```bash
python prepare_road_videos.py --no-slow
```

Acelerar vídeos 2x:

```bash
python prepare_road_videos.py --speed-factor 2.0 --no-frames
```

## Onde os arquivos são salvos

| Saída | Caminho | Exemplo |
|-------|---------|---------|
| Vídeo desacelerado | `processed_videos/` | `4309723_slow_1.5x.mp4` |
| Vídeo acelerado | `processed_videos/` | `4309723_fast_2.0x.mp4` |
| Frames | `frames/<nome_do_video>/` | `frames/4309723/frame_000001.jpg` |
| Relatório JSON | `reports/` | `reports/4309723_report.json` |

Cada relatório JSON contém duração, FPS, resolução, modo de amostragem, metadados por frame (brilho, nitidez, contraste, textura, score final) e candidatos rejeitados com motivo (`too_dark`, `too_bright`, `too_blurry`, `too_similar`).

## Recomendação inicial para o projeto

Para os vídeos POV/dashcam atuais, comece com:

```bash
python prepare_road_videos.py --slow-factor 1.5 --frame-interval 0.05 --sampling-mode hybrid
```

**Por quê:**

- **1.5x mais lento** — facilita identificar danos em cenas rápidas sem gerar arquivos enormes.
- **`--frame-interval 0.05`** — ~20 amostras/segundo de vídeo útil → **milhares de frames** no dataset (5 vídeos ≈ 3500+ imagens).
- **`--sampling-mode hybrid`** — melhor frame por segmento temporal, com filtro de qualidade na ROI da via.
- **Skip de 2 s** no início e fim — evita frames de transição ou cenas instáveis comuns em dashcams.

Ajuste conforme necessário:

- Vídeos longos → aumente `--frames-per-video` (ex: 100).
- Muitos frames escuros/claros descartados → ajuste `--min-brightness` e `--max-brightness`.
- Resultados diferentes entre execuções → altere `--seed` (padrão: 42).

## Parâmetros disponíveis

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--input-dir` | `videos` | Pasta de entrada |
| `--processed-dir` | `processed_videos` | Vídeos processados |
| `--frames-dir` | `frames` | Pasta base dos frames |
| `--reports-dir` | `reports` | Relatórios JSON |
| `--slow-factor` | `1.5` | Fator de desaceleração |
| `--speed-factor` | — | Fator de aceleração (substitui desaceleração) |
| `--frames-per-video` | `600` | Frames por vídeo (fixo) |
| `--frame-interval` | — | Segundos entre amostras (substitui contagem fixa) |
| `--skip-start` | `2.0` | Ignorar início (segundos) |
| `--skip-end` | `2.0` | Ignorar fim (segundos) |
| `--seed` | `42` | Seed da amostragem |
| `--sampling-mode` | `hybrid` | `random`, `uniform`, `smart` ou `hybrid` |
| `--min-brightness` | `25` | Descartar frames muito escuros |
| `--max-brightness` | `235` | Descartar frames muito claros |
| `--overwrite` | desligado | Substituir saídas existentes |
| `--no-slow` | — | Pular desaceleração |
| `--no-frames` | — | Pular extração de frames |

## Modos de amostragem (`--sampling-mode`)

| Modo | Comportamento |
|------|----------------|
| `random` | Timestamps aleatórios distribuídos; filtra brilho e qualidade básica |
| `uniform` | Timestamps uniformemente espaçados; filtra qualidade básica |
| `smart` | Avalia muitos candidatos e seleciona os de maior score global |
| `hybrid` | **Recomendado.** Divide o vídeo em segmentos temporais e escolhe o melhor frame de cada um, com leve aleatoriedade entre os top candidatos |

Métricas calculadas na ROI da via (parte inferior ~55% da imagem):

- **Brilho** — média na ROI; rejeita muito escuro/claro
- **Nitidez** — variância do Laplaciano
- **Contraste** — desvio padrão em escala de cinza
- **Textura** — densidade de bordas Canny
- **Penalidade de duplicata** — similaridade com frames já selecionados
