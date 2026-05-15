# ECAPA-TDNN - Speaker Diarization для вьетнамского языка

Исследовательский проект по диаризации говорящего на вьетнамском языке на основе архитектуры **ECAPA-TDNN** (C=1024, 14.73M параметров).

Пайплайн: верификация говорящего → извлечение эмбеддингов → кластеризация → RTTM.

Оригинальная архитектура: https://github.com/TaoRuijie/ECAPA-TDNN

---

## Структура проекта

```
ECAPA-TDNN/
├── models/                  # архитектура
│   ├── ecapa_tdnn.py        # ECAPA-TDNN encoder
│   └── loss.py              # AAM-Softmax
│
├── train/                   # обучение
│   ├── trainer.py           # ECAPAModel (encoder + loss + optimizer)
│   ├── run.py               # скрипт запуска обучения
│   ├── loader_hf.py         # загрузчик dolly (HuggingFace streaming)
│   └── loader_vlsp.py       # загрузчик VLSP (локальные WAV)
│
├── eval/                    # оценка качества
│   ├── eer.py               # EER / minDCF
│   └── der.py               # DER (Diarization Error Rate)
│
├── diarize/                 # пайплайн диаризации
│   ├── pipeline.py          # audio → RTTM
│   ├── vad.py               # VAD (silero-vad)
│   ├── cluster.py           # AHC / спектральная кластеризация
│   └── rttm_convert.py      # конвертация RTTM ↔ Audacity labels
│
├── utils/
│   └── metrics.py           # EER, minDCF, вспомогательные функции
│
└── exps/
    └── exp1/model/          # чекпоинты модели
```

---

## Датасеты

| Датасет | Спикеров | Использование |
|---|---|---|
| `vuhoanhuy/dolly-audio-1000h-vietnamese` | 152  | предобучение |
| `dungnasa10/vietnamese-speaker-verification` (VLSP 2021) | 1015  | дообучение |
| `VLSP 2021 Public Test` | - | оценка EER |

---

## Установка

```bash
python -m venv ECAPA
source ECAPA/bin/activate
pip install torch torchaudio pyarrow soundfile huggingface_hub httpx[socks] \
            silero-vad scikit-learn scipy pyannote.metrics pyannote.core tqdm requests
```

---

## Обучение

### Предобучение на dolly (HuggingFace)

```bash
python train/run.py \
    --n_class 152 \
    --save_path exps/exp1 \
    --num_frames 200 \
    --batch_size 256 \
    --n_cpu 0 \
    --max_epoch 80
```

### Дообучение на VLSP (локальные WAV)

Переключить загрузчик в `train/run.py`:
```python
from train.loader_vlsp import VLSPLoader as TrainLoader
```

```bash
python train/run.py \
    --n_class 1015 \
    --save_path exps/exp1 \
    --initial_model exps/exp1/model/model_0034.model \
    --num_frames 200 \
    --batch_size 256 \
    --n_cpu 0 \
    --max_epoch 80
```

Обучение автоматически продолжается с последнего чекпоинта при перезапуске.

---

## Оценка качества (EER / minDCF)

```bash
python eval/eer.py \
    --model_path exps/exp1/model/model_0034.model \
    --n_pairs 2000
```

| Модель | Данные оценки | EER |
|---|---|---|
| exp1, эпоха 27 (только dolly) | VLSP 2021 Public Test | 20.28% |
| exp1, эпоха 34 (dolly → VLSP) | VLSP 2021 Public Test | **5.00%** |
| Лучший результат VLSP 2021 | - | 1.755% |

---

## Диаризация

### Запуск пайплайна

```bash
# автоподбор числа спикеров (AHC)
python diarize/pipeline.py audio.wav \
    --model exps/exp1/model/model_0034.model \
    --output diarize/result.rttm

# с указанием числа спикеров (точнее)
python diarize/pipeline.py audio.wav \
    --model exps/exp1/model/model_0034.model \
    --n_speakers 2 \
    --method spectral \
    --output diarize/result.rttm
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--n_speakers` | None (авто) | Число спикеров |
| `--method` | `ahc` | `ahc` или `spectral` |
| `--threshold` | 0.65 | Порог AHC (ниже = больше спикеров) |
| `--vad_threshold` | 0.5 | Порог VAD |

### Оценка DER

```bash
# сравнить два RTTM файла
python eval/der.py --hyp diarize/result.rttm --ref reference.rttm

# запустить диаризацию и сразу посчитать DER
python eval/der.py \
    --audio audio.wav \
    --model exps/exp1/model/model_0034.model \
    --ref reference.rttm
```

### Разметка через Audacity

```bash
# RTTM → метки Audacity (для редактирования)
python diarize/rttm_convert.py to_audacity diarize/result.rttm diarize/labels.txt

# После редактирования → обратно в RTTM
python diarize/rttm_convert.py to_rttm diarize/labels_corrected.txt reference.rttm
```
