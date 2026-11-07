# Vietnamese Speaker Diarization

**Исследование диаризации вьетнамской речи на основе ECAPA-TDNN и алгоритмов кластеризации.**

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Research](https://img.shields.io/badge/status-research-orange)](#текущий-статус)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

> **Who spoke when?** Система определяет, **кто и когда говорил** в аудиозаписи, без предварительного знания числа участников. Проект также предоставляет инструменты исследования эмбеддингов говорящих, сравнения алгоритмов кластеризации и воспроизводимой оценки качества на вьетнамских речевых корпусах.

**Навигация:** [Возможности](#возможности) · [Быстрый старт](#быстрый-старт) · [Методы](#методы-кластеризации) · [Результаты](#результаты-экспериментов) · [Воспроизведение](#воспроизведение-исследований) · [Ограничения](#ограничения-и-научная-корректность) · [Документация](#документация)

## Задача и архитектура

Диаризация (*speaker diarization*) разбивает запись на временные интервалы и назначает каждому интервалу идентификатор говорящего. В отличие от **speaker verification** (проверки, принадлежат ли два речевых фрагмента одному человеку), здесь необходимо восстановить последовательность говорящих на протяжении записи.

```text
                    ┌────────────────────┐
Audio (WAV / audio) ─▶│ Silero VAD         │  обнаружение речи
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Sliding windows    │  сегменты речи
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ ECAPA-TDNN         │  speaker embeddings
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Clustering         │  AHC / Spectral / NME-SC
                    └─────────┬──────────┘
                              ▼
                    ┌────────────────────┐
                    │ Temporal processing│  согласование меток во времени
                    └─────────┬──────────┘
                              ▼
                        RTTM + diagnostics
```

Модель **ECAPA-TDNN** извлекает векторные представления речевых фрагментов; кластеризация группирует представления одного предполагаемого говорящего. Метки кластеров не содержат реальных имён или идентичностей людей.

## Возможности

- **Inference:** диаризация отдельных аудиозаписей, экспорт в стандартный формат **RTTM**, автоматический или заранее заданный подсчёт говорящих.
- **Кластеризация:** fixed-threshold AHC, spectral clustering, NME-SC, экспериментальные `adaptive_ahc` и `adaptive_ahc_robust`.
- **Обучение:** скрипты для ECAPA-TDNN, AAM-Softmax, Dolly/VLSP, опциональной аугментации и fine-tuning.
- **Оценка:** DER (Miss / False Alarm / Speaker Error), EER и minDCF; отдельная оценка ошибок количества говорящих.
- **Исследования:** синтетические разговоры с известными говорящими, естественные разговоры с *silver* разметкой, анализ качества разбиений (ARI, NMI, purity), иерархии AHC и устойчивости к возмущениям.
- **Воспроизводимость:** зафиксированный checkpoint, проверка SHA-256, разделение выборок, provenance manifests, совместимые кэши эмбеддингов и regression-тесты.

## Текущий статус

**Это исследовательский проект, а не готовый production-сервис.** Основной рабочий режим — AHC с фиксированным порогом **cosine distance `0.65`**. Новые методы изучаются отдельно и **не заменяют значения по умолчанию** без независимого обоснования.

Состояние на **7 ноября 2026 года**:

- Исправлена и проверена совместимость архитектуры ECAPA-TDNN с сохранённым checkpoint: порядок операций в `Bottle2neck` — `Conv1d → ReLU → BatchNorm1d`.
- Воспроизведены исторические результаты на всех **100 записях ViYT-Diar**, включая совпадение SHA-256 всех 100 RTTM.
- Проведён отдельный **синтетический benchmark на 600 разговорах** (`300 / 150 / 150` для development / validation / holdout).
- Проведён benchmark на **8 естественных вьетнамских разговорах DataHive** с автоматически построенной двухканальной *silver reference*.
- Добавлены диагностики ошибки выбора среза иерархии, преждевременных объединений говорящих, устойчивости кластеров и последующей обработки меток.
- **Robust Adaptive AHC остаётся экспериментальным:** улучшение на ранее использованном ViYT-Diar не подтвердилось как устойчивое превосходство на независимых синтетических и естественных данных.

## Быстрый старт

### 1. Установка

Требуется **Python 3.11 или 3.12**. Для GPU предварительно установите совместимые версии `torch` и `torchaudio` для вашей CUDA; для CPU — соответствующие CPU-сборки.

```bash
git clone https://github.com/skylak33/diarize-vietnamese-speech.git
cd diarize-vietnamese-speech

python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

python -m pip install -r requirements.txt
```

Для запуска тестов дополнительно установите `requirements-dev.txt`. Опция `--device auto` выбирает CUDA при доступности, иначе CPU.

**Модель:** основной checkpoint находится в репозитории по пути `exps/exp1/model/model_0034.model` (~60 MiB). Проверка SHA-256:

```text
58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325
```

### 2. Диаризация аудио

```bash
python -m diarize.pipeline path/to/audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method ahc \
  --device auto \
  --output results/audio.rttm
```

Результат — файл `RTTM`, содержащий временные границы и метки предполагаемых говорящих. Пример фрагмента формата (иллюстрация, не результат конкретной записи):

```text
SPEAKER audio 1 0.50 2.30 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio 1 3.10 1.70 <NA> <NA> SPEAKER_01 <NA> <NA>
```

Если количество говорящих известно **из внешних условий**, его можно указать явно:

```bash
python -m diarize.pipeline path/to/audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method spectral --n_speakers 2 \
  --output results/audio.rttm
```

Использовать истинное количество говорящих **из тестовой разметки** допустимо только для отдельно обозначенного *oracle experiment*, но не для оценки полностью автоматического режима.

### 3. Настройка

Параметры по умолчанию заданы в [`configs/diarization_default.yaml`](configs/diarization_default.yaml):

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `clustering.method` | `ahc` | Метод кластеризации |
| `clustering.threshold` | `0.65` | Порог **cosine distance** для AHC |
| `clustering.max_speakers` | `10` | Максимальное число говорящих |
| `embedding.window_sec` | `1.5` | Длина окна эмбеддинга, с |
| `embedding.step_sec` | `0.75` | Шаг окна, с |
| `vad.threshold` | `0.5` | Порог VAD |
| `vad.min_silence_ms` | `100` | Минимальная пауза, мс |
| `postprocessing.centroid_merge.enabled` | `false` | Дополнительное объединение кластеров |
| `evaluation.collar` | `0.25` | Допуск при DER, с |

```bash
python -m diarize.pipeline path/to/audio.wav \
  --model exps/exp1/model/model_0034.model \
  --config configs/diarization_default.yaml \
  --threshold 0.75 \
  --output results/audio.rttm
```

Приоритет: **явный аргумент CLI > YAML > встроенное значение**. Для AHC увеличение `--threshold` обычно разрешает больше объединений и уменьшает число кластеров. Порог `centroid_similarity_threshold` имеет другую семантику (**cosine similarity**): больший порог делает объединение строже.

## Методы кластеризации

| Метод | Число говорящих | Статус |
|---|---|---|
| **Fixed AHC** | Определяется порогом cosine distance | **Основной режим** |
| **Spectral** | Заданное `K` или автоматическая оценка | Реализован, сравнивается в экспериментах |
| **NME-SC** | Автоматическая оценка по спектральной структуре | Реализован, включая обработку коротких записей |
| **Legacy Adaptive AHC** | Выбор среза иерархии по gap-эвристике | Экспериментальный |
| **Robust Adaptive AHC** | Leave-one-out Tukey-критерий для наибольшего gap с fallback | **Экспериментальный, opt-in** |

Запуск Robust Adaptive AHC:

```bash
python -m diarize.pipeline path/to/audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method ahc --threshold 0.75 \
  --count_estimator adaptive_ahc_robust \
  --output results/audio_robust.rttm
```

**Идея Robust Adaptive AHC:** построить полную иерархию average-linkage AHC, проанализировать последовательность расстояний слияния и выбрать разрез при статистически выделяющемся максимальном gap. Для расчёта Tukey-порога кандидат исключается из эталонной выборки; неоднозначность и отсутствие выраженного gap приводят к fallback на фиксированный порог. Метод **не использует истинные speaker labels при inference**.

**Важно:** корректное предсказание `K` не означает корректного назначения фрагментов говорящим. Иерархия может необратимо объединить разных людей ещё до выбора разреза. Поэтому оценка включает **DER и качество разбиения**, а не только count accuracy.

## Оценка качества

В проекте используются **разные метрики для разных задач**:

| Задача | Метрики | Что измеряют |
|---|---|---|
| Speaker diarization | **DER**, Miss, False Alarm, Speaker Error | Ошибки активности речи и назначения говорящих |
| Speaker count | Accuracy, MAE, overcount / undercount | Ошибку числа говорящих |
| Speaker partition | ARI, NMI, purity, fragmentation | Качество группировки фрагментов |
| Speaker verification | EER, minDCF | Различимость пар «один / разные говорящие» |

DER считается как отношение суммарного времени ошибок к суммарному времени референсной речи. **Corpus DER агрегирует секунды ошибок по всему корпусу, а не усредняет DER отдельных записей.** Для описанных ниже экспериментов используется collar **0.25 с**; overlap входит в основной подсчёт, если явно не указано иное.

### DER для одной записи или корпуса

```bash
python -m eval.der \
  --hyp results/audio.rttm \
  --ref path/to/reference.rttm

python -m eval.der \
  --hyp_dir results/hypothesis_rttm \
  --ref_dir path/to/reference_rttm \
  --collar 0.25 \
  --output results/der.json \
  --csv results/der.csv
```

Если официальный протокол требует ограниченную область оценки, можно передать `--uem` или `--uem_dir`. `--skip_overlap` **изменяет протокол** и должен указываться явно.

### EER / minDCF для speaker verification

```bash
python -m eval.eer \
  --model_path exps/exp1/model/model_0034.model \
  --dataset vlsp \
  --trials path/to/public_test_trials.txt \
  --trials_format vlsp \
  --audio_root path/to/public_test_audio \
  --device auto \
  --output results/vlsp_verification.json
```

Полный заданный trials-протокол оценивается без случайного сокращения. **EER не равен точности определения говорящих и не заменяет DER.** Исторические значения EER, измеренные до исправления порядка операций ECAPA, нельзя автоматически относить к текущей модели: для корректного сравнения необходимо повторное evaluation с совместимой архитектурой и тем же протоколом.

## Результаты экспериментов

**Ниже приведены измеренные результаты из отчётов проекта, а не прогнозы.** Эти корпуса и протоколы различаются; их абсолютные DER **нельзя напрямую сравнивать между собой**. Меньше DER — лучше.

### 1. ViYT-Diar — воспроизведение и exploratory analysis

**100 записей**, полный аудиосигнал, collar 0.25 с, overlap включён. Результаты получены на уже неоднократно исследованной тестовой выборке.

| Автоматический метод | DER ↓ | Count accuracy ↑ |
|---|---:|---:|
| Fixed AHC 0.75 | **15.124%** | 60% |
| Legacy Adaptive AHC | 13.770% | 68% |
| Robust Adaptive AHC | **13.013%** | **71%** |
| Spectral | 16.649% | 65% |
| NME-SC | 19.956% | 46% |

Внутри этой **исследовательской** серии Robust показывает лучший DER. Однако ViYT-Diar **не является независимым holdout для выбора гиперпараметров**. Исторический fixed-AHC baseline был воспроизведён для 100/100 RTTM

### 2. Синтетический вьетнамский benchmark — отдельный holdout

Создано **600 чистых разговоров из реальных вьетнамских речевых фрагментов**: 300 development, 150 validation, 150 holdout. Длительность одного разговора — **30–45 с**, число говорящих — **1–5**, без добавленного шума и overlap. Идентификаторы говорящих, исходные фрагменты и хеши декодированного контента разделены между выборками. Сегменты речи для reference автоматически уточнялись VAD, поэтому это **не полностью вручную размеченный корпус**.

**Holdout, full pipeline (Silero VAD → ECAPA → clustering → postprocessing):**

| Автоматический метод | DER ↓ | Count accuracy ↑ |
|---|---:|---:|
| **Fixed AHC 0.75** | **4.630%** | **74.67%** |
| Legacy Adaptive AHC | 6.887% | 66.67% |
| Robust Adaptive AHC | 8.609% | 61.33% |
| Spectral | 8.511% | 65.33% |
| NME-SC | 20.893% | 58.00% |

**Вывод:** на этом независимом синтетическом holdout фиксированный AHC работает лучше адаптивных вариантов. Robust может улучшать частный случай с двумя говорящими, но склонен дробить односпикерные записи и ошибаться при большем `K`.

### 3. Natural DataHive — естественные разговоры с silver reference

**8 двухканальных записей**, для которых эталонные интервалы получены независимым Silero VAD для левого и правого канала. В ref сохранено пересечение речи двух участников; проверенная человеком ground truth **отсутствует**. Все восемь записей номинально содержат двух говорящих.

**Full pipeline, overlap included:**

| Автоматический метод | DER ↓ | Count accuracy ↑ | ARI ↑ |
|---|---:|---:|---:|
| Fixed AHC 0.65 | 13.995% | 0/8 | 0.829 |
| Fixed AHC 0.75 | 13.995% | 0/8 | 0.829 |
| Legacy Adaptive AHC | 26.202% | 2/8 | 0.468 |
| Robust Adaptive AHC | 31.149% | 2/8 | 0.288 |
| Spectral | 9.971% | 8/8 | 0.942 |
| NME-SC | **9.698%** | **8/8** | **0.968** |

На этом небольшом корпусе **spectral / NME-SC** оказываются лучше по DER, чем fixed AHC. При этом меньшая ошибка подсчёта у Robust **не приводит** к лучшему разбиению или меньшему DER. Эти результаты не обосновывают смену production default: **всего 8 записей, только `K=2`, silver reference**. 
### Что показывают эксперименты

1. **Нельзя выбирать метод только по speaker-count accuracy:** правильно угаданное количество не гарантирует правильную кластеризацию.
2. **Качество зависит от типа данных:** порядок методов на чистой синтетике и естественных двухканальных разговорах существенно различается.
3. **Robust Adaptive AHC не готов к использованию по умолчанию:** положительный результат на повторно использованном ViYT-Diar не обобщается на два других исследованных сценария.
4. **Ошибки нужно локализовать по этапам:** качество эмбеддингов, необратимые объединения AHC, выбор разреза и temporal postprocessing могут ухудшать итог независимо друг от друга.

## Воспроизведение исследований

### Синтетический benchmark

Генератор использует **локально подготовленные** источники VIVOS / VieSpeaker / DNU-ViSpeech. Данные не загружаются автоматически. Сначала настройте пути в source manifest и проверьте доступность файлов.

```bash
python experiments/audit_local_speaker_datasets.py \
  --vietnamese-asr /path/to/vietnamese_asr \
  --viespeaker /path/to/VieSpeaker-Dataset-opus \
  --dnu-vispeech /path/to/DNU-ViSpeech

python experiments/generate_synthetic_conversations.py --dry-run
python experiments/generate_synthetic_conversations.py

python experiments/run_synthetic_diarization_benchmark.py \
  --conversation-json results/synthetic_vietnamese_benchmark/generated \
  --model exps/exp1/model/model_0034.model \
  --split development --device auto \
  --output-dir results/synthetic_vietnamese_benchmark/experiments/development

python experiments/analyze_synthetic_benchmark.py \
  results/synthetic_vietnamese_benchmark/experiments/development/recording_results.json
```

**Правильный порядок исследования:** выбрать конфигурацию на development → зафиксировать её → проверить на validation → однократно использовать holdout. Не подбирать параметры по holdout.

### Natural DataHive

Требует доступа к соответствующему датасету, свободного места для кэша и соблюдения лицензии источника. Скрипты сохраняют manifest, исходные ревизии и хеши.

```bash
python experiments/acquire_natural_vietnamese_benchmark.py \
  --dataset datahive --download \
  --output results/natural_vietnamese_benchmark/datahive_acquisition_manifest.json

python experiments/prepare_natural_vietnamese_references.py \
  --dataset datahive \
  --acquisition-manifest results/natural_vietnamese_benchmark/datahive_acquisition_manifest.json \
  --output-dir results/natural_vietnamese_benchmark/datahive_references

python experiments/run_natural_vietnamese_benchmark.py \
  --dataset datahive --device auto \
  --annotation-manifest results/natural_vietnamese_benchmark/datahive_references/annotation_manifest.jsonl \
  --output-dir results/natural_vietnamese_benchmark/full_evaluation

python experiments/analyze_natural_vietnamese_benchmark.py \
  results/natural_vietnamese_benchmark/full_evaluation/recording_results.json
```

### Диагностика ошибок кластеризации

```bash
python experiments/cache_natural_diagnostic_embeddings.py --device auto
python experiments/analyze_ahc_merge_quality.py
python experiments/analyze_clustering_stability.py
python experiments/compare_partition_failure_mechanisms.py
```

Анализ различает:

- **count error** — ошибочное число кластеров;
- **partition error** — неправильное объединение эмбеддингов даже при верном `K`;
- **wrong cut** — выбор неудачного уровня уже построенной иерархии;
- **early cross-speaker merge** — необратимое объединение разных говорящих;
- **postprocessing effect** — изменение числа и назначения меток при сглаживании.

Reference speaker IDs используются **только после inference** для диагностики. *Oracle cut* показывает потенциальные свойства уже построенной иерархии, но не является автоматическим алгоритмом.

## Структура репозитория

```text
.
├── configs/          # YAML/JSON конфигурации и протоколы benchmark
├── diarize/          # VAD, embeddings, clustering, RTTM, pipeline
├── eval/             # DER / EER / minDCF
├── experiments/      # runners, data preparation и диагностика
├── models/           # ECAPA-TDNN и loss
├── train/            # обучение, fine-tuning, загрузчики данных
├── utils/            # аудио, модели, метрики, seed, provenance
├── tests/            # unit / integration / regression tests
├── docs/             # научные протоколы и отчёты
├── kaggle/           # исследовательские ноутбуки
├── exps/             # сохранённые checkpoints
└── audit/            # локальные эталоны и результаты ручного аудита
```

Выходные артефакты исследований записываются в `results/` (игнорируется Git): RTTM, JSON/CSV, manifests, кэши эмбеддингов и диагностические графики. Их отсутствие в Git **не означает**, что результаты можно воспроизвести без исходных аудиоданных.

## Тестирование

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall diarize eval experiments tests
git diff --check
```

Тесты и CI проверяют код на CPU-совместимых локальных fixtures и **не скачивают** реальные датасеты или checkpoint. Для воспроизведения опубликованных чисел нужны реальные аудио, точные версии моделей и зафиксированный scoring protocol.

## Ограничения и научная корректность

- **Данных недостаточно для универсальных выводов.** ViYT-Diar уже применялся для исследовательских ablation; синтетический benchmark не воспроизводит всю сложность живых диалогов; DataHive содержит только восемь записей с `K=2`.
- **Разметка неоднородна.** Synthetic reference частично VAD-assisted; DataHive reference — *silver*, а не независимо подтверждённая вручную ground truth.
- **Одновременная речь:** текущий inference назначает только **одну speaker label в момент времени**, поэтому не моделирует overlap как отдельные параллельные потоки.
- **Вероятный перенос между корпусами не доказан.** Разделение источников и speaker IDs снижает риск leakage, но не доказывает отсутствие повторения одного и того же реального человека между внешними датасетами. Полный манифест обучения epoch-34 checkpoint недоступен.
- **Checkpoint compatibility критична.** Кэши и метрики, вычисленные с неправильным порядком операций `Bottle2neck`, нельзя смешивать с результатами исправленного encoder.
- **Безопасная смена параметров:** пороги и новые алгоритмы должны проверяться на отдельной вручную проверенной development-выборке, после чего — на нетронутом test set.

Для воспроизводимой публикации следует сохранять commit репозитория, SHA-256 checkpoint и исходного encoder, ревизии аудиодатасетов, конфигурации VAD/эмбеддингов/кластеризации, состояние overlap/UEM/collar, seed, хеши reference/RTTM и provenance кэшей.


## Лицензия и использование данных

Исходный код распространяется на условиях [MIT License](LICENSE.md)

