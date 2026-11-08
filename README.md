# Vietnamese Speaker Diarization with ECAPA-TDNN

Исследовательский pipeline диаризации вьетнамской речи:

```text
audio -> Silero VAD -> ECAPA-TDNN embeddings -> clustering -> postprocessing -> RTTM
```

Проект разделяет две задачи: speaker verification (`EER`, `minDCF`) и speaker
diarization (`DER`). Основная метрика диаризации — corpus-level DER, рассчитанный
через `pyannote.metrics`.

## Текущий статус

- Production defaults не менялись по результатам исследовательских ablation.
- Зафиксированный checkpoint: `exps/exp1/model/model_0034.model`.
- SHA256 checkpoint:
  `58f3b402afa1de3c0a2543e845267146544be7ba74cad1a38cb4d26303afd325`.
- ECAPA `Bottle2neck` использует совместимый с checkpoint порядок
  `Conv -> ReLU -> BatchNorm`.
- ViYT-Diar уже многократно использовался для exploratory analysis и не считается
  независимым test set для выбора параметров.
- DataHive-разметка получена channel-wise VAD и является silver reference, а не
  вручную подтверждённой ground truth.
- В целевом ручном аудите 24 подозрительных DataHive overlap-фрагментов 23
  подтвердили одновременную речь двух людей; в `vi_000004_01` речь была только
  слева, а справа присутствовал шум. Это проверка выборки, не всего корпуса.

## Структура репозитория

- `models/` — ECAPA-TDNN и AAM-Softmax.
- `train/` — Dolly/VLSP loaders, augmentation и обучение.
- `diarize/` — VAD, embeddings, clustering, postprocessing и RTTM.
- `eval/` — EER/minDCF и diarization DER.
- `experiments/` — воспроизводимые benchmark и diagnostic scripts.
- `configs/` — проверяемые конфигурации.
- `tests/` — CPU unit/regression tests на локальных fixtures.
- `docs/` — подробные протоколы и исследовательские отчёты.

## Установка

Рекомендуется Python 3.11 или 3.12. Сначала установите согласованные сборки
`torch` и `torchaudio` для CPU либо своей версии CUDA, затем остальные зависимости:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
python -m pytest -q
```

`--device auto` использует CUDA при наличии, иначе CPU. `--device cuda` завершает
работу с ошибкой, если CUDA недоступна. CI не скачивает checkpoints или датасеты и
проверяет только CPU-совместимую логику.

## Диаризация

Fixed-threshold AHC:

```bash
python diarize/pipeline.py audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method ahc --threshold 0.65 --device auto --seed 42 \
  --output results/audio.rttm
```

Spectral clustering с известным числом спикеров:

```bash
python diarize/pipeline.py audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method spectral --n_speakers 2 \
  --output results/audio.rttm
```

Основные defaults: VAD threshold `0.5`, embedding window `1.5 s`, step `0.75 s`,
максимум `10` спикеров. AHC threshold — cosine distance: меньший threshold обычно
сохраняет больше кластеров. Pipeline выдаёт одну speaker label на момент времени и
не моделирует overlap отдельным детектором.

Adaptive AHC доступен только opt-in:

```bash
python diarize/pipeline.py audio.wav \
  --model exps/exp1/model/model_0034.model \
  --method ahc --threshold 0.75 --count_estimator adaptive_ahc_robust \
  --output results/audio.rttm
```

Experimental confidence-aware merging is opt-in after Fixed AHC:

```bash
python diarize/pipeline.py audio.wav \
  --model exps/exp1/model/model_0034.model \
  --config configs/diarization_confidence_aware_experimental.yaml \
  --output results/audio.rttm

## Оценка

Один RTTM:

```bash
python eval/der.py --hyp results/audio.rttm --ref data/reference/audio.rttm
```

Корпус с collar `0.25 s`:

```bash
python eval/der.py \
  --hyp_dir results/rttm --ref_dir data/reference_rttm \
  --collar 0.25 --output results/der_results.json \
  --csv results/der_results.csv
```

По умолчанию reference overlap учитывается. `--skip_overlap` исключает его только
явно. Corpus rates агрегируются по сумме секунд компонентов, а не усреднением DER
по записям.

Speaker verification по явному trials-файлу:

```bash
python eval/eer.py \
  --model_path exps/exp1/model/model_0034.model \
  --dataset vlsp --trials /path/to/public_test_trials.txt \
  --audio_root /path/to/public_test_audio --device auto \
  --output results/vlsp_eer.json
```

Trials оцениваются целиком. Training split не используется как скрытая замена test.

## Обучение

Dolly pretraining:

```bash
python train/run.py --dataset dolly --n_class 152 \
  --save_path exps/dolly --max_epoch 80 --device auto --seed 42
```

VLSP fine-tuning с переносом encoder:

```bash
python train/run.py --dataset vlsp --n_class 1015 \
  --save_path exps/vlsp \
  --initial_model exps/exp1/model/model_0034.model \
  --metadata_path /path/to/training_metadata.txt \
  --audio_base /path/to/audio_root --device auto --seed 42
```

`--initial_model` переносит encoder и создаёт новый classifier. Возобновление
обучения остаётся weight-only: optimizer state не сохраняется, поэтому побитовый
resume не заявляется.

## Сохранённые benchmark-результаты

Результаты ниже получены в разных условиях и не образуют единого cross-corpus
рейтинга.

| Dataset | Fixed AHC 0.75 DER | Robust AHC DER | Ограничение |
|---|---:|---:|---|
| ViYT-Diar | 15.124% | 13.013% | reused exploratory test |
| Synthetic Vietnamese holdout | 4.630% | 8.609% | синтетические разговоры из реальной речи |
| Natural DataHive | 13.995% | 31.149% | 8 записей, K=2, silver reference |

На DataHive Robust иногда выбирает статистически выраженный hierarchy gap, но
получает плохое speaker partition. Критический пример `vi_000004`: initial AHC
`K=4`, final postprocessed `K=2`, initial partition ARI около `0.001`. Поэтому
speaker-count accuracy, partition quality и влияние postprocessing всегда
сообщаются раздельно.

Ручное прослушивание 24 выбранных overlap-фрагментов подтвердило reference в 23
случаях и выявило один ложный overlap из-за шума правого канала. Эта локальная
ошибка не объясняет крупные Robust AHC regressions.

## Диагностика разбиений и устойчивости

Все команды используют сохранённые артефакты и не меняют production defaults:

```bash
# Создать/проверить локальный embedding cache. Повторно использует совместимый cache.
python experiments/cache_natural_diagnostic_embeddings.py --device auto

# Partition quality, AHC merges, oracle cut diagnostics и компактные figures.
python experiments/analyze_ahc_merge_quality.py

# Angular noise, ordered subsampling, short prefixes и synthetic controls.
python experiments/analyze_clustering_stability.py

# Silver-reference signal audit; excerpts остаются локальными.
python experiments/audit_silver_reference.py --write-excerpts

# Сопоставление механизмов ошибок без прямого сравнения DER разных корпусов.
python experiments/compare_partition_failure_mechanisms.py
```

Результаты сохраняются в `results/partition_quality_stability/`:

```text
embedding_cache_verified/
partition_quality/
hierarchy/
stability/
silver_reference_audit/
cross_dataset/
figures/
report.md
```

Reference labels используются только после inference для diagnostics. Oracle cut
sweep показывает ёмкость уже построенной иерархии и не является production rule.

## Проверка изменений

```bash
python -m pytest -q
python -m compileall diarize eval experiments tests
git diff --check
```

Для публикации сохраняйте git commit, SHA256 checkpoint и source, dataset revision,
effective config, scoring protocol, hashes audio/reference и manifest использованных
embedding caches.

## Лицензия

См. [`LICENSE.md`](LICENSE.md).
