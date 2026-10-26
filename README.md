# ECAPA-TDNN — speaker diarization для вьетнамского языка

Исследовательский проект: audio → Silero VAD → ECAPA-TDNN embeddings →
AHC/spectral clustering → RTTM. Архитектура ECAPA-TDNN сохранена.
Обучение speaker recognition и оценка speaker diarization — разные этапы.
Основная метрика диаризации — **corpus DER**.

## Структура

- `models/`: ECAPA-TDNN и AAM-Softmax.
- `train/`: обучение, Dolly/VLSP loaders, waveform augmentation.
- `diarize/`: VAD, embeddings, clustering, постобработка и RTTM.
- `eval/`: verification EER/minDCF, адаптеры данных, diarization DER.
- `experiments/`: ablations, сравнение checkpoints, графики обучения.
- `utils/`: metrics, audio decoding, model/device loading, seeds, provenance.
- `configs/`: YAML-рецепты параметров; автоматически не загружаются, используется CLI.
- `tests/`: CPU unit tests на синтетических данных без скачивания датасетов.

## Установка

Рекомендуется Python 3.11/3.12. Установите **согласованные версии torch и torchaudio**
для CPU или своей версии CUDA, затем зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions запускает тот же CPU test suite на Python 3.11. Workflow не скачивает
Git LFS checkpoints и не обращается к Kaggle/Hugging Face: CI проверяет логику на
синтетических fixtures, а реальные corpus/GPU прогоны остаются отдельным этапом.

На Kaggle сначала проверьте уже установленные torch/torchaudio; не заменяйте CUDA-сборку
на CPU-сборку. `--device auto` выбирает CUDA при наличии, иначе CPU;
`--device cuda` при отсутствии CUDA выдаёт ошибку. Обучение автоматически включает
DataParallel только при выбранном CUDA и нескольких GPU (в том числе двух T4).
Inference/evaluation используют одну GPU или CPU.

## Данные и защита от leakage

В существующем коде найдены:

- Dolly training: `dolly-vn/dolly-audio-1000h-vietnamese`, parquet `data/train-*.parquet`,
  колонка `audio`, speaker ID `voice_id` или `speaker_id`.
- VLSP training: локальная metadata, **speaker ID + TAB + относительный путь**.
  Старые Kaggle defaults сохранены в `train/loader_vlsp.py`, но пути можно задавать через CLI.
- VLSP 2021 Public Test и официальный trials-файл **не входят в репозиторий**.
  В предоставленном Kaggle notebook (ячейка 14) найдены пути и формат Public Test;
  команда приведена ниже. Сам trials-файл локально отсутствует. Train metadata
  не используется как test; адаптер требует явного файла/каталога.

Старый EER-скрипт оценивал train-шарды Dolly, переименованные в `test`.
Прежние числа EER из README нельзя подтвердить как результаты VLSP Public Test:
таблица удалена до повторного измерения по проверенному протоколу.

### VLSP Public Test: пути из вашего Kaggle notebook

В `notebook030f0a2c0a(1).ipynb`, ячейка 14 (индекс с нуля), используется
`public_test_pairs_with_label.txt`: **без заголовка, `wav1,wav2,label` через запятую**.
Для этого формата добавлен `--trials_format vlsp`; расширение `.txt` сохраняется.
Запускайте из корня обновлённого репозитория, где есть `eval/eer.py`:

```bash
python eval/eer.py --model_path exps/exp1/model/model_0034.model \
  --dataset vlsp --trials_format vlsp \
  --trials /kaggle/input/datasets/dungnasa10/vietnamese-speaker-verification/data/data/metadata/test/labels/public_test_pairs_with_label.txt \
  --audio_root /kaggle/input/datasets/dungnasa10/vietnamese-speaker-verification/data/data/test/sv_vlsp_2021/public_test/competition_public_test \
  --split public_test --seed 42 --device auto --output results/vlsp_public_test.json
```

Пути взяты из кода ноутбука, их существование в текущей Kaggle-сессии нужно проверить.
В notebook сохранён вывод старого `evalECAPAModel.py` для 152 speakers/10 shards;
отдельная VLSP-ячейка не содержит сохранённого результата. Это не подтверждает
прежние значения EER как VLSP Public Test. Новый scorer обрабатывает весь trials-файл
и останавливается при ошибке вместо `except ...: pass` из notebook.

Notebook копирует только корневые `*.py` старой плоской версии проекта и переключает
loaders редактированием импортов. Для обновлённой версии нужен **весь репозиторий**
с `models/`, `train/`, `eval/`, `diarize/`, `utils/`, `experiments/`;
используйте `train/run.py --dataset dolly|vlsp` и отдельные `save_path` для стадий.
Скрипт проверки GPU и training output в notebook подтверждают использование двух T4
старым кодом; новая версия на двух T4 пока не проверялась. Ячейка удаления старых
checkpoints мешает сравнению эпох: не запускайте её до сохранения нужных моделей.

### Локальные verification данные

Предпочтителен **полный официальный trials protocol** с известными бинарными метками.
Поддерживаются текстовые строки `label utterance1 utterance2`, либо
`utterance1 utterance2 label` с `--trials_order label_last`, либо CSV
с заголовком `label,utterance1,utterance2`. Метки: `1`/`target` и `0`/`nontarget`.
Без manifest идентификаторы являются путями относительно `--audio_root`.
Если официальный формат иной (например, отдельные enrollment/test/key-файлы), сначала
сделайте явное преобразование в этот интерфейс **с сохранением всех trials и меток**;
это интерфейс проекта, а не утверждение о формате VLSP.

Если официальных trials нет, подготовьте CSV manifest с заголовком:

```text
utterance_id,path,speaker_id,split
```

Укажите реальные строки из metadata, `path` относительно `--audio_root` (либо каталога
manifest), `split`: `test`, `public_test`, `validation`, `dev` или `train`.
Для evaluation `train` недопустим. Если manifest содержит train, пересечение speakers
или путей с evaluation вызывает ошибку. Trials вместе с manifest обязаны ссылаться
только на выбранный split; их метки проверяются по speaker IDs.

Для Dolly без официального test split необходим **заранее выделенный speaker-disjoint
held-out набор** и список исключённых speakers для training (`--exclude_speakers`, один ID
в строке). Если модель уже обучалась на этих speakers, такой результат нельзя выдавать
за оценку на невиденных говорящих: нужно переобучение с исключением held-out speakers.
Один только manifest не доказывает отсутствие leakage в старых checkpoints.

## Обучение

Предобучение Dolly:

```bash
python train/run.py --dataset dolly --n_class 152 --save_path exps/dolly \
  --num_frames 200 --batch_size 256 --n_cpu 0 --max_epoch 80 \
  --device auto --seed 42 --augmentation none
```

VLSP finetuning без редактирования импортов:

```bash
python train/run.py --dataset vlsp --n_class 1015 --save_path exps/vlsp \
  --initial_model exps/exp1/model/model_0034.model \
  --metadata_path /path/to/training_metadata.txt --audio_base /path/to/audio_root \
  --num_frames 200 --batch_size 256 --n_cpu 0 --max_epoch 80 \
  --device auto --seed 42 --augmentation none
```

`--initial_model` переносит encoder, оставляя новый AAM classifier: соответствие
индексов классов между датасетами не предполагается. Используйте отдельный `save_path`
для каждой стадии. Без `--initial_model` загружаются последние веса из `save_path`;
**optimizer state не сохраняется**, поэтому это weight-only continuation, не побитовый
resume. Для Dolly рядом с checkpoint сохраняется `.speakers.json`; старый checkpoint
без mapping можно использовать для encoder transfer, но не безопасного восстановления
классификатора. Промежуточный checkpoint внутри эпохи также не является полным resume.

`score.txt` содержит loss, accuracy и LR каждой эпохи; `run_config.json` — аргументы.
Loaders используют `num_workers=0` для стабильного порядка, `--n_cpu` сохранён для
совместимости. Dolly сохраняет prefetch/decode threads, но результаты потребляются
в порядке подачи. Ошибки данных останавливают запуск, а не меняют корпус молча.

### Augmentation

```bash
python train/run.py --dataset vlsp --n_class 1015 --save_path exps/vlsp_aug \
  --metadata_path /path/to/training_metadata.txt --audio_base /path/to/audio_root \
  --augmentation full --musan_path /path/to/musan --rir_path /path/to/rirs \
  --device auto --seed 42
```

- `none`: без waveform augmentation и без SpecAugment.
- `noise`: additive noise, music или babble (3–8 речевых сигналов), плюс clean branch.
- `rir`: нормированная по энергии RIR convolution либо clean branch.
- `full`: выбор clean/noise/music/babble/RIR. SpecAugment включён для noise/rir/full.
- Если параметр не указан, waveform режим выводится из MUSAN/RIR путей;
  прежнее поведение SpecAugment сохраняется.

Под MUSAN должны быть каталоги `noise/`, `music/`, `speech/` с WAV любой глубины;
RIR WAV ищутся рекурсивно. SNR: noise 0–15 dB, music 5–15 dB, babble 13–20 dB.
Аудио приводится к mono 16 kHz; Windows paths поддерживаются. Недостающие ресурсы
для явно выбранного режима вызывают ошибку. Evaluation не использует augmentation.

## Диаризация

```bash
python diarize/pipeline.py audio.wav --model exps/exp1/model/model_0034.model \
  --method ahc --threshold 0.65 --vad_threshold 0.5 \
  --window_sec 1.5 --step_sec 0.75 --device auto --seed 42 --output results/audio.rttm

python diarize/pipeline.py audio.wav --model exps/exp1/model/model_0034.model \
  --method spectral --n_speakers 2 --output results/audio.rttm
```

Старые позиционные аргументы и defaults window=1.5, step=0.75 сохранены.
Число speakers автоматически оценивается без `--n_speakers`; максимум по умолчанию 10.
Порог AHC — **cosine distance**, меньший порог обычно даёт больше кластеров.
Перекрывающиеся embedding windows разделяются по серединам центров, чтобы не создавать
ложную одновременную речь. Этот pipeline выдаёт одну метку на момент времени и не имеет
отдельного overlap detector. Тишина не заполняется при слиянии соседних сегментов.
При отсутствии речи сохраняется пустой RTTM.

## Evaluation

### Speaker verification evaluation: EER и minDCF

Сначала задайте реальные файлы Public Test:

```bash
python eval/eer.py --model_path exps/exp1/model/model_0034.model \
  --dataset vlsp --trials /path/to/public_test_trials.txt \
  --audio_root /path/to/public_test_audio --n_pairs 20000 --seed 42 \
  --device auto --output results/vlsp_eer.json
```

При trials-файле `n_pairs` игнорируется: оценивается **весь** протокол.
Для короткой команды можно задать `VLSP_TRIALS` и `VLSP_AUDIO_ROOT` в окружении:

```bash
python eval/eer.py --model_path exps/exp1/model/model_0034.model --dataset vlsp --n_pairs 20000 --seed 42
```

Без trials, на явно подготовленном held-out manifest:

```bash
python eval/eer.py --model_path exps/exp1/model/model_0034.model \
  --dataset dolly --manifest data/dolly_eval.csv --audio_root data/dolly \
  --n_pairs 20000 --seed 42 --output results/dolly_eer.json
```

Поддерживаются `VLSP_EVAL_MANIFEST`, `DOLLY_EVAL_MANIFEST`, `DOLLY_TRIALS`,
`DOLLY_AUDIO_ROOT`. Для VLSP manifest при необходимости задайте `--split public_test`.
Без локальных файлов Dolly adapter проверяет реальные HF splits и предпочитает test;
если test/выбранного evaluation split нет, выдаёт ошибку. Train fallback отсутствует.
`--max_utterances` ограничивает streaming evaluation; для публикации указывайте его
и HF revision. `--n_shards` теперь выдаёт объяснение, почему прежний протокол небезопасен.

Pairs генерируются с фиксированным seed, примерно 50/50, равномерно по speakers,
**с возвращением** (повторные пары допустимы; это не 20000 независимых наблюдений).
Positive pair никогда не состоит из одного utterance дважды. Embeddings кешируются
на время evaluation, audio resampling общий с pipeline. Ошибки trials не пропускаются.

Сравнение checkpoints (каждый второй **файл в порядке эпох**, начиная с первого):

```bash
python experiments/eval_checkpoints.py --model_dir exps/exp1/model --dataset vlsp \
  --trials /path/to/public_test_trials.txt --audio_root /path/to/public_test_audio \
  --every 2 --seed 42 --output results/eer_by_epoch.csv
```

CSV: `epoch,EER,minDCF`. Соседний JSON содержит checkpoints, hash протокола,
метаданные и версии. Все checkpoints оцениваются на одних trials.

### Speaker diarization evaluation: DER

```bash
python eval/der.py --hyp results/audio.rttm --ref data/reference/audio.rttm
python eval/der.py --hyp_dir results/rttm --ref_dir data/reference_rttm \
  --collar 0.25 --output results/der_results.json --csv results/der_results.csv
python eval/der.py --audio audio.wav --model exps/exp1/model/model_0034.model --ref reference.rttm
```

Основной и единственный production scorer — `pyannote.metrics.DiarizationErrorRate`.
Некорректный `compute_der_simple` и `--simple` удалены. `--skip_overlap` исключает
overlap reference из scoring; по умолчанию overlap учитывается.

RTTM ID внутри строк должен совпадать с audio stem. При directory evaluation RTTM
читаются рекурсивно; дублирующиеся IDs и неизвестные hypothesis IDs — ошибки.
Отсутствующая гипотеза считается пустой, включая полностью пустой hypothesis directory.
Пустой reference RTTM в директории обозначает silence recording с ID = stem файла.
Многоканальные записи нужно оценивать отдельно.

Corpus rates получаются из **суммы секунд компонентов / суммы reference speaker-seconds**,
а не среднего файловых процентов. `duration` — scoring reference speech после collar/
overlap exclusions; overlap может давать несколько speaker-seconds на одну секунду.
Без аудио/UEM scoring region: от 0 до максимального конца reference/hypothesis.
Runner знает длительности аудио и оценивает полные записи. Для неполной разметки
подготовьте полностью размеченную evaluation область; частичный RTTM не задаёт UEM.
Нулевой знаменатель отображается как `undefined`/JSON `null`, FA seconds сохраняются.

### Ablation

```bash
python experiments/run_diarization_experiments.py \
  --audio_dir data/diarization/audio --ref_rttm_dir data/diarization/rttm \
  --model exps/exp1/model/model_0034.model --output_dir experiments/results/clustering \
  --experiment clustering --seed 42
```

Выбор `--experiment`:

- `clustering`: AHC/spectral.
- `ahc_threshold`: 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75; AHC automatic count.
- `vad_threshold`: 0.3, 0.4, 0.5, 0.6, 0.7.
- `window`: 1.0, 1.5, 2.0 s.
- `speaker_count`: known/automatic.
- `all`: объединение этих **однофакторных** серий с удалением дубликатов, не декартово произведение.

`--dry_run` выводит план без запуска; `--max_configs 32` ограничивает его размер.
Baseline выбирается через `--method`, `--threshold`, `--vad_threshold`, `--window_sec`,
`--speaker_mode`. Step по умолчанию window/2; `--step_sec` фиксирует его явно.
Для каждой серии используйте новый output directory. Encoder загружается один раз.
Сохраняются config, RTTM, DER JSON, per-file CSV, общий `results.csv` и manifest
с seed, версиями, git revision и SHA256 модели/audio/reference.

Общий CSV содержит:
`method,ahc_threshold,vad_threshold,window_sec,step_sec,speaker_mode,DER,Miss,FalseAlarm,SpeakerError,SpeakerCountAccuracy,SpeakerCountMAE,config_id`.
Per-file CSV/JSON: `N_ref,N_pred,abs_error,exact_match` плюс DER/components.
Accuracy — доля точных совпадений (0–1), MAE — средняя абсолютная ошибка числа speakers.
`N_pred` измеряется по окончательному RTTM; smoothing может уменьшить число меток даже
при заданном числе кластеров. Если known count больше числа embeddings, запуск выдаёт
явную ошибку: такая запись не пропускается и count не подменяется.

### Plots

```bash
python experiments/plot_training.py --log exps/exp1/score.txt --output_dir results
python experiments/plot_training.py --log exps/exp1/score.txt \
  --eer_csv results/eer_by_epoch.csv --output_dir results
```

Файлы: `training_loss.png`, `training_accuracy.png`, `training_lr.png`,
опционально `eer_by_epoch.png`. Для повторяющихся epochs в append-log используется
последняя запись. Не смешивайте разные стадии обучения в одном графике без пояснения.

## Experimental protocol

**Verification:** cosine similarity между L2-normalized ECAPA embeddings. EER — точка
равенства false rejection и false acceptance, линейно интерполированная на ROC с
группировкой одинаковых scores. Выводимый Threshold — ближайшая эмпирическая рабочая
точка, не обещание точного равенства ошибок на конечной выборке.
minDCF минимизирует `C_miss * P_miss * P_target + C_fa * P_fa * (1-P_target)` и
нормируется на `min(C_miss*P_target, C_fa*(1-P_target))`. Defaults: P_target=0.05,
оба cost=1; доступны CLI overrides. Порог minDCF отдельно записывается в JSON.
Это явно заданная настройка проекта, а не утверждение о конкретном VLSP cost protocol.
См. [NIST evaluation plan](https://www.nist.gov/system/files/documents/2021/07/12/2021_SRE_Evaluation_Plan_V5.pdf).

**Diarization:** `DER = (Miss seconds + False Alarm seconds + Confusion seconds) /
reference speaker-seconds`. Miss — пропущенная речь, FA — лишняя речь,
Confusion — неверный speaker после оптимального взаимно-однозначного сопоставления
меток в записи. Компоненты и DER в CSV выражены в процентах. Collar передаётся
непосредственно в pyannote: `0.25` — исключаемая окрестность общей ширины 0.25 s
вокруг каждой reference границы (по 0.125 s с каждой стороны).
См. [pyannote reference](https://pyannote.github.io/pyannote-metrics/reference.html).

Known count — oracle-условие с числом speakers из reference; automatic count — реальное
предсказание pipeline. Их результаты сообщаются отдельно. AHC использует average linkage
по cosine distance; spectral использует clipped cosine affinity и silhouette selection
для k=2..max_speakers. Для одного speaker и слишком коротких записей применяется
AHC gate с тем же threshold; это эвристика, а не обученный count estimator.
Spectral сам по себе не восстанавливает overlapping speech.

Выбирайте thresholds/window/checkpoint на **development** corpus, а итоговую оценку
проводите один раз на независимом test. При ablations на Public Test результаты следует
обозначать как exploratory, не как независимую финальную оценку. Не объединяйте DER
диаризации и EER верификации в одну метрику. Указывайте train data/stage, test protocol,
collar, overlap policy, count mode, seed и версии для каждой таблицы.
[Описание VLSP 2021](https://vlsp.org.vn/node/123) различает условия использования
обучающих данных; Dolly pretraining нужно явно указывать при сопоставлении с challenge.

`set_seed` задаёт random/numpy/torch/CUDA seeds и deterministic algorithms.
Повторяемость предполагает одинаковые данные, порядок, версии и оборудование;
побитовое совпадение между CPU, одной GPU и DataParallel не гарантируется.
Для публикации сохраните `pip freeze`, git diff/commit, конфиги и evaluation manifests.
Реальные DER/EER в этом README не приводятся: сначала выполните команды выше.

### Smoke checks

Без внешних данных и GPU можно проверить CLI, notebook и локальные checkpoints:

```bash
python -m pytest -q
python -m compileall -q models train diarize eval experiments utils
python train/run.py --help
python eval/eer.py --help
python eval/der.py --help
python diarize/pipeline.py --help
```

Полный Kaggle smoke run требует подключённых datasets, Internet для ViYT-Diar/Silero
и GPU runtime. Считайте успешным только завершённый notebook с `manifest.json`,
`der_results.json` и `status: complete`; локальный синтетический тест notebook не
является измерением качества и не должен попадать в таблицу DER/EER.

## RTTM и Audacity

```bash
python diarize/rttm_convert.py to_audacity results/audio.rttm results/labels.txt
python diarize/rttm_convert.py to_rttm results/labels.txt results/audio.rttm --file_id audio
```

## Какие измерения нужны для НИР

Для первого запуска DER на Kaggle используйте отдельный
[`kaggle/ViYT_Diar_DER.ipynb`](kaggle/ViYT_Diar_DER.ipynb): подключите обновлённый
dataset проекта `ecapa-tdnn` с `model_0034.model`, включите Internet и запустите все
ячейки (GPU рекомендуется). VLSP для этого notebook не требуется.
Notebook скачивает полный test split [ViYT-Diar](https://huggingface.co/datasets/tuanduy1612/ViYT-Diar),
экспортирует WAV/reference RTTM, выполняет diarization с automatic speaker count
и считает corpus DER через pyannote.metrics (collar 0.25 s, overlap учитывается).
Обучения и удаления моделей нет. Каждый запуск создаёт отдельную папку в
`/kaggle/working`; ZIP включает RTTM, DER JSON/CSV, таблицу corpus summary,
параметры, revision dataset, SHA модели, версии зависимостей и копию кода.
Исходные WAV и веса в ZIP не дублируются. Измеренные результаты появятся только
после выполнения notebook; EER/minDCF на VLSP остаются отдельной verification оценкой.

### ViYT-Diar: oracle number of speakers

В том же notebook доступны `SPEAKER_MODE = 'automatic'` (default) и `'oracle'`.
Для oracle подключите маленький `kaggle/viyt-oracle-code.zip` как дополнительный
Kaggle dataset, оставьте подключённый dataset с моделью 34 и добавьте исходный ZIP
результатов automatic как ещё один dataset. Kaggle распакует его; при нескольких
baseline укажите `BASELINE_RESULTS` — каталог с `manifest.json` и `der_results.json`.
Смените только `SPEAKER_MODE` и выполните Run All в новой сессии.

Revision зафиксирован: `c66b118709d68c7538408d086232d850356db374`, test — ровно 100
записей. Oracle передаёт существующему `diarize(n_speakers=...)` только количество
уникальных reference labels. AHC использует fixed `n_clusters`, без применения
threshold к числу кластеров. VAD, preprocessing, encoder, окна, постобработка и
pyannote scoring не меняются. Нехватка окон или пустой VAD при заданном числе
спикеров останавливает прогон с audio_id и manifest `failed`; неполный corpus не
выдаётся за завершённый. Automatic с пустой речью сохраняет прежнее поведение.

Результаты находятся в новом каталоге запуска, внутри `results/baseline` либо
`results/oracle_num_speakers`. `automatic_vs_oracle.csv/json` содержит измеренные
значения и delta = oracle − automatic (DER/components — процентные пункты).
Перед сравнением сверяются SHA модели, WAV/reference, revision, параметры,
устройство, CUDA, Python и записанные версии численных библиотек; baseline DER
пересчитывается по сохранённым RTTM. Исходный baseline не перезаписывается.
Изменения кода pipeline ограничены диагностикой; остальные прежние модули
inference/scoring проверяются по SHA. Снимок кода и manifest включены в новый ZIP.

`diagnostics.csv/json` показывает число окон, размеры кластеров и counts до/после
постобработки. Сглаживание может убрать короткий кластер, поэтому **итоговая**
speaker-count accuracy oracle не обязана быть 100%; точное число проверяется
сразу после AHC. Raw reference/predicted speech duration — объединение интервалов
до collar, а `scored_reference_speaker_time` — знаменатель DER. `manual_review.csv`
выделяет три подозрительные записи для прослушивания, но не исключает их из corpus
и не объявляет reference неполным. Oracle — диагностическое условие, не результат
полностью автоматической системы. Реальные oracle метрики появятся после запуска.

1. На development corpus выполнить отдельные серии clustering, AHC threshold, VAD,
   window и known/automatic speaker count; сохранить DER/components и count accuracy/MAE.
2. Выбрать параметры, зафиксировать их и получить corpus DER на отдельном test corpus.
3. Сравнить Dolly и Dolly → VLSP checkpoints на одном полном verification protocol:
   таблица model / training data / EER / minDCF. Самостоятельно добавить достоверную
   training provenance; имя checkpoint не доказывает, на каких данных он обучался.
4. Построить loss/accuracy/LR и EER vs epoch из реальных score logs/CSV.
5. Для augmentation сравнить none/noise/rir/full при одинаковых split, бюджете обучения
   и seed; при возможности повторить выбранные эксперименты на нескольких seeds.
