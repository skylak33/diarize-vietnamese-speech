# Изменения и проверка

Архитектура в `models/` не изменялась. Веса имеющихся моделей не изменялись.
Ни обучения на реальном корпусе, ни реальных EER/DER измерений не проводилось.

## Изменённые файлы

| Файл | Изменение |
|---|---|
| `eval/eer.py` | Универсальный CLI, явный протокол, cached embeddings, JSON provenance |
| `eval/der.py` | pyannote corpus aggregation, отсутствующие гипотезы, overlap tracks, CSV/JSON, speaker counts |
| `diarize/pipeline.py` | Window/step, device/seed, общий loader, пустые RTTM, разделение перекрывающихся окон |
| `diarize/cluster.py` | Seed для spectral, крайние случаи count, one-speaker gate, ограничение max_speakers |
| `train/loader_hf.py` | Детерминированное потребление threaded decode, стабильные labels, augmentation, held-out exclusions |
| `train/loader_vlsp.py` | CLI paths, строгая metadata, mono/resampling, augmentation, held-out exclusions |
| `train/run.py` | Main guard, dataset/device/seed/augmentation CLI, конфигурация запуска, безопасный transfer |
| `train/trainer.py` | CPU/CUDA/DataParallel, общий encoder loader, epoch LR, empty-loader error, speaker mapping sidecar |
| `utils/metrics.py` | ROC с корректной обработкой ties, интерполяция EER, minDCF с крайними рабочими точками |
| `requirements.txt` | Реальные зависимости и диапазоны версий, удалён неиспользуемый torchvision |
| `README.md` | Команды, протокол НИР, ограничения и разделение verification/diarization; неподтверждённые EER удалены |
| `.gitignore` | Сохранена исходная строка пользователя; добавлены среды и Python/test caches |

## Добавленные файлы

| Файл | Назначение |
|---|---|
| `eval/datasets.py` | Local manifests, explicit trials, HF split inspection; без train fallback |
| `experiments/__init__.py` | Пакет экспериментальных скриптов |
| `experiments/run_diarization_experiments.py` | Контролируемые ablations, RTTM/DER/count outputs |
| `experiments/eval_checkpoints.py` | Каждый N-й checkpoint, одинаковый verification protocol |
| `experiments/plot_training.py` | Loss/accuracy/LR/EER PNG без Jupyter |
| `train/augmentation.py` | MUSAN noise/music/babble и RIR для training |
| `utils/audio.py` | Общее декодирование mono 16 kHz |
| `utils/device.py` | Проверяемый выбор auto/cpu/cuda |
| `utils/model.py` | Строгий ECAPA encoder loader для raw/trainer/DataParallel weights |
| `utils/seed.py` | Random/numpy/torch/CUDA seed и deterministic algorithms |
| `utils/provenance.py` | SHA256, версии библиотек, git revision |
| `configs/ecapa_vlsp.yaml` | Рецепт обучения; не автоматическая YAML-конфигурация |
| `configs/diarization_default.yaml` | Рецепт diarization/evaluation |
| `requirements-dev.txt` | Зависимости тестов |
| `pytest.ini` | Поиск tests и импорт проекта |
| `tests/test_metrics.py` | RTTM, corpus weighting, missing hypotheses, overlap/collar, EER/ties |
| `tests/test_pipeline.py` | Seed, segment merging/partition, deterministic clustering |
| `tests/test_protocols.py` | Sampling, leakage checks, ablation plan, training logs |
| `tests/test_augmentation.py` | Noise/music/babble/RIR на синтетических WAV |
| `tests/test_training.py` | Local loader, CPU training step, checkpoint compatibility |
| `tests/test_experiments.py` | Runner outputs с тестовым pipeline, PNG и checkpoint selection |
| `IMPLEMENTATION_NOTES.md` | Этот отчёт |

## Что проверено

- `python -m pytest -q`: **37 passed**. Все данные синтетические, без GPU и сети.
- `--help`: train/run, eval/eer, eval/der, diarize/pipeline, все три experiment CLI
  (runner, checkpoint evaluator, plotting), а также существующий rttm_convert.
- Импорт всех Python-модулей models/train/eval/diarize/utils/experiments.
- `compileall`, `git diff --check`, `pip check`: успешно.
- Имеющийся `exps/exp1/model/model_0034.model` загружается строго на CPU;
  синтетический сигнал даёт конечный L2-normalized 192-dimensional embedding.
- `--experiment all --dry_run`: 15 уникальных однофакторных конфигураций по defaults.
- Локальное окружение проверки: Python 3.14.4, torch/torchaudio 2.9.1 CPU,
  pyannote.metrics 4.1. CUDA и две T4 здесь аппаратно не проверялись.
- В Windows для subprocess CLI-проверок использовался `PYTHONUTF8=1`:
  исходный rttm_convert содержит символ стрелки, не кодируемый в cp1251.

## Ограничения и реальные запуски

В репозитории нет VLSP Public Test metadata/trials и diarization reference corpus.
Публичное описание VLSP не подтверждает конкретное локальное расположение файлов.
После недоступности веб-страницы пользователь предоставил локальный notebook. Все 20
ячеек прочитаны без исполнения. В ячейке 14 найдены Kaggle paths и headerless формат
`wav1,wav2,label`; он поддержан через `--trials_format vlsp`. Сами metadata/trials
и аудио не предоставлены. Точная команда добавлена в README. Сохранённого результата
VLSP-ячейки нет; вывод другого evaluator не переносится в таблицу VLSP.

Команда для VLSP после подстановки реальных путей:

```bash
python eval/eer.py --model_path exps/exp1/model/model_0034.model --dataset vlsp \
  --trials /path/to/public_test_trials.txt --audio_root /path/to/public_test_audio \
  --n_pairs 20000 --seed 42 --device auto --output results/vlsp_eer.json
```

Trials оцениваются целиком; если их формат отличается, используйте явно описанный
CSV интерфейс в README, сохраняя официальный список и метки. Альтернатива при
отсутствии официальных trials — локальный held-out manifest и seeded sampling.
Это разные протоколы, сравнивать их как один нельзя.

Для НИР нужно самостоятельно выполнить:

1. EER/minDCF всех выбранных Dolly и Dolly→VLSP checkpoints на одном протоколе.
2. Отдельные ablations clustering/AHC threshold/VAD/window/speaker_count на development.
3. Итоговый corpus DER и components на независимом diarization test с фиксированными параметрами.
4. Speaker Count Accuracy/MAE из тех же результатов, отдельно для automatic count.
5. PNG training curves и EER-vs-epoch из реальных logs/CSV.
6. При исследовании augmentation — сопоставимые training runs none/noise/rir/full,
   с исключёнными held-out speakers и одинаковым бюджетом; несколько seeds по возможности.

Продолжение обучения остаётся weight-only: optimizer state не сохраняется. Это явно
документировано; точный resume не заявляется. Legacy Dolly classifier без speaker mapping
не восстанавливается молча. Настройки YAML пока служат рецептами CLI.

## Дополнение после изучения notebook

Изменены `eval/datasets.py`, `eval/eer.py`, `tests/test_protocols.py`, `README.md`
и этот отчёт. Добавлен строгий VLSP CSV adapter для .txt без заголовка.
Поддержка доступна также в `experiments/eval_checkpoints.py` через общий CLI parser.
Notebook пользователя не изменялся; его ячейки копирования, обучения и удаления файлов
не исполнялись. Новые тесты проверяют полный protocol, malformed rows и отсутствующее аудио.

После дополнения: 37 CPU tests passed; --help EER и checkpoint evaluator содержит
--trials_format; git diff --check прошёл. Реальная VLSP evaluation не запускалась.
