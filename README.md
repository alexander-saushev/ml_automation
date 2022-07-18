# Автоматизация обучения моделей

В этом репозитории собраны Spark-задачи для автоматизации обучения моделей: от выгрузки сырых данных из базы данных до определения лучшей модели. Предполагается, что джобы запускаются на Spark-кластере раз в неделю по расписанию.

Ниже представлены описания задач. Посмотреть, как работает каждый шаг конкретной задачи, можно в юпитер-ноутбуках, которые находятся в [notebooks](https://github.com/alexander-saushev/ml_automation/tree/main/notebooks).

### parse_clickhouse_job

Задача выгружает данные из «Кликхауса» за заданный временной период и сохраняет полученную таблицу в parquet-файл.

Аргументы — даты первого и последнего дней временного периода в формате строки.  
**Пример:** `'2022-07-11' '2022-07-17'`

Папка с parquet-файлами будет сохранена в `raw_data/*дата последнего дня*`. Для примера — в `raw_data/2022-07-17`.

### prepare_training_data_job

Задача забирает выгруженные из «Кликхауса» сырые данные и готовит их для обучения моделей:
- рассчитывает новые фичи,
- делит на тестовую и обучающую выборки.

Джоба знает, что сырые данные хранятся в папке `raw_data`, и принимает на вход один аргумент — дату последнего дня одного из интервалов.  
**Пример:** `'2022-07-17'`

Обработанные данные сохраняются в `data/*переданная в задачу дата*`. Для примера — в `data/2022-07-17`. В этой папке находятся и тренировочная, и тестовая выборки — в папках `train` и `test` соответственно.

### train_models_job

Задача обучает и тестирует три вида моделей и сохраняет лучшую по RMSE.

Данные для обучения всегда забираются из папки `data`, поэтому на вход задаче подается один аргумент — дата последнего дня одного из интервалов, описанных выше.

Лучшая модель сохраняется в `best_models/*переданная дата*`.
