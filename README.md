## Research: Optimization of Oblivious Decision Tree Ensembles Evaluation for CPU

Код является частью работы "Оптимизация применения oblivious решающих деревьев на ЦП"

Код предложенных алгоритмов находится в [catboost/libs/model/cpu_avx](catboost/libs/model/cpu_avx)

Тестовый стенд в [catboost/tools/model_perftest](catboost/tools/model_perftest)

Собрать можно так `./ya make catboost/tools/model_perftest -r`

Для запуска нужно иметь модель и данные. Модель можно взять отсюда [catboost/benchmarks/model_evaluation_speed/epsilon8k_64.bin](catboost/benchmarks/model_evaluation_speed/epsilon8k_64.bin).

Данные можно получить, выполнив
```python3
$ python3
>>> import catboost.datasets
>>> catboost.datasets.epsilon()
```
они будут лежать в `catboost_cached_datasets/epsilon`

Так же данные скачать [отсюда](https://storage.mds.yandex.net/get-devtools-opensource/250854/epsilon.tar.gz) (Ссылка здесь [catboost/python-package/catboost/datasets.py#L212](catboost/python-package/catboost/datasets.py#L212))

Запустить можно так `catboost/tools/model_perftest/model_perftest -f catboost_cached_datasets/epsilon/test.tsv --cd catboost_cached_datasets/epsilon/pool.cd -m catboost/benchmarks/model_evaluation_speed/epsilon8k_64.bin --block-size <размер блока>`
