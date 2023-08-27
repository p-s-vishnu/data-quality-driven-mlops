from collections import OrderedDict
from tensorflow_datasets import Split

from snoopy.embedding import EmbeddingConfig, uni_se
from snoopy.reader import TFDSTextConfig

dataset_name = "imdb_reviews"
train_data_config = TFDSTextConfig(dataset_name=dataset_name, split=Split.TRAIN)
test_data_config = TFDSTextConfig(dataset_name=dataset_name, split=Split.TEST)
classes = 2

models = OrderedDict({
    "use": EmbeddingConfig(uni_se, batch_size=10, prefetch_size=1)
}) 

target = 0.1

import numpy as np
import os
import os.path as path
import torch as pt

from snoopy import set_cache_dir
from snoopy.pipeline import run, store_embeddings
from snoopy.result import BerStoringObserver
from snoopy.strategy import SimpleStrategyConfig
from snoopy.reader import ReaderConfig, data_factory

set_cache_dir("cache")

results_folder = "results"
if not path.exists(results_folder):
    os.mkdir(results_folder)

test_data = data_factory(test_data_config)
test_size = float(test_data.size)

observer = BerStoringObserver(classes, test_size)

run(train_data_config=train_data_config,
    test_data_config=test_data_config,
    embedding_configs=models,
    strategy_config=SimpleStrategyConfig(train_size=1000, test_size=5_000),
    observer=observer,
    device=pt.device("cpu"))

observer.store(results_folder)


# Set the random baseline as min achievable error first, then iterate over all models
min_error = (classes - 1.0) / float(classes)
for k in models.keys():
    f = path.join(results_folder, "{0}.npz".format(k))
    if path.exists(f):
        items = np.load(f)
        error = items['ber'][-1]
        min_error = min(min_error, error)

print('Minimal error achievable is {0:.4f}'.format(min_error))
print('Max score:{0:.4f}'.format(1-min_error) )
print('Target Achievable: ', target > min_error)
