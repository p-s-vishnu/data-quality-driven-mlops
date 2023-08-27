from snoopy import set_cache_dir
from snoopy.pipeline import run
from snoopy.result import BerStoringObserver, f1score
from snoopy.strategy import SimpleStrategyConfig
from snoopy.embedding import EmbeddingConfig
from snoopy.embedding.tabular_embedding import TabularEmbeddingModelSpec
from snoopy.reader.tabular import TabularFileConfig, TabularFileReader # Importing the new classes
from collections import OrderedDict
import os
import numpy as np
import torch as pt


classes = 2
path_to_train_csv = 'data/train.csv'
path_to_test_csv = 'data/test.csv'
label_column_number = 19
num_columns = 20
label_values = ['No', 'Yes']
shuffle_buffer_size = 1000
input_dimension = num_columns - 1 # Assuming one column is for labels
output_dimension = 5 # You can specify the desired output dimension for the embedding


tabular_embedding_spec = TabularEmbeddingModelSpec(input_dimension=input_dimension, output_dimension=output_dimension)

models = OrderedDict({
    "tabular_embedding": EmbeddingConfig(tabular_embedding_spec, batch_size=10, prefetch_size=1)
})

target = 0.1    # 1 - acc

set_cache_dir("cache")

results_folder = "results"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

# Using TabularFileConfig and TabularFileReader for reading the tabular data
print("Present working dir:", os.getcwd())
train_data_config = TabularFileConfig(path=path_to_train_csv, header_present=True, label_column_number=label_column_number,
                                       num_columns=num_columns, label_values=label_values, shuffle_buffer_size=shuffle_buffer_size)

test_data_config = TabularFileConfig(path=path_to_test_csv, header_present=True, label_column_number=label_column_number,
                                      num_columns=num_columns, label_values=label_values, shuffle_buffer_size=shuffle_buffer_size)


observer = BerStoringObserver(classes, test_size=1407) # Adjust test_size if needed

run(train_data_config=train_data_config,
    test_data_config=test_data_config,
    embedding_configs=models,
    strategy_config=SimpleStrategyConfig(train_size=5625, test_size=1407),
    observer=observer, 
    device=pt.device("cpu")) # Change to "gpu" if using GPU

observer.store(results_folder)


results_folder = "results"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

min_error = (classes - 1.0) / float(classes)
for k in models.keys():
    f = os.path.join(results_folder, "{0}.npz".format(k))
    if os.path.exists(f):
        items = np.load(f)
        error = items['ber'][-1]
        min_error = min(min_error, error)

print('Minimal error achievable is {0:.4f}'.format(min_error))
print('Max score:{0:.4f}'.format(f1score(min_error)))
print('Target Achievable: ', target > min_error)
