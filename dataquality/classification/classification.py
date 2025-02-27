from json import dump, load as load_json
from logging import info as logging_info
from pathlib import Path
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset_manager import DatasetManager
from classification.experiments import LogRegExperiment, KNeighborsExperiment, DecisionTreeExperiment, \
    MultilayerPerceptronExperiment, SupportVectorMachineExperiment
from polluters import TargetAccuracyPolluter, CompletenessPolluter, FeatureAccuracyPolluter
from util import start_logging

DATA_DIR = Path('data/')
CLEAN_DATA_DIR = DATA_DIR / 'clean/'
POLLUTED_DATA_DIR = DATA_DIR / 'polluted/'
RESULTS_JSON_PATH = 'results.json'
CLASSIFICATION_DATASETS = ['TelcoCustomerChurn.csv']


def main():
    # Polluter classes
    pollution_methods = [CompletenessPolluter, FeatureAccuracyPolluter]
    # Experiment classes
    experiments = [LogRegExperiment, KNeighborsExperiment, DecisionTreeExperiment, MultilayerPerceptronExperiment,
                   SupportVectorMachineExperiment]

    if not POLLUTED_DATA_DIR.exists():
        POLLUTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Read dataset metadata file
    with open('metadata.json', 'r') as f:
        metadata = load_json(f)

    # Iterate over datasets to read, pollute and evaluate them
    for ds_name in tqdm(CLASSIFICATION_DATASETS):
        # Check whether dataset is defined in metadata
        if ds_name not in metadata.keys():
            raise KeyError(
                f'Dataset {ds_name} is not specified in the metadata file as keys are {metadata.keys()}.')
        df = read_csv(CLEAN_DATA_DIR / ds_name)

        # Make a stratified train test split and reset indices
        test_ratio = 0.2
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=1,
                                             stratify=df[metadata[ds_name]['target']])
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df.to_csv(CLEAN_DATA_DIR / 'train_clean.csv', index=False)
        test_df.to_csv(CLEAN_DATA_DIR / 'test_clean.csv', index=False)
        
        # Run experiments with polluted datasets
        for polluter_class in tqdm(pollution_methods, leave=False):
            polluters = polluter_class.configure(metadata, df, ds_name)
            for idx, polluter in enumerate(tqdm(polluters, leave=False)):
                logging_info(
                    f'Polluting dataset {ds_name} with {polluter.__class__.__name__}')

                train_df_polluted, train_df_quality = DatasetManager.read_or_pollute(
                    POLLUTED_DATA_DIR, train_df, 'train_' + ds_name, polluter)
                test_df_polluted, test_df_quality = DatasetManager.read_or_pollute(
                    POLLUTED_DATA_DIR, test_df, 'test_' + ds_name, polluter)

                DatasetManager.persist_dataset(
                    POLLUTED_DATA_DIR, train_df_polluted, 'train_' + ds_name, polluter)
                DatasetManager.persist_dataset(
                    POLLUTED_DATA_DIR, test_df_polluted, 'test_' + ds_name, polluter)

                logging_info(f'Pollution finished, logging results...')
                logging_info(f'Pollution parameters: {polluter.get_pollution_params()}')
                logging_info(f'Quality Train: {train_df_quality}\nQuality Test: {test_df_quality}')

                scenarios = {
                    # 'train_clean_test_clean': (train_df, test_df),
                    # 'train_clean_test_polluted': (train_df, test_df_polluted),
                    # 'train_polluted_test_polluted': (train_df_polluted, test_df_polluted),
                    'train_polluted_test_clean': (train_df_polluted, test_df)
                }

                # Open an existing results.json if it exists, otherwise create a new dictionary
                try:
                    with open('classification_results.json', 'r') as f:
                        existing_results = load_json(f)
                except OSError:
                    existing_results = {}

                # Initialize results dictionary:
                # The results will be saved in a hierarchical structure
                #   |- %Dataset Name%
                #       |- %Polluter Class Name%
                #           | - %Pollution parameter dictionary (represented as string to be able to be used as key)%
                #               | - quality: %Quality Measure%
                #               | - %Scenario Name%
                #                   | - %Experiment Name%
                #                       | - %Experiment results%
                if existing_results.get(ds_name) is None:
                    existing_results[ds_name] = dict()
                if existing_results[ds_name].get(polluter.__class__.__name__) is None:
                    existing_results[ds_name][polluter.__class__.__name__] = dict(
                    )
                if existing_results[ds_name][polluter.__class__.__name__].get(polluter.get_pollution_params()) is None:
                    existing_results[ds_name][polluter.__class__.__name__][polluter.get_pollution_params()] = {
                        'quality': {
                            "train": test_df_quality,
                            "test": train_df_quality
                        },
                    }

                for scenario_name, scenario_datasets in tqdm(scenarios.items(), leave=False):
                    df_train, df_test = scenario_datasets
                    # scenario_name, polluter.pollution_percentages
                    # Experiment loop implementation for the classification experiments
                    # write the pollution
                    if "pollution_percentages" in polluter.get_pollution_params():
                        pollution_pct = str(polluter.pollution_percentages)
                        fname = '_'.join([scenario_name, pollution_pct])+'.csv'
                    else:
                        pollution_pct = str(idx/(len(polluters)-1))
                        fname = '_'.join([scenario_name, pollution_pct])+'.csv'
                    fpath = POLLUTED_DATA_DIR  / polluter.__class__.__name__
                    pd.concat((df_train, df_test)).to_csv(fpath / fname, index=False)
                    exp_predictions = []
                    exp_columns = []
                    for experiment in tqdm(experiments, leave=False):
                        logging_info(
                            f'Starting experiment {experiment} for scenario {scenario_name} and dataset {ds_name} with '
                            f'{polluter.__class__.__name__}')

                        # Instantiate Experiment - need to know categorical features and target feature for encoding and
                        # data split, therefore we just pass the metadata corresponding to the current dataset
                        exp = experiment(df_train, df_test, metadata[ds_name])
                        # Results are metrics (e.g. accuracy) returned as a dictionary
                        results = exp.run()
                        if 'predictions' in results and any(pred := results.get('predictions')):
                            exp_predictions.append(pred.tolist())
                            exp_columns.append(str(exp.name))
                            del results['predictions']

                        # Update the results dictionary with our current experiment
                        if existing_results[ds_name][polluter.__class__.__name__][polluter.get_pollution_params()]\
                                .get(scenario_name) is None:
                            existing_results[ds_name][polluter.__class__.__name__][polluter.get_pollution_params()][
                                scenario_name] = {}

                        existing_results[ds_name][polluter.__class__.__name__][polluter.get_pollution_params()][
                            scenario_name].update(results)

                        # Persist existing results dictionary after each experiment, in case something fails
                        with open('classification_results.json', 'w') as f:
                            dump(existing_results, f, indent=4, sort_keys=True)

                        logging_info(f'{exp.name} results: {results}')
                    pd.DataFrame(dict(zip(exp_columns, exp_predictions))).to_csv(str(fpath / f'predictions{pollution_pct}.csv'), index=False)


if __name__ == "__main__":
    start_logging(cmd_out=True)
    main()
