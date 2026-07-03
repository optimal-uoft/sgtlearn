import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.tree import DecisionTreeClassifier
import gc
import numpy as np
import argparse
import os
import json
import time
import signal

from sklearn.metrics import accuracy_score
from src.data_utils import *

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def sample_hyperparameters(rng):
    """
    Sample hyperparameters randomly matching the previous Optuna search space.

    Args:
        rng: numpy random number generator with seed set

    Returns:
        dict: Dictionary of sampled hyperparameters
    """
    # Categorical choices
    criterion = rng.choice(['gini', 'entropy'])

    # Integer sampling for min_samples_split (log scale: 2^1 to 2^5)
    min_samples_split_exp = rng.integers(1, 6)  # 1 to 5 inclusive
    min_samples_split = 2 ** min_samples_split_exp

    # Integer sampling for min_samples_leaf (1 to 32)
    min_samples_leaf = rng.integers(1, 33)  # 1 to 32 inclusive

    # Categorical choices for min_impurity_decrease
    min_impurity_decrease = rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 0.01])

    # Categorical choices for ccp_alpha
    ccp_alpha = rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    depth = int(rng.choice([2, 3, 4, 5, 6]))

    return {
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'min_impurity_decrease': min_impurity_decrease,
        'ccp_alpha': ccp_alpha,
        'max_depth': depth,
    }

def prepare_and_save_results(results, args):
    """
    Prepare all data for serialization and save to JSON file.

    Args:
        results: Dictionary of results (raw values)
        args: Command line arguments
    """

    # Extract integer/boolean fields before float serialization
    actual_depth = results.pop('actual_depth')
    n_leaves = results.pop('n_leaves')
    failed = results.pop('failed')

    # Convert results to strings
    serialized_results = {}
    for key, value in results.items():
        serialized_results[key] = f"{value:.6f}"

    # Build output dictionary
    output = {
        'model': 'CART',
        'trial_id': str(args.trial_id),
        'fold': str(args.fold),
        'dataset': args.dataset,
        'random_seed': str(args.random_seed),
        'base_seed': str(args.base_seed),
        'max_depth': str(args.depth) if args.depth != -1 else '',
        'actual_depth': str(actual_depth),
        'n_leaves': str(n_leaves),
        'failed': failed,
    }
    output.update(serialized_results)

    # Create output directory structure
    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)

    # Save results to JSON file
    output_file = os.path.join(output_path, f'cart_results_trial_{args.dataset}_{args.trial_id}_{args.fold}.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CART Hyperparameter Trial Runner')
    parser.add_argument('--dataset', type=str, default='room', help='Data name. Default is room')
    parser.add_argument('--trial-id', type=int, required=True, help='Trial ID (typically from SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--base-seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--timeout', type=int, default=15*60, help='Timeout for each trial in seconds')
    parser.add_argument('--home-dir', type=str, default='./', help='Output directory for results')
    parser.add_argument('--fold', type=int, default=0, help='Data fold to use, default is 0')
    parser.add_argument('--min_time', type = int, default=3, help='Minimum time for each trial in seconds')
    args = parser.parse_args()

    args.output_dir = os.path.join(args.home_dir, 'results')
    t0 = time.time()

    # Set random seed based on trial ID
    trial_seed = args.base_seed + args.trial_id
    args.random_seed = trial_seed
    rng = np.random.default_rng(trial_seed)

    # Load data
    data_factory = DataFactory_clf(args.dataset, cache_dir= os.path.join(args.home_dir, 'data'))

    hyperparams = sample_hyperparameters(rng)
    args.depth = hyperparams['max_depth']
    X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(args.fold)

    # Save dummy results before running the model (in case of OOM/crash)
    dummy_clf = DecisionTreeClassifier(max_depth=1, random_state=args.random_seed)
    dummy_clf.fit(X_train, y_train)
    dummy_train_acc  = accuracy_score(y_train, dummy_clf.predict(X_train))
    dummy_val_acc    = accuracy_score(y_val,   dummy_clf.predict(X_val))
    dummy_test_acc   = accuracy_score(y_test,  dummy_clf.predict(X_test))
    dummy_depth      = dummy_clf.get_depth()
    dummy_n_leaves   = dummy_clf.get_n_leaves()
    dummy_results = {
        'train_acc': dummy_train_acc,
        'val_acc':   dummy_val_acc,
        'test_acc':  dummy_test_acc,
        'elapsed_time': 0.0,
        'actual_depth': dummy_depth,
        'n_leaves':     dummy_n_leaves,
        'failed': True,
    }
    prepare_and_save_results(dummy_results, args)

    clf = DecisionTreeClassifier(
        max_depth=hyperparams['max_depth'],
        criterion=hyperparams['criterion'],
        min_samples_split=hyperparams['min_samples_split'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        min_impurity_decrease=hyperparams['min_impurity_decrease'],
        random_state=args.random_seed,
        ccp_alpha=hyperparams['ccp_alpha']
    )

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(args.timeout)
    try:
        t1 = time.time()
        clf.fit(X_train, y_train)
        signal.alarm(0)
        t2 = time.time()
        elapsed_time = t2 - t1

        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        actual_depth = clf.get_depth()
        n_leaves = clf.get_n_leaves()
        failed = False

    except TimeoutException:
        elapsed_time = args.timeout
        train_acc = dummy_train_acc
        val_acc = dummy_val_acc
        test_acc = dummy_test_acc
        actual_depth = dummy_depth
        n_leaves = dummy_n_leaves
        failed = True

    results = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'elapsed_time': elapsed_time,
        'actual_depth': actual_depth,
        'n_leaves': n_leaves,
        'failed': failed,
    }

    # # Prepare and save results
    prepare_and_save_results(results, args)
    t3 = time.time()
    final_elapsed = t3 - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed) # <-- to ensure minimum time for each trial
