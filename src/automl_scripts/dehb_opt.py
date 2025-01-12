import time
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer

# Example documentation to optimize RamdomForest
seed = 123
np.random.seed(seed)
warnings.filterwarnings('ignore')

min_fidelity, max_fidelity = 2, 50

import ConfigSpace as CS


def create_search_space(seed=123):
    """Parameter space to be optimized --- contains the hyperparameters
    """
    cs = CS.ConfigurationSpace(seed=seed)

    cs.add_hyperparameters([
        CS.UniformIntegerHyperparameter(
            'max_depth', lower=1, upper=15, default_value=2, log=False
        ),
        CS.UniformIntegerHyperparameter(
            'min_samples_split', lower=2, upper=128, default_value=2, log=True
        ),
        CS.UniformFloatHyperparameter(
            'max_features', lower=0.1, upper=0.9, default_value=0.5, log=False
        ),
        CS.UniformIntegerHyperparameter(
            'min_samples_leaf', lower=1, upper=64, default_value=1, log=True
        ),
    ])
    return cs

cs = create_search_space(seed)
print(cs)

dimensions = len(cs.get_hyperparameters())
print("Dimensionality of search space: {}".format(dimensions))

from sklearn.model_selection import train_test_split


def prepare_dataset(model_type="classification", dataset=None):
    if model_type == "classification":
        if dataset is None:
            dataset = np.random.choice(list(classification.keys()))
        _data = classification[dataset]()
    else:
        if dataset is None:
            dataset = np.random.choice(list(regression.keys()))
        _data = regression[dataset]()

    train_X, rest_X, train_y, rest_y = train_test_split(
        _data.get("data"),
        _data.get("target"),
        train_size=0.7,
        shuffle=True,
        random_state=seed
    )

    # 10% test and 20% validation data
    valid_X, test_X, valid_y, test_y = train_test_split(
        rest_X, rest_y,
        test_size=0.3333,
        shuffle=True,
        random_state=seed
    )
    return train_X, train_y, valid_X, valid_y, test_X, test_y, dataset

train_X, train_y, valid_X, valid_y, test_X, test_y, dataset = \
    prepare_dataset(model_type="classification")

print(dataset)
print("Train size: {}\nValid size: {}\nTest size: {}".format(
    train_X.shape, valid_X.shape, test_X.shape
))




accuracy_scorer = make_scorer(accuracy_score)


def target_function(config, fidelity, **kwargs):
    # Extracting support information
    seed = kwargs["seed"]
    train_X = kwargs["train_X"]
    train_y = kwargs["train_y"]
    valid_X = kwargs["valid_X"]
    valid_y = kwargs["valid_y"]
    max_fidelity = kwargs["max_fidelity"]

    if fidelity is None:
        fidelity = max_fidelity

    start = time.time()
    # Building model
    model = RandomForestClassifier(
        **config.get_dictionary(),
        n_estimators=int(fidelity),
        bootstrap=True,
        random_state=seed,
    )
    # Training the model on the complete training set
    model.fit(train_X, train_y)

    # Evaluating the model on the validation set
    valid_accuracy = accuracy_scorer(model, valid_X, valid_y)
    cost = time.time() - start

    # Evaluating the model on the test set as additional info
    test_accuracy = accuracy_scorer(model, test_X, test_y)

    result = {
        "fitness": -valid_accuracy,  # DE/DEHB minimizes
        "cost": cost,
        "info": {
            "test_score": test_accuracy,
            "fidelity": fidelity
        }
    }
    return result

from dehb import DEHB

dehb = DEHB(
    f=target_function,
    cs=cs,
    dimensions=dimensions,
    min_fidelity=min_fidelity,
    max_fidelity=max_fidelity,
    n_workers=1,
    output_path="./temp"
)

trajectory, runtime, history = dehb.run(
    total_cost=10,
    # parameters expected as **kwargs in target_function is passed here
    seed=123,
    train_X=train_X,
    train_y=train_y,
    valid_X=valid_X,
    valid_y=valid_y,
    max_fidelity=dehb.max_fidelity
)

print(len(trajectory), len(runtime), len(history), end="\n\n")

# Last recorded function evaluation
last_eval = history[-1]
config_id, config, score, cost, fidelity, _info = last_eval

print("Last evaluated configuration, ")
print(dehb.vector_to_configspace(config), end="")
print("got a score of {}, was evaluated at a fidelity of {:.2f} and "
      "took {:.3f} seconds to run.".format(score, fidelity, cost))
print("The additional info attached: {}".format(_info))

runs = 5

best_config_list = []

for i in range(runs):
    # Resetting to begin optimization again
    dehb.reset()
    # Executing a run of DEHB optimization lasting for 10s
    trajectory, runtime, history = dehb.run(
        total_cost=10,
        seed=123,
        train_X=train_X,
        train_y=train_y,
        valid_X=valid_X,
        valid_y=valid_y,
        max_fidelity=dehb.max_fidelity
    )
    best_config = dehb.vector_to_configspace(dehb.inc_config)

    # Creating a model using the best configuration found
    model = RandomForestClassifier(
        **best_config.get_dictionary(),
        n_estimators=int(max_fidelity),
        bootstrap=True,
        random_state=seed,
    )
    # Training the model on the complete training set
    model.fit(
        np.concatenate((train_X, valid_X)),
        np.concatenate((train_y, valid_y))
    )
    # Evaluating the model on the held-out test set
    test_accuracy = accuracy_scorer(model, test_X, test_y)
    best_config_list.append((best_config, test_accuracy))

print("Mean score across trials: ", np.mean([score for _, score in best_config_list]))
print("Std. dev. of score across trials: ", np.std([score for _, score in best_config_list]))

for config, score in best_config_list:
    print("{} got an accuracy of {} on the test set.".format(config, score))
    print()
