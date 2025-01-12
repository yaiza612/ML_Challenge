import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from pathlib import Path

print(os.getcwd())
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
# load targets
all_targets = []
for i in range(4):
    all_targets.append(np.load(f"{Path(dir_path)}/../../data/train/target_{i}.npy"))

merged_targets = np.hstack(all_targets)
# make histogram of class distributions
def make_hist_of_class_dist(target, name):
    num_occurences = []
    labels = []
    for pos_idx in range(5):
        for val in range(2):
            label = ["x", "x", "x", "x", "x"]
            label[pos_idx] = str(val)
            label = ''.join(label)
            labels.append(label)
            num_occurences.append(
                val * np.sum(target[pos_idx, :]) + (1 - val) * (target.shape[1] - np.sum(target[pos_idx, :])))

    plt.bar(range(10), height=num_occurences, label=labels)
    ax = plt.gca()
    ax.set_xticks(range(len(labels)), labels)
    plt.savefig(f"{Path(dir_path)}/../../data_visualizations/label_distribution/label_distribution_for_target_{name}.png")
    plt.close()


for target_idx, target in enumerate(all_targets):
    make_hist_of_class_dist(target=target, name=target_idx)


make_hist_of_class_dist(target=merged_targets, name="merged")

def get_all_occurences(array):
    # The window size (in this case 5)
    window_size = 5

    # Initialize a dictionary to store counts for different combinations
    occurrences_dict = {}

    # Iterate through the number of constraints (from 1 to 5)
    for num_constraints in range(1, window_size + 1):

        # Get all combinations of positions where constraints could be applied (e.g., (0,), (1,), (0,1), ..., (0,1,2,3,4))
        position_combinations = list(itertools.combinations(range(window_size), num_constraints))

        # For each combination of positions, try all combinations of values (e.g., 0 or 1 at each position)
        for positions in position_combinations:

            # Get all possible value combinations for the selected positions (e.g., for (0,1), it would be (0,0), (0,1), (1,0), (1,1))
            value_combinations = list(itertools.product([0, 1], repeat=len(positions)))

            # Check each value combination for the current set of positions
            for values in value_combinations:
                # Create a mask for the current constraint
                constraint = np.full(window_size, None)

                for pos, val in zip(positions, values):
                    constraint[pos] = val

                # Initialize a counter for this specific combination of constraints
                count_for_constraint = 0

                # Now apply the sliding window and check for this combination of constraints
                for row in array.T:
                    # Create a boolean mask that will check only the constrained positions
                    matches = True
                    for pos, val in zip(positions, values):
                        if row[pos] != val:
                            matches = False
                            break

                    if matches:
                        count_for_constraint += 1

                # Store the count for this constraint combination
                occurrences_dict[(positions, values)] = count_for_constraint

    # Print the result for all constraints
    for key, counts in occurrences_dict.items():
        print(f"Positions: {key[0]}, Values: {key[1]}, Occurrences per row: {counts}")
    return occurrences_dict


def calculate_conditional_probabilities(occurence_dict):
    prior_positions = [0, 1, 2, 3, 4]
    prior_values = [0, 1]

    posterior_positions = [0, 1, 2, 3, 4]
    posterior_values = [0, 1]

    for prior_position in prior_positions:
        for prior_value in prior_values:
            for posterior_position in posterior_positions:
                for posterior_value in posterior_values:
                    if posterior_position < prior_position:
                        merged_position = (posterior_position, prior_position)
                        merged_value = (posterior_value, prior_value)
                    elif prior_position < posterior_position:
                        merged_position = (prior_position, posterior_position)
                        merged_value = (prior_value, posterior_value)
                    else:
                        break
                    print(f"The probability that given a {prior_value} is in position {prior_position}, "
                          f"the value at position {posterior_position} will be {posterior_value} is "
                          f"{occurence_dict[(merged_position, merged_value)] / occurence_dict[((prior_position,), (prior_value,))]}")
                    # TODO compare the conditional probability to the base probability

occurence_dict = get_all_occurences(merged_targets)
calculate_conditional_probabilities(occurence_dict)



