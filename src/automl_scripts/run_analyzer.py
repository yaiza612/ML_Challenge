import numpy as np



def analyze_run(dehb_obj, use_top_fidelities_only):
    hist = dehb_obj.history  # list of tuples (33 elements); the last element is the info dict which includes the kappas for each dataset
    runtimes = dehb_obj.runtime  # list of runtimes in seconds (33 elements)
    fidelities = [elem[4] for elem in hist]
    maximum_fidelity = np.max(fidelities)
    if use_top_fidelities_only:
        top_fidelity_indices = [i for i in range(len(fidelities)) if fidelities[i] >= (maximum_fidelity-1)]
    else:
        top_fidelity_indices = [i for i in range(len(fidelities))]
    def get_config_readable(data_index):
        return dehb_obj.vector_to_configspace(np.array(hist[data_index][1]))


    def get_pareto_front():
        def get_pareto_individual(data_index):
            translate_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: "all"}
            index_of_best_individual = None
            performance_of_best_individual = -1
            for index, res in enumerate(hist):
                if index in top_fidelity_indices:
                    perf_dict = res[5]
                    perf_of_ind = perf_dict[f"data_{translate_dict[data_index]}_score"]
                    if perf_of_ind is not None:
                        if perf_of_ind > performance_of_best_individual:
                            performance_of_best_individual = perf_of_ind
                            index_of_best_individual = index
            return index_of_best_individual
        return list(set([get_pareto_individual(_) for _ in range(5)]))

    def get_top_fitnesses(k):
        def calculate_fitness(perf_dict, index):
            translate_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: "all"}
            perf_list = [perf_dict[f"data_{translate_dict[data_index]}_score"] for data_index in range(5)]
            if None in perf_list or index not in top_fidelity_indices:
                return 5
            else:
                fitness = float(np.linalg.norm(1 - np.array(perf_list)))
            return fitness
        fitness_list = [calculate_fitness(res[5], i) for i, res in enumerate(hist)]
        return list(np.argsort(fitness_list)[:k]), fitness_list
    pareto_front = get_pareto_front()
    print(pareto_front)
    top_fitnesses, fitness_list = get_top_fitnesses(5)
    print(fitness_list)
    print(top_fitnesses)  # first element is best fitness
    overall_best = list(set(pareto_front) | set(top_fitnesses))
    print(overall_best)
    for elem in overall_best:
        print(f"fitness: {fitness_list[elem]} with runtime {runtimes[elem]} and fidelity {fidelities[elem]}")
    for elem in overall_best:
        print(f"{get_config_readable(elem).get_dictionary()}")