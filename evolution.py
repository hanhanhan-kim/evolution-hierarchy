import concurrent.futures

import numpy as np

from layered_system import LayeredSystem, uniform_sphere_gaussian


def compute_cosine_similarity_subset(layer_outputs, optimal_output, dims):
    """Compute cosine similarity on a subset of dimensions for each vector."""
    return np.array(
        [
            np.dot(vec[dims], optimal_output[dims])
            / (np.linalg.norm(vec[dims]) * np.linalg.norm(optimal_output[dims]) + 1e-8)
            for vec in layer_outputs
        ]
    )


def similarity_to_selection_probs(
    cosine_similarities,
    nonlinear_fitness=False,
    temperature=0.7,
    epsilon=1e-6,
    omega=1.0,
):
    # Shift similarities to [0, 1]
    shifted = (np.array(cosine_similarities) + 1) / 2
    # Ensure non-zero entries
    adjusted = shifted + epsilon
    # Nonlinear fitness function
    if nonlinear_fitness:
        # Cosine modulation with max at x=1, never negative
        mod = (np.cos(omega * (1 - adjusted)) + 1) / 2  # ∈ [0,1], =1 at x≈1
        adjusted *= mod
    # Compute softmax
    # scaled = adjusted / temperature
    # adjusted = np.exp(scaled - np.max(scaled))  # for numerical stability
    return adjusted


def mutate_population_subset(population, mutation_std, mutation_rate):
    """Mutate a subset of components in the population vectors (fast, probabilistic version)."""
    num_individuals, dim = population.shape

    mutation_mask = np.random.rand(num_individuals, dim) < mutation_rate
    mutations = np.random.normal(
        loc=0.0, scale=mutation_std / np.sqrt(dim), size=(num_individuals, dim)
    )
    population[mutation_mask] += mutations[mutation_mask]
    return population


def run_evolution(
    n_generations=10,
    population_size=1000,
    mutation_std=0.1,
    eval_fraction=1,
    mutation_rate=1,
    dim=20,
    use_sigmoid=False,
    rank_fraction=1,
    max_depth=20,
    normalize=False,
    seed=None,
    nonlinear_fitness=False,
    omega=1.0,
    fix_initial_pop_distance=False,
):
    """Run evolutionary simulation"""
    if seed is not None:
        np.random.seed(seed)

    # Initialize layeredSystem
    system = LayeredSystem(
        dim=dim,
        max_depth=max_depth,
        rank_fraction=rank_fraction,
        use_sigmoid=use_sigmoid,
    )

    # Generate optimal direction
    optimal_direction = uniform_sphere_gaussian(1, dim=system.dim)[0]

    # Compute optimal outputs at each layer using forward pass
    optimal_outputs = system._forward_pass(optimal_direction.reshape(1, -1))

    # Initialize with random population
    y_star = optimal_outputs[-1].reshape(-1)
    y_star_col = y_star.reshape(-1, 1)
    if fix_initial_pop_distance:
        # generate a population of size population_size, where the cosine similarity between each vector and the optimal direction is zero at the final layer
        I = np.eye(system.dim)
        all_outputs = system._forward_pass(I)
        A = all_outputs[-1]
        v = A @ y_star_col
        v = v.reshape(-1)
        population = sample_input_orthogonal_to_v(v, num_samples=population_size)

        # normalize population
        population = population / np.linalg.norm(population, axis=1, keepdims=True)
    else:
        population = uniform_sphere_gaussian(population_size, dim=system.dim)

    # check orthogonality of population
    outputs = system._forward_pass(population)
    dots = outputs[-1] @ y_star
    print("Max abs dot product:", np.max(np.abs(dots)))

    # Initialize ancestry
    ancestry = np.eye(population_size)

    # Select subset of dimensions for fitness evaluation
    eval_dims = np.random.choice(
        system.dim, size=int(system.dim * eval_fraction), replace=False
    )

    # Track mean and standard deviation at each layer
    layer_stats = {
        "mean": np.zeros((system.max_depth + 1, n_generations)),
        "stdev": np.zeros((system.max_depth + 1, n_generations)),
        "ancestry_proportions": np.zeros((population_size, n_generations)),
        "fitness": np.zeros((population_size, n_generations)),
    }

    for gen in range(n_generations):
        # Forward pass through all layers
        layer_outputs = system._forward_pass(population)

        # Track statistics at each layer
        for layer_idx, layer_out in enumerate(layer_outputs):
            angles = system._compute_cossim(layer_out, optimal_outputs[layer_idx][0])
            layer_stats["mean"][layer_idx, gen] = np.mean(angles)
            layer_stats["stdev"][layer_idx, gen] = np.std(angles)
            layer_stats["ancestry_proportions"][:, gen] = np.mean(
                ancestry, axis=0
            )  # Track ancestry proportions

        # Selection
        cossim = compute_cosine_similarity_subset(
            layer_outputs[-1], optimal_outputs[-1][0], eval_dims
        )
        fitness = similarity_to_selection_probs(
            cossim, nonlinear_fitness=nonlinear_fitness, omega=omega
        )
        layer_stats["fitness"][:, gen] = fitness  # Store fitness

        parent_indices = np.random.choice(
            population_size, size=population_size, p=fitness / np.sum(fitness)
        )
        offspring = population[parent_indices]
        ancestry = ancestry[parent_indices]

        # Mutation
        offspring = mutate_population_subset(offspring, mutation_std, mutation_rate)

        # Create new population
        population = offspring / np.linalg.norm(offspring, axis=1, keepdims=True)

    # Mean standardize to first generation
    if normalize:
        factor = layer_stats["mean"][:, 0, None]
        layer_stats["mean"] /= factor
        layer_stats["stdev"] /= factor

    return layer_stats


def sample_input_orthogonal_to_v(v, num_samples=1, rng=None):
    """
    Returns `num_samples` random row-vectors x such that x·v = 0.
    v: 1D numpy array of shape (d_in,)
    """
    if rng is None:
        rng = np.random.default_rng()

    v = np.asarray(v, dtype=float)
    d_in = v.size
    v_norm_sq = np.dot(v, v)
    if v_norm_sq == 0:
        raise ValueError(
            "v is the zero vector; orthogonality constraint is trivial (any x works)."
        )

    xs = []
    for _ in range(num_samples):
        z = rng.standard_normal(d_in)
        alpha = np.dot(z, v) / v_norm_sq
        x = z - alpha * v
        xs.append(x)

    return np.stack(xs, axis=0)  # shape: (num_samples, d_in)


def run_evolution_with_kwargs(kwargs):
    return run_evolution(**kwargs)


def parallel_run_evolution(n_runs, **kwargs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_evolution_with_kwargs,
                {**kwargs, "seed": np.random.randint(0, 1000000)},
            )
            for _ in range(n_runs)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Aggregate results
    aggregated_results = {}
    for key in [
        "mean",
        "stdev",
        "ancestry_proportions",
        "fitness",
    ]:
        values = np.array([result[key] for result in results])
        if key == "mean":
            aggregated_results[key] = np.mean(values, axis=0)
        elif key == "stdev":
            aggregated_results[key] = np.std(values, axis=0)
        else:
            aggregated_results[key] = values  # don't aggregate
    return aggregated_results


if __name__ == "__main__":
    import plotting

    layer_stats = parallel_run_evolution(
        1,
        n_generations=10000,
        population_size=1000,
        mutation_std=0.3,
        mutation_rate=0.2,
        eval_fraction=0.3,
        max_depth=3,
        dim=10,
        normalize=False,
        use_sigmoid=False,
        nonlinear_fitness=True,
        omega=1.0,
        fix_initial_pop_distance=True,
    )
    plotting.plot_layer_evolution(layer_stats, save=False)
