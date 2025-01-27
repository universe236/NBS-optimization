"""
NSGA-III
"""

import pandas as pd
import numpy as np
from platypus import NSGAIII, Problem, Real, nondominated, Solution, SBX, PM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import copy

warnings.filterwarnings("ignore")


class AdaptiveOperators:
    def __init__(self, population_size):
        self.population_size = population_size
        self.pc_max = 1.0
        self.pc_min = 0.7
        self.pm_max = 0.3
        self.pm_min = 0.1

        self.di_c_max = 20
        self.di_c_min = 2
        self.di_m_max = 15
        self.di_m_min = 2

        self.current_pc = self.pc_max
        self.current_pm = self.pm_min
        self.current_di_c = self.di_c_min
        self.current_di_m = self.di_m_min

        self.sbx = SBX(probability=self.current_pc, distribution_index=self.current_di_c)
        self.pm = PM(probability=self.current_pm, distribution_index=self.current_di_m)

    def update_parameters(self, population, generation, update_freq=5, diversity_threshold=15):
        if generation % update_freq != 0:
            return

        if not population:
            return

        valid_solutions = [sol for sol in population if sol.objectives is not None]
        if len(valid_solutions) < 2:
            return

        objectives = np.array([[-obj for obj in sol.objectives] for sol in valid_solutions])
        diversity = np.mean(np.std(objectives, axis=0))

        if diversity < diversity_threshold:
            self.current_pc = min(self.pc_max, self.current_pc * 1.2)
            self.current_pm = min(self.pm_max, self.current_pm * 1.3)
            self.current_di_c = max(self.di_c_min, self.current_di_c * 0.7)
            self.current_di_m = max(self.di_m_min, self.current_di_m * 0.7)
        else:
            self.current_pc = max(self.pc_min, self.current_pc * 0.95)
            self.current_pm = max(self.pm_min, self.current_pm * 0.95)
            self.current_di_c = min(self.di_c_max, self.current_di_c * 1.05)
            self.current_di_m = min(self.di_m_max, self.current_di_m * 1.05)

        self.sbx = SBX(probability=self.current_pc, distribution_index=self.current_di_c)
        self.pm = PM(probability=self.current_pm, distribution_index=self.current_di_m)

        print(f"\n[Generation {generation}] Population Diversity: {diversity:.2f}")
        print(f"  Crossover Probability: {self.current_pc:.3f} | Mutation Probability: {self.current_pm:.3f}")
        print(
            f"  Crossover Distribution Index: {self.current_di_c:.3f} | Mutation Distribution Index: {self.current_di_m:.3f}")


def load_data(demand_path, effect_path):
    try:
        demand_df = pd.read_excel(demand_path)
        effect_df = pd.read_excel(effect_path)

        demands = demand_df.iloc[:, 1:5].values
        effects = effect_df.iloc[:, 1:5].values

        return demand_df, effect_df, demands, effects

    except Exception as e:
        print(f"Data loading error: {str(e)}")
        raise


def discretize_variables(variables, step=0.001):
    return np.round(np.array(variables) / step) * step


def evaluate_solution(variables, demands, effects, num_plots, num_trees, tolerance=0.005):
    try:
        if hasattr(variables, 'variables'):
            variables = variables.variables
        variables = np.array([v.value if hasattr(v, 'value') else v for v in variables])

        variables = discretize_variables(variables, step=0.001)
        proportions = variables.reshape((num_plots, num_trees))

        for i in range(num_plots):
            row_vars = proportions[i]
            row_vars = np.clip(row_vars, 0, 1)
            total_sum = np.sum(row_vars)

            if total_sum > tolerance and total_sum != 1.0:
                row_vars = row_vars / total_sum

            row_vars = np.clip(row_vars, 0, 1)
            row_vars = discretize_variables(row_vars, step=0.001)
            proportions[i] = row_vars

        objectives = []
        for j in range(4):
            match_values = proportions * effects[:, j] * demands[:, j].reshape(-1, 1)
            total_match = np.sum(match_values)
            objectives.append(-total_match)

        constraints = []
        for i in range(num_plots):
            sum_constraint = max(0, abs(np.sum(proportions[i]) - 1.0) - tolerance)
            constraints.append(sum_constraint)

        return objectives, constraints

    except Exception as e:
        print(f"Objective function calculation error: {str(e)}")
        raise


def create_optimization_problem(demands, effects):
    num_plots = demands.shape[0]
    num_trees = effects.shape[0]
    num_variables = num_plots * num_trees
    tolerance = 0.005

    print(f"Number of decision variables: {num_variables}")

    def evaluate(solution):
        return evaluate_solution(solution, demands, effects, num_plots, num_trees, tolerance)

    problem = Problem(num_variables, 4, num_plots)
    problem.types[:] = Real(0, 1)
    problem.function = evaluate

    return problem


def create_initial_population(problem, population_size, num_plots, num_trees):
    population = []

    for _ in range(population_size):
        solution = Solution(problem)
        variables = np.random.random(num_plots * num_trees)

        for i in range(0, len(variables), num_trees):
            block_vars = variables[i:i + num_trees]
            block_sum = np.sum(block_vars)
            if block_sum > 0:
                block_vars = block_vars / block_sum
            block_vars = discretize_variables(block_vars, step=0.001)
            variables[i:i + num_trees] = block_vars

        solution.variables = variables
        population.append(solution)

    return population


def evaluate_population(problem, population):
    for sol in population:
        if sol.objectives is None:
            problem.evaluate(sol)


def visualize_pareto_front(pareto_df, n_solutions, output_base_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        pareto_df['PM'],
        pareto_df['CS'],
        pareto_df['Runoff'],
        c=pareto_df['LST'],
        cmap='viridis'
    )
    plt.colorbar(scatter, label='Fourth Objective (LST)')
    ax.set_xlabel('PM')
    ax.set_ylabel('CS')
    ax.set_zlabel('Runoff')
    plt.title(f'Pareto Front (Adaptive NSGA-III, {n_solutions} solutions)')

    output_plot_path = output_base_path.rsplit('.', 1)[0] + '_pareto_front.png'
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Pareto front visualization saved to {output_plot_path}")


def main(demand_path, effect_path, output_path):
    print("Loading data...")
    demand_df, effect_df, demands, effects = load_data(demand_path, effect_path)

    print("Creating optimization problem...")
    problem = create_optimization_problem(demands, effects)

    population_size = 400
    n_iterations = 130

    adaptive_ops = AdaptiveOperators(population_size)

    print("Starting NSGA-III optimization...")
    algorithm = NSGAIII(problem,
                        divisions_outer=12,
                        population_size=population_size,
                        variator=adaptive_ops.sbx,
                        mutator=adaptive_ops.pm)

    num_plots = demands.shape[0]
    num_trees = effects.shape[0]

    initial_population = create_initial_population(
        problem, population_size, num_plots, num_trees
    )
    algorithm.population = initial_population

    evaluate_population(problem, algorithm.population)

    try:
        for i in range(n_iterations):
            if i % 5 == 0:
                print(f"\nIteration {i + 1}/{n_iterations}")

            algorithm.step()
            evaluate_population(problem, algorithm.population)
            adaptive_ops.update_parameters(algorithm.population, i, update_freq=5)
            algorithm.variator = adaptive_ops.sbx
            algorithm.mutator = adaptive_ops.pm

            if i % 5 == 0:
                current_solutions = nondominated(algorithm.result)
                print(f"Current non-dominated solutions: {len(current_solutions)}")
                if len(current_solutions) > 0:
                    best_objectives = [-obj for obj in current_solutions[0].objectives]
                    print(f"Current best objective values: {best_objectives}")

    except Exception as e:
        print(f"Optimization error: {str(e)}")
        raise

    final_solutions = nondominated(algorithm.result)
    n_solutions = len(final_solutions)

    pareto_df = pd.DataFrame([
        [-obj for obj in sol.objectives] for sol in final_solutions
    ], columns=['PM', 'CS', 'Runoff', 'LST'])

    all_solutions_proportions = []
    for sol in final_solutions:
        arr = np.array(sol.variables).reshape((num_plots, num_trees))
        all_solutions_proportions.append(arr)

    with pd.ExcelWriter(output_path) as writer:
        pareto_df.to_excel(writer, sheet_name='Objectives', index=True)
        for idx, proportions in enumerate(all_solutions_proportions):
            df = pd.DataFrame(
                proportions,
                columns=effect_df['species'],
                index=demand_df['FID']
            )
            df.to_excel(writer, sheet_name=f'Solution_{idx + 1}')

    print(f"\n=== Optimization completed, results saved to {output_path} ===")

    if n_solutions >= 2:
        visualize_pareto_front(pareto_df, n_solutions, output_path)
    else:
        print("\nSingle solution objective values:")
        for col, val in zip(['PM', 'CS', 'Runoff', 'LST'], pareto_df.iloc[0]):
            print(f"{col}: {val:.4f}")


if __name__ == "__main__":
    DEMAND_PATH = "{DEMAND_FILE_PATH}"
    EFFECT_PATH = "{EFFECT_FILE_PATH}"
    OUTPUT_PATH = "{OUTPUT_FILE_PATH}"

    main(DEMAND_PATH, EFFECT_PATH, OUTPUT_PATH)