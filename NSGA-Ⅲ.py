import pandas as pd
import numpy as np
from platypus import NSGAIII, Problem, Real, nondominated, Solution, SBX, PM, GAOperator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

class AdaptiveOperators:
    def __init__(self, population_size):
        self.population_size = population_size
        self.pc_max = 1.0
        self.pc_min = 0.7
        self.pm_max = 0.3
        self.pm_min = 0.05
        self.di_c_max = 30
        self.di_c_min = 2
        self.di_m_max = 20
        self.di_m_min = 2

        self.current_pc = self.pc_max
        self.current_pm = self.pm_min
        self.current_di_c = self.di_c_min
        self.current_di_m = self.di_m_min

        self.sbx = SBX(probability=self.current_pc, distribution_index=self.current_di_c)
        self.pm = PM(probability=self.current_pm, distribution_index=self.current_di_m)

    def get_variator(self):
        return GAOperator(self.sbx, self.pm)

    def update_parameters(self, population, generation, update_freq=5):
        if generation % update_freq != 0 or not population:
            return

        valid_solutions = [sol for sol in population if sol.objectives is not None]
        if len(valid_solutions) < 2:
            return

        objectives = np.array([sol.objectives for sol in valid_solutions])
        
        obj_min = np.min(objectives, axis=0)
        obj_max = np.max(objectives, axis=0)
        denom = obj_max - obj_min
        denom[denom == 0] = 1.0
        normalized_objs = (objectives - obj_min) / denom
        
        diversity = np.mean(np.std(normalized_objs, axis=0))

        diversity_threshold = 0.1
        if diversity < diversity_threshold:
            self.current_pc = min(self.pc_max, self.current_pc * 1.1)
            self.current_pm = min(self.pm_max, self.current_pm * 1.2)
            self.current_di_c = max(self.di_c_min, self.current_di_c * 0.9)
            self.current_di_m = max(self.di_m_min, self.current_di_m * 0.9)
        else:
            self.current_pc = max(self.pc_min, self.current_pc * 0.95)
            self.current_pm = max(self.pm_min, self.current_pm * 0.95)
            self.current_di_c = min(self.di_c_max, self.current_di_c * 1.1)
            self.current_di_m = min(self.di_m_max, self.current_di_m * 1.1)

        self.sbx = SBX(probability=self.current_pc, distribution_index=self.current_di_c)
        self.pm = PM(probability=self.current_pm, distribution_index=self.current_di_m)

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

def evaluate_solution(solution_vars, demands, effects, num_plots, num_trees):
    try:
        continuous_vars = np.array(solution_vars)
        discrete_vars = np.round(continuous_vars / 0.001) * 0.001
        
        proportions = discrete_vars.reshape((num_plots, num_trees))

        row_sums = np.sum(proportions, axis=1)
        tolerance = 0.001
        constraints = np.maximum(0, np.abs(row_sums - 1.0) - tolerance)
        
        safe_sums = row_sums.reshape(-1, 1)
        safe_sums[safe_sums == 0] = 1.0 
        normalized_props = proportions / safe_sums
        
        objectives = []
        for j in range(4): 
            match_values = normalized_props * effects[:, j] * demands[:, j].reshape(-1, 1)
            total_match = np.sum(match_values)
            objectives.append(-total_match)

        return objectives, constraints.tolist()

    except Exception as e:
        return [1e9]*4, [1e9]*num_plots

def create_optimization_problem(demands, effects):
    num_plots = demands.shape[0]
    num_trees = effects.shape[0]
    num_variables = num_plots * num_trees
    
    def evaluate(solution):
        objs, constrs = evaluate_solution(solution.variables, demands, effects, num_plots, num_trees)
        solution.objectives[:] = objs
        solution.constraints[:] = constrs

    problem = Problem(num_variables, 4, num_plots)
    problem.types[:] = Real(0, 1)
    problem.constraints[:] = "<=0"
    problem.function = evaluate

    return problem

def perform_topsis(pareto_objectives):
    benefit_matrix = -np.array(pareto_objectives)
    
    row_sq_sum = np.sqrt(np.sum(benefit_matrix**2, axis=0))
    row_sq_sum[row_sq_sum == 0] = 1.0
    normalized_matrix = benefit_matrix / row_sq_sum
    
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    weighted_normalized = normalized_matrix * weights
    
    ideal_best = np.max(weighted_normalized, axis=0)
    ideal_worst = np.min(weighted_normalized, axis=0)
    
    dist_best = np.sqrt(np.sum((weighted_normalized - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_normalized - ideal_worst)**2, axis=1))
    
    total_dist = dist_best + dist_worst
    total_dist[total_dist == 0] = 1.0
    scores = dist_worst / total_dist
    
    best_index = np.argmax(scores)
    return best_index, scores

def visualize_results(pareto_df, best_idx, output_base_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    p = ax.scatter(
        pareto_df['PM'], 
        pareto_df['CS'], 
        pareto_df['Runoff'], 
        c=pareto_df['LST'], 
        cmap='viridis', 
        alpha=0.6,
        s=30
    )
    
    ax.scatter(
        pareto_df.iloc[best_idx]['PM'],
        pareto_df.iloc[best_idx]['CS'],
        pareto_df.iloc[best_idx]['Runoff'],
        color='red', s=150, marker='*', label='TOPSIS Best'
    )

    plt.colorbar(p, label='LST Reduction')
    ax.set_xlabel('PM')
    ax.set_ylabel('CS')
    ax.set_zlabel('Runoff')
    plt.legend()
    plt.title('Pareto Front (Standard NSGA-III) & TOPSIS')

    output_plot_path = output_base_path.rsplit('.', 1)[0] + '_standard_topsis.png'
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_plot_path}")

def main(demand_path, effect_path, output_path):
    print("Loading data...")
    demand_df, effect_df, demands, effects = load_data(demand_path, effect_path)
    
    print("Initializing Standard NSGA-III problem...")
    problem = create_optimization_problem(demands, effects)

    population_size = 400 
    n_iterations = 500    
    divisions_outer = 12  
    
    adaptive_ops = AdaptiveOperators(population_size)
    
    algorithm = NSGAIII(problem,
                        divisions_outer=divisions_outer,
                        population_size=population_size,
                        variator=adaptive_ops.get_variator())

    print("Running optimization (Standard Mode)...")
    for i in range(n_iterations):
        algorithm.step()
        adaptive_ops.update_parameters(algorithm.population, i)
        algorithm.variator = adaptive_ops.get_variator()
        
        if (i + 1) % 50 == 0:
            print(f"Generation {i + 1}/{n_iterations} completed.")

    print("Optimization finished.")
    final_solutions = nondominated(algorithm.result)
    
    feasible_solutions = [s for s in final_solutions if s.constraint_violation <= 0.001]
    
    if not feasible_solutions:
        print("Warning: Using best available solutions (Standard constraints).")
        feasible_solutions = final_solutions
    
    print(f"Found {len(feasible_solutions)} solutions.")

    pareto_data = [[-obj for obj in sol.objectives] for sol in feasible_solutions]
    pareto_df = pd.DataFrame(pareto_data, columns=['PM', 'CS', 'Runoff', 'LST'])

    best_idx, scores = perform_topsis(pareto_data)
    pareto_df['TOPSIS_Score'] = scores
    
    print(f"Best Solution Index: {best_idx}, Score: {scores[best_idx]:.4f}")

    with pd.ExcelWriter(output_path) as writer:
        pareto_df.to_excel(writer, sheet_name='Pareto_Objectives')
        
        raw_vars = np.array(feasible_solutions[best_idx].variables)
        
        final_discrete_vars = np.round(raw_vars / 0.001) * 0.001
        
        best_props = final_discrete_vars.reshape((demands.shape[0], effects.shape[0]))
        
        row_sums = np.sum(best_props, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        best_props = best_props / row_sums
        
        df_best = pd.DataFrame(
            best_props,
            columns=effect_df['species'] if 'species' in effect_df.columns else None,
            index=demand_df['FID'] if 'FID' in demand_df.columns else None
        )
        df_best.to_excel(writer, sheet_name='TOPSIS_Best_Solution')

    visualize_results(pareto_df, best_idx, output_path)
    print("Done.")

if __name__ == "__main__":
    DEMAND_PATH = "demand_data.xlsx"
    EFFECT_PATH = "effect_data.xlsx"
    OUTPUT_PATH = "final_result_standard.xlsx"
    
    main(DEMAND_PATH, EFFECT_PATH, OUTPUT_PATH)
