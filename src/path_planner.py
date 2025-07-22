import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import random


class PathPlanner:
    def __init__(self):
        self.distance_weight = 1.0
        self.importance_weight = 0.5
        self.smoothness_weight = 0.3
        self.max_iterations = 1000
        
    def calculate_distance_matrix(self, points: List[Tuple[int, int, float]]) -> np.ndarray:
        coords = np.array([(p[0], p[1]) for p in points])
        distances = squareform(pdist(coords, metric='euclidean'))
        return distances
    
    def calculate_importance_scores(self, points: List[Tuple[int, int, float]]) -> np.ndarray:
        return np.array([p[2] for p in points])
    
    def calculate_path_cost(self, path: List[int], distance_matrix: np.ndarray, 
                           importance_scores: np.ndarray) -> float:
        total_distance = 0
        total_importance = 0
        smoothness_penalty = 0
        
        for i in range(len(path)):
            current_idx = path[i]
            next_idx = path[(i + 1) % len(path)]
            
            total_distance += distance_matrix[current_idx, next_idx]
            total_importance += importance_scores[current_idx]
            
            if i >= 2:
                prev_idx = path[i - 1]
                current_point = np.array([current_idx % 1000, current_idx // 1000])  # Simplified
                prev_point = np.array([prev_idx % 1000, prev_idx // 1000])
                next_point = np.array([next_idx % 1000, next_idx // 1000])
                
                angle1 = np.arctan2(current_point[1] - prev_point[1], current_point[0] - prev_point[0])
                angle2 = np.arctan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
                angle_diff = abs(angle1 - angle2)
                
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                smoothness_penalty += angle_diff
        
        normalized_distance = total_distance / len(path) if len(path) > 0 else 0
        normalized_importance = -total_importance / len(path) if len(path) > 0 else 0
        normalized_smoothness = smoothness_penalty / len(path) if len(path) > 0 else 0
        
        cost = (self.distance_weight * normalized_distance + 
                self.importance_weight * normalized_importance + 
                self.smoothness_weight * normalized_smoothness)
        
        return cost
    
    def nearest_neighbor_tsp(self, distance_matrix: np.ndarray, 
                            importance_scores: np.ndarray, start_idx: int = 0) -> List[int]:
        n_points = distance_matrix.shape[0]
        unvisited = set(range(n_points))
        path = [start_idx]
        unvisited.remove(start_idx)
        
        current = start_idx
        
        while unvisited:
            best_next = None
            best_score = float('inf')
            
            for candidate in unvisited:
                distance = distance_matrix[current, candidate]
                importance_bonus = -importance_scores[candidate] * self.importance_weight
                score = distance + importance_bonus
                
                if score < best_score:
                    best_score = score
                    best_next = candidate
            
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        return path
    
    def improve_path_2opt(self, path: List[int], distance_matrix: np.ndarray, 
                         importance_scores: np.ndarray) -> List[int]:
        best_path = path.copy()
        best_cost = self.calculate_path_cost(best_path, distance_matrix, importance_scores)
        improved = True
        
        iteration = 0
        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path)):
                    if j - i == 1:
                        continue
                    
                    new_path = path.copy()
                    new_path[i:j] = path[i:j][::-1]
                    
                    new_cost = self.calculate_path_cost(new_path, distance_matrix, importance_scores)
                    
                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
            
            path = best_path
        
        return best_path
    
    def genetic_algorithm_tsp(self, distance_matrix: np.ndarray, 
                             importance_scores: np.ndarray, 
                             population_size: int = 100, 
                             generations: int = 200) -> List[int]:
        n_points = distance_matrix.shape[0]
        
        def create_individual():
            path = list(range(n_points))
            random.shuffle(path)
            return path
        
        def mutate(individual, mutation_rate=0.1):
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
            return individual
        
        def crossover(parent1, parent2):
            start, end = sorted(random.sample(range(len(parent1)), 2))
            child = [-1] * len(parent1)
            child[start:end] = parent1[start:end]
            
            remaining = [item for item in parent2 if item not in child]
            j = 0
            for i in range(len(child)):
                if child[i] == -1:
                    child[i] = remaining[j]
                    j += 1
            
            return child
        
        population = [create_individual() for _ in range(population_size)]
        
        for generation in range(generations):
            costs = [self.calculate_path_cost(individual, distance_matrix, importance_scores) 
                    for individual in population]
            
            sorted_indices = sorted(range(len(costs)), key=lambda i: costs[i])
            population = [population[i] for i in sorted_indices]
            
            new_population = population[:population_size//4]
            
            while len(new_population) < population_size:
                parent1 = random.choice(population[:population_size//2])
                parent2 = random.choice(population[:population_size//2])
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        final_costs = [self.calculate_path_cost(individual, distance_matrix, importance_scores) 
                      for individual in population]
        best_individual = population[np.argmin(final_costs)]
        
        return best_individual
    
    def find_optimal_starting_point(self, points: List[Tuple[int, int, float]]) -> int:
        best_start = 0
        best_importance = points[0][2]
        
        for i, (x, y, importance) in enumerate(points):
            edge_bonus = 0
            if x < 50 or y < 50 or x > 450 or y > 450:
                edge_bonus = 0.2
            
            total_score = importance + edge_bonus
            if total_score > best_importance:
                best_importance = total_score
                best_start = i
        
        return best_start
    
    def plan_path(self, points: List[Tuple[int, int, float]], 
                  method: str = 'genetic') -> List[int]:
        if len(points) < 2:
            return list(range(len(points)))
        
        distance_matrix = self.calculate_distance_matrix(points)
        importance_scores = self.calculate_importance_scores(points)
        
        start_idx = self.find_optimal_starting_point(points)
        
        if method == 'nearest_neighbor':
            path = self.nearest_neighbor_tsp(distance_matrix, importance_scores, start_idx)
            path = self.improve_path_2opt(path, distance_matrix, importance_scores)
        
        elif method == 'genetic':
            path = self.genetic_algorithm_tsp(distance_matrix, importance_scores)
            
            if start_idx != path[0]:
                start_pos = path.index(start_idx)
                path = path[start_pos:] + path[:start_pos]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return path
    
    def get_path_coordinates(self, points: List[Tuple[int, int, float]], 
                           path: List[int]) -> List[Tuple[int, int]]:
        return [(points[i][0], points[i][1]) for i in path]