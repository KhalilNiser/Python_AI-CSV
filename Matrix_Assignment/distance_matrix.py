

# IMPORT_NUMPY
import numpy as np

# Define the distance matrix
distance_matrix = np.array( [
    [ 8, 6, 3, 0, 5 ],
    [ 3, 0, 7, 0, 8 ],
    [ 4, 3, 0, 7, 9 ],
    [ 3, 6, 7, 0, 1 ],
    [ 4, 6, 7, 8, 0 ]
] )


# Define Number of Cities
numberOfCities = distance_matrix.shape[ 0 ]

# Define Number Of Ants
numberOfAnts = 30

# Initialize pheromone level matrix: A matrix that represents
# the amount of pheromone on each path
def ant_colony_optimization( numberOfIterations ):
    # Initialize pheromone level matrix
    pheromone_level = np.ones( ( numberOfCities, numberOfCities ) )
    # Initialize heuristic information matrix: A matrix providing guidance 
    # # for path selection based on the problem-specific knowledge.

    # Initialize heuristic information matrix. NOTE: This code is commonly 
    # used when dealing with distance calculations, where a division by 
    # zero can occur, specifically where two data points are identical. 
    # Adding the machine epsilon helps avoid this issue and ensures that 
    # the resulting values are well-defined. 
    # NOTE: This code performs the following operations: np.finfo(float).eps: 
    # This retrieves the machine epsilon for the float data type. Machine 
    # epsilon is the smallest number that, when added to 1.0, results in a value 
    # different from 1.0 due to the limitations of floating-point representation. 
    # Adding it to the denominator prevents potential division by zero errors and 
    # improves numerical stability.
    # distance_matrix + np.finfo(float).eps: This adds the machine epsilon to every 
    # element in the distance_matrix. 
    # 1 / (...): This calculates the reciprocal of each element in the resulting matrix.
    heuristic_info = 1 / ( distance_matrix + np.finfo( float ).eps )
    # Define alpha and beta parameters: Parameters that determine the relative importance
    # of pheromone and heuristic information in ant decision-making, during optimization
    # Pheromone importance
    alpha = 1.0
    # Heuristic importance
    beta = 2.0
    
    # Initialize the best path and distance, and the ant path and distance
    best_distance = float( 'inf' )
    best_path = [ ]
    # Initialize/define variable best_iteration
    best_iteration = -1
    
    
    for iteration in range( numberOfIterations ):
        # Initialize ant's paths and distances
        ant_paths = np.zeros( ( numberOfAnts, numberOfCities ), dtype = int )
        ant_distances = np.zeros( numberOfAnts )
        
        # Randomly choose the starting city, and construct the path
        for ant in range( numberOfAnts ):
            # Randomly choose the starting city
            current_city = np.random.randint( numberOfCities )
            visited = [ current_city ]
            
            # Construct the path
            for _ in range( numberOfCities - 1 ):
                # Calculate the selection probabilities
                selection_probabilities = ( pheromone_level [ current_city ] ** alpha ) * ( heuristic_info [ current_city ] ** beta )
                # Set the selection_probabilities of visited cities to zero (0)
                selection_probabilities[ np.array( visited ) ] = 0
                
                # Choose the next city based on the selection_probabilities
                next_city = np.random.choice( np.arange( numberOfCities ), probs = ( selection_probabilities / np.sum( selection_probabilities ) ) )
                
                
                # Update the path and the visited (current_city) list
                ant_paths[ ant, _ + 1 ] = next_city 
                visited.append( next_city )
                
                # update the distance
                ant_distances[ ant ] += distance_matrix[ current_city, next_city ]
                
                current_city = next_city
                
            # Update the distance to return to the starting city
            ant_distances[ ant ] += distance_matrix[ current_city, ant_paths[ ant, 0 ] ] 
            
            
        # update the pheromone level based on the ant paths
        # evaporation
        pheromone_level *= 0.5
        
        for ant in range( numberOfAnts ):
            
            for city in range( numberOfCities - 1 ):
                
                pheromone_level[ ant_paths[ ant, city], ant_paths[ ant, city + 1 ] ] += 1 / ant_distances[ ant ] 
            pheromone_level[ ant_paths[ ant, -1 ], ant_paths[ ant, 0 ] ] += 1 / ant_distances[ ant ]
            
        # Updates best path and distance, if a better solution is found
        min_distance_idx = np.argmin( ant_distances )
        
        if ant_distances[ min_distance_idx ] < best_distance:
            best_distance = ant_distances[ min_distance_idx ]
            best_path = ant_paths[ min_distance_idx ]
            # Update the best iteration
            best_iteration = iteration
            
    return best_path, best_distance, best_iteration

# RUN_THE_ALGORITHM_AND_PRINT_THE_RESULTS

# Run the ant_colony_optimization algorithm
# Number of iterations
numberOfIterations = 200
best_path, best_distance, best_iteration = ant_colony_optimization( numberOfIterations )

# Display the best path, distance and iteration
print( "Best Path: ", best_path )
print( "Best Distance: ", best_distance )
print( "Iteration With the Bset Distance: ", best_iteration )
                