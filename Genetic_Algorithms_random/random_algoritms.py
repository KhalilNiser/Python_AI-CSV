
# Import Random: Method inside Python's classes
import random

#               ---- CONTANT_VARIABLES ----
# Target phrase to be matched
TARGET_PHRASE = "It's Bette to Burnout. Than Fade Away...!"
# Number of inviduals in the population
POPULATION_SIZE = 500
# Probability of mutations
MUTATION_RATE = 0.03

# Generate initial population
def generate_population():
    # Empty variable to hold the population size
    population = []
    
    for _ in range( POPULATION_SIZE ):
        
        invidual = '' .join( random.choice( 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!' ) 
                            for _ in range( len ( TARGET_PHRASE ) ) )
        population.append( invidual )
        
        return population
    
    
    #               ---- Calculate the fitness score ----
    # Calculate fitness score
    def calculate_fitness( invidual ):
        
        score = 0
        
        for i in range( len( TARGET_PHRASE ) ):
            
            if invidual[ i ] == TARGET_PHRASE[ i ]:
                
                score += 1
                
        return score
    
    
    #               ---- Select the parents based on their fitness ----
    # Select parents based on fitness
    def select_parents( population ):
        
        # Empty variable to hold the parents size
        parents = []
        
        for _ in range( 2 ):
            
            parents.append( max( population, key = calculate_fitness ) )
            
            return parents
        
    #               ---- CREATE THE OFFSPRING THROUGH CROSSOVER ----
    # Create offspring through crossover
    def crossover( parents ):
        
        offspring = ""
        
        crossoverPoint = random.randint( 0, len( TARGET_PHRASE ) - 1 )
        
        for i in range( len( TARGET_PHRASE ) ):
            
            if i <= crossoverPoint:
                
                offspring += parents[ 0 ][ i ]
            else:
                offspring += parents[ 1 ][ i ]
            return offspring
        
        
    #               ---- MUTATE THE OFFSPRING 
    # Mutate the offspring
    def mutate( offspring ):
        
        mutated_offspring = ""
        
        for i in range( len ( offspring ) ):
            
            if random.random() < MUTATION_RATE:
                
                mutated_offspring += random.choice( 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.!' )
            else:
                mutated_offspring += offspring[i]
                
            return mutated_offspring
        
        
    # Create the main portion of the genetic algorithm
    # Main genetic algorithm
    def genetic_algorithm():
        population = generate_population()
        generation = 1
        
        while True:
            print( f"Generation { generation } - Best Fit: { max( population, key = calculate_fitness ) }" )
            
            if TARGET_PHRASE in population:
                break
            
            newPopulation = [] 
            
            # Floor division drops the remainder, only gives integers
            for _ in range( POPULATION_SIZE // 2 ):
                
                parents = select_parents( population )
                
                offspring = crossover( parents )
                
                mutated_offspring = mutate( offspring )
                
                newPopulation.extend( [ offspring, mutated_offspring ] ) 
                
            population = newPopulation
            generation += 1
            
    # Run the genetic_algorithm
    genetic_algorithm()
                
            

        
                            
    

