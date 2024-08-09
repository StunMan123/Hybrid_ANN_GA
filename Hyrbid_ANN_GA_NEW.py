#ANN is a widely accepted machine learning method that uses past data to predict future trend, 
#while GA is an algorithm that can find better subsets of input variables for importing into ANN, 
#hence enabling more accurate prediction by its efficient feature selection.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import deap.creator #deap = distributed evolutionary algorithm in python
import time
import psutil
import os
from deap import base, creator, tools, algorithms

# Function to get memory usage of the current process
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

memory_before = memory_usage()# Before execution

start_time=time.time() #start time


# Load the dataset to see the first few rows and the data structure
data_path = 'C:/Users/ASUS/Desktop/year 4 sem 2/Aritifical Intelligence in Chemical Engineering/Project 1/Obesity.csv'
data = pd.read_csv(data_path)

# Store predictions for individuals
predictions_dict = {}

# Fitness function
def eval_ann(individual):
    # Convert genes to hidden layers configuration
    hidden_layers = tuple([max(10, int(x)) for x in individual])
    
    # Creating and training the ANN with genetic configuration
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    # Store predictions using individual's string representation as a key
    predictions_dict[str(individual)] = predictions
    return (accuracy,) #(,) -> comma show tuple class

# Preprocessing
mapping_dicts = {
    'Gender': {'Female': 0, 'Male': 1},
    'family_history_with_overweight': {'yes': 1, 'no': 0},
    'FAVC': {'yes': 1, 'no': 0},
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'SMOKE': {'yes': 1, 'no': 0},
    'SCC': {'yes': 1, 'no': 0},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'MTRANS': {'Automobile': 0, 'Motorbike': 1, 'Bike': 2, 'Public_Transportation': 3, 'Walking': 4}
}

for col, mapping in mapping_dicts.items():
    data[col] = data[col].map(mapping)


# Split the data into features and target
X = data.drop(['NObeyesdad'], axis=1)
y = data['NObeyesdad']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #weight 1.0 means maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox() #Initialize toolbox cointaining genetic operators
toolbox.register("attr_float", np.random.uniform, 10, 100) #random number (10,100) for number of neurons
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3) #create individual with 3 layers, 'individual' will go eval_ann function
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #create population
toolbox.register("evaluate", eval_ann) #return eval_ann's output
toolbox.register("mate", tools.cxBlend, alpha=0.7) #crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2) #mu is mean, sigma is standard deviation, indpb independent probability each attributes get mutated
toolbox.register("select", tools.selTournament, tournsize=4) #choose best out from random group of 4, repeat until enough is selected, then they paired and crossover/mutation (some individual might get selected multiple times)

# Run genetic algorithm and store fitness values
population = toolbox.population(n=10)
ngen = 10  # Number of generations
fitness_history = []
for gen in range(ngen):
    population = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    fitness_history.append([ind.fitness.values[0] for ind in population])
    toolbox.select(population, len(population))

# Plotting
plt.figure(figsize=(10, 5))
for gen, fitnesses in enumerate(fitness_history):
    plt.plot([gen + 1] * len(fitnesses), fitnesses, 'o', label='Generation ' + str(gen + 1))

plt.title('Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc='center left', bbox_to_anchor=(0.99, 0.5))
plt.show()

# Finalize and report
best_ind = tools.selBest(population, 1)[0]
print("Best individual is %s, Accuracy: %s" % (best_ind, best_ind.fitness.values))
print("Predictions for the best individual:", predictions_dict[str(best_ind)])

end_time = time.time()  # End time
print(f"Execution time: {end_time - start_time} seconds")

memory_after = memory_usage()# After execution
print(f"Memory used: {memory_after - memory_before} MB")

