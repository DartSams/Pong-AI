"""
[NEAT]
fitness_criterion     = max #set to max when 1 genome gets the max fitness threshold (mean,max,min)
fitness_threshold     = 400 #number that says when a genome is good
pop_size              = 50 #how many genomes at 1 time in a population so 50 genomes each with their own neural network
reset_on_extinction   = False

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2

[DefaultGenome]
# node activation options
activation_default      = relu #activation function takes a value from previous node to mutate and change to a specific range for next node (relu min value = 0)
activation_mutate_rate  = 1.0
activation_options      = relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 3.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True #inputs are being sent to the next node
initial_connection      = full_direct #fully connected nodes (every input node connects to every hidden node then every hidden connects to output node)

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 2 #hidden layers
num_inputs              = 3 #input layers
num_outputs             = 3 #output layers

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0
"""