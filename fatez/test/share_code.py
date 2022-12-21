"""
Blackboard to share codes
"""

import itertools


list_1 = ['a', 'b', 'c','d']
list_2 = [1,4,9]


unique_combinations = []

for comb in itertools.permutations(list_1, len(list_2)):
    zipped = zip(comb, list_2)
    unique_combinations.append(list(zipped))


print(unique_combinations)
