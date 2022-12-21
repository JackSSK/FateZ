"""
Blackboard to share codes
"""

import itertools


test_list = [1, 7, 4, 3]

res = list(itertools.combinations(test_list, 2))
print(res)

# iter mode
for ele in itertools.combinations(test_list, 2):
    print(ele)
