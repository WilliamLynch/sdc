
# coding: utf-8

# In[ ]:

"""
This script builds and runs a graph with miniflow.

v1
"""

from miniflow import *

x, y = Input(), Input()

f = Add(x, y)

feed_dict = {x: 10, y: 5}

sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f, sorted_neurons)

# NOTE: because topological_sort set the values for the `Input` neurons we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))

