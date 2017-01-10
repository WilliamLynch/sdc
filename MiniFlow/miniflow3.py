
# coding: utf-8

"""
This script builds and runs a graph with miniflow.

v1
"""

class Neuron:
    """
    This represents a generic node
    
    We know that each node might receive input from multiple other nodes. 
    We also know that each node creates a single output, 
    which will likely be passed to other nodes. 
    Let's add two lists: one to store references to the inbound nodes,
    and the other to store references to the outbound nodes.
    """
    def __init__(self, inbound_neurons=[]):
        # List of Neurons from which this Node receives values
        self.inbound_neurons = inbound_neurons
        # List of Neurons to which this Node passes values
        self.outbound_neurons = []
        # A calculated value, initialized to zero but will have something
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplemented


class Input(Neuron):
    """
    Unlike the other subclasses of Neuron, 
    Input doesn't actually calculate anything. 
    The Input subclass just holds a value,
    (as a data feature or a model parameter (weight/bias)).
    You can set value either explicitly or with the forward() method. 
    This value is then fed through the rest of the neural network.
    """
    def __init__(self):
        # an Input neuron has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Neuron.__init__(self)

    # NOTE: Input node is the only node where the value
    # is passed as an argument to forward().
    #
    # All other neuron implementations should get the value
    # of the previous neurons from self.inbound_neurons
    #
    # Example:
    # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value:
            self.value = value


class Add(Neuron):
    """
    The Add class takes 2 inbound neurons, x and y, and adds the values of those neurons.
    """
    def __init__(self, x, y):
        # You could access `x` and `y` in forward with
        # self.inbound_neurons[0] (`x`) and self.inbound_neurons[1] (`y`)
        Neuron.__init__(self, [x, y])
        
        # x = self.inbound_neurons[0]
        # y = self.inbound_neurons[1]

    def forward(self):
        """
        Set the value of this neuron (`self.value`) to the sum of it's inbound_nodes.
                """
        self.value = self.inbound_neurons[0].value +self.inbound_neurons[1].value

        
class Mul(Neuron):
    def __init__(self, *inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        
        if 0 in [var.value for var in self.inbound_neurons]:
            self.value = 0
        elif len(self.inbound_neurons):
            self.value = 1
            for var in self.inbound_neurons:
                self.value *= var.value        

class Linear(Neuron):
    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)
        
        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other neurons.
        # The weight and bias values are stored within the
        # respective neurons.
        self.weights = weights
        self.bias = bias
        
    def forward(self):
        """
        Set self.value to the value of the linear function output.
        """
        self.value = self.bias.value
        for w, x in zip(self.weights, self.inbound_neurons):
            self.value = self.value + (w.value * x.value)
                

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    
    It's a way to flatten the NN graph so that all the input dependencies
    for each node are resolved before trying to run its calculation. 
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_neuron`: A neuron in the graph, should be the output neuron (have no outgoing edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_neuron.value



