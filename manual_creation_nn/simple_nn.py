import numpy as np


class Operation(object):
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self, inputs=[]):
        self.inputs = inputs


class add(Operation):
    def __init__(self, x, y):
        super(add, self).__init__([x, y])

    def compute(self, x_var, y_var):
        super(add, self).compute([x_var, y_var])
        return x_var + y_var


class multiply(Operation):
    def __init__(self, x, y):
        super(multiply, self).__init__([x, y])

    def compute(self, x_var, y_var):
        super(multiply, self).compute([x_var, y_var])
        return x_var * y_var


class matmul(Operation):
    def __init__(self, x, y):
        super(matmul, self).__init__([x, y])

    def compute(self, x_var, y_var):
        super(matmul, self).compute([x_var, y_var])
        return x_var.dot(y_var)


class Placeholder(object):
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable(object):
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


def traverse_postorder(operation):
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session(object):
    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                node.inputs = [
                    input_node.output for input_node in node.input_nodes
                ]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output


g = Graph()
g.set_as_default()

a = Variable(10)
b = Variable(1)
x = Placeholder()

y = multiply(a, x)
z = add(y, b)

sess = Session()
result = sess.run(z, feed_dict={x: 10})
print('**************************')
print('session is runned')
print('**************************')
print(f'result: {result}')

g = Graph()
g.set_as_default()

a = Variable([[10, 20], [30, 40]])
b = Variable([1, 1])
x = Placeholder()

y = matmul(a, x)
z = add(y, b)

sess = Session()
result = sess.run(z, feed_dict={x: 10})
print('**************************')
print('session is runned')
print('**************************')
print(f'result: {result}')
