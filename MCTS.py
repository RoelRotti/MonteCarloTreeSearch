import numpy, math, scipy.stats, norm
import matplotlib.pyplot as plt
import binarytree as bt
import time
from binarytree import Node

# Assume that the number MCTS-iterations starting in a specific root node is limited (e.g. to 10 or 50). Make a similar
# assumption for the number of roll-outs starting in a particular (”snowcap”) leaf node (e.g. 1 or 5).
number_of_MCTS_iterations_in_root_node = 30
number_of_roll_outs_snowcap = 3
depth = 12
c = 2

# Construct a binary tree (each node has two child nodes) of depth d = 12 (or more – if you’re feeling lucky) and assign
# different values to each of the 2d leaf-nodes. Specifically, pick the leaf values to be real numbers randomly
# distributed between 0 and 100 (use the uniform continuous distribution U(0, 100), so don’t restrict yourself to
# integer values!).

# Essential: in package binarytree, I added the following to the definition __init__ (also in definition parameters)
# class Node(object):
#     def __init__(self, value, left=None, right=None  tNode = 0, numberVisits = 0):
#         ...
#         self.tNode = tNode
#         self.numberVisits = numberVisits

def buildTree(doprint = False):
    # First create a list with random values
    nodes = []
    for node in range((2**depth)-1):
        nodes.append(numpy.random.uniform(0, 100))
    # tNode and numberVisits added in the package itself, set to 0
    binary_tree = bt.build(nodes)
    if doprint:
        print('Binary tree from list :\n', binary_tree)
        print('\nList from binary tree :', binary_tree.values)
    return binary_tree


def MCTS(node, doprint = False):
    height = depth-1 #height is # of connections, depth = # of nodes
    while node.height > 1:
        node = MCTS_snowcap(node, height) #returns new root
        if doprint: print("Height = ", height), node.pprint()
        height = height - 1 # ensures backup doesn't backs too far up
    return selection(node)


def MCTS_snowcap(node, height):
    current_root = node
    for i in range(number_of_roll_outs_snowcap):
        current_root = MCTS_once(current_root, height) #same root node
    # TODO: do you always go down one layer?
    # Tree policy: choose child node with best (finite) UCB
    current_root = selection(current_root)
    return current_root


# Implement the MCTS algorithm and apply it to the above tree to search for the optimal (i.e. highest) value.
def MCTS_once(node, height): #root node
    # 1. Selection
    current = selection(node) #new selection node
    # 2. Expansion
    while current.numberVisits > 0:
        if current.left is None and current.right is None:  # no children
            break
        current = selection(current) #expanded node
    # 3. Simulation: Roll out
    terminalValue = rollOut(current)
    # 4. Back-up
    current = backup(current, terminalValue, height) #backup returns root node
    return current #root node again


def backup(node, terminalValue, height):
    node.tNode = node.tNode + terminalValue
    node.numberVisits = node.numberVisits + 1
    if node.height == height:
        return node
    else:
        parent = bt.get_parent(root, node)
        return backup(parent, terminalValue, height)


def rollOut(node):
    if node.left is None and node.right is None: # no children
        return node.value
    #TODO: Better rollout
    elif numpy.random.uniform(0, 1) > 0.5:
        return rollOut(node.left)
    else:
        return rollOut(node.right)


def selection(node):
    # Safety check:
    if node.left is None and node.right is None:
        return node
    # 1. Selection
    left = node.left
    right = node.right
    ucbLeft, terminal = UCB(left, node.numberVisits)
    ucbRight, terminal = UCB(right, node.numberVisits)
    if ucbLeft > ucbRight:
        return node.left
    else:
        return node.right


def UCB(node, numberVisitsParent):
    # 1.1 Tree policy: UCB(node_i) = mean_node_value + c*sqrt( (logNumberVisitsParents)/(numberVisitsNode) )
    # Ensure UCB does not crash/return random value when leaf node is encountered. Now it chooses max value in selection
    if node.left is None and node.right is None:  # no children
        return node.value, 1
    elif node.numberVisits == 0:
        return math.inf, 0
    else:
        ucb = node.tNode / node.numberVisits + c * math.sqrt(
            math.log(numberVisitsParent, 10) / node.numberVisits)
        return ucb, 0


def getMaxLeaves(node):
    levels = node.levels
    lastLevel = levels[depth-1]
    values = []
    for index in range(len(lastLevel)):
        values.append(lastLevel[index].value)
    return sorted(values, reverse=True)

# Collect statistics on the performance and discuss the role of the hyperparameter c in the UCB-score.
#TODO: Collect statistics

start_time = time.time()
average = []
for i in range(100):
    tree = buildTree(doprint=False)
    root = tree #root is essential for finding parent
    max = getMaxLeaves(root)
    returned = MCTS(tree).value  # root node
    average.append(max.index(returned))
    print(max.index(returned),"/", 2**(depth-1),"\nvalue = ", returned)

elapsed_time = time.time() - start_time
print("Average = ", sum(average)/len(average)) #Ran on 09/12 16:18, average = 335.62
print("Time = ", elapsed_time)

