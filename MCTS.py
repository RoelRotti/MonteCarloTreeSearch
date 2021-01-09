import numpy, math, scipy.stats, norm
import matplotlib.pyplot as plt
import binarytree as bt
import time
from binarytree import Node



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
    for i in range(number_of_MCTS_iterations_in_root_node):
        current_root = MCTS_once(current_root, height) #same root node
    # Tree policy: choose child node with best (finite) UCB
    current_root = selection(current_root)
    return current_root


# Implement the MCTS algorithm and apply it to the above tree to search for the optimal (i.e. highest) value.
def MCTS_once(node, height): #root node
    # 1. Selection
    current = selection(node) #new selection node
    #TODO: check expansion
    # 2. Expansion
    while current.numberVisits > 0:
        # Safety check
        if current.left is None and current.right is None:  # no children
            break
        current = selection(current) #expanded node
    for leaf in range(number_of_roll_outs_snowcap-1):
        # 3. Simulation: Roll out
        terminalValue = rollOut(current)
        # 4. Back-up
        backdown = []
        current, backdown = backup(current, terminalValue, height, backdown) #backup returns root node
        current = backDown(current, backdown)
    backdown = []
    terminalValue = rollOut(current)
    current, backdown = backup(current, terminalValue, height, backdown)  # backup returns root node
    return current #root node again

def backDown(node, backdown):
    for nodes in range(len(backdown)):
        if backdown[nodes] == "left":
            node = node.left
        else:
            node = node.right
    return node



def backup(node, terminalValue, height, backdown):
    node.tNode = node.tNode + terminalValue
    node.numberVisits = node.numberVisits + 1
    if node.height == height:
        return node, backdown
    else:
        parent = bt.get_parent(root, node)
        # Saves path so path root -> this node remembered, for multiple rollouts
        if parent.left == node:
            #TODO implemente FILO structure
            backdown.insert(0, 'left')
        else:
            backdown.insert(0, 'right')
        return backup(parent, terminalValue, height, backdown)


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
def test():
    start_time = time.time()
    avgIndexI = []
    avgValueI = []
    avgPercentI = []

    global cList
    # while c < cMax+cIncrement:
    #     print(cMax)
    for i in range(len(cList)):
        global c
        c = cList[i]
        avgIndex = []
        avgValue = []
        for j in range(test_iteration_per_c_value):
            tree = buildTree(doprint=False)
            global root
            root = tree  # root is essential for finding parent
            maxLeaves = getMaxLeaves(root)
            returned = MCTS(tree).value  # root value
            avgIndex.append(maxLeaves.index(returned))
            avgValue.append(returned)
            print("c = ", c, "\nj = ", j, '\n', maxLeaves.index(returned), "/", 2 ** (depth - 1), "\nvalue = ", returned, '\n')

        averageIndex = sum(avgIndex) / len(avgIndex)
        averagePercentage = 100 - (averageIndex / (2 ** (depth - 1) / 100))
        averageValue = sum(avgValue) / len(avgValue)
        print("Average index= ", averageIndex, "\nAverage value= ", averageValue, "\nAverage percent= ", averagePercentage)  # Ran on 09/12 16:18, average = 335.62
        avgIndexI.append(averageIndex)
        avgValueI.append(averageValue)
        avgPercentI.append(averagePercentage)

        # print("c = ", c)
        # c = c + cIncrement
        # print("c = ", c)

    elapsed_time = time.time() - start_time
    print("Time = ", elapsed_time)
    return avgIndexI, avgValueI, avgPercentI


# Assume that the number MCTS-iterations starting in a specific root node is limited (e.g. to 10 or 50). Make a similar
# assumption for the number of roll-outs starting in a particular (”snowcap”) leaf node (e.g. 1 or 5).
number_of_MCTS_iterations_in_root_node = 3
number_of_roll_outs_snowcap = 3
test_iteration_per_c_value = 10
depth = 15
root = Node
cList = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
c = cList[0]

# cMax = 0.5
# cIncrement = 0.5


def plotP(avgIndex, avgValue, avgPercent):
    increment = cList[-1]/0.5
    #cxaxis = numpy.linspace(cList[0], cList[-1], int(increment))
    cxaxis = cList
    # baseline = []
    # cxaxis = []
    i = 0
    #while i < cMax + cIncrement:
    # for i in range(len(c)):
    #     baseline.append(50)
    #     cxaxis.append(i)
    #     i = i+cIncrement
    plt.plot(cxaxis, avgValue, label = "Average Value")
    plt.plot(cxaxis, avgPercent, label="Average Percentage")
    baseline = plt.hlines(50, cList[0], cList[-1], label="Baseline")
    #plt.plot(cxaxis, baseline, label="Baseline")
    plt.xlabel("c")
    plt.ylabel("Value / Percentage")
    plt.title("Average value and Average percentage for different c in MCTS with UCB\n #rollouts = {}, #MCTS iterations root node = {}, #iterations per c = {}".format(number_of_roll_outs_snowcap, number_of_MCTS_iterations_in_root_node, test_iteration_per_c_value))
    #plt.title("MCTS iterations = ", number_of_MCTS_iterations_in_root_node, ", #rollouts = ", number_of_roll_outs_snowcap)#,
              #"test iterations = ", test_iteration_per_c_value)
    plt.legend()
    # plt.savefig('MCTS.png')
    # plt.savefig('MCTS.pdf')
    plt.show()


avgIndexI, avgValueI, avgPercentI = test()
print(avgIndexI)
print(avgValueI)
print(avgPercentI)
plotP(avgIndexI, avgValueI, avgPercentI)


