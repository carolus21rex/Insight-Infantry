import math

import Node
import random

# settings
weightRange = 16


def initLeftMatrix(nodes, questions):
    maximum = 2**(questions-1)-2
    index = 0
    for x in range(index, maximum):
        nodes[x].append(nodes[2*x+1], nodes[2*x+2])
        nodes[x].addQuestion()

    return nodes


def initRightMatrix(nodes, questions):
    maximum = 2**questions+2**(questions-1)-2
    index = 2**(questions-1)-1
    for x in range(index, maximum):
        nodes[x].append(nodes[index+math.ceil((x+maximum-2*index)/2)])

    return nodes


def generateMatrix(questions):
    nodes = [
        Node(random.randint(-weightRange, weightRange), False)
        for _ in range(2**questions+2**(questions-1)-2)
    ]
    nodes = initLeftMatrix(nodes, questions)
    nodes = initRightMatrix(nodes, questions)

    return nodes[0]


class DecisionMatrix:
    def __init__(self, questions):
        self.nodes = generateMatrix(questions)

