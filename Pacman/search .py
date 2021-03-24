# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from audioop import reverse

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def similarTinyMaze(problem):
    import random

    current = problem.getStartState()
    solution = []

    while (not (problem.isGoalState(current))):
        succ = problem.getSuccessors(current)
        no_of_successors = len(succ)
        random_succ_index = int(random.random() * no_of_successors)
        next = succ[random_succ_index]
        current = next[0]
        solution.append(next[1])

    print "The solution is ", solution

    return solution


class Node:
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def getState(self):
        return self.state

    def getParent(self):
        return self.parent

    def getAction(self):
        return self.action

    def getPathCost(self):
        return self.path_cost

    def setPathCost(self, cost):
        self.path_cost = cost


def depthFirstSearch(problem):
    node = Node(problem.getStartState(), None, None, 0)

    frontier = util.Stack()
    frontier.push(node)
    explored = []

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.getState()):
            solution = []
            while node.getParent() is not None:
                solution.append(node.getAction())
                node = node.getParent()
            solution.reverse()
            return solution

        for successor in problem.getSuccessors(node.getState()):
            if successor[0] not in explored:
                child = Node(successor[0], node, successor[1], successor[2])
                frontier.push(child)
        explored.append(node.getState())

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    node = Node(problem.getStartState(), None, None, 0)
    frontier = util.Queue()
    frontier.push(node)
    explored = []
    nodesInQueue = [node.getState()]
    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.getState()):
            solution = []
            while node.getParent() is not None:
                solution.append(node.getAction())
                node = node.getParent()
            solution.reverse()
            return solution
        for successor in problem.getSuccessors(node.getState()):
            child = Node(successor[0], node, successor[1], successor[2])
            if child.getState() not in explored and child.getState() not in nodesInQueue:
                frontier.push(child)
                nodesInQueue.append(child.getState())
        explored.append(node.getState())
    util.raiseNotDefined()


def uniformCostSearch(problem):
    node = Node(problem.getStartState(), None, None, 0)
    frontier = util.PriorityQueue()
    frontier.push(node, 0)
    explored = []
    nodesInQueue = [node.getState()]

    cost_so_far = dict()
    cost_so_far[node.getState()] = 0

    while not frontier.isEmpty():
        node = frontier.pop()
        if problem.isGoalState(node.getState()):
            solution = []
            while node.getParent() is not None:
                solution.append(node.getAction())
                node = node.getParent()
            solution.reverse()
            return solution
        for successor in problem.getSuccessors(node.getState()):
            child = Node(successor[0], node, successor[1], successor[2])
            new_cost = cost_so_far[node.getState()] + successor[2]
            if child.getState() not in explored and child.getState() not in nodesInQueue:
                frontier.push(child, new_cost)
                cost_so_far[child.getState()] = new_cost
                nodesInQueue.append(child.getState())
            if child.getState() in cost_so_far and new_cost < cost_so_far[child.getState()]:
                cost_so_far[child.getState()] = new_cost
                frontier.update(child, new_cost)
        explored.append(node.getState())


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    node = Node(problem.getStartState(), None, None, 0)
    frontier = util.PriorityQueue()
    frontier.push(node, 0)
    explored = []
    nodsInQueue = [node.getState()]

    cost_so_far = dict()
    cost_so_far[node.getState()] = 0

    while not problem.isGoalState(node.getState()):
        node = frontier.pop()
        if problem.isGoalState(node.getState()):
            solution = []
            while node.getParent() is not None:
                solution.append(node.getAction())
                node = node.getParent()
            solution.reverse()
            return solution

        explored.append(node.getState())
        for successor in problem.getSuccessors(node.getState()):
            child = Node(successor[0], node, successor[1], successor[2])
            new_cost = cost_so_far[node.getState()] + successor[2]
            if child.getState() not in explored and child.getState() not in nodsInQueue:
                frontier.push(child, new_cost + heuristic(child.getState(), problem))
                cost_so_far[child.getState()] = new_cost
                nodsInQueue.append(child.getState())
            if child.getState() in cost_so_far and new_cost < cost_so_far[child.getState()]:
                cost_so_far[child.getState()] = new_cost
                frontier.update(child, new_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
