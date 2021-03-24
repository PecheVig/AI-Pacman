# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalMoves[chosenIndex]

    def findNext(self, pacman, matrix):
        dist = 100000
        for i, matrix_i in enumerate(matrix):
            for j, value in enumerate(matrix_i):
                if value:
                    aux = abs(pacman[0] - i) + abs(pacman[1] - j)
                    if aux < dist:
                        dist = aux
        return dist

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        ghostPosition = newGhostStates[0].getPosition()
        ghostPosiblePositions = []
        ghostPosiblePositions.append((ghostPosition[0], ghostPosition[1]))
        ghostPosiblePositions.append((ghostPosition[0], ghostPosition[1] + 1))
        ghostPosiblePositions.append((ghostPosition[0], ghostPosition[1] - 1))
        ghostPosiblePositions.append((ghostPosition[0] + 1, ghostPosition[1]))
        ghostPosiblePositions.append((ghostPosition[0] - 1, ghostPosition[1]))

        pacmanPos = currentGameState.getPacmanPosition()
        if newPos in ghostPosiblePositions:
            return -99999
        initialDistance = self.findNext(pacmanPos, newFood)
        newDistance = self.findNext(newPos, newFood)
        if initialDistance > newDistance:
            return successorGameState.getScore()
        return successorGameState.getScore() - 1


class RandomAgent(Agent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        chosenIndex = random.choice(range(0, len(legalMoves)))
        return legalMoves[chosenIndex]


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return self.max_value(0, gameState)

    def max_value(self, currentDepth, gameState):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        val = float('-inf')
        actionF = ""
        for action in gameState.getLegalActions():
            suc = gameState.generateSuccessor(0, action)
            mini = self.min_value(currentDepth, suc, 1)
            if mini > val:
                val = mini
                actionF = action
        return actionF if currentDepth == 0 else val

    def min_value(self, currentDepth, gameState, ghostIndex):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        val = float('inf')
        for action in gameState.getLegalActions(ghostIndex):
            suc = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == gameState.getNumAgents() - 1:
                max_value = self.max_value(currentDepth + 1, suc)
            else:
                max_value = self.min_value(currentDepth, suc, ghostIndex + 1)
            val = min(val, max_value)
        return val


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return self.max_value(0, gameState, float('-inf'), float('inf'))

    def max_value(self, currentDepth, gameState, alpha, beta):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        val = float('-inf')
        actionF = ""
        for action in gameState.getLegalActions():
            suc = gameState.generateSuccessor(0, action)
            mini = self.min_value(currentDepth, suc, 1, alpha, beta)
            if mini > val:
                val = mini
                actionF = action
            alpha = max(alpha, mini)
            if beta < alpha:
                break
        return actionF if currentDepth == 0 else val

    def min_value(self, currentDepth, gameState, ghostIndex, alpha, beta):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        val = float('inf')
        for action in gameState.getLegalActions(ghostIndex):
            suc = gameState.generateSuccessor(ghostIndex, action)
            if ghostIndex == gameState.getNumAgents() - 1:
                max_value = self.max_value(currentDepth + 1, suc, alpha, beta)
            else:
                max_value = self.min_value(currentDepth, suc, ghostIndex + 1, alpha, beta)
            val = min(val, max_value)
            beta = min(beta, max_value)
            if beta < alpha:
                break
        return val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
