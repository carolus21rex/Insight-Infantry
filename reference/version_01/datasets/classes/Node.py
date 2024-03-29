import questions.Question as Question


class Node:
    def __init__(self, weight, expanding):
        self.weight = weight
        self.question = None
        self.expanding = expanding
        self.left = None
        self.right = None

    def append(self, left, right):
        self.left = left
        self.right = right

    def append(self, left):
        self.left = left

    def addQuestion(self):
        self.question = Question.randomQuestion()


