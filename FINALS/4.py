import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('./dataset/ToyotaCorolla.csv')

plt.boxplot([data["Price"],data["HP"],data["KM"]])

plt.xticks([1,2,3],["Price","HP","KM"])

plt.show()

#----------------------------------


class TreeNode:
    def __init__(self, value, children=[]):
        self.value = value
        self.children = children

def alphabeta(node, depth, alpha, beta, maximizing_player):
    global pruned_count

    if depth == 0 or not node.children:
        return node.value, [node.value]

    if maximizing_player:
        max_value, max_path = float("-inf"), []
        for child in node.children:
            value, path = alphabeta(child, depth - 1, alpha, beta, False)
            if value > max_value:
                max_value, max_path = value, [node.value] + path
            alpha = max(alpha, max_value)
            if alpha >= beta:
                pruned_count += len(child.children) + 1
                break
        return max_value, max_path

    else:
        min_value, min_path = float("inf"), []
        for child in node.children:
            value, path = alphabeta(child, depth - 1, alpha, beta, True)
            if value < min_value:
                min_value, min_path = value, [node.value] + path
            beta = min(beta, min_value)
            if alpha >= beta:
                pruned_count += len(child.children) + 1
                break
        return min_value, min_path

game_tree = TreeNode(0, [
    TreeNode(0, [
        TreeNode(0, [TreeNode(3), TreeNode(5)]),
        TreeNode(0, [TreeNode(6), TreeNode(9)])
    ]),
    TreeNode(0, [
        TreeNode(0, [TreeNode(1), TreeNode(2)]),
        TreeNode(0, [TreeNode(0), TreeNode(-1)])
    ])
])

pruned_count = 0
optimal_value, optimal_path = alphabeta(game_tree, 3, float('-inf'), float('inf'), True)

print("Optimal value:", optimal_value)
print("Total pruned nodes:", pruned_count)
