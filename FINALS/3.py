import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic n-dimensional data
np.random.seed(42)
data_nd = np.random.rand(10,10)

# For demonstration, let's assume we want to visualize this data directly as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data_nd, cmap='viridis', annot=True)
plt.title('Heatmap of n-Dimensional Data (Direct Visualization)')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.show()
#------------------------------

class TreeNode:
    def __init__(self, value, children=[]):
        self.value = value
        self.children = children

def minimax(node, depth, maximizing_player):
    if depth == 0 or not node.children:
        return node.value, [node.value]

    best_value = float("-inf") if maximizing_player else float("inf")
    best_path = []
    
    for child in node.children:
        value, path = minimax(child, depth - 1, not maximizing_player)
        if (maximizing_player and value > best_value) or (not maximizing_player and value < best_value):
            best_value, best_path = value, [node.value] + path

    return best_value, best_path

game_tree = TreeNode(0, [
    TreeNode(1, [TreeNode(3), TreeNode(12)]),
    TreeNode(4, [TreeNode(8), TreeNode(2)])
])

optimal_value, optimal_path = minimax(game_tree, 2, True)
print("Optimal value:", optimal_value)
print("Optimal path:", optimal_path)
