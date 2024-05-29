from mpl_toolkits.mplot3d import Axes3D

# 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D Surface Plot of 3D Data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

------------------------------

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
