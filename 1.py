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
-----------------------------------------

from collections import defaultdict

def bfs(graph, start, goal):
    visited = set()
    queue = [(start, [start])]
    
    while queue:
        node, path = queue.pop(0)
        visited.add(node)
        
        if node == goal:
            return path
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
                visited.add(neighbor)
    
    return None

# Function to create a graph from user input
def create_graph():
    graph = defaultdict(list)
    while True:
        node = input("Enter node (or 'done' to finish): ").strip()
        if node.lower() == 'done':
            break
        neighbors = input("Enter neighbors of {} separated by space: ".format(node)).split()
        graph[node] = neighbors
    return graph

# Example usage
graph = create_graph()
start_node = input("Enter start node: ").strip()
goal_node = input("Enter goal node: ").strip()

path = bfs(graph, start_node, goal_node)
if path:
    print("Path found:", path)
else:
    print("Path not found")
