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
def best_first_search(graph,start,goal,heuristic, path=[]):
    open_list = [(0,start)]
    closed_list = set()
    closed_list.add(start)

    while open_list:
        open_list.sort(key = lambda x: heuristic[x[1]], reverse=True)
        cost, node = open_list.pop()
        path.append(node)

        if node==goal:
            return cost, path

        closed_list.add(node)
        for neighbour, neighbour_cost in graph[node]:
            if neighbour not in closed_list:
                closed_list.add(node)
                open_list.append((cost+neighbour_cost, neighbour))

    return None


graph = {
    'A': [('B', 20), ('C', 15), ('D',16)],
    'B': [('A', 20)],
    'C': [('A', 15), ('E', 12)],
    'D': [('A', 16), ('E', )],
    'E': []
}

start = 'A'
goal = 'E'

heuristic = {
    'A': 25,
    'B': 20,
    'C': 12,
    'D': 13,
    'E': 0
}

result = best_first_search(graph, start, goal, heuristic)

if result:
    print(f"Minimum cost path from {start} to {goal} is {result[1]}")
    print(f"Cost: {result[0]}")
else:
    print(f"No path from {start} to {goal}")
