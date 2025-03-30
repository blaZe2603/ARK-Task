import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Create the map with obstacles
def create_map(size, num_obstacles):
    obstacles = []
    for _ in range(num_obstacles):
        x, y = np.random.uniform(0, size, 2)
        radius = np.random.uniform(2, 5)
        obstacles.append((x, y, radius))
    return obstacles

# Visualize the map with obstacles
def plot_map(size, obstacles):
    fig, ax = plt.subplots()
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
        ax.add_patch(circle)
    plt.grid()
    plt.show()

# Generate random nodes
def generate_random_nodes(size, num_nodes, obstacles):
    nodes = []
    while len(nodes) < num_nodes:
        x, y = np.random.uniform(0, size, 2)
        if not any((np.sqrt((x - obs[0])**2 + (y - obs[1])**2) < obs[2]) for obs in obstacles):
            nodes.append((x, y))
    return nodes

# Calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Check if the path between two points is collision-free
def is_collision_free(p1, p2, obstacles):
    for obs in obstacles:
        closest_point = np.array([p1[0], p1[1]], dtype=np.float64)  # Ensure float type
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float64)
        t = np.clip(np.dot([obs[0] - p1[0], obs[1] - p1[1]], direction) / np.dot(direction, direction), 0, 1)
        closest_point += t * direction
        distance_to_obstacle = np.sqrt((closest_point[0] - obs[0])**2 + (closest_point[1] - obs[1])**2)
        if distance_to_obstacle < obs[2]:
            return False
    return True

# Create the PRM graph
def create_prm_graph(nodes, obstacles, threshold):
    graph = nx.Graph()
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j and euclidean_distance(node1, node2) < threshold:
                if is_collision_free(node1, node2, obstacles):
                    graph.add_edge(i, j, weight=euclidean_distance(node1, node2))
    return graph

# Parameters
map_size = 50
num_obstacles = 0
num_nodes = 100
connection_threshold = 10  # Maximum distance for connecting nodes

# Generate map, nodes, and PRM graph
obstacles = create_map(map_size, num_obstacles)
nodes = generate_random_nodes(map_size, num_nodes, obstacles)
prm_graph = create_prm_graph(nodes, obstacles, connection_threshold)

# Define start and goal points
start = (5, 5)
goal = (45, 45)
nodes_with_start_goal = nodes + [start, goal]
start_idx = len(nodes)
goal_idx = len(nodes) + 1

# Connect start and goal points to the graph
for i, node in enumerate(nodes_with_start_goal[:-2]):
    if euclidean_distance(start, node) < connection_threshold and is_collision_free(start, node, obstacles):
        prm_graph.add_edge(start_idx, i, weight=euclidean_distance(start, node))
    if euclidean_distance(goal, node) < connection_threshold and is_collision_free(goal, node, obstacles):
        prm_graph.add_edge(goal_idx, i, weight=euclidean_distance(goal, node))

# Check if start and goal nodes are connected
if not prm_graph.has_node(start_idx) or not prm_graph.has_node(goal_idx):
    raise ValueError("Start or goal node is not connected to the PRM graph. Try increasing the connection_threshold.")

# Find the shortest path using Dijkstra's algorithm
shortest_path = nx.shortest_path(prm_graph, source=start_idx, target=goal_idx, weight='weight')

# Visualize the PRM graph and the shortest path
plt.figure(figsize=(8, 8))
plot_map(map_size, obstacles)
nx.draw(prm_graph, pos={i: node for i, node in enumerate(nodes_with_start_goal)}, node_size=50, with_labels=False, edge_color="gray")
path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
nx.draw_networkx_edges(prm_graph, pos={i: node for i, node in enumerate(nodes_with_start_goal)}, edgelist=path_edges, edge_color="blue", width=2)
plt.scatter(*zip(*[start, goal]), color='green', s=100, label="Start/Goal")
plt.title("Shortest Path (PRM*)")
plt.legend()
plt.show()