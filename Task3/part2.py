from autonavsim2d import AutoNavSim2D
import cv2 as cv
import numpy as np



def my_planner(grid, matrix, start_loc, goal_loc):
    # your own custom path planning algorithm here
    path = []
    runtime = ''

    # Load the image
    image = cv.imread("Task3/maze.png")
    height, width, _ = image.shape
    image = cv.imread("Task3/maze.png", cv.IMREAD_GRAYSCALE)

    # Define start and end points
    start_1 = (40, height-30)
    start_2 = (150, 30)
    end_1 = (100, height-30)
    end_2 = (450, height-30)

    NODES = 30
    RADIUS = 100
    RADIUS_SQ = RADIUS ** 2
    node = np.random.randint([20, 30], [120, height-30], size=(NODES, 2), dtype=np.int16)

    node_possible = [start_1]
    # node_possible = [start_2]
    nodes = []
    graph = {}

    # Faster distance computation
    def squ_dist(p1, p2):
        return np.sum((p1 - p2) ** 2)

    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # Check if we can connect two points without an obstacle
    def check_white_pixels(image, point1, point2):
        line_pixels = image.copy()
        cv.line(line_pixels, point1, point2, 128, 1) 
        line_points = cv.line(np.zeros_like(image), point1, point2, 128, 1)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if line_points[y, x] == 128:  
                    if not image[y, x] == 255 and not image[y, x] == 128:
                        return False
        return True

    #checks if we satisfy both 
    def dist_check(image, p1, p2):
        if check_white_pixels(image, p1, p2):
            return squ_dist(p1, p2) < RADIUS_SQ
        else:
            return False  

    # Add valid nodes to node_possible
    for i in node:
        if (image[i[1], i[0]] == 255).all():
            cv.circle(image, tuple(i), 2, (128, 0, 0), -1)  
            node_possible.append(i)  

    node_possible.append(end_1)  
    node_possible = np.array(node_possible)
    cv.circle(image, start_1, 2, 128, -1)
    cv.circle(image, end_1, 2, 128, -1)

    # Check connectivity between nodes
    for i in range(len(node_possible)):
        for j in range(i + 1, len(node_possible)):  
            if dist_check(image, node_possible[i], node_possible[j]):
                cv.line(image, tuple(node_possible[i]), tuple(node_possible[j]), 128, thickness=1, lineType=8, shift=0)
                nodes.append((node_possible[i], node_possible[j]))


    # Build the graph with distances
    for (node1, node2) in nodes:
        node1_tuple = tuple(node1)
        node2_tuple = tuple(node2)
        dist = squ_dist(node1, node2)
        
        if node1_tuple not in graph:
            graph[node1_tuple] = []
        if node2_tuple not in graph:
            graph[node2_tuple] = []
        
        graph[node1_tuple].append((node2_tuple, dist))
        graph[node2_tuple].append((node1_tuple, dist))  

    # Dijkstra's algorithm
    def dijkstra(graph, start, goal):
        distances = {start: 0}
        previous_nodes = {start: None}
        unvisited_nodes = list(graph.keys())  # List of all nodes to visit
        
        while unvisited_nodes:
            current_node = None
            for node in unvisited_nodes:
                if current_node is None:
                    current_node = node
                elif distances.get(node, float('inf')) < distances.get(current_node, float('inf')):
                    current_node = node

            if distances.get(current_node, float('inf')) == float('inf'):
                break

            if current_node == goal:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                return path[::-1], distances[goal]  # Return reversed path and distance

            for neighbor, weight in graph[current_node]:
                new_distance = distances[current_node] + weight
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node

            unvisited_nodes.remove(current_node)

        return None, float('inf')

    # Convert start and goal nodes to tuples of integers
    start_node = tuple(map(int, node_possible[0]))
    goal_node = tuple(map(int, node_possible[-1]))

    # Find the shortest path
    path, total_distance = dijkstra(graph, start_node, goal_node)



        

    return (path, runtime)


# parameter configuration
config = {
    "show_frame": True,
    "show_grid": False,
    "map": None
}
nav = AutoNavSim2D(
    custom_planner=my_planner,
    custom_motion_planner='default',
    window='amr',
    config=config
)

nav.run()