import pygame
import graphUI
from node_color import white, yellow, black, red, blue, purple, orange, green

"""
Feel free print graph, edges to console to get more understand input.
Do not change input parameters
Create new function/file if necessary
"""
from collections import deque
import collections
from queue import Queue, PriorityQueue
#from queues import PriorityQueue
import math
import bisect

def BFS(graph, edges, edge_id, start, goal):
    """
    BFS search
    """
    # TODO: your code

    # Queue for traversing the
    # graph in the BFS
    parents = {start: None}
    queue = deque([start])
    visited=set()
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        if node not in visited:
            visited.add(node)
            graph[node][3] = yellow
            graphUI.updateUI()

            for neighbor in graph[node][1]:
                graph[neighbor][2] = white
                graphUI.updateUI()
                edges[edge_id(node, neighbor)][1] = white

                # graph[node][3] = yellow
                # graph[neighbor][2] = white
                # graphUI.updateUI()
                # edges[edge_id(node, neighbor)][1] = white
                if neighbor not in parents:
                    parents[neighbor] = node

                    graph[neighbor][3] = red
                    graphUI.updateUI()

                    # queue.append(neighbor)
                    # if node == goal:
                    # break
                    queue.append(neighbor)
            graph[node][3] = blue
            graphUI.updateUI()


    graph[goal][3] = purple
    graphUI.updateUI()
    graph[start][3] = orange
    graphUI.updateUI()
    path = [goal]
    while parents[goal] is not None:
        path.insert(0, parents[goal])
        goal = parents[goal]

    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = green
    graphUI.updateUI()

    print("Implement BFS algorithm.")
    pass


def DFS(graph, edges, edge_id, start, goal):
    """
    DFS search
    """
    # TODO: your code
    visited = set()
    stack = [start]
    parents = {start: None}

    while stack:
        node = stack.pop()
        if node == goal:
            break
        if node not in visited:
            visited.add(node)
            graph[node][3] = yellow
            graphUI.updateUI()

            for neighbor in graph[node][1]:

                graph[neighbor][2] = white
                graphUI.updateUI()
                edges[edge_id(node, neighbor)][1] = white

                if neighbor not in visited:
                    parents[neighbor] = node

                    graph[neighbor][3] = red
                    graphUI.updateUI()
                    stack.append(neighbor)
            graph[node][3] = blue
            graphUI.updateUI()

    graph[goal][3] = purple
    graphUI.updateUI()
    graph[start][3] = orange
    graphUI.updateUI()
    path = [goal]
    while parents[goal] is not None:
        path.insert(0, parents[goal])
        goal = parents[goal]

    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = green
    graphUI.updateUI()
                    
    print("Implement DFS algorithm.")
    pass

#Hàm lấy chi phí giữa 2 đỉnh
def weight(graph, start, goal):
    return math.sqrt((graph[start][0][0] - graph[goal][0][0])**2 + (graph[start][0][1] - graph[goal][0][1])**2)

def UCS(graph, edges, edge_id, start, goal):
    """
    Uniform Cost Search search
    """
    frontier = PriorityQueue()
    frontier.put((0, start))  # (priority, node)
    explored = []
    parents = {start: None}
    while frontier:
        if frontier.empty():
            raise Exception("No way Exception")

        ucs_w, current_node = frontier.get()

        if current_node == goal:
            break

        explored.append(current_node)
        graph[current_node][3] = yellow
        graphUI.updateUI()

        # if current_node == goal:
        #   break

        for node in graph[current_node][1]:
            graph[node][2] = white
            graphUI.updateUI()
            edges[edge_id(current_node, node)][1] = white

            if node not in explored:
                frontier.put((ucs_w + weight(graph, current_node, node), node))
                parents[node] = current_node

                graph[node][3] = red
                graphUI.updateUI()
        graph[current_node][3] = blue
        graphUI.updateUI()

    graph[goal][3] = purple
    graphUI.updateUI()
    graph[start][3] = orange
    graphUI.updateUI()
    path = [goal]
    while parents[goal] is not None:
        path.insert(0, parents[goal])
        goal = parents[goal]

    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = green
    graphUI.updateUI()

    print("Implement Uniform Cost Search algorithm.")
    pass

def manhattan(graph,node, goal):
    return abs(graph[node][0][0]-graph[goal][0][0])+abs(graph[node][0][1]-graph[goal][0][1])
    #return abs(node.x - goal.x) + abs(node.y - goal.y)

def euclidean_distance(graph,node, goal):
    return math.sqrt((graph[node][0][0]-graph[goal][0][0]) ** 2 + (graph[node][0][1]-graph[goal][0][1]) ** 2)
    #return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)

def reconstruct_path(came_from, start, end):
    """
    Reconstructs the came_from dictionary to be a list of tiles
    we can traverse and draw later.
    :param came_from: dictionary
    :param start: Tile
    :param end: Tile
    :return: List path
    """
    current = end
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start)  # optional
    path.reverse()      # optional
    return path

def AStar(graph, edges, edge_id, start, goal):
    """
    A star search
    """
    #parents = {start: None}
    # The open and closed sets
    openset = set()
    closedset = set()
    # Current point is the starting point
    current = start
    # Add the starting point to the open set
    openset.add(current)
    # While the open set is not empty
    while openset:
        # Find the item in the open set with the lowest G + H score
        #current = min(openset, key=lambda o: o.G + o.H)
        current = min(openset)
        # If it is the item we want, retrace the path and return it
        graph[current][3] = yellow
        graphUI.updateUI()

        if current == goal:
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(current)
            #return path[::-1]
            return
        # Remove the item from the open set
        openset.remove(current)
        # Add it to the closed set
        closedset.add(current)
        # Loop through the node's children/siblings
        for node in graph[current][1]:
            graph[node][2] = white
            graphUI.updateUI()
            edges[edge_id(current, node)][1] = white
            # If it is already in the closed set, skip it
            if node in closedset:
                continue
            # Otherwise if it is already in the open set
            if node in openset:
                # Check if we beat the G score
                new_g = current.G + current.move_cost(node)
                if node.G > new_g:
                    # If so, update the node to have a new parent
                    node.G = new_g
                    node.parent = current
                graph[node][3] = red
                graphUI.updateUI()
            else:
                # If it isn't in the open set, calculate the G and H score for the node
                node.G = current.G + current.move_cost(node)
                node.H = manhattan(graph,node, goal)
                # Set the parent to our current item
                node.parent = current
                # Add it to the set
                openset.add(node)
                graph[node][3] = red
                graphUI.updateUI()
        graph[current][3] = blue
        graphUI.updateUI()

    graph[goal][3] = purple
    graphUI.updateUI()
    graph[start][3] = orange
    graphUI.updateUI()
    print("Implement A* algorithm.")
    pass


def example_func(graph, edges, edge_id, start, goal):
    """
    This function is just show some basic feature that you can use your project.
    @param graph: list - contain information of graph (same value as global_graph)
                    list of object:
                     [0] : (x,y) coordinate in UI
                     [1] : adjacent node indexes
                     [2] : node edge color
                     [3] : node fill color
                Ex: graph = [
                                [
                                    (139, 140),             # position of node when draw on UI
                                    [1, 2],                 # list of adjacent node
                                    (100, 100, 100),        # grey - node edged color
                                    (0, 0, 0)               # black - node fill color
                                ],
                                [(312, 224), [0, 4, 2, 3], (100, 100, 100), (0, 0, 0)],
                                ...
                            ]
                It means this graph has Node 0 links to Node 1 and Node 2.
                Node 1 links to Node 0,2,3 and 4.
    @param edges: dict - dictionary of edge_id: [(n1,n2), color]. Ex: edges[edge_id(0,1)] = [(0,1), (0,0,0)] : set color
                    of edge from Node 0 to Node 1 is black.
    @param edge_id: id of each edge between two nodes. Ex: edge_id(0, 1) : id edge of two Node 0 and Node 1
    @param start: int - start vertices/node
    @param goal: int - vertices/node to search
    @return:
    """

    # Ex1: Set all edge from Node 1 to Adjacency node of Node 1 is green edges.
    node_1 = graph[3]
    for adjacency_node in node_1[1]:
        graph[adjacency_node][3] = orange
        graphUI.updateUI()

    # Ex2: Set color of Node 2 is Red
    graph[2][2] = white
    graphUI.updateUI()

    # Ex3: Set all edge between node in a array.
    path = [4, 7, 9]  # -> set edge from 4-7, 7-9 is blue
    for i in range(len(path) - 1):
        edges[edge_id(path[i], path[i + 1])][1] = blue
    graphUI.updateUI()
    print(graph[0][0][0])
    print(graph[0][0][1])
    print(graph[1][0][0])
    print(graph[1][0][1])
    print(weight(graph, 0, 1))
