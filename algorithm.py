"""
Created on 10/02/2025

@author: Aryan

Filename: algorithm.py

Relative Path: algorithm.py
"""

import numpy as np
from path_generator import draw_maze_state, get_neighbors


def solve_maze_dfs(maze, entry, goal, cell_size, frames, record_every=5):
    """
    Solves the maze using DFS starting from the entry (value 2) until the goal (value 3)
    is reached. During the search, the state is recorded every few steps into the
    provided frames list (to later produce a video).
    
    Parameters:
      - maze: 2D list representing the maze.
      - entry: starting coordinate (x, y)
      - goal: goal coordinate (x, y)
      - cell_size: pixel size for drawing each cell.
      - frames: list to which the visualization frames (as numpy arrays) are appended.
      - record_every: record a frame every N steps (to reduce total frame count).
      
    Returns:
      A tuple (final_path, visited) where final_path is a list of coordinates from entry to goal.
    """
    stack = [(entry, [entry])]
    visited = set([entry])
    step = 0
    while stack:
        current, path = stack.pop()
        step += 1
        # Record a frame every 'record_every' steps:
        if step % record_every == 0:
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal)
            frames.append(np.array(img))
        if current == goal:
            # When the goal is found, record a final frame with the solution highlighted.
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal, final_path=path)
            frames.append(np.array(img))
            return path, visited
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))
    return None, visited


def solve_maze_bfs(maze, entry, goal, cell_size, frames, record_every=5):
    """
    Solves the maze using Breadth-First Search (BFS) starting from the entry until the goal is reached.
    It records frames of the search progress for visualization.
    
    Parameters:
      - maze: 2D list representing the maze.
      - entry: starting coordinate (x, y)
      - goal: goal coordinate (x, y)
      - cell_size: pixel size for drawing each cell.
      - frames: list to which the visualization frames (as numpy arrays) are appended.
      - record_every: record a frame every N steps (to reduce total frame count).
      
    Returns:
      A tuple (final_path, visited) where final_path is a list of coordinates from entry to goal,
      and visited is a set of all coordinates that were visited during the search.
    """
    from collections import deque

    queue = deque()
    queue.append((entry, [entry]))
    visited = set([entry])
    step = 0

    while queue:
        current, path = queue.popleft()
        step += 1

        # Record a frame every 'record_every' steps.
        if step % record_every == 0:
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal)
            frames.append(np.array(img))

        if current == goal:
            # Record a final frame with the solution highlighted.
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal, final_path=path)
            frames.append(np.array(img))
            return path, visited

        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    return None, visited


def solve_maze_astar(maze, entry, goal, cell_size, frames, record_every=5):
    """
    Solves the maze using the A* search algorithm starting from the entry until the goal is reached.
    It records frames of the search progress for visualization.
    
    Parameters:
      - maze: 2D list representing the maze.
      - entry: starting coordinate (x, y)
      - goal: goal coordinate (x, y)
      - cell_size: pixel size for drawing each cell.
      - frames: list to which the visualization frames (as numpy arrays) are appended.
      - record_every: record a frame every N steps (to reduce total frame count).
      
    Returns:
      A tuple (final_path, visited) where final_path is a list of coordinates from entry to goal,
      and visited is a set of all coordinates that were visited during the search.
    """
    import heapq

    def heuristic(cell, goal):
        # Using Manhattan distance as the heuristic.
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    open_set = []
    start_cost = 0
    start_f = start_cost + heuristic(entry, goal)
    heapq.heappush(open_set, (start_f, start_cost, entry, [entry]))

    # For visualization, we'll mark nodes as visited when they are discovered.
    visited = set([entry])
    step = 0

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        step += 1

        # Record a frame every 'record_every' steps.
        if step % record_every == 0:
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal)
            frames.append(np.array(img))

        if current == goal:
            # Record a final frame with the solution highlighted.
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal, final_path=path)
            frames.append(np.array(img))
            return path, visited

        for neighbor in get_neighbors(maze, current):
            # Assuming each move has a cost of 1.
            new_cost = g + 1
            if neighbor not in visited:
                visited.add(neighbor)
                f_new = new_cost + heuristic(neighbor, goal)
                new_path = path + [neighbor]
                heapq.heappush(open_set, (f_new, new_cost, neighbor, new_path))

    return None, visited
