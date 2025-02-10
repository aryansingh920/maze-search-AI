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


def solve_maze_greedy(maze, entry, goal, cell_size, frames, record_every=5):
    """
    Solves the maze using Greedy Best-First Search, which prioritizes cells based solely on
    their heuristic (Manhattan distance) to the goal. This algorithm does not guarantee the
    shortest path but often finds a quick solution.

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
        # Manhattan distance as the heuristic.
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    # Initialize the priority queue with the starting cell.
    open_set = []
    heapq.heappush(open_set, (heuristic(entry, goal), entry, [entry]))
    visited = set([entry])
    step = 0

    while open_set:
        h, current, path = heapq.heappop(open_set)
        step += 1

        # Record a frame every 'record_every' steps.
        if step % record_every == 0:
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal)
            frames.append(np.array(img))

        if current == goal:
            # Record the final frame with the solution highlighted.
            img = draw_maze_state(maze, cell_size, visited,
                                  current, path, entry, goal, final_path=path)
            frames.append(np.array(img))
            return path, visited

        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                heapq.heappush(open_set, (heuristic(
                    neighbor, goal), neighbor, new_path))

    return None, visited


def solve_maze_dijkstra(maze, entry, goal, cell_size, frames, record_every=5):
    """
    Solves the maze using Dijkstra's algorithm, which finds the shortest path by considering the
    cumulative cost from the start. Since each move has a cost of 1 in the maze, this algorithm
    will guarantee the shortest solution path.

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

    # Priority queue holds tuples of (cumulative_cost, current_cell, path_taken)
    open_set = []
    heapq.heappush(open_set, (0, entry, [entry]))
    # distances keeps track of the best cost found to reach each cell.
    distances = {entry: 0}
    visited = set()
    step = 0

    while open_set:
        g, current, path = heapq.heappop(open_set)
        if current in visited:
            continue  # Skip if this cell was already finalized.
        visited.add(current)
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
            new_cost = g + 1  # Each move has a cost of 1.
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_cost, neighbor, new_path))

    return None, visited


def mdp_value_iteration(maze, discount=0.9, threshold=1e-4):
    """
    Solves the maze as an MDP using Value Iteration.
    
    Maze representation:
       - 1: Wall (impassable)
       - 0: Free cell
       - 2: Entry (free cell)
       - 3: Goal (terminal state)
       
    Each move gives a reward of -1 (i.e. a cost of 1), and reaching the goal gives 0 reward.
    
    Parameters:
       maze: 2D list of integers representing the maze.
       discount: Discount factor (gamma) for future rewards.
       threshold: Convergence threshold.
       
    Returns:
       V: A dictionary mapping state (x, y) to its value.
       policy: A dictionary mapping state (x, y) to the optimal action (dx, dy).
               Terminal states (goal) will have a policy value of None.
    """
    height = len(maze)
    width = len(maze[0])

    # Define possible actions: up, right, down, left.
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def get_next_state(state, action):
        """Given a state and an action, return the next state.
        If the action would lead into a wall or off the grid, return the current state."""
        x, y = state
        dx, dy = action
        new_x, new_y = x + dx, y + dy
        if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height or maze[new_y][new_x] == 1:
            return state
        return (new_x, new_y)

    # Build state space (all non-wall cells) and record the goal states.
    states = set()
    goal_states = set()
    for y in range(height):
        for x in range(width):
            if maze[y][x] != 1:
                states.add((x, y))
                if maze[y][x] == 3:
                    goal_states.add((x, y))

    # Initialize value function to 0 for all states.
    V = {s: 0 for s in states}

    # --- Value Iteration Loop ---
    while True:
        delta = 0
        new_V = {}
        for s in states:
            # Terminal states keep a fixed value (0).
            if s in goal_states:
                new_V[s] = 0
                continue
            # Evaluate all actions for state s.
            action_values = []
            for a in actions:
                s_next = get_next_state(s, a)
                # Each move costs -1.
                value = -1 + discount * V[s_next]
                action_values.append(value)
            best_value = max(action_values)
            new_V[s] = best_value
            delta = max(delta, abs(best_value - V[s]))
        V = new_V
        if delta < threshold:
            break

    # --- Derive Policy from the Computed Value Function ---
    policy = {}
    for s in states:
        if s in goal_states:
            policy[s] = None
        else:
            best_action = None
            best_value = -float('inf')
            for a in actions:
                s_next = get_next_state(s, a)
                value = -1 + discount * V[s_next]
                if value > best_value:
                    best_value = value
                    best_action = a
            policy[s] = best_action

    return V, policy


def mdp_policy_iteration(maze, discount=0.9, threshold=1e-4):
    """
    Solves the maze as an MDP using Policy Iteration.
    
    Maze representation:
       - 1: Wall (impassable)
       - 0: Free cell
       - 2: Entry (free cell)
       - 3: Goal (terminal state)
       
    Each move gives a reward of -1, and the goal is terminal (reward 0).
    
    Parameters:
       maze: 2D list of integers representing the maze.
       discount: Discount factor (gamma) for future rewards.
       threshold: Convergence threshold used during policy evaluation.
       
    Returns:
       policy: A dictionary mapping state (x, y) to the optimal action (dx, dy).
               Terminal states (goal) will have a policy value of None.
       V: A dictionary mapping state (x, y) to its value under the optimal policy.
    """
    height = len(maze)
    width = len(maze[0])

    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    def get_next_state(state, action):
        """Given a state and an action, return the next state (or the same state if the move is invalid)."""
        x, y = state
        dx, dy = action
        new_x, new_y = x + dx, y + dy
        if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height or maze[new_y][new_x] == 1:
            return state
        return (new_x, new_y)

    # Build state space and identify goal states.
    states = set()
    goal_states = set()
    for y in range(height):
        for x in range(width):
            if maze[y][x] != 1:
                states.add((x, y))
                if maze[y][x] == 3:
                    goal_states.add((x, y))

    # --- Initialize Policy Arbitrarily ---
    import random
    policy = {}
    for s in states:
        if s in goal_states:
            policy[s] = None
        else:
            policy[s] = random.choice(actions)

    # Initialize value function.
    V = {s: 0 for s in states}

    # --- Policy Iteration Loop ---
    policy_stable = False
    while not policy_stable:
        # Policy Evaluation: update V until convergence under the current policy.
        while True:
            delta = 0
            new_V = {}
            for s in states:
                if s in goal_states or policy[s] is None:
                    new_V[s] = 0
                else:
                    a = policy[s]
                    s_next = get_next_state(s, a)
                    new_V[s] = -1 + discount * V[s_next]
                delta = max(delta, abs(new_V[s] - V[s]))
            V = new_V
            if delta < threshold:
                break

        # Policy Improvement: update the policy based on the current value function.
        policy_stable = True
        for s in states:
            if s in goal_states:
                continue
            best_action = None
            best_value = -float('inf')
            for a in actions:
                s_next = get_next_state(s, a)
                value = -1 + discount * V[s_next]
                if value > best_value:
                    best_value = value
                    best_action = a
            if best_action != policy[s]:
                policy[s] = best_action
                policy_stable = False

    return policy, V
