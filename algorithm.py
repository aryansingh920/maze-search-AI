"""
Created on 10/02/2025

@author: Aryan

Filename: algorithm.py

Relative Path: algorithm.py
"""

from PIL import Image, ImageDraw
import numpy as np
from path_generator import draw_maze_state, get_neighbors

# MDP
REWARD_FOR_GOAL = 1000  # Large enough to offset a long path
STEP_COST = 0     # Zero cost per step
DISCOUNT = 0.9999
THETA = 1e-3



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


def draw_mdp_policy(maze, policy, path, cell_size=20, entry=None, goal=None):
    """
    Creates a visualization of the MDP policy:
      - Each cell in 'maze' that is not a wall has an arrow indicating the action in 'policy'.
      - Optionally highlights 'entry' and 'goal'.
      - Optionally highlights the final 'path' in a distinct color.
    
    Parameters:
      maze (list of lists): 2D maze layout (0=open, 1=wall, 2=entry, 3=goal).
      policy (dict): Mapping from (x, y) -> action ('up', 'down', 'left', 'right', or None).
      path (list): Sequence of (x, y) coordinates that forms the path from entry to goal.
      cell_size (int): Number of pixels per cell when drawing.
      entry (tuple): (x, y) for the entry cell.
      goal (tuple): (x, y) for the goal cell.

    Returns:
      PIL.Image: The generated policy visualization as an Image object.
    """
    height = len(maze)
    width = len(maze[0])

    # Create a blank image with a white background
    img = Image.new("RGB", (width * cell_size, height * cell_size), "white")
    draw = ImageDraw.Draw(img)

    # Helper function to convert maze coordinates to pixel coordinates
    def cell_to_pixel(x, y):
        return (x * cell_size, y * cell_size)

    # Draw the maze (walls, free cells) plus optional entry/goal highlights
    for y in range(height):
        for x in range(width):
            top_left = cell_to_pixel(x, y)
            bottom_right = (top_left[0] + cell_size, top_left[1] + cell_size)

            if maze[y][x] == 1:
                # Wall
                draw.rectangle([top_left, bottom_right],
                               fill=(50, 50, 50))  # dark gray
            else:
                # Open cell
                draw.rectangle([top_left, bottom_right], fill=(230, 230, 230))

            # If this cell is on the final path, color it slightly differently
            if (x, y) in path:
                draw.rectangle([top_left, bottom_right],
                               fill=(173, 216, 230))  # light blue

    # Highlight entry and goal distinctly
    if entry is not None:
        ex, ey = entry
        draw.rectangle([
            (ex * cell_size, ey * cell_size),
            (ex * cell_size + cell_size, ey * cell_size + cell_size)
        ], fill=(144, 238, 144))  # light green

    if goal is not None:
        gx, gy = goal
        draw.rectangle([
            (gx * cell_size, gy * cell_size),
            (gx * cell_size + cell_size, gy * cell_size + cell_size)
        ], fill=(255, 182, 193))  # light pink

    # Now draw arrows for the policy, skipping walls and the goal itself
    arrow_color = (0, 0, 0)  # black
    for (x, y), action in policy.items():
        if maze[y][x] == 1:
            continue  # skip walls
        if (x, y) == goal:
            continue  # skip drawing arrow in goal cell
        if action is None:
            continue  # no action at terminal or non-defined

        # Center of the cell
        cx = x * cell_size + cell_size // 2
        cy = y * cell_size + cell_size // 2

        # Depending on the action, draw an arrow
        arrow_length = cell_size * 0.35
        if action == 'up':
            # arrow from (cx, cy) to (cx, cy - arrow_length)
            draw.line([(cx, cy), (cx, cy - arrow_length)],
                      fill=arrow_color, width=2)
            # small triangle head
            draw.polygon([
                (cx - 3, cy - arrow_length + 6),
                (cx + 3, cy - arrow_length + 6),
                (cx, cy - arrow_length - 2)
            ], fill=arrow_color)
        elif action == 'down':
            draw.line([(cx, cy), (cx, cy + arrow_length)],
                      fill=arrow_color, width=2)
            draw.polygon([
                (cx - 3, cy + arrow_length - 6),
                (cx + 3, cy + arrow_length - 6),
                (cx, cy + arrow_length + 2)
            ], fill=arrow_color)
        elif action == 'left':
            draw.line([(cx, cy), (cx - arrow_length, cy)],
                      fill=arrow_color, width=2)
            draw.polygon([
                (cx - arrow_length + 6, cy - 3),
                (cx - arrow_length + 6, cy + 3),
                (cx - arrow_length - 2, cy)
            ], fill=arrow_color)
        elif action == 'right':
            draw.line([(cx, cy), (cx + arrow_length, cy)],
                      fill=arrow_color, width=2)
            draw.polygon([
                (cx + arrow_length - 6, cy - 3),
                (cx + arrow_length - 6, cy + 3),
                (cx + arrow_length + 2, cy)
            ], fill=arrow_color)

    return img


def solve_maze_value_iteration(maze, entry, goal, discount=DISCOUNT, theta=THETA):
    """
    Solves the maze using MDP value iteration.
    
    Parameters:
      - maze: 2D list representing the maze (walls=1, open=0, entry=2, goal=3)
      - entry: Tuple (x, y) representing the start position.
      - goal: Tuple (x, y) representing the goal (terminal) state.
      - discount: Discount factor for future rewards.
      - theta: Convergence threshold.
      
    Returns:
      - policy: Dictionary mapping state (x,y) to the best action ('up', 'down', etc.)
      - V: Dictionary mapping state (x,y) to its computed value.
      - path: List of states (tuples) representing the path from entry to goal following the derived policy.
    """
    height = len(maze)
    width = len(maze[0])

    # Create the state space: all coordinates that are not walls.
    states = [(x, y) for y in range(height)
              for x in range(width) if maze[y][x] != 1]

    # Initialize the value function.
    V = {s: 0 for s in states}

    # Define possible actions.
    actions = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }

    def get_next_state(state, action):
        """Returns the next state when taking an action from a given state."""
        dx, dy = action
        x, y = state
        nx, ny = x + dx, y + dy
        # Check grid bounds and wall condition.
        if nx < 0 or nx >= width or ny < 0 or ny >= height or maze[ny][nx] == 1:
            return state  # Invalid move; stay in place.
        return (nx, ny)

    # Value Iteration loop.
    while True:
        delta = 0
        for s in states:
            # Skip update for terminal state.
            if s == goal:
                continue
            max_value = float("-inf")
            for a in actions.values():
                next_state = get_next_state(s, a)
                # Reward: 0 if moving into the goal, otherwise -1.
                r = REWARD_FOR_GOAL if next_state == goal else STEP_COST
                value = r + discount * V[next_state]
                if value > max_value:
                    max_value = value
            delta = max(delta, abs(max_value - V[s]))
            V[s] = max_value
        if delta < theta:
            break

    # Extract the optimal policy.
    policy = {}
    for s in states:
        if s == goal:
            policy[s] = None  # No action needed at terminal.
        else:
            best_action = None
            best_value = float("-inf")
            for a_name, a in actions.items():
                next_state = get_next_state(s, a)
                r = REWARD_FOR_GOAL if next_state == goal else STEP_COST
                value = r + discount * V[next_state]
                if value > best_value:
                    best_value = value
                    best_action = a_name
            policy[s] = best_action

    # Derive a path from entry to goal following the policy.
    path = []
    current = entry
    visited_path = set()
    while current != goal and current not in visited_path:
        path.append(current)
        visited_path.add(current)
        action_name = policy.get(current)
        if action_name is None:
            break
        next_state = get_next_state(current, actions[action_name])
        # If no progress is made, break out.
        if next_state == current:
            break
        current = next_state
    path.append(current)  # Append the final state.

    return policy, V, path


def solve_maze_policy_iteration(maze, entry, goal, discount=DISCOUNT, theta=THETA):
    """
    Solves the maze using MDP policy iteration.
    
    Parameters:
      - maze: 2D list representing the maze (walls=1, open=0, entry=2, goal=3)
      - entry: Tuple (x, y) for the starting cell.
      - goal: Tuple (x, y) for the terminal goal cell.
      - discount: Discount factor for future rewards.
      - theta: Convergence threshold for policy evaluation.
      
    Returns:
      - policy: Dictionary mapping state (x,y) to the optimal action.
      - V: Dictionary mapping state (x,y) to its evaluated value.
      - path: List of states representing the path from entry to goal following the optimal policy.
    """
    height = len(maze)
    width = len(maze[0])

    # Create the state space.
    states = [(x, y) for y in range(height)
              for x in range(width) if maze[y][x] != 1]

    actions = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }

    def get_next_state(state, action):
        dx, dy = action
        x, y = state
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= width or ny < 0 or ny >= height or maze[ny][nx] == 1:
            return state
        return (nx, ny)

    # Initialize an arbitrary policy for non-terminal states.
    policy = {}
    for s in states:
        policy[s] = None if s == goal else list(actions.keys())[0]

    # Initialize value function.
    V = {s: 0 for s in states}

    policy_stable = False
    while not policy_stable:
        # Policy Evaluation: Iteratively update V for the current policy.
        while True:
            delta = 0
            for s in states:
                if s == goal:
                    continue
                a_name = policy[s]
                a = actions[a_name]
                next_state = get_next_state(s, a)
                r = REWARD_FOR_GOAL if next_state == goal else STEP_COST
                v_new = r + discount * V[next_state]
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break

        # Policy Improvement: Update the policy based on the new value function.
        policy_stable = True
        for s in states:
            if s == goal:
                continue
            old_action = policy[s]
            best_action = None
            best_value = float("-inf")
            for a_name, a in actions.items():
                next_state = get_next_state(s, a)
                r = REWARD_FOR_GOAL if next_state == goal else STEP_COST
                value = r + discount * V[next_state]
                if value > best_value:
                    best_value = value
                    best_action = a_name
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

    # Derive a path from entry to goal following the optimal policy.
    path = []
    current = entry
    visited_path = set()
    while current != goal and current not in visited_path:
        path.append(current)
        visited_path.add(current)
        action_name = policy.get(current)
        if action_name is None:
            break
        next_state = get_next_state(current, actions[action_name])
        if next_state == current:
            break
        current = next_state
    path.append(current)

    return policy, V, path


def solve_maze_value_iteration_wrapper(maze, entry, goal, cell_size, frames, record_every=5):
    policy, V, path = solve_maze_value_iteration(maze, entry, goal)

    if path and path[-1] == goal:
        visited = set(path)
        # Return (policy, final_path, visited)
        return policy, path, visited
    else:
        return None, None, set()


def solve_maze_policy_iteration_wrapper(maze, entry, goal, cell_size, frames, record_every=5):
    policy, V, path = solve_maze_policy_iteration(maze, entry, goal)

    if path and path[-1] == goal:
        visited = set(path)
        return policy, path, visited
    else:
        return None, None, set()
