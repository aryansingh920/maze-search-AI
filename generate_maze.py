"""
Created on 27/02/2025

@author: Aryan

Filename: generate_maze.py

Relative Path: generate_maze.py
"""

import random
from PIL import Image, ImageDraw


def generate_maze(width, height):
    """
    Generate a maze using iterative DFS. The maze is represented as a 2D grid,
    where 1 represents a wall and 0 represents an open path.
    
    Note: width and height should be odd numbers.
    """
    # Initialize maze grid with walls (1)
    maze = [[1 for _ in range(width)] for _ in range(height)]

    # Use (1, 1) as the starting point for maze generation.
    stack = [(1, 1)]
    maze[1][1] = 0  # Mark start as open

    while stack:
        cx, cy = stack[-1]  # Peek at the top of the stack

        # Define the four possible directions: Up, Right, Down, Left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        random.shuffle(directions)  # Randomize order of directions

        carved = False
        for dx, dy in directions:
            nx, ny = cx + dx * 2, cy + dy * 2  # Look two cells ahead
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                # Carve a path between the current cell and the new cell
                maze[cy + dy][cx + dx] = 0  # Remove the wall between
                maze[ny][nx] = 0            # Mark the new cell as open
                stack.append((nx, ny))      # Push the new cell onto the stack
                carved = True
                break  # Proceed with the new cell

        if not carved:
            # If no directions are available, backtrack.
            stack.pop()

    return maze


def add_entry_and_goal(maze):
    """
    Opens up a cell on the top border as the entry and a cell on the bottom border as the goal.
    Returns the coordinates of (entry, goal).
    """
    height = len(maze)
    width = len(maze[0])

    # For entry: Choose an odd-indexed column on the top row where the cell below is open.
    possible_entries = [x for x in range(1, width, 2) if maze[1][x] == 0]
    entry_x = random.choice(possible_entries) if possible_entries else 1
    maze[0][entry_x] = 0  # Open the top border cell to create an entry
    entry = (entry_x, 0)

    # For goal: Choose an odd-indexed column on the bottom row where the cell above is open.
    possible_goals = [x for x in range(
        1, width, 2) if maze[height - 2][x] == 0]
    goal_x = random.choice(possible_goals) if possible_goals else width - 2
    # Open the bottom border cell to create a goal
    maze[height - 1][goal_x] = 0
    goal = (goal_x, height - 1)

    return entry, goal


def save_maze_as_image(maze, cell_size=10, filename="maze.png", entry=None, goal=None):
    """
    Saves the maze grid as an image.
    
    - Walls (1) are drawn in black.
    - Free cells (0) are white.
    - The entry is highlighted in green.
    - The goal is highlighted in red.
    """
    height = len(maze)
    width = len(maze[0])

    # Create a new image with a white background.
    img = Image.new("RGB", (width * cell_size, height * cell_size), "white")
    draw = ImageDraw.Draw(img)

    # Draw the maze walls.
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == 1:  # Draw walls in black.
                top_left = (x * cell_size, y * cell_size)
                bottom_right = ((x + 1) * cell_size - 1,
                                (y + 1) * cell_size - 1)
                draw.rectangle([top_left, bottom_right], fill="black")

    # Highlight the entry point in green.
    if entry:
        ex, ey = entry
        top_left = (ex * cell_size, ey * cell_size)
        bottom_right = ((ex + 1) * cell_size - 1, (ey + 1) * cell_size - 1)
        draw.rectangle([top_left, bottom_right], fill="green")

    # Highlight the goal point in red.
    if goal:
        gx, gy = goal
        top_left = (gx * cell_size, gy * cell_size)
        bottom_right = ((gx + 1) * cell_size - 1, (gy + 1) * cell_size - 1)
        draw.rectangle([top_left, bottom_right], fill="red")

    img.save(filename)
    print(f"Maze saved as {filename}")


def save_maze_as_values(maze, filename="maze_values.csv", entry=None, goal=None):
    """
    Save the maze grid to a CSV file with numeric values.
    
    The grid is a 2D list where:
      - Walls are represented as 1.
      - Open paths are represented as 0.
      - If provided, the entry is marked as 2.
      - If provided, the goal is marked as 3.
    
    Parameters:
      - maze: A 2D list representing the maze.
      - filename: The name of the file to which the values will be saved.
      - entry: Tuple (x, y) for the entry cell (optional).
      - goal: Tuple (x, y) for the goal cell (optional).
    """
    # Create a copy of the maze so that the original maze is not modified.
    maze_copy = [row[:] for row in maze]

    # Mark the entry and goal in the maze copy, if provided.
    if entry:
        ex, ey = entry
        maze_copy[ey][ex] = 2  # Use 2 to mark the entry.
    if goal:
        gx, gy = goal
        maze_copy[gy][gx] = 3  # Use 3 to mark the goal.

    # Write the maze grid values to a CSV file.
    with open(filename, "w") as file:
        for row in maze_copy:
            # Convert each row to a comma-separated string.
            row_values = ",".join(str(cell) for cell in row)
            file.write(row_values + "\n")

    print(f"Maze values saved as {filename}")
