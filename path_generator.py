"""
Created on 10/02/2025

@author: Aryan

Filename: path_generator.py

Relative Path: path_generator.py
"""

import csv
import imageio
import numpy as np
from PIL import Image, ImageDraw


def load_maze_from_csv(filename):
    """
    Loads the maze from a CSV file.
    
    Maze cell values:
      - 1: Wall
      - 0: Open path
      - 2: Entry
      - 3: Goal
      
    Returns:
      maze (2D list of ints), entry (tuple), goal (tuple)
    """
    maze = []
    entry = None
    goal = None
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for y, row in enumerate(reader):
            row_values = [int(cell) for cell in row]
            maze.append(row_values)
            for x, cell in enumerate(row_values):
                if cell == 2:
                    entry = (x, y)
                elif cell == 3:
                    goal = (x, y)
    return maze, entry, goal


def get_neighbors(maze, cell):
    """
    Given a cell (x,y), return its adjacent neighbors (up, right, down, left)
    that are within bounds and are not walls.
    """
    x, y = cell
    neighbors = []
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]):
            if maze[ny][nx] != 1:  # Not a wall
                neighbors.append((nx, ny))
    return neighbors


def draw_maze_state(maze, cell_size, visited, current, path, entry, goal, final_path=None):
    """
    Draws the current state of the maze and DFS progress.
    
    Colors used:
      - Walls: Black
      - Open cells: White
      - Visited cells: Light blue (#ADD8E6)
      - The DFS path from entry to the current cell: Yellow
      - The current cell: Orange
      - Final solution path (if provided): Purple
      - Entry: Green
      - Goal: Red
    """
    height = len(maze)
    width = len(maze[0])
    img = Image.new("RGB", (width * cell_size, height * cell_size), "white")
    draw = ImageDraw.Draw(img)

    # Draw walls (cells with value 1)
    for y in range(height):
        for x in range(width):
            if maze[y][x] == 1:
                top_left = (x * cell_size, y * cell_size)
                bottom_right = ((x + 1) * cell_size - 1,
                                (y + 1) * cell_size - 1)
                draw.rectangle([top_left, bottom_right], fill="black")

    # Draw visited cells (light blue)
    for (x, y) in visited:
        top_left = (x * cell_size, y * cell_size)
        bottom_right = ((x + 1) * cell_size - 1, (y + 1) * cell_size - 1)
        draw.rectangle([top_left, bottom_right], fill="#ADD8E6")

    # Draw the current DFS path (yellow)
    for (x, y) in path:
        top_left = (x * cell_size, y * cell_size)
        bottom_right = ((x + 1) * cell_size - 1, (y + 1) * cell_size - 1)
        draw.rectangle([top_left, bottom_right], fill="yellow")

    # If a final solution path is provided, draw it (purple)
    if final_path:
        for (x, y) in final_path:
            top_left = (x * cell_size, y * cell_size)
            bottom_right = ((x + 1) * cell_size - 1, (y + 1) * cell_size - 1)
            draw.rectangle([top_left, bottom_right], fill="purple")

    # Highlight the current cell as orange
    if current:
        x, y = current
        top_left = (x * cell_size, y * cell_size)
        bottom_right = ((x + 1) * cell_size - 1, (y + 1) * cell_size - 1)
        draw.rectangle([top_left, bottom_right], fill="orange")

    # Finally, re-draw the entry and goal so they’re visible
    if entry:
        ex, ey = entry
        top_left = (ex * cell_size, ey * cell_size)
        bottom_right = ((ex + 1) * cell_size - 1, (ey + 1) * cell_size - 1)
        draw.rectangle([top_left, bottom_right], fill="green")
    if goal:
        gx, gy = goal
        top_left = (gx * cell_size, gy * cell_size)
        bottom_right = ((gx + 1) * cell_size - 1, (gy + 1) * cell_size - 1)
        draw.rectangle([top_left, bottom_right], fill="red")

    return img




def create_video(frames, output_filename, fps=30):
    """
    Saves the collected frames as a video file (MP4 format).
    
    Parameters:
      - frames: List of numpy arrays (each representing an image frame).
      - output_filename: Name of the output video file.
      - fps: Frames per second for the output video.
    """
    writer = imageio.get_writer(output_filename, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


