"""
Created on 10/02/2025

@author: Aryan

Filename: main.py

Relative Path: main.py
"""

import numpy as np
from generate_maze import generate_maze,add_entry_and_goal,save_maze_as_image,save_maze_as_values
from path_generator import load_maze_from_csv,draw_maze_state,create_video
from algorithm import solve_maze_dfs,solve_maze_bfs,solve_maze_astar

functionObject = {
    "dfs_maze": solve_maze_dfs,
    "bfs_maze": solve_maze_bfs,
    "astar_maze": solve_maze_astar
}

def main():
    maze_width, maze_height = 61, 61
    maze = generate_maze(maze_width, maze_height)

    # Add entry and goal to the maze.
    entry, goal = add_entry_and_goal(maze)

    # Save the maze as an image.
    save_maze_as_image(maze, cell_size=10,
                       filename="maze.png", entry=entry, goal=goal)

    # Save the maze grid as CSV values for further processing.
    save_maze_as_values(maze, filename="maze_values.csv",
                        entry=entry, goal=goal)
    
    
    # time.sleep(1)
    
    # Load the maze (this CSV file should have been produced by your maze generator)
    maze, entry, goal = load_maze_from_csv("maze_values.csv")
    cell_size = 10  # Must match the cell size used when generating the maze

    frames = []  # List that will hold all frames of our DFS animation.

    # Solve the maze using DFS (and record progress)
    final_path, visited = solve_maze_dfs(
        maze, entry, goal, cell_size, frames, record_every=5)

    if final_path is None:
        print("No solution found!")
    else:
        print("Solution found!")
        # Add extra frames at the end to “hold” the final solution on-screen (e.g. 2 seconds at 30 fps)
        final_img = draw_maze_state(
            maze, cell_size, visited, final_path[-1], final_path, entry, goal, final_path=final_path)
        for _ in range(60):
            frames.append(np.array(final_img))

    # Create a video from the frames
    output_video = "dfs_maze.mp4"
    create_video(frames, output_video, fps=30)
    print(f"DFS solution video saved as {output_video}")

    
    

    

if __name__ == "__main__":
    main()
