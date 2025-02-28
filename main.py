"""
Created on 10/02/2025

@author: Aryan

Filename: main.py

Relative Path: main.py
"""

import numpy as np
from generate_maze import generate_maze,add_entry_and_goal,save_maze_as_image,save_maze_as_values
from path_generator import load_maze_from_csv,draw_maze_state,create_video
from algorithm import (
    solve_maze_dfs, solve_maze_bfs, solve_maze_astar, solve_maze_dijkstra, solve_maze_greedy,
    solve_maze_value_iteration_wrapper, solve_maze_policy_iteration_wrapper
)

functionObject = {
    # "dfs_maze": solve_maze_dfs,
    # "bfs_maze": solve_maze_bfs,
    # "astar_maze": solve_maze_astar,
    # "dijkstra_maze": solve_maze_dijkstra,
    # "greedy_maze": solve_maze_greedy,
    "mdp_value_iteration": solve_maze_value_iteration_wrapper,
    "mdp_policy_iteration": solve_maze_policy_iteration_wrapper
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

    # Load the maze from CSV
    maze, entry, goal = load_maze_from_csv("maze_values.csv")
    cell_size = 10  # Must match the cell size used when generating the maze

    for name, function in functionObject.items():
        frames = []

        # Notice now we do an *unpacking* to get `policy, final_path, visited`
        if name in ["mdp_value_iteration", "mdp_policy_iteration"]:
            # MDP solvers return 3 values
            policy, final_path, visited = function(
                maze, entry, goal, cell_size, frames, record_every=5)
        else:
            # Non-MDP solvers return 2 values
            final_path, visited = function(
                maze, entry, goal, cell_size, frames, record_every=5)
            policy = None  # Not applicable for BFS/DFS, etc.

        if final_path is None or len(final_path) == 0:
            print(f"{name}: No solution found!")
        else:
            print(f"{name}: Solution found, path length = {len(final_path)}")

            # Build a final frame to hold on the solution path
            final_img = draw_maze_state(
                maze, cell_size, visited, final_path[-1],
                final_path, entry, goal, final_path=final_path
            )
            for _ in range(60):
                frames.append(np.array(final_img))

            # If this is an MDP, draw the policy image!
            if policy is not None:
                from algorithm import draw_mdp_policy
                policy_img = draw_mdp_policy(maze, policy, final_path,
                                             cell_size=cell_size,
                                             entry=entry, goal=goal)
                policy_img.save(f"{name}_policy.png")
                print(f"{name} policy image saved as {name}_policy.png")

        # Save the video
        output_video = f"{name}.mp4"
        create_video(frames, output_video, fps=30)
        print(f"{name} solution video saved as {output_video}")


    
    

    

if __name__ == "__main__":
    main()
