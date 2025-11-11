"""Main script to run Knight's Tour algorithms and display results.

This script demonstrates:
- P1: Basic Backtracking algorithm
- P2: Q-Learning Reinforcement Learning agent
"""
from P1 import BasicBacktracking
from P2 import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np


def print_separator(char='-', length=70):
    """Print a separator line."""
    print(char * length)


def display_p1_results(rows: int = 8, cols: int = 8, start_pos=None):
    """Run and display results for Basic Backtracking algorithm.
    
    Args:
        rows: Number of rows on the board
        cols: Number of columns on the board
        start_pos: Starting position as (row, col)
    """
    print_separator('=')
    print(f"KNIGHT'S TOUR - BASIC BACKTRACKING ALGORITHM (P1)")
    print_separator('=')
    print(f"Board Size: {rows}x{cols} ({rows * cols} squares)")
    
    if start_pos is None:
        start_pos = (0, 0)
    print(f"Starting Position: {start_pos}")
    print_separator()
    
    # Create solver and run
    solver = BasicBacktracking(rows, cols)
    print("\nSearching for knight's tour...")
    
    success = solver.solve(start_pos)
    
    print_separator()
    
    if success:
        print("✓ TOUR FOUND!")
    else:
        print("✗ NO TOUR FOUND")
    
    print_separator()
    
    # Get and display metrics
    metrics = solver.get_metrics()
    
    print("\nMETRICS:")
    print(f"  Recursive Calls:    {metrics['recursive_calls']:,}")
    print(f"  Execution Time:     {metrics['execution_time']:.6f} seconds")
    print(f"  Solution Complete:  {metrics['is_complete']}")
    print(f"  Tour Type:          {'Closed' if metrics['is_closed'] else 'Open'}")
    print(f"  Tour Length:        {metrics['tour_length']}/{metrics['board_size']}")
    
    # Display the tour
    if success:
        print_separator()
        solver.print_tour_board()
        
        print("\n\nTOUR PATH (as coordinates):")
        tour_path = solver.get_tour_path()
        
        # Print in rows of 8 for readability
        for i in range(0, len(tour_path), 8):
            row_coords = tour_path[i:i+8]
            print(f"  Moves {i:2d}-{min(i+7, len(tour_path)-1):2d}: ", end="")
            print(" → ".join(f"{coord}" for coord in row_coords))
        
        # Show if it's a closed tour
        if metrics['is_closed']:
            print(f"\n  Knight can return to start: {tour_path[-1]} → {tour_path[0]} ✓")
    
    print_separator('=')


def display_p2_results(rows: int = 5, cols: int = 5, 
                      num_episodes: int = 1000,
                      start_pos=None):
    """Run and display results for Q-Learning RL agent.
    
    Args:
        rows: Number of rows on the board
        cols: Number of columns on the board
        num_episodes: Number of training episodes
        start_pos: Starting position as (row, col)
    """
    print_separator('=')
    print(f"KNIGHT'S TOUR - Q-LEARNING REINFORCEMENT LEARNING (P2)")
    print_separator('=')
    print(f"Board Size: {rows}x{cols} ({rows * cols} squares)")
    
    if start_pos is None:
        start_pos = (0, 0)
    print(f"Starting Position: {start_pos}")
    print(f"Training Episodes: {num_episodes}")
    print_separator()
    
    # Create and train agent
    agent = QLearningAgent(
        rows=rows, 
        cols=cols,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("\nTraining Q-Learning agent...")
    print_separator('-')
    
    agent.train(num_episodes=num_episodes, start_pos=start_pos, verbose=False)
    
    print_separator('=')
    
    # Get metrics
    metrics = agent.get_metrics()
    
    print("\nTRAINING METRICS:")
    print(f"  Total Episodes:         {metrics['total_episodes']:,}")
    print(f"  Successful Episodes:    {metrics['total_successes']:,}")
    print(f"  Overall Success Rate:   {metrics['overall_success_rate']:.2%}")
    
    if 'last_100_success_rate' in metrics:
        print(f"  Last 100 Success Rate:  {metrics['last_100_success_rate']:.2%}")
        print(f"  Last 100 Avg Reward:    {metrics['last_100_avg_reward']:.2f}")
    
    print(f"  Final Epsilon:          {metrics['final_epsilon']:.4f}")
    print(f"  Q-Table Size:           {metrics['q_table_size']:,} entries")
    
    # Test the learned policy
    print_separator()
    print("\nTesting learned policy...")
    
    tour, success = agent.run_episode(start_pos)
    
    if success:
        print("✓ SUCCESSFUL TOUR FOUND!")
    else:
        print(f"✗ Partial tour ({len(tour)}/{rows * cols} squares)")
    
    print_separator()
    
    # Display the tour
    agent.print_tour_board(tour)
    
    print("\n\nTOUR PATH (as coordinates):")
    tour_path = [agent.board.index_to_rc(idx) for idx in tour]
    
    # Print in rows of 8 for readability
    for i in range(0, len(tour_path), 8):
        row_coords = tour_path[i:i+8]
        print(f"  Moves {i:2d}-{min(i+7, len(tour_path)-1):2d}: ", end="")
        print(" → ".join(f"{coord}" for coord in row_coords))
    
    print_separator()
    
    # Plot learning curves
    print("\nGenerating learning curves...")
    plot_learning_curves(agent)
    
    print_separator('=')
    
    return agent


def plot_learning_curves(agent: QLearningAgent):
    """Plot learning curves for the RL agent.
    
    Args:
        agent: Trained Q-Learning agent
    """
    try:
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(18, 5))
        
        # Plot 1: Average Reward vs Episodes
        ax1 = plt.subplot(1, 3, 1)
        episodes, smoothed_rewards = agent.get_learning_curve(window=100)
        ax1.plot(episodes, smoothed_rewards, linewidth=2, color='blue', alpha=0.8)
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.set_title('Learning Curve\n(100-episode moving average)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # Plot 2: Success Rate vs Episodes
        ax2 = plt.subplot(1, 3, 2)
        episodes, success_rates = agent.get_success_rate_curve(window=100)
        ax2.plot(episodes, success_rates, linewidth=2, color='green', alpha=0.8)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.set_title('Convergence Behavior\n(100-episode moving average)', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.set_xlim(left=0)
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal line at 100% success
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect Success')
        ax2.legend(loc='lower right')
        
        # Add percentage formatting to y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Plot 3: Epsilon Decay (Exploration Rate)
        ax3 = plt.subplot(1, 3, 3)
        # Use actual epsilon values from training history
        if hasattr(agent, 'epsilon_history') and agent.epsilon_history:
            epsilon_values = agent.epsilon_history
        else:
            # Fallback: reconstruct epsilon values
            epsilon_values = []
            epsilon = 1.0
            epsilon_decay = agent.epsilon_decay
            epsilon_min = agent.epsilon_min
            for _ in range(len(agent.episode_rewards)):
                epsilon_values.append(epsilon)
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        ax3.plot(range(len(epsilon_values)), epsilon_values, linewidth=2, color='orange', alpha=0.8)
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
        ax3.set_title('Exploration vs Exploitation\n(Epsilon Decay)', fontsize=13, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.set_xlim(left=0)
        ax3.grid(True, alpha=0.3)
        
        # Add horizontal line at min epsilon
        ax3.axhline(y=epsilon_min, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
                   label=f'Min ε = {epsilon_min}')
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
        print("✓ Learning curves saved to 'learning_curves.png'")
        plt.show()
        
    except Exception as e:
        print(f"Note: Could not generate plots (matplotlib may not be available): {e}")


def main():
    """Main function to run demonstrations."""
    
    # Configuration
    n = 6  # Board size (6x6)
    start_pos = (0, 0)  # Starting position
    num_episodes = 5000  # Training episodes for Q-Learning
    
    print("\n" + "=" * 70)
    print("KNIGHT'S TOUR SOLVER - AUTOMATED TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Board Size: {n}x{n}")
    print(f"  Starting Position: {start_pos}")
    print(f"  RL Training Episodes: {num_episodes}")
    print("=" * 70)
    
    # Run P1: Basic Backtracking
    print("\n\n")
    print("=" * 70)
    print("RUNNING ALGORITHM 1: BASIC BACKTRACKING (P1)")
    print("=" * 70)
    display_p1_results(rows=n, cols=n, start_pos=start_pos)
    
    # Run P2: Q-Learning RL Agent
    print("\n\n")
    print("=" * 70)
    print("RUNNING ALGORITHM 2: Q-LEARNING REINFORCEMENT LEARNING (P2)")
    print("=" * 70)
    display_p2_results(rows=n, cols=n, num_episodes=num_episodes, start_pos=start_pos)


if __name__ == "__main__":
    main()
