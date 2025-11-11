"""Reinforcement Learning Agent for Knight's Tour Problem.

This module implements a Q-Learning agent that learns to solve the Knight's Tour
through trial and error, using rewards to guide its learning.
"""
from typing import List, Tuple, Optional, Dict
from board import BoardGraph
import random
import time
import numpy as np
from collections import defaultdict


class QLearningAgent:
    """Q-Learning agent for Knight's Tour problem.
    
    State: (current_position, visited_cells_mask)
    Action: Legal knight moves (neighbor positions)
    
    Attributes:
        board (BoardGraph): The board graph representation
        q_table (dict): Q-values for state-action pairs
        learning_rate (float): Alpha - learning rate
        discount_factor (float): Gamma - discount factor for future rewards
        epsilon (float): Exploration rate for epsilon-greedy policy
        epsilon_decay (float): Decay rate for epsilon
        epsilon_min (float): Minimum epsilon value
    """
    
    def __init__(self, rows: int = 5, cols: int = 5, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """Initialize the Q-Learning agent.
        
        Args:
            rows: Number of rows on the board
            cols: Number of columns on the board
            learning_rate: Alpha parameter for Q-learning
            discount_factor: Gamma parameter for Q-learning
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.board = BoardGraph(rows, cols)
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_episodes: List[bool] = []
        
    def _state_to_tuple(self, position: int, visited: List[bool]) -> Tuple:
        """Convert state to hashable tuple for Q-table.
        
        Args:
            position: Current position index
            visited: List of visited cells
            
        Returns:
            Hashable state representation
        """
        return (position, tuple(visited))
    
    def _get_valid_actions(self, position: int, visited: List[bool]) -> List[int]:
        """Get list of valid actions (unvisited neighbors).
        
        Args:
            position: Current position index
            visited: List of visited cells
            
        Returns:
            List of valid neighbor indices
        """
        neighbors = self.board.neighbors(position)
        return [n for n in neighbors if not visited[n]]
    
    def _get_reward(self, action: int, visited: List[bool], is_valid: bool, 
                    tour_complete: bool) -> float:
        """Calculate reward for taking an action.
        
        Args:
            action: Action taken (position moved to)
            visited: Current visited state
            is_valid: Whether the move was valid
            tour_complete: Whether the tour is complete
            
        Returns:
            Reward value
        """
        if not is_valid:
            return -10.0  # Penalty for invalid move
        
        if tour_complete:
            return 100.0  # Large reward for completing tour
        
        return 1.0  # Small reward for visiting new cell
    
    def _select_action(self, position: int, visited: List[bool], 
                       training: bool = True) -> Optional[int]:
        """Select action using epsilon-greedy policy.
        
        Args:
            position: Current position
            visited: Current visited state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action (position to move to) or None if no valid actions
        """
        valid_actions = self._get_valid_actions(position, visited)
        
        if not valid_actions:
            return None
        
        # Epsilon-greedy selection
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.choice(valid_actions)
        else:
            # Exploit: best action according to Q-table
            state = self._state_to_tuple(position, visited)
            q_values = [(action, self.q_table[(state, action)]) 
                       for action in valid_actions]
            
            # If all Q-values are equal, choose randomly
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [a for a, q in q_values if q == max_q]
            
            return random.choice(best_actions)
    
    def _update_q_value(self, state: Tuple, action: int, reward: float, 
                       next_state: Tuple, valid_next_actions: List[int]) -> None:
        """Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            valid_next_actions: Valid actions from next state
        """
        current_q = self.q_table[(state, action)]
        
        # Calculate max Q-value for next state
        if valid_next_actions:
            max_next_q = max(self.q_table[(next_state, a)] 
                           for a in valid_next_actions)
        else:
            max_next_q = 0.0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state, action)] = new_q
    
    def train_episode(self, start_pos: Optional[Tuple[int, int]] = None) -> Tuple[float, int, bool]:
        """Run one training episode.
        
        Args:
            start_pos: Starting position as (row, col), or None for random
            
        Returns:
            Tuple of (total_reward, episode_length, success)
        """
        # Initialize episode
        if start_pos is None:
            position = random.randint(0, self.board.size - 1)
        else:
            position = self.board.rc_to_index(*start_pos)
        
        visited = [False] * self.board.size
        visited[position] = True
        
        total_reward = 0.0
        steps = 0
        max_steps = self.board.size * self.board.size  # Prevent infinite loops
        
        # Run episode
        while steps < max_steps:
            state = self._state_to_tuple(position, visited)
            
            # Select action
            action = self._select_action(position, visited, training=True)
            
            # Check if episode is done
            if action is None:
                # No valid moves - episode ends
                break
            
            # Take action
            is_valid = not visited[action]
            visited[action] = True
            tour_complete = all(visited)
            
            # Get reward
            reward = self._get_reward(action, visited, is_valid, tour_complete)
            total_reward += reward
            steps += 1
            
            # Get next state and valid actions
            next_state = self._state_to_tuple(action, visited)
            valid_next_actions = self._get_valid_actions(action, visited)
            
            # Update Q-value
            self._update_q_value(state, action, reward, next_state, valid_next_actions)
            
            # Move to next position
            position = action
            
            # Check if tour is complete
            if tour_complete:
                break
        
        success = all(visited)
        return total_reward, steps, success
    
    def train(self, num_episodes: int = 1000, 
              start_pos: Optional[Tuple[int, int]] = None,
              verbose: bool = True) -> None:
        """Train the agent for multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            start_pos: Starting position (None for random)
            verbose: Whether to print progress
        """
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_episodes = []
        self.epsilon_history = []  # Track actual epsilon values
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            reward, length, success = self.train_episode(start_pos)
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.success_episodes.append(success)
            self.epsilon_history.append(self.epsilon)  # Store current epsilon
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                recent_success_rate = sum(self.success_episodes[-100:]) / 100
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Success Rate: {recent_success_rate:.2%}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        end_time = time.time()
        
        if verbose:
            print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
            print(f"Final epsilon: {self.epsilon:.3f}")
            print(f"Q-table size: {len(self.q_table):,} entries")
    
    def run_episode(self, start_pos: Optional[Tuple[int, int]] = None) -> Tuple[List[int], bool]:
        """Run one episode using learned policy (no exploration).
        
        Args:
            start_pos: Starting position as (row, col)
            
        Returns:
            Tuple of (tour_path, success)
        """
        if start_pos is None:
            start_pos = (0, 0)
        
        position = self.board.rc_to_index(*start_pos)
        visited = [False] * self.board.size
        visited[position] = True
        tour = [position]
        
        max_steps = self.board.size
        
        while len(tour) < max_steps:
            action = self._select_action(position, visited, training=False)
            
            if action is None:
                break
            
            visited[action] = True
            tour.append(action)
            position = action
            
            if all(visited):
                break
        
        success = all(visited)
        return tour, success
    
    def get_learning_curve(self, window: int = 100) -> Tuple[List[float], List[float]]:
        """Get smoothed learning curve data.
        
        Args:
            window: Window size for moving average
            
        Returns:
            Tuple of (episode_numbers, smoothed_rewards)
        """
        if len(self.episode_rewards) < window:
            window = max(1, len(self.episode_rewards))
        
        smoothed = []
        for i in range(len(self.episode_rewards)):
            start_idx = max(0, i - window + 1)
            window_rewards = self.episode_rewards[start_idx:i + 1]
            smoothed.append(np.mean(window_rewards))
        
        return list(range(len(smoothed))), smoothed
    
    def get_success_rate_curve(self, window: int = 100) -> Tuple[List[float], List[float]]:
        """Get success rate over training.
        
        Args:
            window: Window size for moving average
            
        Returns:
            Tuple of (episode_numbers, success_rates)
        """
        if len(self.success_episodes) < window:
            window = max(1, len(self.success_episodes))
        
        success_rates = []
        for i in range(len(self.success_episodes)):
            start_idx = max(0, i - window + 1)
            window_successes = self.success_episodes[start_idx:i + 1]
            success_rates.append(sum(window_successes) / len(window_successes))
        
        return list(range(len(success_rates))), success_rates
    
    def print_tour_board(self, tour: List[int]) -> None:
        """Print the tour as a board showing the move sequence.
        
        Args:
            tour: List of position indices representing the tour
        """
        if not tour:
            print("No tour to display.")
            return
        
        # Create a board with move numbers
        board = [[-1 for _ in range(self.board.cols)] for _ in range(self.board.rows)]
        
        for move_num, idx in enumerate(tour):
            r, c = self.board.index_to_rc(idx)
            board[r][c] = move_num
        
        # Print the board
        max_num = len(tour) - 1
        width = len(str(max_num)) + 1
        
        print("\nKnight's Tour (move numbers):")
        for row in board:
            print(" ".join(f"{num:{width}d}" if num >= 0 else f"{'.':{width}s}" for num in row))
    
    def get_metrics(self) -> dict:
        """Get training and performance metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        total_episodes = len(self.episode_rewards)
        total_successes = sum(self.success_episodes)
        
        metrics = {
            'total_episodes': total_episodes,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / total_episodes if total_episodes > 0 else 0,
            'final_epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'board_size': self.board.size,
            'board_dimensions': f"{self.board.rows}x{self.board.cols}"
        }
        
        if total_episodes >= 100:
            metrics['last_100_success_rate'] = sum(self.success_episodes[-100:]) / 100
            metrics['last_100_avg_reward'] = np.mean(self.episode_rewards[-100:])
        
        return metrics
