"""Basic Backtracking Algorithm for Knight's Tour Problem.

This module implements a recursive depth-first search through all possible
knight moves to find a tour that visits every square on the board exactly once.
"""
from typing import List, Optional, Tuple
from board import BoardGraph
import time


class BasicBacktracking:
    """Knight's Tour solver using basic backtracking (DFS).
    
    Attributes:
        board (BoardGraph): The board graph representation
        tour (List[int]): The sequence of squares visited (indices)
        visited (List[bool]): Tracks which squares have been visited
        recursive_calls (int): Counter for number of recursive calls
        start_time (float): Time when search started
        end_time (float): Time when search completed
    """
    
    def __init__(self, rows: int = 8, cols: int = 8):
        """Initialize the backtracking solver.
        
        Args:
            rows: Number of rows on the board
            cols: Number of columns on the board
        """
        self.board = BoardGraph(rows, cols)
        self.tour: List[int] = []
        self.visited: List[bool] = [False] * self.board.size
        self.recursive_calls = 0
        self.start_time = 0.0
        self.end_time = 0.0
    
    def solve(self, start_pos: Optional[Tuple[int, int]] = None) -> bool:
        """Find a knight's tour starting from the given position.
        
        Args:
            start_pos: Starting position as (row, col). Defaults to (0, 0).
            
        Returns:
            True if a tour was found, False otherwise.
        """
        if start_pos is None:
            start_pos = (0, 0)
        
        # Reset state
        self.tour = []
        self.visited = [False] * self.board.size
        self.recursive_calls = 0
        
        # Convert start position to index
        start_idx = self.board.rc_to_index(*start_pos)
        
        # Start timing
        self.start_time = time.time()
        
        # Start backtracking
        result = self._backtrack(start_idx)
        
        # End timing
        self.end_time = time.time()
        
        return result
    
    def _backtrack(self, current: int) -> bool:
        """Recursive backtracking function.
        
        Args:
            current: Current square index
            
        Returns:
            True if a complete tour is found from this position, False otherwise.
        """
        self.recursive_calls += 1
        
        # Mark current square as visited and add to tour
        self.tour.append(current)
        self.visited[current] = True
        
        # Base case: all squares visited
        if len(self.tour) == self.board.size:
            return True
        
        # Try all possible knight moves from current position
        for neighbor in self.board.neighbors(current):
            if not self.visited[neighbor]:
                if self._backtrack(neighbor):
                    return True
        
        # Backtrack: unmark and remove from tour
        self.visited[current] = False
        self.tour.pop()
        
        return False
    
    def get_tour_path(self) -> List[Tuple[int, int]]:
        """Get the tour as a list of (row, col) coordinates.
        
        Returns:
            List of (row, col) tuples representing the tour path.
        """
        return [self.board.index_to_rc(idx) for idx in self.tour]
    
    def get_execution_time(self) -> float:
        """Get the execution time in seconds.
        
        Returns:
            Execution time in seconds.
        """
        return self.end_time - self.start_time
    
    def is_closed_tour(self) -> bool:
        """Check if the tour is closed (can return to start).
        
        Returns:
            True if the last square can reach the first square.
        """
        if len(self.tour) < self.board.size:
            return False
        
        first = self.tour[0]
        last = self.tour[-1]
        
        return first in self.board.neighbors(last)
    
    def is_complete(self) -> bool:
        """Check if the tour visits all squares.
        
        Returns:
            True if all squares are visited.
        """
        return len(self.tour) == self.board.size
    
    def get_metrics(self) -> dict:
        """Get all metrics about the solution.
        
        Returns:
            Dictionary containing all metrics.
        """
        return {
            'recursive_calls': self.recursive_calls,
            'execution_time': self.get_execution_time(),
            'is_complete': self.is_complete(),
            'is_closed': self.is_closed_tour(),
            'tour_length': len(self.tour),
            'board_size': self.board.size,
            'board_dimensions': f"{self.board.rows}x{self.board.cols}"
        }
    
    def print_tour_board(self) -> None:
        """Print the tour as a board showing the move sequence."""
        if not self.tour:
            print("No tour found.")
            return
        
        # Create a board with move numbers
        board = [[-1 for _ in range(self.board.cols)] for _ in range(self.board.rows)]
        
        for move_num, idx in enumerate(self.tour):
            r, c = self.board.index_to_rc(idx)
            board[r][c] = move_num
        
        # Print the board
        max_num = len(self.tour) - 1
        width = len(str(max_num)) + 1
        
        print("\nKnight's Tour (move numbers):")
        for row in board:
            print(" ".join(f"{num:{width}d}" if num >= 0 else " " * width for num in row))
