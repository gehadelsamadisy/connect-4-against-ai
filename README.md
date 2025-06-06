# Connect 4 Against AI Agent

## Overview

This project implements a Connect 4 game with a graphical user interface (GUI) where a human player competes against an AI agent. The AI supports three algorithms:

- **Minimax (without alpha-beta pruning)**
- **Minimax with alpha-beta pruning**
- **Expected Minimax** (with probabilistic disc drop as described in the assignment)

The game is played on a 6x7 board (default, but can be changed in code), and the AI uses a heuristic evaluation function to limit the search tree to a configurable depth \( K \).

## Features

- **Human vs. Computer** mode only.
- Choose the AI algorithm and search depth before starting.
- Visualizes the board and moves using Pygame.
- Prints the minimax tree for each AI move to the console (optionally generates a tree image using Graphviz).
- Heuristic evaluation function for non-terminal states.
- Displays heuristic values for the board after each move.
- Tracks and displays time taken and nodes expanded for AI moves.
- End-of-game summary and option to restart or quit.

## Requirements

- Python 3.7+
- [Pygame](https://www.pygame.org/)
- [Graphviz](https://graphviz.gitlab.io/download/) (for tree visualization)
- [graphviz Python package](https://pypi.org/project/graphviz/)

Install dependencies with:

```bash
pip install pygame graphviz
```

You may also need to install Graphviz system binaries (see their website for your OS).

## How to Run

1. **Clone or download the project files.**
2. **Install the required Python packages** (see above).
3. **Run the game:**
   ```bash
   python board.py
   ```
   (Do **not** run `game.py`—it is a different or legacy implementation.)

## How to Play

1. **Start the program.**
2. **Enter the search depth \( K \)** when prompted. This controls how many levels deep the AI will search before using the heuristic.
3. **Select the AI algorithm** from the menu:
   - Minimax without Alpha-Beta Pruning
   - Minimax with Alpha-Beta Pruning
   - Expected Minimax
4. The game window will open. The human player (red) always goes first.
5. **To make a move:** Click on the desired column.
6. The AI will make its move, print the minimax tree to the console, and display the board with heuristic values.
7. The game continues until the board is full.
8. At the end, the winner is announced based on the number of connected-fours.
9. You can restart (`R`) or quit (`Q`) after the game ends.

## Project Structure

- `board.py` — Main game logic, GUI, AI algorithms, and entry point.
- `minimax_tree.png`, `minimax_pruning_tree.png`, `expected_minimax_tree.png` — Example tree visualizations generated by the program.
- `game.py` — (Not used for main project; legacy or alternate implementation.)
- `README.md` — This file.

## Algorithms

### 1. Minimax (without alpha-beta pruning)

- Explores all possible moves up to depth \( K \).
- Evaluates leaf nodes using the heuristic function.

### 2. Minimax with Alpha-Beta Pruning

- Same as above, but prunes branches that cannot affect the final decision, improving efficiency.

### 3. Expected Minimax

- Models uncertainty in disc placement: 0.6 probability for chosen column, 0.2 for left/right (or 0.4 if only one neighbor).
- Computes expected value over possible outcomes.

### Heuristic Function

- Evaluates the board for both players.
- Considers number of connected-fours, potential threats, and other features.
- Returns a higher value when the AI is closer to winning, and lower when the human is closer.

## Console Output

- **Minimax Tree:** Printed in a readable format for each AI move.
- **Heuristic Values:** Displayed after each move.
- **Performance:** Time taken and nodes expanded are shown at the end.

## Restarting or Quitting

- After the game ends, press `R` to restart or `Q` to quit.

## Example Run

```
Enter the depth (K) for heuristic pruning: 4
Select the algorithm:
1. Minimax without Alpha-Beta Pruning
2. Minimax with Alpha-Beta Pruning
3. Expected Minimax
> 2
Selected algorithm: Minimax with Alpha-Beta Pruning
[Game window opens...]
[Minimax tree printed to console after each AI move...]
Total Nodes Expanded: 12345
Total Time Taken by AI: 12.34 seconds
Game Over
AI Wins! Score: 3 vs 2
```

## Notes

- The board size and other constants can be changed at the top of `board.py`.
- The program will generate and open a tree image at the end of the game if Graphviz is installed.
- For best results, use a recent version of Python and ensure all dependencies are installed.

## Troubleshooting

- **Pygame errors:** Make sure you have a working display and the correct version of Pygame.
- **Graphviz errors:** Ensure both the Python package and system binaries are installed and on your PATH.
- **Other issues:** Check the console for error messages and ensure all dependencies are installed.

## Credits

- Developed for the Artificial Intelligence course, Alexandria University, Faculty of Engineering, Computer and Systems Department.
