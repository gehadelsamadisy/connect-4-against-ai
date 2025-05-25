import random
import pygame
import sys
from graphviz import Digraph
import copy
import time


ROWS = 6
COLUMNS = 7
CELL_SIZE = 100
RADIUS = CELL_SIZE // 2 - 5

WIDTH = COLUMNS * CELL_SIZE
HEIGHT = (ROWS + 1) * CELL_SIZE
SIZE = (WIDTH, HEIGHT)

PLAYER_SYMBOL = '1'
AI_SYMBOL = '2'
EMPTY_CELL = '0'

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

nodes_expanded = 0
total_time = 0

modes = ["Minimax without Alpha-Beta Pruning",
        "Minimax with Alpha-Beta Pruning",
        "Expected Minimax"]
selected_mode = None


def create_board():
    return str(EMPTY_CELL * ROWS * COLUMNS)


def drop_piece(board, row, column, symbol):
    index = row * COLUMNS + column
    return board[:index] + symbol + board[index + 1:]


def is_valid_location(board, column):
    return board[column] == EMPTY_CELL


def get_next_open_row(board, column):
    for row in range(ROWS - 1, -1, -1):
        index = row * COLUMNS + column
        if board[index] == EMPTY_CELL:
            return row
    return -1


def draw_board(screen, board):
    screen.fill(BLACK)
    for c in range(COLUMNS):
        for r in range(ROWS):
            pygame.draw.rect(screen, BLUE, (c * CELL_SIZE, (r + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            center = (c * CELL_SIZE + CELL_SIZE // 2, (r + 1) * CELL_SIZE + CELL_SIZE // 2)
            index = r * COLUMNS + c
            color = BLACK
            if board[index] == PLAYER_SYMBOL:
                color = RED
            elif board[index] == AI_SYMBOL:
                color = YELLOW
            pygame.draw.circle(screen, color, center, RADIUS)
    pygame.display.update()


def find_connected_fours_and_score(board, symbol):
    # 3mlna set 3shan n avoid el duplicates w n3rf el connected indices
    # elly 3ndna 3shan n7sb score w nsib el connected bs 3la el board
    #5od balak en el board 3amla keda
    #[(0,0) , (0,1)...........................]
    #[...................................(1,1)]
    
    #y3ny el rows btzid mn fo2 l ta7t
    #w columns btzid mn yemin lel shemal
    connected_indices = set()
    score = 0

    #bashof el rows el awel , el 4 elly bel 3rd
    for r in range(ROWS):
        for c in range(COLUMNS - 3):  #hena columns-3 3shan ha3mel columns -2 w columns -1 , f 3shan mab2ash out of index
            count = 0
            for i in range(4):
                if board[r * COLUMNS + c + i] == symbol:
                    count += 1
            if count == 4:
                for i in range(4):
                    connected_indices.add(r * COLUMNS + c + i)
                score += 1

    #hena bashof el columns, el 4 elly bel tool
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            count = 0
            window = []
            for i in range(4):
                index = (r + i) * COLUMNS + c
                window.append(board[index])
            for cell in window:
                if cell == symbol:
                    count += 1
            if count == 4:
                for i in range(4):
                    connected_indices.add((r + i) * COLUMNS + c)
                score += 1  # Each connected four adds 1 to the score

    # bashof el diagonals elly nazla mn fo2 l ta7t
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = []
            for i in range(4):
                index = (r + i) * COLUMNS + (c + i)
                window.append(board[index])
            
            is_connected = True
            for cell in window:
                if cell != symbol:
                    is_connected = False
                    break
            
            if is_connected:
                for i in range(4):
                    connected_indices.add((r + i) * COLUMNS + (c + i))
                score += 1  

    # el diagonals elly mn t7t l fo2
    for r in range(3, ROWS):
        for c in range(COLUMNS - 3):
            window = []
            for i in range(4):
                index = (r - i) * COLUMNS + (c + i)
                window.append(board[index])
            
            is_connected = True
            for cell in window:
                if cell != symbol:
                    is_connected = False
                    break
            
            if is_connected:
                for i in range(4):
                    connected_indices.add((r - i) * COLUMNS + (c + i))
                score += 1  

    return connected_indices, score





# def connected_fours_score(board,symbol):
#     connected_indices = set()
#     score = 0

#     # Check horizontal locations
#     for r in range(ROWS):
#         for c in range(COLUMNS - 3):
#             count = 0
#             for i in range(4):
#                 if board[r * COLUMNS + c + i] == symbol:
#                     count += 1
#             if count == 4:
#                 for i in range(4):
#                     connected_indices.add(r * COLUMNS + c + i)
#                 score += 1

#     # Check vertical locations
#     for c in range(COLUMNS):
#         for r in range(ROWS - 3):
#             count = 0
#             window = []
#             for i in range(4):
#                 index = (r + i) * COLUMNS + c
#                 window.append(board[index])
#             for cell in window:
#                 if cell == symbol:
#                     count += 1
#             if count == 4:
#                 for i in range(4):
#                     connected_indices.add((r + i) * COLUMNS + c)
#                 score += 1
                
#     # Check positively sloped diagonals
#     for r in range(ROWS):
#         for c in range(COLUMNS - 3):
#             count = 0
#             for i in range(4):
#                 if board[r * COLUMNS + c + i] == symbol:
#                     count += 1
#             if count == 4:
#                 for i in range(4):
#                     connected_indices.add(r * COLUMNS + c + i)
#                 score += 1
                
#     # Check negatively sloped diagonals
#     for r in range(ROWS - 3):
#         for c in range(COLUMNS - 3):
#             count = 0
#             for i in range(4):
#                 if board[r * COLUMNS + c + i] == symbol:
#                     count += 1
#             if count == 4:
#                 for i in range(4):
#                     connected_indices.add((r + i) * COLUMNS + (c + i))
#                 score += 1

#     return connected_indices, score


def select_algorithm():
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("Select Algorithm")
    font = pygame.font.Font(None, 30)

    buttons = []
    for i, mode in enumerate(modes):
        button_rect = pygame.Rect(50, 50 + i * 70, 400, 50)
        buttons.append((button_rect, mode))

    selected = None
    running = True
    while running:
        screen.fill(BLACK)
        for button_rect, mode in buttons:
            pygame.draw.rect(screen, BLUE, button_rect)
            text = font.render(mode, True, YELLOW)
            screen.blit(text, (button_rect.x + 10, button_rect.y + 10))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for button_rect, mode in buttons:
                    if button_rect.collidepoint(mouse_pos):
                        selected = mode
                        running = False

    return selected

def evaluate_windows(window, symbol): #bashof el board 3la shakl windows of 4
    # wel score bta3 el window da byeb2a 7sb 3dd el pieces elly shbh b3d fih
    score = 0
    opponent_symbol = None
    if symbol == PLAYER_SYMBOL:
        opponent_symbol = AI_SYMBOL
    else:
        opponent_symbol = PLAYER_SYMBOL
    
    count = 0
    for cell in window:
        if cell == symbol:
            count += 1
        elif cell == opponent_symbol:
            count -= 1
    if count == 4:
        score += 100  #keda ana kasban fel window da ana gm3t 4 f ana gamed w 7elw keda
    elif count == 3:
        if window.count(EMPTY_CELL) == 1:
            score += 5
        elif window.count(opponent_symbol) == 1:
            score -= 4
    elif count == 2:
        if window.count(EMPTY_CELL) == 2:
            score += 2
        elif window.count(opponent_symbol) == 2:
            score -= 3
    return score
        
    
def old_heuristic_function(board):
    score = 0

    # Score center column
    center_column = [board[r * COLUMNS + COLUMNS // 2] for r in range(ROWS)]
    center_count_ai = center_column.count(AI_SYMBOL)
    center_count_player = center_column.count(PLAYER_SYMBOL)
    score += center_count_ai * 3  # lw el ai wa5ed el center ba2ollo bravo 3lek 3la ad el 7eta elly a5adha
    score -= center_count_player * 3  # law el user wa5ed el center ba penalize el ai 3la ad elly el user a5ado 3shan kan el mfrod y control el center kollo

    # ba3ady 3al rows kolohom 3la shakl window of 4 byet7arak m3aya
    for r in range(ROWS):
        row = [board[r * COLUMNS + c] for c in range(COLUMNS)] #ba5od el row kollo f array wab3ato lel evaluate windows function
        print(type(row))
        for c in range(COLUMNS - 3):
            window = row[c:c + 4]
            score += evaluate_windows(window, AI_SYMBOL)
            score -= evaluate_windows(window, PLAYER_SYMBOL)

    # Score vertical
    for c in range(COLUMNS):
        col = [board[r * COLUMNS + c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col[r:r + 4]
            score += evaluate_windows(window, AI_SYMBOL)
            score -= evaluate_windows(window, PLAYER_SYMBOL)

    # Score positively sloped diagonals
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[(r + i) * COLUMNS + (c + i)] for i in range(4)]
            score += evaluate_windows(window, AI_SYMBOL)
            score -= evaluate_windows(window, PLAYER_SYMBOL)

    # Score negatively sloped diagonals
    for r in range(3, ROWS):
        for c in range(COLUMNS - 3):
            window = [board[(r - i) * COLUMNS + (c + i)] for i in range(4)]
            score += evaluate_windows(window, AI_SYMBOL)
            score -= evaluate_windows(window, PLAYER_SYMBOL)

    return score




def heuristic_function(board):
    score = 0

    # Score center column
    center_column = [board[r * COLUMNS + COLUMNS // 2] for r in range(ROWS)]
    center_count_ai = center_column.count(AI_SYMBOL)
    center_count_player = center_column.count(PLAYER_SYMBOL)
    score += center_count_ai * 3  # lw el ai wa5ed el center ba2ollo bravo 3lek 3la ad el 7eta elly a5adha
    score -= center_count_player * 3  # law el user wa5ed el center ba penalize el ai 3la ad elly el user a5ado 3shan kan el mfrod y control el center kollo
    
    # ba3ady 3al rows wa7ed wa7ed
    for r in range(ROWS):
        # f kol row ba5od window of 4 ab3ato lel evaluate window traga3ly score bta3o
        for c in range(COLUMNS - 3):
            window = board[r * COLUMNS + c:r * COLUMNS + c + 4]  # da el window
            score += evaluate_windows(window, AI_SYMBOL) #evaluate score el ai
            score -= evaluate_windows(window, PLAYER_SYMBOL) #evaluate score el user f nfs el window

    # hena el 3aks ba3ady 3al columns a5od 4 b 4 keda bel tool w a3mel nfs el kalam bs bb3ato ka string brdo
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            window = "".join([board[(r + i) * COLUMNS + c] for i in range(4)]) 
            score += evaluate_windows(window, AI_SYMBOL)
            score -= evaluate_windows(window, PLAYER_SYMBOL)

    # hena bashof el diagonal elly mn fo2 l ta7t
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = "".join([board[(r + i) * COLUMNS + (c + i)] for i in range(4)])  # Collect 4 cells in the diagonal
            score += evaluate_windows(window, AI_SYMBOL)
            score -= evaluate_windows(window, PLAYER_SYMBOL)

    # nfs el kalam bs lel diagonal elly mn ta7t l fo2
    for r in range(3, ROWS):
        for c in range(COLUMNS - 3):
            window = "".join([board[(r - i) * COLUMNS + (c + i)] for i in range(4)])  # Collect 4 cells in the diagonal
            score += evaluate_windows(window, AI_SYMBOL)
            score -= evaluate_windows(window, PLAYER_SYMBOL)

    return score

def display_board_with_heuristic(board):
    print("\nCurrent Board:")
    for r in range(ROWS):
        row = board[r * COLUMNS:(r + 1) * COLUMNS]
        print(" ".join(row))
    heuristic_score = heuristic_function(board)
    print(f"Heuristic Score: {heuristic_score}\n")

def minimax(board, depth, maximizing_player):
    global nodes_expanded
    nodes_expanded +=1
    valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
    is_terminal = len(valid_columns) == 0 or find_connected_fours_and_score(board, PLAYER_SYMBOL)[1] > 0 or find_connected_fours_and_score(board, AI_SYMBOL)[1] > 0 #hena el board full (msh 3ndy columns aktr mn kda) aw el ai kasab aw el player kasab
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if(find_connected_fours_and_score(board, AI_SYMBOL)[1] > 0):
                return random.choice(valid_columns), 100000 #hena el ai kasab w ana 3ayez el state dy
            elif(find_connected_fours_and_score(board, PLAYER_SYMBOL)[1] > 0):
                return random.choice(valid_columns), -100000 #hena el player kasab w ana msh 3ayz el state dy
            else:
                return random.choice(valid_columns), 0 #hena Tie msh ha3mel 7aga ()
            # mah ha7ot el heuristic function 3shan dy terminal state w el heuristic basta5demha eny a evaluate el non terminal states ewl game shaghala
            
            # gher keda bab3at 0 3shan el ai y3rf y yfara2 ben el terminal states wel non terminal
        else: #el depth b 0 el tree 5elset bs ana msh terminal state (hena lazem awa2af iteratrion 3shan el tree 5las)
            return random.choice(valid_columns), heuristic_function(board)
        # hena momken ab3at el heuristic function 3shan el game ma5alasetsh
    if maximizing_player:
        value = -1*float('inf') #hena 3shan maximizing player f 3ayez a3la score
        best_column = random.choice(valid_columns) # ba5od random column ka starting point w abda2 a evaluate ba2y el columns mn 3ndo
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, AI_SYMBOL) #hena bashof lw 3mlt drop fel column dy el score hayeb2a eih
            _, new_score = minimax(temp_board, depth - 1, False) # ba call el fucntion recursively 3shan a3rf el score ely hayegy mn el column dy
            if new_score > value:
                value = new_score
                best_column = col
        return best_column, value
    else:
        value = float('inf') #hena lw ana msh el maximizing player w 3ayz a2ll score
        best_column = random.choice(valid_columns) 
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
            _, new_score = minimax(temp_board, depth - 1, True)
            if new_score < value:
                value = new_score
                best_column = col
        return best_column, value



              

def minimax_with_alpha_beta_pruning(board, depth,alpha,beta, maximizing_player):
    global nodes_expanded
    nodes_expanded +=1
    valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]

    is_terminal = len(valid_columns) == 0 or find_connected_fours_and_score(board, PLAYER_SYMBOL)[1] > 0 or find_connected_fours_and_score(board, AI_SYMBOL)[1] > 0

    
    if depth == 0 or is_terminal:
        if is_terminal:
            if(find_connected_fours_and_score(board, AI_SYMBOL)[1] > 0):
                return random.choice(valid_columns), 100000 #hena el ai kasab w ana 3ayez el state dy
            elif(find_connected_fours_and_score(board, PLAYER_SYMBOL)[1] > 0):
                return random.choice(valid_columns), -100000 #hena el player kasab w ana msh 3ayz el state dy
            else:
                return random.choice(valid_columns), 0
        else:
            return random.choice(valid_columns), heuristic_function(board)
    if maximizing_player:
        value = -1 * float('inf')
        best_column = valid_columns[0] if valid_columns else None
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, AI_SYMBOL)
            _, new_score = minimax_with_alpha_beta_pruning(temp_board, depth - 1, alpha, beta, False)
            if new_score > value:
                value = new_score
                best_column = col
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return best_column, value
    else:
        value = float('inf')
        best_column = valid_columns[0] if valid_columns else None
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
            _, new_score = minimax_with_alpha_beta_pruning(temp_board, depth - 1, alpha, beta, True)
            if new_score < value:
                value = new_score
                best_column = col
            beta = min(beta, value)
            if beta <= alpha:
                break
        return best_column, value


def expected_minimax(board, depth, maximizing_player):
    global nodes_expanded
    nodes_expanded +=1
    valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
    is_terminal = len(valid_columns) == 0 or find_connected_fours_and_score(board, PLAYER_SYMBOL)[1] > 0 or find_connected_fours_and_score(board, AI_SYMBOL)[1] > 0
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if(find_connected_fours_and_score(board, AI_SYMBOL)[1] > 0):
                return random.choice(valid_columns), 100000
            elif(find_connected_fours_and_score(board, PLAYER_SYMBOL)[1] > 0):
                return random.choice(valid_columns), -100000
            else:
                return random.choice(valid_columns), 0
        else:
            return random.choice(valid_columns), heuristic_function(board)
        
    if maximizing_player:
        value = -1 * float('inf')
        best_column = random.choice(valid_columns)
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, AI_SYMBOL)
            _, new_score = expected_minimax(temp_board, depth - 1, False)
            if new_score > value:
                value = new_score
                best_column = col
        return best_column, value
    else: #hena ba handle door el player fa bagarab kol el possible moves elly momken yel3abha w ashof average score bta3hom
        expected_val = 0
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
            _, new_score = expected_minimax(temp_board, depth - 1, True)
            expected_val += new_score / len(valid_columns)
        return random.choice(valid_columns), expected_val 


def print_minimax_tree(board, depth, alpha, beta, maximizing_player, prefix=""):
    if depth == 0 or len([c for c in range(COLUMNS) if is_valid_location(board, c)]) == 0:
        heuristic_score = heuristic_function(board)
        print(f"{prefix}Leaf Node: Heuristic Score = {heuristic_score}")
        return heuristic_score

    print(f"{prefix}Depth {depth}, Maximizing: {maximizing_player}, Alpha: {alpha}, Beta: {beta}")
    valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
    if maximizing_player:
        value = float('-inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, AI_SYMBOL)
            child_score = print_minimax_tree(temp_board, depth - 1, alpha, beta, False, prefix + "    ")
            value = max(value, child_score)
            alpha = max(alpha, value)
            if alpha >= beta:
                print(f"{prefix}    Pruned at Column {col}")
                break
        print(f"{prefix}Returning Value: {value}")
        return value
    else:
        value = float('inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
            child_score = print_minimax_tree(temp_board, depth - 1, alpha, beta, True, prefix + "    ")
            value = min(value, child_score)
            beta = min(beta, value)
            if alpha >= beta:
                print(f"{prefix}    Pruned at Column {col}")
                break
        print(f"{prefix}Returning Value: {value}")
        return value
            
def print_minimax_tree_no_pruning(board, depth, maximizing_player, prefix=""):
    if depth == 0 or len([c for c in range(COLUMNS) if is_valid_location(board, c)]) == 0:
        heuristic_score = heuristic_function(board)
        print(f"{prefix}Leaf Node: Heuristic Score = {heuristic_score}")
        return heuristic_score

    print(f"{prefix}Depth {depth}, Maximizing: {maximizing_player}")
    valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
    if maximizing_player:
        value = float('-inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, AI_SYMBOL)
            child_score = print_minimax_tree_no_pruning(temp_board, depth - 1, False, prefix + "    ")
            value = max(value, child_score)
        print(f"{prefix}Returning Value: {value}")
        return value
    else:
        value = float('inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
            child_score = print_minimax_tree_no_pruning(temp_board, depth - 1, True, prefix + "    ")
            value = min(value, child_score)
        print(f"{prefix}Returning Value: {value}")
        return value


def print_minimax_tree_combined(board, depth, alpha=None, beta=None, maximizing_player=True, pruning=False, prefix=""):

    global nodes_expanded
    nodes_expanded += 1  # Increment the counter for each node processed

    # Base case: terminal state or depth limit reached
    if depth == 0 or len([c for c in range(COLUMNS) if is_valid_location(board, c)]) == 0:
        heuristic_score = heuristic_function(board)
        print(f"{prefix}Leaf Node: Heuristic Score = {heuristic_score}")
        return heuristic_score

    # Print the current node
    if pruning:
        print(f"{prefix}Depth {depth}, Maximizing: {maximizing_player}, Alpha: {alpha}, Beta: {beta}")
    else:
        print(f"{prefix}Depth {depth}, Maximizing: {maximizing_player}")

    valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
    if maximizing_player:
        value = float('-inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, AI_SYMBOL)
            if pruning:
                child_score = print_minimax_tree_combined(
                    temp_board, depth - 1, alpha, beta, False, pruning, prefix + "    "
                )
                value = max(value, child_score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    print(f"{prefix}    Pruned at Column {col}")
                    break
            else:
                child_score = print_minimax_tree_combined(
                    temp_board, depth - 1, None, None, False, pruning, prefix + "    "
                )
                value = max(value, child_score)
        print(f"{prefix}Returning Value: {value}")
        return value
    else:
        value = float('inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
            if pruning:
                child_score = print_minimax_tree_combined(
                    temp_board, depth - 1, alpha, beta, True, pruning, prefix + "    "
                )
                value = min(value, child_score)
                beta = min(beta, value)
                if alpha >= beta:
                    print(f"{prefix}    Pruned at Column {col}")
                    break
            else:
                child_score = print_minimax_tree_combined(
                    temp_board, depth - 1, None, None, True, pruning, prefix + "    "
                )
                value = min(value, child_score)
        print(f"{prefix}Returning Value: {value}")
        return value




# def generate_minimax_tree(board, depth, alpha=None, beta=None, maximizing_player=True, pruning=False, graph=None, parent=None, node_id=0):
#     """
#     Generates a visual Minimax tree with or without Alpha-Beta Pruning.

#     Parameters:
#     - board: The current game board.
#     - depth: The depth of the Minimax tree.
#     - alpha: The alpha value for pruning (only used if pruning=True).
#     - beta: The beta value for pruning (only used if pruning=True).
#     - maximizing_player: True if it's the maximizing player's turn, False otherwise.
#     - pruning: True to use Alpha-Beta Pruning, False for standard Minimax.
#     - graph: The Graphviz Digraph object.
#     - parent: The parent node ID.
#     - node_id: The current node ID.

#     Returns:
#     - The heuristic score of the node.
#     - The updated Graphviz Digraph object.
#     - The next available node ID.
#     """
#     if graph is None:
#         graph = Digraph(format='png')
#         graph.attr('node', shape='circle')

#     # Base case: terminal state or depth limit reached
#     if depth == 0 or len([c for c in range(COLUMNS) if is_valid_location(board, c)]) == 0:
#         heuristic_score = heuristic_function(board)
#         node_label = f"Leaf\nScore: {heuristic_score}"
#         graph.node(str(node_id), label=node_label)
#         if parent is not None:
#             graph.edge(str(parent), str(node_id))
#         return heuristic_score, graph, node_id + 1

#     # Create a node for the current state
#     node_label = f"Depth: {depth}\nMaximizing: {maximizing_player}"
#     if pruning:
#         node_label += f"\nAlpha: {alpha}\nBeta: {beta}"
#     graph.node(str(node_id), label=node_label)
#     if parent is not None:
#         graph.edge(str(parent), str(node_id))

#     valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
#     if maximizing_player:
#         value = float('-inf')
#         for col in valid_columns:
#             row = get_next_open_row(board, col)
#             temp_board = drop_piece(board, row, col, AI_SYMBOL)
#             child_id = node_id + 1
#             if pruning:
#                 child_score, graph, child_id = generate_minimax_tree(temp_board, depth - 1, alpha, beta, False, pruning, graph, node_id, child_id)
#                 value = max(value, child_score)
#                 alpha = max(alpha, value)
#                 if alpha >= beta:
#                     break  # Prune
#             else:
#                 child_score, graph, child_id = generate_minimax_tree(temp_board, depth - 1, None, None, False, pruning, graph, node_id, child_id)
#                 value = max(value, child_score)
#         return value, graph, child_id
#     else:
#         value = float('inf')
#         for col in valid_columns:
#             row = get_next_open_row(board, col)
#             temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
#             child_id = node_id + 1
#             if pruning:
#                 child_score, graph, child_id = generate_minimax_tree(temp_board, depth - 1, alpha, beta, True, pruning, graph, node_id, child_id)
#                 value = min(value, child_score)
#                 beta = min(beta, value)
#                 if alpha >= beta:
#                     break  # Prune
#             else:
#                 child_score, graph, child_id = generate_minimax_tree(temp_board, depth - 1, None, None, True, pruning, graph, node_id, child_id)
#                 value = min(value, child_score)
#         return value, graph, child_id


def generate_minimax_tree(board, depth, alpha=None, beta=None, maximizing_player=True, pruning=False, graph=None, parent=None, node_id=0):
    """
    Generates a visual Minimax tree with or without Alpha-Beta Pruning.

    Parameters:
    - board: The current game board.
    - depth: The depth of the Minimax tree.
    - alpha: The alpha value for pruning (only used if pruning=True).
    - beta: The beta value for pruning (only used if pruning=True).
    - maximizing_player: True if it's the maximizing player's turn, False otherwise.
    - pruning: True to use Alpha-Beta Pruning, False for standard Minimax.
    - graph: The Graphviz Digraph object.
    - parent: The parent node ID.
    - node_id: The current node ID.

    Returns:
    - The heuristic score of the node.
    - The updated Graphviz Digraph object.
    - The next available node ID.
    """
    if graph is None:
        graph = Digraph(format='png')
        graph.attr('node', shape='circle')

    # Base case: terminal state or depth limit reached
    if depth == 0 or len([c for c in range(COLUMNS) if is_valid_location(board, c)]) == 0:
        heuristic_score = heuristic_function(board)
        node_label = f"Leaf\nScore: {heuristic_score}"
        graph.node(str(node_id), label=node_label)
        if parent is not None:
            graph.edge(str(parent), str(node_id))
        return heuristic_score, graph, node_id + 1

    # Create a node for the current state
    node_label = f"Depth: {depth}\nMaximizing: {maximizing_player}"
    if pruning:
        node_label += f"\nAlpha: {alpha}\nBeta: {beta}"
    graph.node(str(node_id), label=node_label)
    if parent is not None:
        graph.edge(str(parent), str(node_id))

    valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
    next_node_id = node_id + 1  # Start assigning IDs for child nodes
    if maximizing_player:
        value = float('-inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, AI_SYMBOL)
            if pruning:
                child_score, graph, next_node_id = generate_minimax_tree(
                    temp_board, depth - 1, alpha, beta, False, pruning, graph, node_id, next_node_id
                )
                value = max(value, child_score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Prune
            else:
                child_score, graph, next_node_id = generate_minimax_tree(
                    temp_board, depth - 1, None, None, False, pruning, graph, node_id, next_node_id
                )
                value = max(value, child_score)
        return value, graph, next_node_id
    else:
        value = float('inf')
        for col in valid_columns:
            row = get_next_open_row(board, col)
            temp_board = drop_piece(board, row, col, PLAYER_SYMBOL)
            if pruning:
                child_score, graph, next_node_id = generate_minimax_tree(
                    temp_board, depth - 1, alpha, beta, True, pruning, graph, node_id, next_node_id
                )
                value = min(value, child_score)
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Prune
            else:
                child_score, graph, next_node_id = generate_minimax_tree(
                    temp_board, depth - 1, None, None, True, pruning, graph, node_id, next_node_id
                )
                value = min(value, child_score)
        return value, graph, next_node_id



def main():
    global K
    K = int(input("Enter the depth (K) for heuristic pruning: "))
    
    USER_SCORE = 0
    AI_SCORE = 0
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("Connect 4")
    
    algorithm = select_algorithm()
    print(f"Selected algorithm: {algorithm}")

    board = create_board()
    draw_board(screen, board)

    font = pygame.font.Font(None, 74)

    turn = 0
    running = True
    game_over = False

    while running:
        if not game_over:
            pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, CELL_SIZE))
            mouse_x, _ = pygame.mouse.get_pos()
            if turn % 2 == 0:
                pygame.draw.circle(screen, RED, (mouse_x, CELL_SIZE // 2), RADIUS)
            pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not game_over and event.type == pygame.MOUSEBUTTONDOWN and turn == 0:
                col = mouse_x // CELL_SIZE
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    board = drop_piece(board, row, col, PLAYER_SYMBOL)
                    turn = not turn
                    draw_board(screen, board)

            if not game_over and turn == 1 and running:
                valid_columns = [c for c in range(COLUMNS) if is_valid_location(board, c)]
                if valid_columns:
                    global total_time  # Use the global variable to accumulate time
                    if algorithm == "Minimax without Alpha-Beta Pruning":
                        print("\nRunning Minimax without Alpha-Beta Pruning...")
                        start_time = time.time()  # Start time
                        col, _ = minimax(board, K, True)
                        end_time = time.time()  # End time
                        move_time = end_time - start_time
                        total_time += move_time
                        print(f"Time Taken for Move: {move_time:.4f} seconds")
                    elif algorithm == "Minimax with Alpha-Beta Pruning":
                        print("\nRunning Minimax with Alpha-Beta Pruning...")
                        start_time = time.time()  # Start time
                        col, _ = minimax_with_alpha_beta_pruning(board, K, float('-inf'), float('inf'), True)
                        end_time = time.time()  # End time
                        move_time = end_time - start_time
                        total_time += move_time
                        print(f"Time Taken for Move: {move_time:.4f} seconds")
                    elif algorithm == "Expected Minimax":
                        print("\nRunning Expected Minimax...")
                        start_time = time.time()  # Start time
                        col, _ = expected_minimax(board, K, True)
                        end_time = time.time()  # End time
                        move_time = end_time - start_time
                        total_time += move_time
                        print(f"Time Taken for Move: {move_time:.4f} seconds")
                    else:
                        col = random.choice(valid_columns)

                    row = get_next_open_row(board, col)
                    board = drop_piece(board, row, col, AI_SYMBOL)
                    turn = not turn
                    draw_board(screen, board)

                    display_board_with_heuristic(board)

            if EMPTY_CELL not in board:
                game_over = True

        if game_over:
            global nodes_expanded
            print(f"Total Nodes Expanded: {nodes_expanded}")
            # global total_time
            print(f"\nTotal Time Taken by AI: {total_time:.4f} seconds")
            # global total_time
            # total_time = 0
            # Find connected fours and calculate scores for both players
            player_fours, USER_SCORE = find_connected_fours_and_score(board, PLAYER_SYMBOL)
            ai_fours, AI_SCORE = find_connected_fours_and_score(board, AI_SYMBOL)
                        
            # Clear the board
            board = create_board()

            # Add only the connected fours back to the board
            for index in player_fours:
                row, col = divmod(index, COLUMNS)
                board = drop_piece(board, row, col, PLAYER_SYMBOL)
            for index in ai_fours:
                row, col = divmod(index, COLUMNS)
                board = drop_piece(board, row, col, AI_SYMBOL)

            # Redraw the board with only the connected fours
            draw_board(screen, board)

            # Display "Game Over" message above the board
            text = font.render("Game Over", True, YELLOW)
            if USER_SCORE > AI_SCORE:
                winner_text = "Player Wins!" + " Score: " + str(USER_SCORE) + " vs " + str(AI_SCORE)
            elif AI_SCORE > USER_SCORE:
                winner_text = "AI Wins!" + " Score: " + str(AI_SCORE) + " vs " + str(USER_SCORE)
            else:
                winner_text = "It's a Tie!" + " Score: " + str(USER_SCORE) + " vs " + str(AI_SCORE)
            winner = font.render(winner_text, True, RED)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, CELL_SIZE // 4))
            screen.blit(winner, (WIDTH // 2 - winner.get_width() // 2, CELL_SIZE // 2 + 10))
            pygame.display.update()

            # Generate and display the Minimax tree at the end of the game
            print("\nGenerating Minimax Tree...")
            if algorithm == "Minimax without Alpha-Beta Pruning":
                _, graph, _ = generate_minimax_tree(board, K, maximizing_player=True, pruning=False)
                print_minimax_tree_combined(board, K, maximizing_player=True, pruning=False)

                graph.render('minimax_tree', view=True)  # Save and open the tree image
            elif algorithm == "Minimax with Alpha-Beta Pruning":
                _, graph, _ = generate_minimax_tree(board, K, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, pruning=True)
                print_minimax_tree_combined(board, K, alpha=float('-inf'), beta=float('inf'), maximizing_player=True, pruning=True)

                graph.render('minimax_pruning_tree', view=True)  # Save and open the tree image
            elif algorithm == "Expected Minimax":
                _, graph, _ = generate_minimax_tree(board, K, maximizing_player=True, pruning=False)
                print_minimax_tree_combined(board, K, maximizing_player=True, pruning=False)
                graph.render('expected_minimax_tree', view=True)  # Save and open the tree image
            
            # global nodes_expanded
            # nodes_expanded = 0

            # Wait for user input to restart or quit
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        game_over = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            board = create_board()
                            draw_board(screen, board)
                            turn = 0
                            game_over = False
                            USER_SCORE = 0
                            AI_SCORE = 0
                            break
                        if event.key == pygame.K_q:
                            running = False
                            game_over = False
                            break
                if not running or not game_over:
                    break

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()