import chess

# Rewards

# Define the material balance function
# Calculate the material balance of the board based on the piece values
def material_balance(board):
    # Define the piece values for each chess piece
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0,
    }
    # Calculate material balance of board
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            balance += piece_values[str(piece)]
    return balance

# Board tempo/control related rewards:
# Define the piece mobility function
# Calculates the mobility of pieces by counting legal moves for each side.
def piece_mobility(board):
    mobility = 0
    for move in board.legal_moves:
        if board.turn:  
            mobility += 1
        else:
            mobility -= 1
    return mobility

# Define the king safety function
# Reward castling
def king_safety(board):
    safety = 0
    last_move = board.peek()  

    if board.is_kingside_castling(last_move):
        safety += 15
    elif board.is_queenside_castling(last_move):
        safety += 10

    return safety

# Define the center control function
# Calculate the control of the center squares by each side
def center_control(board):
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    control = 0

    for square in center_squares:
        attacker_color = board.turn
        defender_color = not board.turn
        if board.is_attacked_by(attacker_color, square):
            control += 1
        if board.is_attacked_by(defender_color, square):
            control -= 1
    return control

# Pawn related positional rewards:
# Define the function to check if a move is a double pawn push
def is_double_pawn_push(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    return abs(move.from_square - move.to_square) == 16

# Define the function to check if a move results in a passed pawn
def is_passed_pawn(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False

    # Check if there are opponent pawns in the adjacent files
    adjacent_files = [max(0, move.to_square % 8 - 1), min(7, move.to_square % 8 + 1)]
    direction = -8 if board.turn == chess.WHITE else 8

    for file in adjacent_files:
        for i in range(1, 6):
            adjacent_square = move.to_square + i * direction + file - move.to_square % 8
            if chess.square_is_valid(adjacent_square) and board.piece_at(adjacent_square) == chess.Piece(chess.PAWN, not board.turn):
                return False
    return True

# Define the function to check if a pawn is isolated
def is_isolated_pawn(board, square, color):
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    for r in range(8):
        if r == rank:
            continue
        if file > 0 and board.piece_at(chess.square(r, file - 1)) == chess.Piece(chess.PAWN, color):
            return False
        if file < 7 and board.piece_at(chess.square(r, file + 1)) == chess.Piece(chess.PAWN, color):
            return False
    return True

# Define the pawn structure function
# This function calculates the pawn structure score based on double pawn pushes, passed pawns, and isolated pawns
def pawn_structure(board):
    structure = 0
    isolated_pawns = 0
    last_move = board.peek() 

    # Update structure score based on double pawn push or passed pawn
    if is_double_pawn_push(board, last_move):
        structure -= 10
    elif is_passed_pawn(board, last_move):
        structure += 20

    # Calculate isolated pawns and update structure score
    isolated_pawns = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            if is_isolated_pawn(board, square, piece.color):
                isolated_pawns += 1 if piece.color == board.turn else -1
    structure -= 10 * isolated_pawns

    return structure

