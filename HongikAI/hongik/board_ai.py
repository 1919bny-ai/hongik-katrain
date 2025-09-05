# Copyright (c) 2020 Sanderland (KaTrain Original Author)
# Copyright (c) 2025 BNY (Modifications for Hongik AI)

# Implements the world of Go, defining the game board, its rules
# (moves, liberties, scoring, etc.), and determining the final winner.
#
# Author: Gemini 2.5 Pro, Gemini 2.5 Flash

import numpy as np
from collections import deque
import random

class IllegalMoveError(ValueError):
    """Exception class raised for an illegal move."""
    pass

class Board:
    """
    Represents the Go board and enforces game rules.
    """
    EMPTY, BLACK, WHITE, WALL = 0, 1, 2, 3
    PASS_LOC = 0

    def __init__(self, size):
        """Initializes the board, sets up walls, and history."""
        if isinstance(size, tuple):
            self.x_size, self.y_size = size
        else:
            self.x_size = self.y_size = size
        
        self.arrsize = (self.x_size + 1) * (self.y_size + 2) + 1
        self.dy = self.x_size + 1
        self.adj = [-self.dy, -1, 1, self.dy]
        
        self.board = np.zeros(shape=(self.arrsize), dtype=np.int8)
        self.pla = Board.BLACK
        self.prisoners = {Board.BLACK: 0, Board.WHITE: 0}
        self.ko_points = set() 
        self.consecutive_passes = 0
        self.turns = 0
        self.ko_recapture_counts = {}
        self.position_history = set()        
        self.position_history.add(self.board.tobytes())

        for i in range(-1, self.x_size + 1):
            self.board[self.loc(i, -1)] = Board.WALL
            self.board[self.loc(i, self.y_size)] = Board.WALL
        for i in range(-1, self.y_size + 1):
            self.board[self.loc(-1, i)] = Board.WALL
            self.board[self.loc(self.x_size, i)] = Board.WALL

    def copy(self):
        """Creates a deep copy of the current board state."""
        new_board = Board((self.x_size, self.y_size))
        new_board.board = np.copy(self.board)
        new_board.pla = self.pla
        new_board.prisoners = self.prisoners.copy()
        new_board.ko_points = self.ko_points.copy()
        new_board.consecutive_passes = self.consecutive_passes
        new_board.turns = self.turns
        new_board.ko_recapture_counts = self.ko_recapture_counts.copy()
        new_board.position_history = self.position_history.copy()
        return new_board
    
    @staticmethod
    def get_opp(player):
        """Gets the opponent of the given player."""
        return 3 - player

    def loc(self, x, y):
        """Converts (x, y) coordinates to a 1D array location."""
        return (x + 1) + self.dy * (y + 1)

    def loc_to_coord(self, loc):
        """Converts a 1D array location back to (x, y) coordinates."""
        return (loc % self.dy) - 1, (loc // self.dy) - 1

    def is_on_board(self, loc):
        """Checks if a location is within the board boundaries (not a wall)."""
        return self.board[loc] != Board.WALL

    def _get_group_info(self, loc):
        """Scans and returns the stones and liberties of a group at a specific location."""
        if not self.is_on_board(loc) or self.board[loc] == self.EMPTY:
            return None, None
            
        player = self.board[loc]
        group_stones, liberties = set(), set()
        q, visited = deque([loc]), {loc}
        
        while q:
            current_loc = q.popleft()
            group_stones.add(current_loc)
            for dloc in self.adj:
                adj_loc = current_loc + dloc
                if self.is_on_board(adj_loc):
                    adj_stone = self.board[adj_loc]
                    if adj_stone == self.EMPTY:
                        liberties.add(adj_loc)
                    elif adj_stone == player and adj_loc not in visited:
                        visited.add(adj_loc)
                        q.append(adj_loc)
        return group_stones, liberties

    def would_be_legal(self, player, loc):
        """Checks if a move would be legal without actually playing it."""
        if loc == self.PASS_LOC: return True
        if not self.is_on_board(loc) or self.board[loc] != self.EMPTY or loc in self.ko_points:
            return False

        temp_board = self.copy()
        temp_board.board[loc] = player
        
        opponent = self.get_opp(player)
        captured_any = False
        captured_stones = set()
        
        for dloc in temp_board.adj:
            adj_loc = loc + dloc
            if temp_board.board[adj_loc] == opponent:
                group, libs = temp_board._get_group_info(adj_loc)
                if not libs:
                    captured_any = True
                    captured_stones.update(group)

        if captured_any:
            for captured_loc in captured_stones:
                temp_board.board[captured_loc] = self.EMPTY

        next_board_hash = temp_board.board.tobytes()
        if next_board_hash in self.position_history:
            return False
        
        if captured_any:
            return True

        _, my_libs = temp_board._get_group_info(loc)
        return bool(my_libs)
    
    def get_features(self):
        """Generates a feature tensor for the neural network input."""
        features = np.zeros((self.y_size, self.x_size, 3), dtype=np.float32)
        
        current_player = self.pla
        opponent_player = self.get_opp(self.pla)

        for y in range(self.y_size):
            for x in range(self.x_size):
                loc = self.loc(x, y)
                stone = self.board[loc]
                if stone == current_player:
                    features[y, x, 0] = 1
                elif stone == opponent_player:
                    features[y, x, 1] = 1

        if self.pla == self.WHITE:
            features[:, :, 2] = 1.0
            
        return features
    
    def is_game_over(self):
        """Checks if the game is over (due to consecutive passes)."""
        return self.consecutive_passes >= 2

    def play(self, player, loc):
        """Plays a move on the board, captures stones, and updates the game state."""
        if not self.would_be_legal(player, loc):
            raise IllegalMoveError("This move is against the rules.")

        self.ko_points.clear()

        if loc == self.PASS_LOC:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
            self.board[loc] = player
            opponent = self.get_opp(player)
            
            captured_stones = set()
            
            for dloc in self.adj:
                adj_loc = loc + dloc
                if self.board[adj_loc] == opponent:
                    group, libs = self._get_group_info(adj_loc)
                    if not libs:
                        captured_stones.update(group)

            if captured_stones:
                self.prisoners[player] += len(captured_stones)
                for captured_loc in captured_stones:
                    self.board[captured_loc] = self.EMPTY
            
            my_group, my_libs = self._get_group_info(loc)

            if len(captured_stones) == 1 and len(my_group) == 1 and len(my_libs) == 1:
                ko_loc = captured_stones.pop()
                self.ko_points.add(ko_loc)
            
            board_hash = self.board.tobytes()
            self.position_history.add(board_hash)
        self.pla = self.get_opp(player)
        self.turns += 1
    
    def _is_group_alive_statically(self, group_stones: set, board_state: np.ndarray) -> bool:
        """Statically analyzes if a group is alive by checking for two eyes."""
        if not group_stones: return False
        owner_player = board_state[next(iter(group_stones))]
        eye_locations = set()
        for stone_loc in group_stones:
            for dloc in self.adj:
                adj_loc = stone_loc + dloc
                if board_state[adj_loc] == self.EMPTY: eye_locations.add(adj_loc)
        real_eye_count, visited_eye_locs = 0, set()
        for eye_loc in eye_locations:
            if eye_loc in visited_eye_locs: continue
            eye_region, q, is_real_eye = set(), deque([eye_loc]), True
            visited_eye_locs.add(eye_loc); eye_region.add(eye_loc)
            while q:
                current_loc = q.popleft()
                for dloc in self.adj:
                    adj_loc = current_loc + dloc
                    if self.is_on_board(adj_loc):
                        if board_state[adj_loc] == self.get_opp(owner_player):
                            is_real_eye = False; break
                        elif board_state[adj_loc] == self.EMPTY and adj_loc not in visited_eye_locs:
                            visited_eye_locs.add(adj_loc); eye_region.add(adj_loc); q.append(adj_loc)
                if not is_real_eye: break
            if is_real_eye:
                eye_size = len(eye_region)
                if eye_size >= 6: real_eye_count += 2
                else: real_eye_count += 1
            if real_eye_count >= 2: return True
        return real_eye_count >= 2

    def _is_group_alive_by_rollout(self, group_stones_initial: set) -> bool:
        """Determines if a group is alive via Monte Carlo rollouts for ambiguous cases."""
        NUM_ROLLOUTS = 20
        MAX_ROLLOUT_DEPTH = self.x_size * self.y_size // 2
        
        owner_player = self.board[next(iter(group_stones_initial))]
        attacker = self.get_opp(owner_player)
        deaths = 0

        for _ in range(NUM_ROLLOUTS):
            rollout_board = self.copy()
            rollout_board.pla = attacker
            
            for _ in range(MAX_ROLLOUT_DEPTH):
                first_stone_loc = next(iter(group_stones_initial))
                if rollout_board.board[first_stone_loc] != owner_player:
                    deaths += 1
                    break

                legal_moves = [loc for loc in range(1, self.arrsize) if rollout_board.board[loc] == self.EMPTY]
                random.shuffle(legal_moves)
                
                move_made = False
                for move in legal_moves:
                    if rollout_board.would_be_legal(rollout_board.pla, move):
                        rollout_board.play(rollout_board.pla, move)
                        move_made = True
                        break
                
                if not move_made:
                    rollout_board.play(rollout_board.pla, self.PASS_LOC)

                if rollout_board.is_game_over():   
                    break
            
        death_rate = deaths / NUM_ROLLOUTS
        print(f"[Life/Death Log] Group survival probability: {1-death_rate:.0%}")
        return death_rate < 0.5

    def get_winner(self, komi=6.5):
        """Calculates the final score and determines the winner, handling life and death."""
        temp_board_state = np.copy(self.board)
        total_captives = self.prisoners.copy()
        
        all_groups = self._find_all_groups(temp_board_state)
        for player, groups in all_groups.items():
            for group_stones in groups:
                is_alive = self._is_group_alive_statically(group_stones, temp_board_state)
                
                if not is_alive:
                    is_alive = self._is_group_alive_by_rollout(group_stones)

                if not is_alive:
                    total_captives[self.get_opp(player)] += len(group_stones)
                    for stone_loc in group_stones:
                        temp_board_state[stone_loc] = self.EMPTY

        final_board_with_territory = self._calculate_territory(temp_board_state)
        black_territory = np.sum((final_board_with_territory == self.BLACK) & (temp_board_state == self.EMPTY))
        white_territory = np.sum((final_board_with_territory == self.WHITE) & (temp_board_state == self.EMPTY))
        
        black_score = black_territory + total_captives.get(self.BLACK, 0)
        white_score = white_territory + total_captives.get(self.WHITE, 0) + komi
        winner = self.BLACK if black_score > white_score else self.WHITE
        
        return winner, black_score, white_score, total_captives

    def _find_all_groups(self, board_state: np.ndarray) -> dict:
        """Finds all stone groups on the board for a given board state."""
        visited, all_groups = set(), {self.BLACK: [], self.WHITE: []}
        for loc in range(self.arrsize):
            if board_state[loc] in [self.BLACK, self.WHITE] and loc not in visited:
                player, group_stones, q = board_state[loc], set(), deque([loc])
                visited.add(loc); group_stones.add(loc)
                while q:
                    current_loc = q.popleft()
                    for dloc in self.adj:
                        adj_loc = current_loc + dloc
                        if board_state[adj_loc] == player and adj_loc not in visited:
                            visited.add(adj_loc); group_stones.add(adj_loc); q.append(adj_loc)
                all_groups[player].append(group_stones)
        return all_groups

    def _calculate_territory(self, board_state: np.ndarray) -> np.ndarray:
        """Calculates the territory for each player on a given board state."""
        territory_map, visited = np.copy(board_state), set()
        for loc in range(self.arrsize):
            if territory_map[loc] == self.EMPTY and loc not in visited:
                region_points, border_colors, q = set(), set(), deque([loc])
                visited.add(loc); region_points.add(loc)
                while q:
                    current_loc = q.popleft()
                    for dloc in self.adj:
                        adj_loc = current_loc + dloc
                        if board_state[adj_loc] in [self.BLACK, self.WHITE]: border_colors.add(board_state[adj_loc])
                        elif board_state[adj_loc] == self.EMPTY and adj_loc not in visited:
                            visited.add(adj_loc); region_points.add(adj_loc); q.append(adj_loc)
                if len(border_colors) == 1:
                    owner = border_colors.pop()
                    for point in region_points: territory_map[point] = owner
        return territory_map
