# Copyright (c) 2020 Sanderland (KaTrain Original Author)
# Copyright (c) 2025 BNY (Modifications for Hongik AI)

# The engine file that serves as the AI's brain, responsible for training
# the model through reinforcement learning and performing move analysis.
#
# Author: Gemini 2.5 Pro, Gemini 2.5 Flash

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import threading
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import time
import json
import random
import traceback
from collections import deque
import pickle
import csv
from huggingface_hub import hf_hub_download

from hongik.hongik_ai import HongikAIPlayer,CNNTransformerHybrid
from katrain.core.sgf_parser import Move
from katrain.core.constants import *
from kivy.clock import Clock
from hongik.board_ai import Board, IllegalMoveError
from katrain.core.game_node import GameNode
from katrain.gui.theme import Theme

class BaseEngine:
    """Base class for KaTrain engines."""
    def __init__(self, katrain, config):
        self.katrain, self.config = katrain, config
    def on_error(self, message, code=None, allow_popup=True):
        print(f"ERROR: {message}")
        if allow_popup and hasattr(self.katrain, "engine_recovery_popup"):
            Clock.schedule_once(lambda dt: self.katrain("engine_recovery_popup", message, code))

class HongikAIEngine(BaseEngine):
    """
    Main AI engine that manages the model, self-play, training, and analysis.
    It orchestrates the entire reinforcement learning loop.
    """
    BOARD_SIZE, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF = 19, 7, 256, 8, 1024
    SAVE_WEIGHTS_EVERY_STEPS, EVALUATION_EVERY_STEPS = 5, 20
    REPLAY_BUFFER_SIZE, TRAINING_BATCH_SIZE = 200000, 32
    CHECKPOINT_EVERY_GAMES = 10 
    
    RULES = {
        "tromp-taylor": {"name": "Tromp-Taylor", "komi": 7.5, "scoring": "area"},
        "korean": {"name": "korean", "komi": 6.5, "scoring": "territory"},
        "chinese": {"name": "Chinese", "komi": 7.5, "scoring": "area"}
    }
    
    @staticmethod
    def get_rules(ruleset: str):
        """Returns the ruleset details for a given ruleset name."""
        if not ruleset or ruleset.lower() not in HongikAIEngine.RULES:
            ruleset = "korean"
        return HongikAIEngine.RULES[ruleset.lower()]
    
    def __init__(self, katrain, config):
        """
        Initializes the Hongik AI Engine. This involves setting up paths, loading
        the neural network model and replay buffer, and preparing for training.
        """
        super().__init__(katrain, config)
        print("Initializing Hongik AI Integrated Engine...")

        from appdirs import user_data_dir        
        APP_NAME = "HongikAI"
        APP_AUTHOR = "NamyongPark" 
        
        self.BASE_PATH = user_data_dir(APP_NAME, APP_AUTHOR)
        print(f"Data will be stored in: {self.BASE_PATH}") 
        
        self.REPLAY_BUFFER_PATH = os.path.join(self.BASE_PATH, "replay_buffer.pkl")
        self.WEIGHTS_FILE_PATH = os.path.join(self.BASE_PATH, "hongik_ai_memory.weights.h5")
        self.BEST_WEIGHTS_FILE_PATH = os.path.join(self.BASE_PATH, "hongik_ai_best.weights.h5")
        self.CHECKPOINT_BUFFER_PATH = os.path.join(self.BASE_PATH, "replay_buffer_checkpoint.pkl")
        self.CHECKPOINT_WEIGHTS_PATH = os.path.join(self.BASE_PATH, "hongik_ai_checkpoint.weights.h5")
        self.TRAINING_LOG_PATH = os.path.join(self.BASE_PATH, "training_log.csv")
        
        os.makedirs(self.BASE_PATH, exist_ok=True)        

        REPO_ID = "puco21/HongikAI"         
        files_to_download = [
            "replay_buffer.pkl",
            "hongik_ai_memory.weights.h5",
            "hongik_ai_best.weights.h5"
        ]

        print("Checking for AI data files...")
        for filename in files_to_download:
            local_path = os.path.join(self.BASE_PATH, filename)
            if not os.path.exists(local_path):
                print(f"Downloading {filename} from Hugging Face Hub...")
                try:
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=filename,
                        local_dir=self.BASE_PATH,
                        local_dir_use_symlinks=False 
                    )
                    print(f"'{filename}' download complete.")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
    
        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)
        self.load_replay_buffer(self.REPLAY_BUFFER_PATH)
        
        self.hongik_model = CNNTransformerHybrid(self.NUM_LAYERS, self.D_MODEL, self.NUM_HEADS, self.D_FF, self.BOARD_SIZE)
        _ = self.hongik_model(np.zeros((1, self.BOARD_SIZE, self.BOARD_SIZE, 3), dtype=np.float32))
        
        load_path = self.CHECKPOINT_WEIGHTS_PATH if os.path.exists(self.CHECKPOINT_WEIGHTS_PATH) else (self.WEIGHTS_FILE_PATH if os.path.exists(self.WEIGHTS_FILE_PATH) else self.BEST_WEIGHTS_FILE_PATH)
        if os.path.exists(load_path):
            try: 
                self.hongik_model.load_weights(load_path)
                print(f"Successfully loaded weights: {load_path}")
            except Exception as e: 
                print(f"Failed to load weights (starting new training): {e}")

        try:
            max_visits = int(config.get("max_visits",150))
        except (ValueError, TypeError):
            print(f"Warning: Invalid max_visits value in config. Using default (150).")
            max_visits = 150
        self.hongik_player = HongikAIPlayer(self.hongik_model, n_simulations=max_visits)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        self.policy_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.value_loss_fn = tf.keras.losses.MeanSquaredError()
        self.training_step_counter, self.game_history, self.self_play_active = 0, [], False
        
        class MockProcess: poll = lambda self: None
        self.katago_process = self.hongik_process = MockProcess()
        self.sound_index = False
        print("Hongik AI Engine ready!")

    def save_replay_buffer(self, path):
        """Saves the current replay buffer to a specified file path using pickle."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.replay_buffer, f)
            print(f"Successfully saved experience data ({len(self.replay_buffer)} items) to '{path}'.")
        except Exception as e:
            print(f"Error saving experience data: {e}")

    def load_replay_buffer(self, path):
        """Loads a replay buffer from a file, prioritizing a checkpoint file if it exists."""
        load_path = self.CHECKPOINT_BUFFER_PATH if os.path.exists(self.CHECKPOINT_BUFFER_PATH) else path
        if os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    self.replay_buffer = pickle.load(f)
                
                if self.replay_buffer.maxlen != self.REPLAY_BUFFER_SIZE:
                    new_buffer = deque(maxlen=self.REPLAY_BUFFER_SIZE)
                    new_buffer.extend(self.replay_buffer)
                    self.replay_buffer = new_buffer
                print(f"Successfully loaded experience data ({len(self.replay_buffer)} items) from '{load_path}'.")
            except Exception as e:
                print(f"Error loading experience data: {e}")

    def _checkpoint_save(self):
        """Saves a training checkpoint, including both the replay buffer and model weights."""
        print(f"\n[{time.strftime('%H:%M:%S')}] Saving checkpoint...")
        self.save_replay_buffer(self.CHECKPOINT_BUFFER_PATH)
        self.hongik_model.save_weights(self.CHECKPOINT_WEIGHTS_PATH)
        print("Checkpoint saved.")

    def _log_training_progress(self, details: dict):
        """Logs the progress of the training process to a CSV file for later analysis."""
        try:
            file_exists = os.path.isfile(self.TRAINING_LOG_PATH)
            with open(self.TRAINING_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=details.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(details)
        except Exception as e:
            print(f"Error logging training progress: {e}")

    def _node_to_board(self, node: GameNode) -> Board:
        """Converts a KaTrain GameNode object to the internal Board representation used by the engine."""
        board_snapshot = Board(node.board_size) 
        root_node = node.root
        for player, prop_name in [(Board.BLACK, 'AB'), (Board.WHITE, 'AW')]:
            setup_stones = root_node.properties.get(prop_name, [])
            if setup_stones:
                for coords in setup_stones:
                    loc = board_snapshot.loc(coords[0], coords[1])
                    if board_snapshot.board[loc] == Board.EMPTY:
                        board_snapshot.play(player, loc)
        
        current_player = Board.BLACK
        for scene_node in node.nodes_from_root[1:]:
            move = scene_node.move
            if move:
                loc = Board.PASS_LOC if move.is_pass else board_snapshot.loc(move.coords[0], move.coords[1])
                board_snapshot.play(current_player, loc)
                current_player = board_snapshot.pla

        board_snapshot.pla = Board.BLACK if node.next_player == 'B' else Board.WHITE
        return board_snapshot

    def request_analysis(self, analysis_node: GameNode, callback: callable, **kwargs):
        """
        Requests an analysis of a specific board position. The analysis is run
        in a separate thread to avoid blocking the GUI.
        """
        if not self.katrain.game: return
        game_id = self.katrain.game.game_id
        board = self._node_to_board(analysis_node)
        threading.Thread(target=self._run_analysis, args=(game_id, board, analysis_node, callback), daemon=True).start()

    def _run_analysis(self, game_id, board, analysis_node, callback):
        """
        The target function for the analysis thread. It runs MCTS and sends the
        formatted results back to the GUI via the provided callback.
        """
        try:
            policy_logits, _ = self.hongik_player.model(np.expand_dims(board.get_features(), 0), training=False)
            policy = tf.nn.softmax(policy_logits[0]).numpy()

            _, root_node = self.hongik_player.get_best_move(board)
            analysis_result = self._format_analysis_results("analysis", root_node, board, policy)
                        
            analysis_node.analysis = analysis_result

            def guarded_callback(dt):
                if self.katrain.game and self.katrain.game.game_id == game_id:
                    callback(analysis_result, False)
            Clock.schedule_once(guarded_callback)
        except Exception as e:
            print(f"Error during AI analysis execution: {e}")
            traceback.print_exc()

    def _format_analysis_results(self, query_id, root_node, board, policy=None): # <-- policy=None 인자 추가
        """
        MCTS 분석 데이터를 KaTrain GUI가 이해할 수 있는 딕셔너리 형식으로 변환합니다.
        """
        move_infos, moves_dict = [], {}
        
        if root_node and root_node.children:
            sorted_children = sorted(root_node.children.items(), key=lambda i: i[1].n_visits, reverse=True)
            
            best_move_q = sorted_children[0][1].q_value if sorted_children else 0

            for i, (action, child) in enumerate(sorted_children):
                coords = board.loc_to_coord(self.hongik_player._action_to_loc(action, board))
                move_gtp = Move(coords=coords).gtp()
                
                current_player_winrate = (child.q_value + 1) / 2
                display_winrate = 1.0 - current_player_winrate if board.pla == Board.WHITE else current_player_winrate
                display_score = -child.q_value * 20 if board.pla == Board.WHITE else child.q_value * 20
                
                points_lost = (best_move_q - child.q_value) * 20
                
                move_data = {
                    "move": move_gtp, 
                    "visits": child.n_visits, 
                    "winrate": display_winrate, 
                    "scoreLead": display_score,
                    "pointsLost": points_lost,
                    "pv": [move_gtp], 
                    "order": i
                }
                
                move_infos.append(move_data)
                moves_dict[move_gtp] = move_data

        current_player_winrate = (root_node.q_value + 1) / 2 if root_node else 0.5
        display_winrate = 1.0 - current_player_winrate if board.pla == Board.WHITE else current_player_winrate
        display_score = -root_node.q_value * 20 if (root_node and board.pla == Board.WHITE) else (root_node.q_value * 20 if root_node else 0.0)
        
        root_info = {"winrate": display_winrate, "scoreLead": display_score, "visits": root_node.n_visits if root_node else 0}        
        return {"id": query_id, "moveInfos": move_infos, "moves": moves_dict, "root": root_info, "rootInfo": root_info, "policy": policy.tolist() if policy is not None else None, "completed": True}    
    
    def start_self_play_loop(self):
        """Starts the main self-play loop, which continuously plays games to generate training data."""
        print(f"\n===========================================\n[{time.strftime('%H:%M:%S')}] Starting new self-play game.\n===========================================")
        self.stop_self_play_loop()
        self.self_play_active = True
        self.game_history = []
        Clock.schedule_once(self._self_play_turn, 0.3)

    def request_score(self, game_node, callback):
        """Requests a score calculation for the current game node, run in a separate thread."""
        threading.Thread(target=lambda: callback(self.get_score(game_node)), daemon=True).start()

    def stop_self_play_loop(self):
        """Stops the active self-play loop."""
        if not self.self_play_active: return
        self.self_play_active = False
        Clock.unschedule(self._self_play_turn)

    def _self_play_turn(self, dt=None):
        """
        Executes a single turn of a self-play game. It gets the best move from the
        AI, plays it on the board, and stores the state for later training.
        """
        if not self.self_play_active: return
        game = self.katrain.game
        try:
            current_node = game.current_node
            board_snapshot = self._node_to_board(current_node)
            if game.end_result or board_snapshot.is_game_over():
                self._process_game_result(game)
                return
            move_loc, root_node = self.hongik_player.get_best_move(board_snapshot, is_self_play=True)
            coords = None if move_loc == Board.PASS_LOC else board_snapshot.loc_to_coord(move_loc)
            move_obj = Move(player='B' if board_snapshot.pla == Board.BLACK else 'W', coords=coords)
            game.play(move_obj)
            if not move_obj.is_pass and self.sound_index:
                self.katrain.play_sound()
                black_player_type = self.katrain.players_info['B'].player_type
                white_player_type = self.katrain.players_info['W'].player_type

                if black_player_type == PLAYER_AI and white_player_type == PLAYER_AI:
                    if self.katrain.game.current_node.next_player == 'B':
                        self.katrain.controls.players['B'].active = True
                        self.katrain.controls.players['W'].active = False
                    else:
                        self.katrain.controls.players['B'].active = False
                        self.katrain.controls.players['W'].active = True

            policy = np.zeros(self.BOARD_SIZE**2 + 1, dtype=np.float32)
            if root_node and root_node.children:
                total_visits = sum(c.n_visits for c in root_node.children.values())
                if total_visits > 0:
                    for action, child in root_node.children.items(): policy[action] = child.n_visits / total_visits

            blacks_win_rate = 0.5
            if root_node:
                player_q_value = root_node.q_value 
                player_win_rate = (player_q_value + 1) / 2 
                blacks_win_rate = player_win_rate if board_snapshot.pla == Board.BLACK else (1 - player_win_rate)
            
            self.game_history.append([board_snapshot.get_features(), policy, board_snapshot.pla, blacks_win_rate])           
            self.katrain.update_gui(game.current_node)
            self.sound_index = True
            Clock.schedule_once(self._self_play_turn, 0.3)
        except Exception as e:
            print(f"Critical error during self-play: {e}"); traceback.print_exc(); self.stop_self_play_loop()

    def _process_game_result(self, game: 'Game'):
        """
        Processes the result of a finished game. It requests a final score and
        then triggers the callback to handle training data generation.
        """
        try:
            self.katrain.controls.set_status("Scoring...", STATUS_INFO) 
            self.katrain.board_gui.game_over_message = "Scoring..."
            
            self.katrain.board_gui.game_is_over = True
            self.request_score(game.current_node, self._on_score_calculated)
        except Exception as e: 
            print(f"Error requesting score calculation: {e}")
            self.katrain._do_start_hongik_selfplay()

    def _on_score_calculated(self, score_details):
        """
        Callback function that handles the game result after scoring. It assigns rewards,
        augments the data, adds it to the replay buffer, and triggers a training step.
        """
        try:
            if not score_details:
                print("Game ended but no result. Starting next game.")
                return

            game_num = self.training_step_counter + 1
            winner_text = "Black" if score_details['winner'] == 'B' else "White"
            b_score, w_score, diff = score_details['black_score'], score_details['white_score'], score_details['score']
            final_message = f"{winner_text} wins by {abs(diff):.1f} points"
            self.katrain.board_gui.game_over_message = final_message
            print(f"\n==========================================\n[{time.strftime('%H:%M:%S')}]    Game #{game_num} Finished\n-----------------------------------------\n    Winner: {winner_text}\n    Margin: {abs(diff):.1f} points\n    Details: Black {b_score:.1f} vs White {w_score:.1f}\n--------------------------------------------")

            winner = Board.BLACK if score_details['winner'] == 'B' else Board.WHITE

            REVERSAL_THRESHOLD = 0.2  
            WIN_REWARD = 1.0
            LOSS_REWARD = -1.0
            BRILLIANT_MOVE_BONUS = 0.5 
            CONSOLATION_REWARD = 0.5   

            for i, (features, policy, player_turn, blacks_win_rate_after) in enumerate(self.game_history):
                blacks_win_rate_before = self.game_history[i-1][3] if i > 0 else 0.5                
                if player_turn == Board.BLACK:
                    win_rate_swing = blacks_win_rate_after - blacks_win_rate_before
                else: # player_turn == Board.WHITE
                    white_win_rate_before = 1 - blacks_win_rate_before
                    white_win_rate_after = 1 - blacks_win_rate_after
                    win_rate_swing = white_win_rate_after - white_win_rate_before

                is_brilliant_move = win_rate_swing > REVERSAL_THRESHOLD
                
                if player_turn == winner:
                    reward = WIN_REWARD
                    if is_brilliant_move:
                        reward += BRILLIANT_MOVE_BONUS 
                else:
                    reward = LOSS_REWARD
                    if is_brilliant_move:
                        reward = CONSOLATION_REWARD 
                
                for j in range(8):
                    aug_features = self._augment_data(features, j, 'features')
                    aug_policy = self._augment_data(policy, j, 'policy')
                    self.replay_buffer.append([aug_features, aug_policy, reward])

            self.training_step_counter += 1
            loss = self._train_model() if len(self.replay_buffer) >= self.TRAINING_BATCH_SIZE else None
            if loss is not None:
                print(f"   Training complete! (Final loss: {loss:.4f})\n==========================================")

            log_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'game_num': game_num,
                'winner': score_details['winner'],
                'score_diff': diff,
                'total_moves': len(self.game_history),
                'loss': f"{loss:.4f}" if loss else "N/A"
            }
            self._log_training_progress(log_data)

            if self.training_step_counter % self.SAVE_WEIGHTS_EVERY_STEPS == 0:
                self.hongik_model.save_weights(self.WEIGHTS_FILE_PATH)

            if self.training_step_counter % self.CHECKPOINT_EVERY_GAMES == 0:
                self._checkpoint_save()

            if self.training_step_counter % self.EVALUATION_EVERY_STEPS == 0:
                self._evaluate_model()

        except Exception as e:
            print(f"Error during post-game processing: {e}")
            traceback.print_exc()
        finally:
            self.katrain._do_start_hongik_selfplay()

    def _train_model(self):
        """
        Performs one training step. It samples a minibatch from the replay buffer
        and uses it to update the neural network's weights.
        """
        if len(self.replay_buffer) < self.TRAINING_BATCH_SIZE: return None
        total_loss, TRAIN_ITERATIONS = 0, 5 
        for _ in range(TRAIN_ITERATIONS):
            minibatch = random.sample(self.replay_buffer, self.TRAINING_BATCH_SIZE)
            features, policies, values = (np.array(e) for e in zip(*minibatch))
            with tf.GradientTape() as tape:
                pred_p, pred_v = self.hongik_model(features, training=True)
                value_loss = self.value_loss_fn(values[:, None], pred_v)
                policy_loss = self.policy_loss_fn(policies, pred_p)
                loss = policy_loss + value_loss
            self.optimizer.apply_gradients(zip(tape.gradient(loss, self.hongik_model.trainable_variables), self.hongik_model.trainable_variables))
            total_loss += loss.numpy()
        return total_loss / TRAIN_ITERATIONS

    def _augment_data(self, data, index, data_type):
        """
        Augments the training data by applying 8 symmetries (rotations and flips)
        to the board features and policy target.
        """
        if data_type == 'features':
            augmented = data
            if index & 1: augmented = np.fliplr(augmented)
            if index & 2: augmented = np.flipud(augmented)
            if index & 4: augmented = np.rot90(augmented, 1)
            return augmented
        elif data_type == 'policy':
            policy_board = data[:-1].reshape(self.BOARD_SIZE, self.BOARD_SIZE)
            augmented_board = policy_board
            if index & 1: augmented_board = np.fliplr(augmented_board)
            if index & 2: augmented_board = np.flipud(augmented_board)
            if index & 4: augmented_board = np.rot90(augmented_board, 1)
            return np.append(augmented_board.flatten(), data[-1])
        return data

    def get_score(self, game_node):
        """Calculates the final score of a game using the board's internal scoring method."""
        try:
            board = self._node_to_board(game_node)
            winner, black_score, white_score, _ = board.get_winner(self.katrain.game.komi)
            score_diff = black_score - white_score
            return {"winner": "B" if winner == Board.BLACK else "W", "score": score_diff, "black_score": black_score, "white_score": white_score}
        except Exception as e:
            print(f"Error during internal score calculation: {e}"); traceback.print_exc(); return None

    def _game_turn(self):
        """
        Handles the AI's turn in a game against a human or another AI. It runs
        in a separate thread to avoid blocking the GUI.
        """
        if self.self_play_active or self.katrain.game.end_result: return
        next_player_info = self.katrain.players_info[self.katrain.game.current_node.next_player]
        if next_player_info.player_type == PLAYER_AI:
            def ai_move_thread():
                try:
                    board_snapshot = self._node_to_board(self.katrain.game.current_node)
                    move_loc, _ = self.hongik_player.get_best_move(board_snapshot, is_self_play=False)
                    coords = None if move_loc == Board.PASS_LOC else board_snapshot.loc_to_coord(move_loc)
                    Clock.schedule_once(lambda dt: self.katrain._do_play(coords))
                except Exception as e:
                    print(f"\n--- Critical error during AI thinking (in thread) ---\n{traceback.format_exc()}\n---------------------------------------\n")
            threading.Thread(target=ai_move_thread, daemon=True).start()
    
    def _evaluate_model(self):
        """
        Periodically evaluates the currently training model against the best-known
        'champion' model to measure progress and update the best weights if the
        challenger is stronger.
        """
        print("\n--- [Championship Match Start] ---")
        challenger_player = self.hongik_player
        best_weights_path = self.BEST_WEIGHTS_FILE_PATH
        if not os.path.exists(best_weights_path):
            print("[Championship Match] Crowning the first champion!")
            self.hongik_model.save_weights(best_weights_path)
            return
        champion_model = CNNTransformerHybrid(self.NUM_LAYERS, self.D_MODEL, self.NUM_HEADS, self.D_FF, self.BOARD_SIZE)
        _ = champion_model(np.zeros((1, self.BOARD_SIZE, self.BOARD_SIZE, 3), dtype=np.float32))
        champion_model.load_weights(best_weights_path)
        champion_player = HongikAIPlayer(champion_model, int(self.config.get("max_visits", 150)))
        EVAL_GAMES, challenger_wins = 5, 0
        for i in range(EVAL_GAMES):
            print(f"\n[Championship Match] Game {i+1} starting...")
            board = Board(self.BOARD_SIZE)
            players = {Board.BLACK: challenger_player, Board.WHITE: champion_player} if i % 2 == 0 else {Board.BLACK: champion_player, Board.WHITE: challenger_player}
            while not board.is_game_over():
                current_player_obj = players[board.pla]
                move_loc, _ = current_player_obj.get_best_move(board)
                board.play(board.pla, move_loc)
            winner, _, _, _ = board.get_winner()
            if (winner == Board.BLACK and i % 2 == 0) or (winner == Board.WHITE and i % 2 != 0):
                challenger_wins += 1; print(f"[Championship Match] Game {i+1}: Challenger wins!")
            else:
                print(f"[Championship Match] Game {i+1}: Champion wins!")
        print(f"\n--- [Championship Match End] ---\nFinal Score: Challenger {challenger_wins} wins / Champion {EVAL_GAMES - challenger_wins} wins")
        if challenger_wins > EVAL_GAMES / 2:
            print("A new champion is born! Updating 'best' weights.")
            self.hongik_model.save_weights(best_weights_path)
        else:
            print("The champion defends the title. Keeping existing weights.")

    def on_new_game(self):
        """Called when a new game starts."""
        pass
    def start(self):
        """Starts the engine."""
        self.katrain.game_controls.set_player_selection()
    def shutdown(self, finish=False):
        """Shuts down the engine, saving progress and cleaning up checkpoint files."""
        self.stop_self_play_loop()
        self.save_replay_buffer(self.REPLAY_BUFFER_PATH)
        try:
            if os.path.exists(self.CHECKPOINT_BUFFER_PATH): os.remove(self.CHECKPOINT_BUFFER_PATH)
            if os.path.exists(self.CHECKPOINT_WEIGHTS_PATH): os.remove(self.CHECKPOINT_WEIGHTS_PATH)
        except OSError as e:
            print(f"Error deleting checkpoint files: {e}")
    def stop_pondering(self):
        """Stops pondering."""
        pass
    def queries_remaining(self):
        """Returns the number of remaining queries."""
        return 0
    def is_idle(self):
        """Checks if the engine is idle (i.e., not in a self-play loop)."""
        return not self.self_play_active