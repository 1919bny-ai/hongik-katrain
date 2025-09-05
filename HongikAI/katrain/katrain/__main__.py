# Copyright (c) 2020 Sanderland (KaTrain Original Author)
# Copyright (c) 2025 BNY (Modifications for Hongik AI)

# Main entry point for the HongikAI-KaTrain application.
# It initializes the Kivy/KaTrain GUI and integrates the custom HongikAIEngine.
#
# Author: Gemini 2.5 Pro, Gemini 2.5 Flash

import os
import sys
import signal
import threading
import time
import traceback
import random
from queue import Queue
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_folder = os.path.dirname(project_root)

if grandparent_folder not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, grandparent_folder)

os.environ["KCFG_KIVY_LOG_LEVEL"] = os.environ.get("KCFG_KIVY_LOG_LEVEL", "warning")
import kivy
kivy.require("2.0.0")
from kivy.app import App
from kivy.base import ExceptionHandler, ExceptionManager
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, BooleanProperty
from kivy.clock import Clock
from kivy.config import Config
from kivymd.app import MDApp

from katrain.core.utils import find_package_resource, PATHS
from katrain.core.base_katrain import KaTrainBase
from katrain.core.lang import DEFAULT_LANGUAGE, i18n
from katrain.core.constants import *
from katrain.core.game import Game, KaTrainSGF, IllegalMoveException
from katrain.core.sgf_parser import Move, ParseError
from katrain.gui.theme import Theme

import pygame

from hongik.board_ai import Board
from hongik.engine_ai import HongikAIEngine

from katrain.gui.kivyutils import *
from katrain.gui.widgets import MoveTree
from katrain.gui.badukpan import BadukPanWidget
from katrain.gui.controlspanel import ControlsPanel

if 'USER' not in PATHS:
    USER_DATA_PATH = os.path.expanduser(os.path.join("~", ".katrain"))
    os.makedirs(USER_DATA_PATH, exist_ok=True)
    PATHS['USER'] = USER_DATA_PATH

ICON = find_package_resource("katrain/img/icon.ico")
Config.set("kivy", "window_icon", ICON)
Config.set("input", "mouse", "mouse,multitouch_on_demand")
SOUNDS_DIR = find_package_resource("katrain/sounds")

class KaTrainGui(Screen, KaTrainBase):
    """
    The main GUI class for the application. It inherits from Kivy's Screen and
    KaTrainBase, managing all visual components and user interactions.
    """
    zen = NumericProperty(0)
    controls = ObjectProperty(None); engine = ObjectProperty(None); game = ObjectProperty(None)
    board_gui = ObjectProperty(None); board_controls = ObjectProperty(None); play_mode = ObjectProperty(None)
    show_move_numbers = BooleanProperty(False)
    analysis_controls = ObjectProperty(None) 

    @property
    def play_analyze_mode(self):
        return self.play_mode.mode
    
    def __init__(self, **kwargs):
        """Initializes the GUI, linking it to the main app and setting up necessary variables."""
        self.katrain_app = kwargs.get('katrain_app')
        self.engine, self.message_queue, self.pondering = None, Queue(), False
        self.contributing, self.animate_contributing = False, False
        super().__init__(**kwargs)

    def config_set(self, section, option, value):
        """Sets a configuration value and writes it to the config file."""
        self.katrain_app.config.set(section, option, value)
        self.katrain_app.config.write()

    def save_config(self, sections=None):
        """Writes the current configuration to disk."""
        self.katrain_app.config.write()
        
    def play_sound(self):
        """Randomly plays a stone placement sound from the sounds directory."""
        try:
            sound_files = [f for f in os.listdir(SOUNDS_DIR) if f.startswith('stone') and f.endswith(('.wav', '.ogg'))]
            if sound_files:
                sound_to_play = random.choice(sound_files)
                pygame.mixer.Sound(os.path.join(SOUNDS_DIR, sound_to_play)).play()
        except pygame.error as e:
            print(f"Pygame sound playback error: {e}")

    def start(self):
        """
        Starts the main application logic, initializes the AI engine, starts the
        message loop, and creates a new game.
        """
        if self.engine: return
        self.board_gui.trainer_config = self.config("trainer")
        self.engine = HongikAIEngine(self, self.config("engine"))
        threading.Thread(target=self._message_loop_thread, daemon=True).start()
        self._do_new_game()
        Clock.schedule_interval(self.handle_animations, 0.1)
        Window.request_keyboard(None, self, "").bind(on_key_down=self._on_keyboard_down)
        
    def update_player(self, bw, **kwargs):
        """Updates the information and type for a given player (Black or White)."""
        player_type = kwargs.get('player_type')
        if player_type == PLAYER_AI:
            self.players_info[bw].player_type, self.players_info[bw].player_subtype = PLAYER_AI, "홍익 AI"
            self.players_info[bw].sgf_rank = ""
            self.players_info[bw].calculated_rank = ""
        elif player_type == PLAYER_HUMAN:
            self.players_info[bw].player_type, self.players_info[bw].player_subtype = PLAYER_HUMAN, "Human"

        self.players_info[bw].periods_used = 0
        self.players_info[bw].being_taught = False
        self.players_info[bw].player = bw 
        if self.game: self.players_info[bw].name = self.game.root.get_property("P" + bw)
        if self.controls: self.controls.update_players(); self.update_state()

    def update_gui(self, cn, redraw_board=False):
        """Updates all GUI elements with the latest game state information."""
        if not self.game: return
        prisoners = self.game.prisoner_count
        self.controls.players["B"].captures, self.controls.players["W"].captures = prisoners.get("W", 0), prisoners.get("B", 0)
        if not self.engine or not self.engine.is_idle(): self.board_controls.engine_status_col = Theme.ENGINE_BUSY_COLOR
        else: self.board_controls.engine_status_col = Theme.ENGINE_READY_COLOR
        if redraw_board: self.board_gui.draw_board()
        self.board_gui.redraw_board_contents_trigger()
        self.controls.update_evaluation(); self.controls.move_tree.current_node = self.game.current_node

    def update_state(self, redraw_board=False):
        """A shortcut to send an 'update-state' message to the message queue."""
        self("update-state", redraw_board=redraw_board)

    def _message_loop_thread(self):
        """
        The main message loop that runs in a separate thread, processing commands
        from the message queue to avoid blocking the GUI.
        """
        while True:
            game_id, msg, args, kwargs = self.message_queue.get()
            try:
                if self.game and game_id != self.game.game_id: continue
                fn = getattr(self, f"_do_{msg.replace('-', '_')}")
                fn(*args, **kwargs)
                if msg != "update_state": self._do_update_state()
            except Exception as exc:
                self.log(f"Message loop exception: {exc}", OUTPUT_ERROR); traceback.print_exc()

    def __call__(self, message, *args, **kwargs):
        """Adds a message to the thread-safe message queue for processing."""
        if message.endswith("popup"): Clock.schedule_once(lambda _dt: getattr(self, f"_do_{message.replace('-', '_')}")(*args, **kwargs), -1)
        else: self.message_queue.put([self.game.game_id if self.game else None, message, args, kwargs])

    def _do_update_state(self, redraw_board=False):
        """
        Handles the 'update-state' message, refreshing the GUI to reflect the
        current game state, player turn, and engine status.
        """
        if not self.game or not self.game.current_node: return

        if self.controls:
            self.controls.update_players()
            next_player_is = self.game.current_node.next_player
            self.controls.active_player = self.game.current_node.next_player
         
            self.controls.players['B'].active = (next_player_is == 'B')
            self.controls.players['W'].active = (next_player_is == 'W')
        
        is_game_active = self.game and not self.game.end_result      
        is_game_over = not is_game_active

        if self.board_gui.game_is_over != is_game_over:
            self.board_gui.game_is_over = is_game_over
            if is_game_over:
                self.board_gui.game_over_message = "Game Over"
        
        is_ai_vs_ai = (self.players_info['B'].player_type == PLAYER_AI and self.players_info['W'].player_type == PLAYER_AI)
        
        if self.controls and self.controls.ids.get('undo_button'):
            self.controls.ids.undo_button.disabled = not is_game_active or is_ai_vs_ai
            self.controls.ids.resign_button.disabled = not is_game_active or is_ai_vs_ai
        
        if self.engine and self.pondering:
            self.game.analyze_extra("ponder")
        else:
            self.engine.stop_pondering()
            
        Clock.schedule_once(lambda _dt: self.update_gui(self.game.current_node, redraw_board), -1)
        self.engine._game_turn()

    def _do_play(self, coords):
        """Handles a 'play' event, creating a Move object and playing it on the board."""
        try:
            move = Move(coords, player=self.game.current_node.next_player)
            self.game.play(move)
            self.update_state()
            if not move.is_pass: self.play_sound()
        except IllegalMoveException as e:
            self.controls.set_status(f"Illegal move: {str(e)}", STATUS_ERROR)

    def _do_new_game(self, player_types=(PLAYER_HUMAN, PLAYER_HUMAN), move_tree=None, sgf_filename=None):
        """Handles a 'new-game' event, setting up a new game with specified players."""
        self.pondering = False
        self.engine.sound_index = False
        if self.engine: self.engine.stop_self_play_loop(); self.engine.on_new_game()
        self.game = Game(self, self.engine, move_tree=move_tree, sgf_filename=sgf_filename)
       
        self.board_controls.ids.game_mode_reset_btn.state = 'down'
        self.update_player('B', player_type=player_types[0])
        self.update_player('W', player_type=player_types[1])
        if self.controls and self.controls.graph:
            self.controls.graph.initialize_from_game(self.game.root)
        self.update_state(redraw_board=True)

        try:
            self.analysis_controls.hamburger.disabled = False 
            self.analysis_controls.show_children.disabled =False
            self.analysis_controls.hints.disabled =False
            self.analysis_controls.policy.disabled =False            
            self.controls.ids.undo.disabled = False
            self.board_controls.ids.pass_btn.disabled = False
            self.controls.ids.timer.ids.pause.disabled = False
        except Exception as e:
            self.log(f"Error enabling button: {e}", OUTPUT_ERROR)

    
    def _do_start_hongik_selfplay(self):
        """Starts a new self-play game between two Hongik AI instances."""
        self._do_new_game(player_types=(PLAYER_AI, PLAYER_AI))
        self.engine.start_self_play_loop()

        try:
            self.analysis_controls.hamburger.disabled = True
            self.analysis_controls.show_children.checkbox.active = False
            self.analysis_controls.show_children.disabled =True
            self.analysis_controls.hints.checkbox.active = False
            self.analysis_controls.hints.disabled =True
            self.analysis_controls.policy.checkbox.active = False
            self.analysis_controls.policy.disabled =True
            self.controls.ids.undo.disabled = True
            self.board_controls.ids.pass_btn.disabled = True
            self.controls.ids.timer.ids.pause.disabled = True
        except Exception as e:
            self.log(f"Error enabling hamburger button: {e}", OUTPUT_ERROR)

    def _do_start_hongik_vshuman(self):
        """Starts a new game between a human player and Hongik AI."""
        self._do_new_game(player_types=(PLAYER_HUMAN, PLAYER_AI))
        try:
            hamburger_button = self.analysis_controls.ids.get('hamburger')
            if hamburger_button:
                hamburger_button.disabled = True
        except Exception as e:
            self.log(f"Error enabling hamburger button: {e}", OUTPUT_ERROR)
        
    def _do_undo(self, n_times=1):
        """Handles an 'undo' event, going back a specified number of moves."""
        try: n_times = int(n_times)
        except (ValueError, TypeError): n_times = 1
        self.game.undo(n_times); self.update_state()
        
    def _do_resign(self):
        """Handles a 'resign' event, ending the game and resetting the GUI."""
        if self.game:
            winner = 'W' if self.game.current_node.next_player == 'B' else 'B'
            self.game.root.set_property("RE", f"{winner}+Resign")
            try:
                self_play_button = self.board_controls.ids.hongik_selfplay_btn
                vs_human_button = self.board_controls.ids.hongik_vs_human_btn
                self_play_button.state = 'normal'
                vs_human_button.state = 'normal'
            except Exception as e:
                self.log(f"Failed to change button state: {e}", OUTPUT_ERROR)
            self.game = Game(self, self.engine)
            self._do_new_game()
    
    def load_sgf_file(self, file_path):
        """Initiates loading of an SGF file in a separate thread."""
        self.controls.set_status(f"Loading SGF file: {os.path.basename(file_path)}", STATUS_INFO)
        threading.Thread(target=self._load_sgf_thread_target, args=(file_path,), daemon=True).start()

    def _load_sgf_thread_target(self, file_path):
        """The target function for the SGF loading thread."""
        try:
            move_tree = KaTrainSGF.parse_file(os.path.abspath(file_path))
            Clock.schedule_once(lambda dt: self._do_new_game(move_tree=move_tree, sgf_filename=file_path))
        except Exception as e:
            self.log(f"SGF file loading failed: {e}", OUTPUT_ERROR)
            Clock.schedule_once(lambda dt: self.controls.set_status(f"SGF loading failed", STATUS_ERROR))

    def handle_animations(self, *_args): pass
    
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        """Handles keyboard shortcuts, such as toggling move numbers."""
        if not self.game: return True
        key = keycode[1]
        if key == 'n':
            self.show_move_numbers = not self.show_move_numbers
            self.board_gui.redraw_board_contents_trigger()
            return True
        return False
    
    def _do_score(self, *_args):
        """Handles a 'score' event, requesting the engine to score the current position."""
        self.board_gui.game_over_message = "Scoring..."
        self.controls.set_status("Scoring...", STATUS_INFO)
        def score_callback(score_details):
            if score_details:
                winner = score_details['winner']
                score = score_details['score']
                self.game.set_result(f"{winner}+{abs(score)}")
                self.update_state()
            else:
                self.controls.set_status("Failed to score the game.", STATUS_ERROR)
        self.engine.request_score(self.game.current_node, score_callback)
    
    def _bind_widgets(self, dt):
        """위젯이 모두 생성된 후, 이벤트를 파이썬에서 직접 바인딩합니다."""
        if self.analysis_controls and self.analysis_controls.show_children:
            self.analysis_controls.show_children.checkbox.bind(active=self._handle_show_children_toggle)

        if self.nav_drawer_contents and 'player_type_spinner_W' in self.nav_drawer_contents.ids:
            self.nav_drawer_contents.ids.player_type_spinner_W.disabled = True
            print("W player type spinner successfully disabled via Python.") # 확인용 로그

    def on_nav_drawer_close(self):
        """Handles the closing of the navigation drawer, forcing a redraw."""
        self.update_state() 
        if self.board_gui:
            self.board_gui.draw_board()
            self.board_gui.redraw_board_contents_trigger() 
        self.canvas.ask_update() 

    def _do_contribute_popup(self,*_args):pass
    def _do_config_popup(self, *_args):pass    
    def _do_new_game_popup(self,*_args):pass    
    def _do_save_game(self,*_args):pass
    def _do_save_game_as_popup(self,*_args):pass
    def _do_analyze_sgf_popup(self,*_args):pass
    def _do_teacher_popup(self,*_args):pass
    def _do_ai_popup(self,*_args):pass
    def _do_timer_popup(self,*_args):pass

class KaTrainApp(MDApp):
    """
    The main application class that inherits from KivyMD's MDApp. It builds the
    GUI, manages the configuration, and handles application lifecycle events.
    """
    gui = ObjectProperty(None)
    language = StringProperty(DEFAULT_LANGUAGE, allownone=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._resize_event = None

    def build_config(self, config):
        """Sets up the default configuration for the application."""
        if 'SGF' not in PATHS:
            PATHS['SGF'] = os.path.join(PATHS.get('USER', '.'), 'sgf')
            os.makedirs(PATHS['SGF'], exist_ok=True)
        config.setdefaults("general",{"lang": DEFAULT_LANGUAGE, "show_player_rank": True, "last_sgf_directory": PATHS["SGF"],})
        config.setdefaults("engine", {"max_visits": "100"})
        
        threshold_str = "-1,0.5,1.5,3,5,7.5,10"
        thresholds_as_floats = [float(v) for v in threshold_str.split(',')]
        
        config.setdefaults("trainer", {
            "eval_thresholds": thresholds_as_floats,
            "theme": "theme:normal" 
        })
        
        config.setdefaults("uistate", {"size": "[1300, 1000]"})

    def build(self):
        """Builds the application's widget tree and sets up window bindings."""
        pygame.mixer.init()
        self.icon, self.title = ICON, "홍익 AI - KaTrain"
        self.theme_cls.theme_style, self.theme_cls.primary_palette = "Dark", "Gray"
        for p in [os.path.join(PATHS["PACKAGE"], d) for d in ["fonts","sounds","img", "lang"]] + [os.path.abspath(PATHS["USER"])]:
            resource_add_path(p)
        Builder.load_file(find_package_resource("katrain/gui.kv"))
        Builder.load_file(find_package_resource("katrain/popups.kv"))
        Window.bind(on_request_close=self.on_request_close)
        Window.bind(on_dropfile=lambda win, file: self.gui.load_sgf_file(file.decode("utf8")))
        Window.bind(on_resize=self.on_resize)
        self.gui = KaTrainGui(katrain_app=self, config=self.config)
        Window.size = Window.system_size
        return self.gui
    
    def on_resize(self, window, width, height):
        """Controls the storm of resize events by debouncing them with a short delay."""
        if self._resize_event:
            self._resize_event.cancel()
        self._resize_event = Clock.schedule_once(self._redraw_all, 0.15)

    def _redraw_all(self, dt):
        """The actual function that redraws the entire screen after a resize."""
        if self.gui:
            self.gui.update_state(redraw_board=True)
    
    def on_start(self):
        """Called when the application is starting."""
        self.language = self.gui.config("general/lang") or DEFAULT_LANGUAGE
        self.gui.start()
        Window.show()        

    def on_language(self, _instance, language):
        """Handles language changes."""
        i18n.switch_lang(language)
        self.gui.config_set("general", "lang", language)
        
    def on_request_close(self, *_args, **_kwargs):
        """Handles the window close event, saving the window size and shutting down the engine."""
        if getattr(self, "gui", None):
            size_str = json.dumps([int(d) for d in Window.size])
            self.gui.config_set("uistate", "size", size_str)
            self.gui.save_config("uistate")
            if self.gui.engine: self.gui.engine.shutdown()
            
    def signal_handler(self, _signal, _frame):
        """Handles signals like Ctrl+C."""
        self.stop()

def run_app():
    """Initializes and runs the application."""
    class CrashHandler(ExceptionHandler):
        def handle_exception(self, inst):
            trace = "".join(traceback.format_tb(sys.exc_info()[2]))
            app = MDApp.get_running_app()   
            message = f"Exception {inst.__class__.__name__}: {inst}\n{trace}"
            if app and app.gui: app.gui.log(message, OUTPUT_ERROR)
            else: print(message)
            return ExceptionManager.PASS
    ExceptionManager.add_handler(CrashHandler())
    
    Config.set('graphics', 'window_state', 'hidden')

    app = KaTrainApp(); signal.signal(signal.SIGINT, app.signal_handler); app.run()

if __name__ == "__main__":
    run_app()