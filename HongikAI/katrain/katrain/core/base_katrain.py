# base_katrain.py (진짜 최종 완성본)

import json
import os
from configparser import ConfigParser, NoSectionError, NoOptionError

# --- 필요한 모든 부품을 정확하게 import ---
from katrain.core.constants import *
from katrain.core.lang import i18n
from katrain.core.utils import PATHS
from katrain.core.ai import ai_rank_estimation

class KaTrainBase:
    """KaTrain의 GUI와 엔진 로직 사이에 공유되는 기본 클래스"""

    def __init__(self, katrain_app, config: ConfigParser):
        self.katrain = katrain_app
        self._config = config
        self.players_info = {}
        self.debug_level = self.config("general/debug_level", 0, int)

        for bw in "BW":
            self.players_info[bw] = self.PlayerInfo(bw, self)

    def config(self, key, default=None, vartype=str):
        try:
            if "/" in key:
                parts = key.split("/")
                section = parts[0]
                name = "/".join(parts[1:])
                if vartype == bool:
                    return self._config.getboolean(section, name)
                if vartype == int:
                    return self._config.getint(section, name)
                return self._config.get(section, name)
            else: # section만 요청된 경우
                return dict(self._config.items(key))
        except (ValueError, KeyError, NoSectionError, NoOptionError):
            return default if "/" in key else (default or {})

    def save_config(self, sections=None):
        config_path = os.path.join(PATHS["USER"], "config.json")
        save_sections = [sections] if isinstance(sections, str) else (sections or self._config.sections())

        output_dict = {}
        for s in save_sections:
            if self._config.has_section(s):
                output_dict[s] = dict(self._config.items(s))

        with open(config_path, "w") as f:
            json.dump(output_dict, f, indent=4)
        
    def log(self, message, level=OUTPUT_INFO):
        if self.debug_level is not None and level <= self.debug_level:
            print(message)

    def update_player(self, bw, **kwargs):
        self.players_info[bw].update(**kwargs)
        self.save_config("players")
        self.update_calculated_ranks()
        
    def update_calculated_ranks(self):
        for bw, player_info in self.players_info.items():
            if player_info.player_type == PLAYER_AI:
                settings = {"komi": self.config("game/komi"), "rules": self.config("game/rules")}
                player_info.calculated_rank = ai_rank_estimation(player_info.player_subtype, settings)

    @property
    def next_player_info(self):
        if hasattr(self, 'game') and self.game and self.game.current_node:
            return self.players_info[self.game.current_node.next_player]
        return self.players_info["B"]

    @property
    def last_player_info(self):
        if hasattr(self, 'game') and self.game and self.game.current_node and self.game.current_node.player:
            return self.players_info[self.game.current_node.player]
        return self.players_info["W"]

    class PlayerInfo:
        def __init__(self, bw, katrain_base):
            self.bw = bw
            self.katrain_base = katrain_base
            self.name = katrain_base.config(f"players/{bw}/name", None)
            self.player_type = katrain_base.config(f"players/{bw}/type", PLAYER_HUMAN)
            self.player_subtype = katrain_base.config(f"players/{bw}/subtype", AI_DEFAULT)
            self.sgf_rank = None
            self.calculated_rank = None
            
        def update(self, **kwargs):
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                    if not self.katrain_base._config.has_section("players"):
                        self.katrain_base._config.add_section("players")
                    self.katrain_base._config.set("players", f"{self.bw}/{k}", str(v))