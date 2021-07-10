from envs.base_env import BaseEnv
from envs.cover import Cover
from envs.painting import Painting
from envs.blocks import Blocks


def create_env(config):
    name_to_cls = {
        "cover": Cover,
        "painting": Painting,
        "blocks": Blocks,
    }
    return name_to_cls[config.env](config)
