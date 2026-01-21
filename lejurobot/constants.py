from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
try:
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    XVLA_AVAILABLE = True
except ImportError:
    XVLA_AVAILABLE = False
    XVLAConfig = None

LEROBOT_POLICIES_NAMES = [
    "tdmpc",
    "diffusion",
    "act",
    "vqbet",
    "pi0",
    "pi05",  # lerobot's PI05
    "sac",
    "reward_classifier",
    "smolvla",
]
if XVLA_AVAILABLE:
    LEROBOT_POLICIES_NAMES.append("xvla")

LEROBOT_POLICIES_CONFIGS_CLASSES = [
    ACTConfig,
    DiffusionConfig,
    PI0Config,
    PI05Config,  # Original lerobot PI05Config
    SACConfig,
    RewardClassifierConfig,
    SmolVLAConfig,
    TDMPCConfig,
    VQBeTConfig,
]
if XVLA_AVAILABLE:
    LEROBOT_POLICIES_CONFIGS_CLASSES.append(XVLAConfig)

LEROBOT_TELEOPERATORES_NAMES = [
    "keyboard",
    "koch_leader",
    "so100_leader",
    "so101_leader",
    "mock_teleop",
    "gamepad",
    "keyboard_ee",
    "homunculus_glove",
    "homunculus_arm",
    "bi_so100_leader",
    "reachy2_teleoperator"
]

LEROBOT_CAMERAS_NAMES = [
    "opencv",
    "intelrealsense",
    "reachy2_camera",
]
