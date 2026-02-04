from lejurobot.configs.train import TrainPipelineConfigLejuRobot
from lejurobot.train.trainer import Trainer
from accelerate import Accelerator
from lerobot.configs import parser

@parser.wrap()
def train(cfg: TrainPipelineConfigLejuRobot, accelerator: Accelerator | None = None):
    """
    Main function to train LejuRobot policy.
    
    Args:
        cfg: A `TrainPipelineConfigLejuRobot` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    trainer = Trainer(cfg, accelerator)
    trainer.setup(loss_key="loss")
    trainer.train()

if __name__ == "__main__":
    train()