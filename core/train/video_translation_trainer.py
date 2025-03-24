from .train import Trainer


class VideoTranslationTrainer(Trainer):
    def train_one_step(self, x: tuple, y: object) -> None:
        pass

    def validate_one_step(self, x: tuple, y: object) -> None:
        pass
