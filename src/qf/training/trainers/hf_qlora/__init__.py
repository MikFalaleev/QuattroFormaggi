"""`hf_trainer_qlora` adapter: HF Trainer + peft QLoRA (plan step 12). Imports torch,
transformers and peft; loaded only through the lazy entry in `qf.cli.wiring`."""

from qf.training.trainers.hf_qlora.train import HFQLoRATrainer, TrainingStopped, run_training

__all__ = ["HFQLoRATrainer", "TrainingStopped", "run_training"]
