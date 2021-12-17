from transformers import Trainer
from torch.cuda.amp import autocast
import torch
import os


from .utils import clear_gpu, optimal_workers
from .model import clip
from .data import train_ds, test_ds
from ..config import args, tokenizer, vision_preprocessor


class CLIPTrainer(Trainer):
    # computes loss w/o label smoothing
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_loss=True)
        return outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        return (loss, None, None)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    clear_gpu()
    args.dataloader_num_workers = optimal_workers()
    trainer = CLIPTrainer(clip, args,
                          train_dataset=train_ds,
                          eval_dataset=test_ds,
                          )

    trainer.train()

    # save pretrained models
    clip.text_model.save_pretrained('clip-fa-text')
    tokenizer.save_pretrained('clip-fa-text')
    clip.vision_model.save_pretrained('clip-fa-vision')
    vision_preprocessor.save_pretrained('clip-fa-vision')