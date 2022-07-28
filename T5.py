from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn


class T5(nn.Module):
  def __init__(self, cfg):
    """
    Setting for T5 model

    Parameters
    ----------
    cfg: DictConfig
    """
    super(T5, self).__init__()
    self.cfg = cfg

    if cfg.model.pretrained_model=="t5-small":
      self.model = T5ForConditionalGeneration.from_pretrained(cfg.model.pretrained_model)
    elif cfg.model.pretrained_model=="t5-base":
      self.model = T5ForConditionalGeneration.from_pretrained(cfg.model.pretrained_model)
    else:
      raise ValueError("No such model!")

  def forward(self, batch, is_training:bool):
    outputs = self.model(
      input_ids=batch[0],
      attention_mask=batch[1],
      decoder_attention_mask=batch[3],
      labels=batch[2])
    return outputs[0]


  def generate(self,batch):
    return self.model.generate(
      input_ids=batch[0], 
      attention_mask=batch[1], 
      max_length=self.cfg.model.max_output_length,
      temperature=1.0,
      repetition_penalty=1.5
    )