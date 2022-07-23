from transformers import T5Tokenizer, T5ForConditionalGeneration


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
    return self.model(input_ids=)


  def generate(self,batch)
    return self.model.generate(
      input_ids=batch,
      attention_mask=attention_mask,
      cross_attention_mask=cross_attention_mask,
      decoder_input_ids=decoder_input_ids,
      decoder_attention_mask=decoder_attention_mask,
      labels=labels,
    )