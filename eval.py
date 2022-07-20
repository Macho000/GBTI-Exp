import hydra
import logging
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from omegaconf import DictConfig, OmegaConf
from utils import *
import torch
from JointGT import JointGT

from transformers import BartTokenizer, T5Tokenizer
from data import EntityTypingJointGTDataset, EntityTypingJointGTDataLoader

from tqdm import tqdm
from tqdm import trange

import nltk
import nltk.translate.bleu_score as bleu

@hydra.main(config_path="config", config_name='config')
def main(cfg : DictConfig) -> None:
  set_logger(cfg)
  use_cuda = cfg.model.cuda and torch.cuda.is_available()
  data_path = os.path.join(cfg.data.data_dir, cfg.data.name, 'clean')
  save_path = os.path.join(cfg.model.save_dir, cfg.data.name)

  # Initialize tokenizer
  if cfg.model.pretrained_model == "Bart":
      tokenizer = BartTokenizer.from_pretrained(cfg.model.tokenizer_path)
  elif cfg.model.pretrained_model == "T5":
      tokenizer = T5Tokenizer.from_pretrained(cfg.model.tokenizer_path)

  data_path = os.path.join(cfg.data.data_dir, cfg.data.name, 'clean')
  test_dataset = EntityTypingJointGTDataset(cfg, data_path, tokenizer, "test")

  # dataloader
  test_dataloader = EntityTypingJointGTDataLoader(cfg, test_dataset, "test")

  # model
  if cfg.model.name == 'JointGT':
      model = JointGT(cfg)
  elif cfg.model.name == 'T5':
      model = T5()
  else:
      raise ValueError('No such model')

  if use_cuda:
      model = model.to('cuda')
  for name, param in model.named_parameters():
      logging.debug('Parameter %s: %s, require_grad=%s' % (name, str(param.size()), str(param.requires_grad)))

  # optimizer
  optimizer = torch.optim.Adam(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=cfg.model.optimizer.learning_rate,
  )

  # test
  with torch.no_grad():
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pkl')))
    model.eval()
    sources = []
    predictions = []
    targets = []
    for batch in tqdm(test_dataloader, desc="Iteration"):
      if torch.cuda.is_available():
          batch = [b.to(torch.device("cuda")) for b in batch]
      if use_cuda:
          batch = [b.to(torch.device('cuda')) for b in batch]
      outputs = model.generate(batch)
      # Convert ids to tokens
      for input_, output, target in zip(batch[0], outputs, batch[2]):
          source = tokenizer.decode(input_, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
          sources.append(source.strip())
          pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
          predictions.append(pred.strip())
          target = tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=cfg.model.valid.clean_up_spaces)
          targets.append(target.strip())
    if cfg.model.test.save_outputs:
        with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_inputs_test.txt'), 'w', encoding='utf-8') as f:
            for src_txt in sources:
                f.write(src_txt+"\n")
        with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_outputs_test.txt'), 'w', encoding='utf-8') as f:
            for pred_txt in predictions:
                f.write(pred_txt+"\n")
        with open(os.path.join(save_path, f'{cfg.model.test.test_dataset.split(".")[0]}_targets_test.txt'), 'w', encoding='utf-8') as f:
            for target_txt in targets:
                f.write(target_txt+"\n")
    evaluation(cfg, sources, predictions)

def evaluation(cfg, sources, predictions):
  assert len(sources)==len(predictions)
  total_score_1gram = 0
  total_score_2gram = 0
  total_score_3gram = 0
  total_score_4gram = 0

  for src, pred in zip(sources, predictions):
    hyp = pred.split()
    ref = src.split()
    total_score_1gram += bleu.sentence_bleu([ref], hyp, weights=(1,0,0,0))
    total_score_2gram += bleu.sentence_bleu([ref], hyp, weights=(0,1,0,0))
    total_score_3gram += bleu.sentence_bleu([ref], hyp, weights=(0,0,1,0))
    total_score_4gram += bleu.sentence_bleu([ref], hyp, weights=(0,0,0,1))
  score_1gram = total_score_1gram/len(sources)
  score_2gram = total_score_2gram/len(sources)
  score_3gram = total_score_3gram/len(sources)
  score_4gram = total_score_4gram/len(sources)

  logging.info("test data set : {}".format(cfg.model.test.test_dataset))
  logging.info("N-grams: 1-{}, 2-{}, 3-{}, 4-{}".format(score_1gram, score_2gram, score_3gram, score_4gram))

if __name__=='__main__':
  main()