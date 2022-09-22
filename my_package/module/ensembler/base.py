import os
import re
from importlib import import_module
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoConfig, logging

from utils import tqdm


logging.set_verbosity_error()


class BaseBlender:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        assert re.fullmatch(r'[0-9]', args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device(f'cuda')

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.pretrained_model_name_or_path)
        self.dev_data_loader, self.test_data_loader = map(
            lambda x: self.get_data_loader(x), ['dev', 'test']
        )
        self.models = [self.load_model(seed).to(self.device) for seed in args.seed]

        self.best = {
            'score': -1.,
            'ratio': ...
        }

    def get_input_path(self, corpus_cfg, split: str) -> Path:
        input_dir = Path(self.cfg.dataset.root).joinpath(corpus_cfg.name)
        if self.args.fold:
            return input_dir.joinpath(f'fold{self.args.fold}').joinpath(f'{split}{self.cfg.dataset.ext}')
        else:
            return input_dir.joinpath(f'{split}{self.cfg.dataset.ext}')

    @staticmethod
    def get_collate_fn():
        return None

    def get_data_loader(self, split: str) -> dict[str, DataLoader]:
        class_ = getattr(import_module(self.cfg.dataset.module), self.cfg.dataset.class_)
        collate_fn = self.get_collate_fn()
        corpus2dataset = {
            corpus_cfg.name: class_(
                self.get_input_path(corpus_cfg, split),
                split,
                self.tokenizer,
                self.cfg.hparams.max_seq_len,
                confirm_inputs=False,
                **self.cfg.dataset.kwargs,
                **corpus_cfg
            )
            for corpus_cfg in getattr(self.cfg.dataset.split, split)
        }
        data_loader = {
            corpus: DataLoader(
                dataset,
                batch_size=self.cfg.hparams.eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=True
            )
            for corpus, dataset in corpus2dataset.items()
        }
        return data_loader

    def get_basename(self) -> str:
        basename = [self.cfg.id]

        if self.cfg.dataset.kwargs and any(self.cfg.dataset.kwargs.values()):
            basename.append('_'.join(key for key, value in self.cfg.dataset.kwargs.items() if value))

        for corpus_cfg in self.cfg.dataset.split.train:
            if corpus_cfg.num_examples:
                basename.append(f'{corpus_cfg.name}-{corpus_cfg.num_examples}')

            if corpus_cfg.name == 'Pseudo':
                basename.append(f'w{corpus_cfg.weight}')

        if self.cfg.dataset.class_ in {'KUCI', 'KWDLC', 'BJWSC', 'JCQA'}:
            basename.append(str(self.cfg.hparams.seed))

        return '_'.join(basename)

    def get_load_dir(self) -> Path:
        basename = self.get_basename()
        if self.args.fold:
            return Path(self.cfg.save_dir).joinpath(basename).joinpath(f'fold{self.args.fold}')
        else:
            return Path(self.cfg.save_dir).joinpath(basename)

    def load_model(self, seed: int):
        self.cfg.hparams.seed = seed
        load_dir = self.get_load_dir()

        config = AutoConfig.from_pretrained(load_dir)
        config.update(self.cfg.model.post_hoc)
        model = getattr(import_module(self.cfg.model.module), self.cfg.model.class_).from_pretrained(
            load_dir, config=config
        )
        # tokenizer.vocab_size doesn't take additional_special_tokens into account
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def truncate_and_to(self, batch: dict[str, torch.Tensor], max_seq_len: int) -> None:
        for key, value in batch.items():
            if key == 'inputs':
                batch[key] = {k: v[..., :max_seq_len].to(self.device) for k, v in value.items()}
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = value.to(self.device)

    def get_output(self, data_loader: DataLoader):
        for model in self.models:
            if model.training:
                model.eval()

        buf = {key: [] for key in ['example_id', 'output', 'label']}
        with torch.no_grad():
            eval_bar = tqdm(data_loader)
            for batch in eval_bar:
                batch_size, *_ = batch['inputs']['input_ids'].shape
                self.truncate_and_to(batch, torch.max(batch['length']).item())

                output = torch.stack([model(**batch) for model in self.models], dim=-1)  # b, out_dim, num_models

                buf['example_id'].append(batch['example_id'])
                buf['output'].append(output)
                buf['label'].append(batch['label'])
            else:
                for key in buf.keys():
                    buf[key] = torch.cat(buf[key], dim=0)
                assert len(buf['example_id']) == len(data_loader.dataset)

        return buf['example_id'], buf['output'], buf['label']

    @staticmethod
    def ensemble(outputs: torch.Tensor, ratio: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ratio is not None:
            return torch.sum(outputs * ratio, dim=-1)
        else:
            return torch.mean(outputs, dim=-1)

    def compute_metrics(
        self,
        data_loader: DataLoader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[float, dict[str, float]]:
        raise NotImplementedError

    def error_analysis(
        self,
        data_loader: DataLoader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor
    ) -> list:
        raise NotImplementedError
