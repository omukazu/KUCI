import json
import os
import random
from datetime import timedelta
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from pyknp import Juman
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoConfig

from utils import tqdm


class BaseTester:
    def __init__(self, args, cfg):
        cfg.hparams.seed = args.seed
        self.args = args
        self.cfg = cfg

        self.set_environment_variables()
        assert \
            cfg.hparams.train_batch_size % dist.get_world_size() == 0, \
            'train_batch_size should be divisible by the world size'
        self.set_seed(cfg.hparams.seed)
        self.device = torch.device(f'cuda:{args.local_rank}')

        self.load_dir = self.get_load_dir()
        print(f'model: {self.load_dir}')

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.pretrained_model_name_or_path)
        if args.split:
            setattr(self, f'{args.split}_data_loader', self.get_data_loader(args.split))
        else:
            self.jumanpp = Juman()
        self.model = self.get_model()
        self.model = DDP(
            self.model.to(self.device),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

    def set_environment_variables(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        # Deprecated
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = self.args.port
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=300))   # init_method="env://"

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)  # optional?

    def get_load_dir(self) -> Path:
        basename = self.get_basename()
        if self.args.fold:
            return Path(self.cfg.save_dir).joinpath(basename).joinpath(f'fold{self.args.fold}')
        else:
            return Path(self.cfg.save_dir).joinpath(basename)

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
        assert split in {'dev', 'test'}, 'specify dev or test'
        corpus2dataset = {
            corpus_cfg.name: class_(
                self.get_input_path(corpus_cfg, split),
                split,
                self.tokenizer,
                self.cfg.hparams.max_seq_len,
                confirm_inputs=(self.args.local_rank == 0),
                **self.cfg.dataset.kwargs,
                **corpus_cfg,
            )
            for corpus_cfg in getattr(self.cfg.dataset.split, split)
        }
        data_loader = {
            corpus: DataLoader(
                dataset,
                batch_size=self.cfg.hparams.eval_batch_size,
                sampler=DistributedSampler(dataset, shuffle=False),
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=True
            )
            for corpus, dataset in corpus2dataset.items()
        }
        return data_loader

    def get_model(self):
        config = AutoConfig.from_pretrained(self.load_dir)
        config.update(self.cfg.model.post_hoc)
        model = getattr(import_module(self.cfg.model.module), self.cfg.model.class_).from_pretrained(
            self.load_dir, config=config
        )
        # tokenizer.vocab_size doesn't take additional_special_tokens into account
        model.resize_token_embeddings(len(self.tokenizer))
        return model

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

    def truncate_and_to(self, batch: dict[str, torch.Tensor], max_seq_len: int) -> None:
        for key, value in batch.items():
            if key == 'inputs':
                batch[key] = {k: v[..., :max_seq_len].to(self.device) for k, v in value.items()}
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = value.to(self.device)

    def compute_loss(
        self,
        output: torch.Tensor,
        batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def all_gather(tensor: torch.Tensor) -> torch.Tensor:
        # zeros_like: take over the dtype and device of tensor
        placeholders = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(placeholders, tensor)
        return torch.cat(placeholders, dim=0)

    def predict(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def unique(tensor: torch.Tensor, dim: int = None) -> torch.Tensor:
        output, inverse_indices = torch.unique(tensor, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
        inverse_indices, perm = inverse_indices.flip([0]), perm.flip([0])
        return inverse_indices.new_empty(output.size(0)).scatter_(0, inverse_indices, perm)

    def compute_metrics(
        self,
        data_loader: DataLoader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[float, dict[str, float]]:
        raise NotImplementedError

    def error_analysis(
        self,
        data_loader: DataLoader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor,
        predictions: torch.Tensor
    ) -> list:
        raise NotImplementedError

    def evaluation_loop(
        self,
        data_loader: DataLoader,
        do_error_analysis: bool = False
    ) -> list:
        if self.model.training:
            self.model.eval()

        buf = {key: [] for key in ['example_id', 'output', 'prediction', 'label', 'loss']}
        ret = []
        with torch.no_grad():
            eval_bar = tqdm(data_loader) if self.args.local_rank == 0 else data_loader
            for batch in eval_bar:
                if 'length' in batch:
                    self.truncate_and_to(batch, torch.max(batch.pop('length')).item())

                output = self.model(**batch)

                buf['example_id'].append(self.all_gather(batch['example_id']))
                buf['output'].append(self.all_gather(output))
                buf['prediction'].append(self.all_gather(self.predict(output, batch)))
                buf['label'].append(self.all_gather(batch['label']))
                buf['loss'].append(self.all_gather(self.compute_loss(output, batch)))
            else:
                sorted_indexes = self.unique(torch.cat(buf['example_id'], dim=0))
                for key in buf.keys():
                    buf[key] = torch.cat(buf[key], dim=0)[sorted_indexes]
                assert len(buf['example_id']) == len(data_loader.dataset)

                split = data_loader.dataset.split
                sum_loss = (torch.sum(buf['loss']) / len(buf['example_id'])).item()
                objective_metric, metrics = self.compute_metrics(
                    data_loader, buf['example_id'], buf['output'], buf['prediction'], buf['label']
                )
                if self.args.local_rank == 0:
                    snapshot = {f'{split}_loss': round(sum_loss, 6)}
                    snapshot.update(metrics)
                    print(json.dumps(snapshot))
                    if do_error_analysis:
                        ret = self.error_analysis(data_loader, buf['example_id'], buf['output'], buf['prediction'])

        return ret

    def preprocess(self, input_: str) -> str:
        raise NotImplementedError

    def interactive(self, input_: str) -> torch.Tensor:
        inputs = self.tokenizer(
            *self.preprocess(input_),
            max_length=self.cfg.hparams.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        length = torch.max(torch.sum(inputs['attention_mask'], axis=1)).item()
        self.truncate_and_to(inputs, length)
        output = self.model(**{'inputs': inputs}).squeeze()
        return output
