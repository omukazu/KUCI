import json
import os
import random
from collections import defaultdict as ddict
from datetime import timedelta
from math import ceil, log10
from pathlib import Path
from importlib import import_module
from socket import gethostname
from typing import Literal, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig, get_linear_schedule_with_warmup

from module.dataset.custom_dataset import CustomConcatDataset
from utils import tqdm, get_logger, set_file_handlers


class Watcher:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.coordinates = ddict(list)

    def add_coordinates(self, x, snapshot):
        for label, y in snapshot.items():
            self.coordinates[label].append([x, y])

    def dump(self):
        with self.output_path.open(mode='w') as f:
            json.dump(self.coordinates, f, ensure_ascii=False, indent=2)


class BaseTrainer:
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.pretrained_model_name_or_path)
        for split in self.cfg.dataset.split.keys():
            setattr(self, f'{split}_data_loader', self.get_data_loader(split))
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.model = DDP(
            self.model.to(self.device),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

        self.save_dir = self.get_save_dir()
        if args.local_rank == 0:
            self.logger = self.pre_execute()
            self.watcher = Watcher(self.save_dir.joinpath('snapshot.json'))

        self.current_training_steps = 0
        self.best = {
            'epoch': ...,
            'state_dict': ...,
            'score': -1.,
            'patience': 0
        }

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

    def get_input_path(self, corpus_cfg, split: str) -> Path:
        input_dir = Path(self.cfg.dataset.root).joinpath(corpus_cfg.name)
        if self.args.fold:
            return input_dir.joinpath(f'fold{self.args.fold}').joinpath(f'{split}{self.cfg.dataset.ext}')
        else:
            return input_dir.joinpath(f'{split}{self.cfg.dataset.ext}')

    def get_collate_fn(self):
        if self.cfg.dataset.collate_fn == 'DataCollatorForLanguageModeling':
            # mlm_probability=0.15
            return DataCollatorForLanguageModeling(tokenizer=self.tokenizer)

    def get_data_loader(self, split: str):
        class_ = getattr(import_module(self.cfg.dataset.module), self.cfg.dataset.class_)
        collate_fn = self.get_collate_fn()
        if split == 'train':
            dataset = CustomConcatDataset(
                [
                    class_(
                        self.get_input_path(corpus_cfg, split),
                        split,
                        self.tokenizer,
                        self.cfg.hparams.max_seq_len,
                        confirm_inputs=(self.args.local_rank == 0),
                        **self.cfg.dataset.kwargs,
                        **corpus_cfg,
                    )
                    for corpus_cfg in getattr(self.cfg.dataset.split, split)
                ]
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self.cfg.hparams.train_batch_size // dist.get_world_size(),
                sampler=DistributedSampler(dataset, shuffle=True, seed=self.cfg.hparams.seed),
                num_workers=0,
                collate_fn=collate_fn,
                # pin_memory=True
            )
        elif split in {'dev', 'test'}:
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
        else:
            raise ValueError('invalid split')
        return data_loader

    def get_model(self):
        config = AutoConfig.from_pretrained(self.cfg.model.pretrained_model_name_or_path)
        config.update(self.cfg.model.post_hoc)
        model = getattr(import_module(self.cfg.model.module), self.cfg.model.class_).from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, config=config
        )
        # tokenizer.vocab_size doesn't take additional_special_tokens into account
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.hparams.lr,
            betas=tuple(self.cfg.hparams.betas)
        )
        return optimizer

    def get_scheduler(self):
        num_iter_per_epoch = len(getattr(self, f'train_data_loader'))
        if self.cfg.hparams.num_training_steps:
            num_training_steps = self.cfg.hparams.num_training_steps
        else:
            num_step_per_epoch = ceil(num_iter_per_epoch / self.cfg.hparams.accumulation_steps)
            num_training_steps = num_step_per_epoch * self.cfg.hparams.epoch

        if self.cfg.hparams.num_warmup_steps:
            num_warmup_steps = self.cfg.hparams.num_warmup_steps
        else:
            num_warmup_steps = int(num_training_steps * self.cfg.hparams.warmup_proportion)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler

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

    def get_save_dir(self) -> Path:
        basename = self.get_basename()
        if self.args.fold:
            return Path(self.cfg.save_dir).joinpath(basename).joinpath(f'fold{self.args.fold}')
        else:
            return Path(self.cfg.save_dir).joinpath(basename)

    def pre_execute(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger = get_logger(__name__)
        set_file_handlers(logger, output_path=self.save_dir.joinpath('fine_tuning.log'))

        logger.info(f'hostname: {gethostname()[:2]}')
        logger.info(f'config: {json.dumps(self.cfg, indent=2)}')
        num_examples = [
            len(getattr(self, f'{split}_data_loader').dataset)
            if split == 'train' else
            [len(data_loader.dataset) for data_loader in getattr(self, f'{split}_data_loader').values()]
            for split in self.cfg.dataset.split.keys()
        ]
        logger.info(f'number of examples: {num_examples}')
        logger.info(f'model: {self.save_dir.name}')
        return logger

    def truncate_and_to(self, batch: dict[str, torch.Tensor], max_seq_len: int) -> None:
        for key, value in batch.items():
            if key == 'inputs':
                batch[key] = {k: v[..., :max_seq_len].to(self.device) for k, v in value.items()}
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = value.to(self.device)

    def compute_loss(
        self,
        output: torch.Tensor,
        batch: dict[str, torch.Tensor],
        mode: Literal['train', 'eval']
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        clone = tensor.clone()
        dist.all_reduce(clone)
        return clone

    def training_loop(self, epoch: int) -> None:
        if not self.model.training:
            self.model.train()

        train_data_loader = getattr(self, 'train_data_loader')
        train_data_loader.sampler.set_epoch(epoch)

        step = max(10 ** (int(log10(len(train_data_loader))) - 1), 1)
        num_training_examples, sum_loss = 0, 0.
        remainder = len(train_data_loader) % self.cfg.hparams.accumulation_steps

        train_bar = tqdm(train_data_loader) if self.args.local_rank == 0 else train_data_loader
        for batch_idx, batch in enumerate(train_bar):
            if (
                self.cfg.hparams.num_training_steps and
                self.current_training_steps >= self.cfg.hparams.num_training_steps
            ):
                break

            if self.cfg.dataset.class_ == 'AMLM':
                batch_size, *_ = batch['input_ids'].shape
                _ = batch.pop('example_id')
            else:
                batch_size, *_ = batch['inputs']['input_ids'].shape

            self.truncate_and_to(batch, torch.max(batch.pop('length')).item())

            output = self.model(**batch)

            batch_sum_loss = self.compute_loss(output, batch, mode='train')
            if (remainder > 0) and (len(train_data_loader) - batch_idx < self.cfg.hparams.accumulation_steps):
                divisor = remainder
            else:
                divisor = self.cfg.hparams.accumulation_steps
            if self.cfg.dataset.class_ != 'AMLM':
                divisor *= batch_size
            batch_mean_loss = batch_sum_loss / divisor

            batch_mean_loss.backward()
            if (batch_idx % self.cfg.hparams.accumulation_steps == 0) or (batch_idx == len(train_data_loader) - 1):
                if self.cfg.hparams.max_norm:
                    # norm_type=2 (l2)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.hparams.max_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.current_training_steps += 1

            num_training_examples += self.all_reduce(torch.tensor(batch_size, device=self.device)).item()
            sum_loss += self.all_reduce(batch_sum_loss).item()

            if self.args.local_rank == 0:
                snapshot = {
                    'lr': round(self.scheduler.get_last_lr()[0], 9),
                    'train_loss': round(sum_loss / num_training_examples, 6)
                }
                train_bar.set_postfix(snapshot)
                if (batch_idx + 1) % step == 0:
                    x = round(epoch + (batch_idx + 1) / len(train_bar), 6)
                    self.watcher.add_coordinates(x, snapshot)

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

    def evaluation_loop(
        self,
        data_loader: DataLoader,
        epoch: int,
        snap: bool = True
    ) -> Union[float, dict[str, float]]:
        if self.model.training:
            self.model.eval()

        buf = {key: [] for key in ['example_id', 'output', 'prediction', 'label', 'loss']}
        with torch.no_grad():
            eval_bar = tqdm(data_loader) if self.args.local_rank == 0 else data_loader
            for batch in eval_bar:
                self.truncate_and_to(batch, torch.max(batch.pop('length')))

                output = self.model(**batch)

                buf['example_id'].append(self.all_gather(batch['example_id']))
                buf['output'].append(self.all_gather(output))
                buf['prediction'].append(self.all_gather(self.predict(output, batch)))
                buf['label'].append(self.all_gather(batch['label']))
                buf['loss'].append(self.all_gather(self.compute_loss(output, batch, mode='eval')))
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
                    if snap:
                        self.watcher.add_coordinates(epoch + 1, snapshot)

        return objective_metric

    def store_state_dict(self) -> None:
        self.best['state_dict'] = {
            key: value.cpu() for key, value in self.model.module.state_dict().items()
        }

    def save(self, ext: str = None) -> None:
        save_directory = self.save_dir.parent.joinpath(self.save_dir.name + ext) if ext else self.save_dir
        print(f'*** save model in {save_directory} ***')
        self.model.module.save_pretrained(save_directory)

    def load(self, key: str) -> None:
        if key == 'best':
            print('*** load the best model ***')
            # self.model.module = self.model.module.from_pretrained(self.save_dir).to(self.device)
            self.model.module.cpu()
            self.model.module.load_state_dict(self.best['state_dict'])
            self.model.module.to(self.device)
        elif key == 'init':
            print('*** load the initial model ***')
            config = AutoConfig.from_pretrained(self.cfg.model.pretrained_model_name_or_path)
            config.update(self.cfg.model.post_hoc)
            self.model.module = self.model.module.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path, config=config
            ).to(self.device)
        else:
            raise KeyError('invalid key')
