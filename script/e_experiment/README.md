# Experiments

Conduct experiments to investigate model performance on KUCI and the related tasks

### Requirements

- [Juman++ Version: 2.0.0-rc3](https://github.com/ku-nlp/jumanpp)
- [KUCI](https://nlp.ist.i.kyoto-u.ac.jp/EN/?KUCI)
- [Winograd-Schema-Challenge-Ja](https://github.com/ku-nlp/Winograd-Schema-Challenge-Ja)
- [JCommonsenseQA](https://github.com/yahoojapan/JGLUE)
- [NICT BERT](https://alaginrc.nict.go.jp/nict-bert/index.html)

### Command Examples

```shell
torchrun --nnodes 1 --nproc_per_node 2 --rdzv_endpoint=$(hostname):<port> train.py config/KUCI/BERT.json --gpu 0,1 [--fold] [--seed] [--dry-run]
torchrun --nnodes 1 --nproc_per_node 2 --rdzv_endpoint=$(hostname):<port> test.py config/KUCI/BERT.json --gpu 0,1 [--fold] [--seed] [--split] [--dump]
python ensemble.py config/KUCI/BERT.json --ensembler module.ensembler.kuci-KUCIBlender --seed 0 --seed 1 --seed 2 --gpu 0 --metric acc [--num-fold]
```
