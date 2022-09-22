# Generation of Commonsense Inference/Pseudo Problems

### Description of config.json

- params
  - length: range of "Length" condition (fixed)
  - choice_similarity: range of "Choice-Sililarity" condition
  - context_similarity: range of "Context-Sililarity" condition
  - num_choices: number of choices
  - seed: random seed value
- path
  - table: path to table (cbeps.tsv / reduced.tsv)
  - kuci: path to KUCI
  - w2v: path to word2vec
  - blacklist: blacklist of paraphrases

### Command Examples

```shell
# python ../b_extract_event_pairs/snippet/split_into_or_merge_data_frames.py test/table.tsv test/split/ --ext .cbep.tsv --num-splits 10
# if basenames are not zero-padded, they won't be sorted numerically, which may cause an unexpected discrepancy
python select_candidates.py config.json test/00000000.cbep.tsv test/00000000.candidate.npy && gzip test/00000000.candidate.npy
python generate_problems.py config.json test/ test/problems.jsonl
python build_dataset.py config.json test/problems.jsonl test/dataset/

# for generating pseudo problems
# python reduce_data_leakage.py config.json test/pseudo_table.tsv [-j]
```

### Multiprocessing

```shell
# python ../b_extract_event_pairs/snippet/split_into_or_merge_data_frames.py test/table.tsv test/split/ --ext .cbep.tsv --num-splits 10
make -f makefile/cbep2candidate.mk -j 2
```
