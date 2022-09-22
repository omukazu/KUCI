# Investigation of Human Accuracy of Commonsense Inference Problems

We used [Yahoo! crowdsourcing](https://crowdsourcing.yahoo.co.jp/).

### Command Examples

```shell
shuf ../d_generate_problems/test/problems.jsonl | head -n 500 > test/sample.jsonl
python create_input_file.py config.json test/sample.jsonl test/input.tsv [--id]
# do crowdsourcing
python aggregate.py test/result.tsv --problem test/problems.jsonl [--num-choices]
```
