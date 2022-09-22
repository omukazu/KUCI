# Verification of Contingent Basic Event Pairs using Crowdsourcing

We used [Yahoo! crowdsourcing](https://crowdsourcing.yahoo.co.jp/).

### Command Examples

```shell
python create_input_file.py config.json ../b_extract_event_pairs/test/cbeps.tsv test/input.tsv [--dummy] [--id]
# do crowdsourcing
python aggregate.py test/result.tsv --output test/aggregated.tsv
python extract_verified_cbeps.py ../b_extract_event_pairs/test/cbeps.tsv test/verified_cbeps.tsv --aggregated test/aggregated.tsv
```

* split cbeps.tsv using ../b_extract_cbeps/snippet/split_into_or_merge_data_frames.py
