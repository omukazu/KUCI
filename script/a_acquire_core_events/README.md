# Acquisition of Core Events from Case Frames

### Requirements

- [Kyoto University Case Frames](https://www.gsk.or.jp/catalog/gsk2018-b)
  - A little preprocessing was done

### Description of config.json

- params
  - predicate_threshold: acquire top-<predicate_threshold> frequent single-word predicates in the active form
  - case_frame_rate: acquire top-k frequent case frames until the ratio of
                     the cumulative frequency to frequency of the predicate exceeds <case_frame_rate>
  - case_rate: acquire top-k frequent cases until the ratio of
               the cumulative frequency to frequency of the case frame exceeds <case_rate>
  - possessive_case_rate: acquire possessive case if the ratio of the frequency to that of the parent case 
                          exceeds <possessive_case_rate>
    - e.g. the parent case of "ノ格~ガ格" is "ガ格"
  - argument_rate: acquire top-k frequent arguments until the ratio of 
                   the cumulative frequency to frequency of the case exceeds <argument_rate>
  - sep_rate: threshold for the percentage of sep3 (examples of which subject is often a person)
- path
  - case_frame_dict: path to case frame dict (in json format)

### Command Examples

```shell
# python convert_xml_to_json.py test/case_frame_dict.xml test/case_frame_dict.json
python acquire_core_events.py config.json test/ [--request]
```
