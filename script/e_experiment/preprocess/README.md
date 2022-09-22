# Preprocessing datasets

### Command Examples

```shell
# python kwdlc.py [--crowdworker] [--knp] [--expert] --output_root dataset/KWDLC/

git clone git@github.com:ku-nlp/Winograd-Schema-Challenge-Ja.git
# manually modify train.txt/test.txt according to jwsc_change_log.txt
python jwsc.py Winograd-Schema-Challenge-Ja/train.txt dataset/JWSC/ [--format]
python jwsc.py Winograd-Schema-Challenge-Ja/test.txt dataset/JWSC/ --test [--format]

git clone git@github.com:yahoojapan/JGLUE.git
python jcqa.py JGLUE/datasets/jcommonsenseqa-v1.0/{train,valid}-v1.0.jsonl dataset/JCQA/{train,dev}.jsonl
```

##### Note

- jwsc.py is supposed to use [Juman++ Version: 2.0.0-rc3](https://github.com/ku-nlp/jumanpp/releases/tag/v2.0.0-rc3) (cf. l.27 - l.32)
