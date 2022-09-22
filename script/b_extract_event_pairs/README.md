# Automatic Extraction of Contingent Basic Event Pairs

### Requirements

- [Juman++](https://github.com/ku-nlp/jumanpp)
- [KNP](https://github.com/ku-nlp/knp)

### Description of each column

- sid: sentence id
- orig: the original sentence
- morphemes1: word-segmented former event
- morphemes1_wo_modifier: " (without modifier, for debugging)
- normalized_morphemes1: " (with normalizing the ending)
- normalized_morphemes2: word-segmented latter event (with normalizing the ending)
- event1: former event
- normalized_event1: former event (with normalizing the ending)
- normalized_event2: latter event (with normalizing the ending)
- num_base_phrases2: number of base phrases in latter event
- num_morphemes2: number of morphemes in latter event
- pas1: predicate argument structure contained in former event
- pas2: " latter event
- content_words1: content words contained in former event
- ya_content_words1: " (according to the criteria of EventGraph)
- content_words2: content words contained in latter event
- ya_content_words2: " (according to the criteria of EventGraph)
- type1: type of former event ∈ {verb, adjective, noun}
- type2: " latter event
- rel: discourse relation between events (clauses)
- rel_surf: discourse marker
- reliable: whether there is ambiguity in dependency between event pair or not  
  = whether event pair is the last two clauses in the sentence or not
- core_event_pair: core event pair contained in event pair
- basic: whether event pair satisfies Basic condition or not
- pou1: whether former event contains demonstratives or undefined words or not
- pou2: whether latter event "
- negation1: negation of former event
- negation2: " latter event
- potential1: whether former event is in the potential form or not
- potential2: whether latter event "
- normalized_predicate1: normalized predicate of former event
- normalized_predicate2: " latter event

### Command Examples

```shell
# prepare knp parsing results
# e.g. cat test/sentences.txt | jumanpp | knp -tab > test/example.knp
python extract_event_pairs.py test/example.knp test/event_pairs.tsv --core-events test/core_events.txt  # (../a_acquire_core_events/test/core_events.txt)
python filter_by_conditions.py test/event_pairs.tsv test/filtered.tsv
python create_blacklist.py test/filtered.tsv test/blacklist.txt
# manually comment out a few in blacklist.txt if necessary
python post_process.py test/filtered.tsv test/cbeps.tsv --blacklist test/blacklist.txt
```

### Multiprocessing

```shell
# prepare knp parsing results, the filenames of which are in the format of "{idx:08}.knp.gz"
# e.g. echo "雨が降る。" | jumanpp | knp -tab > 00000001.knp && gzip 00000001.knp
# if basenames are not zero-padded, they won't be sorted numerically, which may cause an unexpected discrepancy in a later step
make -f makefile/knp2event_pair.mk -j 2
make -f makefile/event_pair2filtered.mk -j 2
```
