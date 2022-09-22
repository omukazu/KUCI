# A Method for Building a Commonsense Inference Dataset based on Basic Events

This repository contains the scripts used in building [KUCI (pseudo-problems)](https://nlp.ist.i.kyoto-u.ac.jp/EN/?KUCI).

### Set up Virtual Environment

```shell
poetry install
poetry shell
echo $(find `pwd` -name "my_package") > $(python -c 'import sys; print(sys.path)' | grep -o "[^']*site-packages")/my_package.pth
```

### Citation

```text
@Inproceedings{Omura_and_Kurohashi_COLING2022,
    title = "{I}mproving {C}ommonsense {C}ontingent {R}easoning by {P}seudo-data and its {A}pplication to the {R}elated {T}asks",
    author = "Omura, Kazumasa and
        Kurohashi, Sadao",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    note = "to appear"
}

@Inproceedings{Omura_et_al_EMNLP2020,
    title = "{A} {M}ethod for {B}uilding a {C}ommonsense {I}nference {D}ataset based on {B}asic {E}vents",
    author = "Omura, Kazumasa and
        Kawahara, Daisuke and
        Kurohashi, Sadao",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.192",
    doi = "10.18653/v1/2020.emnlp-main.192",
    pages = "2450--2460"
}
```
