# Integration of Specifications

Integrate specification changes made during dataset construction

### Command Examples

```shell
# python rename_columns.py test/<before> test/<after> [-v]
python assign_worker_ids_to_table.py test/table/ test/result/ test/table_w_worker_ids.tsv [-v]
python assign_worker_ids_to_cbeps.py ../b_extract_cbeps/test/cbeps.tsv test/verified_cbeps_w_worker_ids.tsv --table_w_worker_ids test/table_w_worker_ids.tsv
# python integrate_canonical_form.py test/verified_cbeps_w_worker_ids.tsv
```
