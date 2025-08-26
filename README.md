# TropicalCNN

## Regularizer options

`experiment.py` supports optional Proximity regularizers via the
`--regularizer` flag or the `regularizer` column in job files. Available
values are:

* `none` (default)
* `pcl_l2` – Proximity loss with L2 metric
* `pcl_tropical` – Proximity loss with tropical metric

When using job files, place the `regularizer` column after
`last_layer_name`.