#!/bin/bash
mkdir datasets;
curl https://raw.githubusercontent.com/KGQA/QALD_9_plus/refs/heads/main/data/qald_9_plus_test_wikidata.json > datasets/qald.json;

curl http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json > datasets/hotpot.json;

curl https://raw.githubusercontent.com/amazon-science/mintaka/refs/heads/main/data/mintaka_train.json > datasets/mintaka.json;

mkdir translations;