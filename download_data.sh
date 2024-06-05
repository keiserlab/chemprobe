#!/bin/bash/sh

mkdir -p data/{cells,cpds}

## Cell models data
# CCLE
wget https://ndownloader.figshare.com/files/24613394 -O data/cells/sample_info.csv
# DepMap Public 20Q3
wget https://ndownloader.figshare.com/files/24613325 -O data/cells/CCLE_expression.csv

## Drug sensitivity data
# CTRP
wget https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip -O data/cpds/CTRPv2.0_2015_ctd2_ExpandedDataset.zip

# Unzip the downloaded file
unzip data/cpds/CTRPv2.0_2015_ctd2_ExpandedDataset.zip -d data/cpds/