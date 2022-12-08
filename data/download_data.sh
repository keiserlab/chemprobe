#!/bin/bash/sh

mkdir -p {cells,cpds}

## Cell models data
# CCLE
wget https://ndownloader.figshare.com/files/24613394 -O cells/sample_info.csv
# DepMap Public 20Q3
wget https://ndownloader.figshare.com/files/24613325 -O cells/CCLE_expression.csv

## Drug sensitivity data
# CTRP
wget ftp://caftpd.nci.nih.gov/pub/OCG-DCC/CTD2/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip -O cpds/CTRPv2.0_2015_ctd2_ExpandedDataset.zip