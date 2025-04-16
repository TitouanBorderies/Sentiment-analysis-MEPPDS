#!/bin/bash

# Définir le nom du bucket personnel
BUCKET_PERSONNEL="titouanborderies"

# Copier le fichier vers S3
mc cp twitter_validation.csv s3/${BUCKET_PERSONNEL}/ensae-reproductibilite/data/raw/twitter_validation.csv
