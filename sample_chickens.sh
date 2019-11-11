# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

DATE=`date +%d_%m_%Y`
CODE=src
OUTPUT=results/${DATE}/chickenNet

echo "Sampling"
python3 ${CODE}/data_samples.py --data /media/owen/ba9d40b5-89de-4832-bad4-156b118e4a66/musicnet/preprocessed/Beethoven_String_Quartet --output ${OUTPUT} -n 2 --seq 80000

echo "Generating"
python3 ${CODE}/run_on_files.py --files ${OUTPUT} --batch-size 2 --checkpoint checkpoints/chickenNet/bestmodel --output-next-to-orig --decoders 0 --py
