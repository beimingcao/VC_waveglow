#!/usr/bin/env bash

python3 inference.py -f <(ls mel_spectrograms/*.pt) -w waveglow_256channels_universal_v5.pt -o . --is_fp16 -s 0.6
