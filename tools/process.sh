#!/usr/bin/env bash
## Process VQA-RAD data
## This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
python3 tools/create_dictionary.py
python3 tools/compute_softscore.py
