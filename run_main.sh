#!/bin/bash
# 1. 执行训练
python main.py --file_Key_words Preparation Antagonist Crown

# 2. 执行测试
python main.py --test --use_crown --ckpts "experiments/PoinTr/Tooth_models/default/ckpt-best.pth" --file_key_words Preparation Antagonist Crown