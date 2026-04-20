#/bin/bash
MUSA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -u scripts/inference_align.py     --checkpoint ../training_data/boltzgen1_structuretrained_small.ckpt     --moldir     ../training_data/mols     --yaml       example/vanill
a_protein/1g13prot.yaml     --out        infer_musa_1.pt     --seed       20260420     --recycling-steps 1     --sampling-steps  20
