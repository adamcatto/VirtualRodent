#!/bin/bash
cd src/website/frontend
npm install
cd ../../..
python scripts/train.py experiment_name=test_run
python scripts/evaluate.py checkpoint_path=outs/checkpoints/test_run/best.ckpt

python src/website/app.py  # Terminal 1
cd src/website/frontend && npm run dev  # Terminal 2
# Then open http://localhost:5173 in your web browser