.PHONY: help setup label train-a train-b embed eval clean

help:
	@echo "Two-Tower Job Matcher — make targets:"
	@echo "  setup       — Create AWS bucket, upload Phase 0 postings"
	@echo "  label       — Label 400 postings via Claude API"
	@echo "  train-a     — Train Tower A from scratch (SageMaker)"
	@echo "  train-b     — Train Tower B fine-tuned (SageMaker)"
	@echo "  embed       — Batch embed all jobs, build FAISS indices"
	@echo "  eval        — Evaluate all three systems"
	@echo "  clean       — Remove local artifacts"

setup:
	bash scripts/setup_aws.sh

label:
	python src/data/label.py --sample-size 400

train-a:
	python scripts/submit_training_job.py --tower a --epochs 20

train-b:
	python scripts/submit_training_job.py --tower b --epochs 10

embed:
	python src/embed/batch_embed.py --tower a --model-path models/tower_a_model.pt --vocab-path models/tower_a_vocab.json
	python src/embed/batch_embed.py --tower b --model-path models/tower_b_model.pt
	python src/embed/build_faiss.py --tower a --embeddings-local /tmp/embeddings_a.jsonl
	python src/embed/build_faiss.py --tower b --embeddings-local /tmp/embeddings_b.jsonl

eval:
	python src/eval/run_eval.py --resume-file data/resume.txt --model-a-path models/tower_a_model.pt --vocab-a-path models/tower_a_vocab.json --model-b-path models/tower_b_model.pt

clean:
	rm -rf data/ models/ indices/ *.pt *.faiss *.jsonl
