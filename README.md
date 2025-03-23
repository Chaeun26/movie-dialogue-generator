# movie-dialogue-generator
A fine-tuned DistilGPT-2 model for generating movie-style dialogue, deployed on AWS EC2

## Features
- Generates dialogue from prompts like "What’s your plan?"
- Tracks server uptime via a GET endpoint.
- Deployed on AWS EC2 with Gunicorn.

## Setup
### Training
- `train_movie_dialogue_model.py`: Fine-tunes DistilGPT-2 on movie dialogues.
- Requirements: `transformers`, `datasets`, `torch`.
- Run: `python3 train_movie_dialogue_model.py`

### Deployment
- `app.py`: Flask API code.
- `deploy.sh`: EC2 setup script.
- Steps:
  1. Launch an EC2 instance (e.g., t3.medium, 20 GB EBS).
  2. Upload `movie_dialogue_model_final.zip` and `app.py`.
  3. Run `deploy.sh`.

## Requirements
- Python 3.12
- Flask, Transformers, Torch, Gunicorn

## Notes
- Trained on Cornell Movie Dialogs Corpus.
- Hosted on AWS EC2; IP may change unless using an Elastic IP.
