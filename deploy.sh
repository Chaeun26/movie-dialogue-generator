#!/bin/bash
# Setup EC2 environment and deploy API
sudo apt update
sudo apt install python3-pip python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install flask transformers torch gunicorn
unzip movie_dialogue_model_final.zip -d movie_dialogue_model_final
nohup gunicorn --bind 0.0.0.0:5000 app:app --workers 1 --timeout 60 &
