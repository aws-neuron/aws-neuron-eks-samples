#!/bin/bash -x

#tar -xzf /app/model.tar.gz 
pip install --upgrade pip
pip install diffusers==0.20.2 transformers==4.33.1 accelerate==0.22.0 safetensors==0.3.1 matplotlib Pillow ipython -U      
uvicorn run:app --host=0.0.0.0
#while true; do sleep 1000; done
