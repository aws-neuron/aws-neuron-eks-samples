ARG REGION

# Base image: PyTorch training image for NeuronX
FROM public.ecr.aws/neuron/pytorch-training-neuronx:2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04

# Install Ray for distributed computing
RUN pip3 install aiohttp \
    && rm -rf /root/.cache

# Install additional Python dependencies
RUN pip3 install wget awscli regex boto3 pyarrow \
    && rm -rf /root/.cache/

# Copy the Llama3 training code into the container
# (Separate layer to rebuild only if the code changes)
COPY ./llama3_finetune /llama3_finetune

# Make shell scripts executable
RUN chmod +x /llama3_finetune/*.sh

# Set the working directory
WORKDIR /llama3_finetune

# Installing the requirements 
RUN pip install -r requirements.txt --extra-index-url https://pip.repos.neuron.amazonaws.com 

# Installing the requirements 
RUN pip install transformers==4.32.1 --no-warn-conflicts
