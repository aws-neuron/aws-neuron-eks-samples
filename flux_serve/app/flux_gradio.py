import gradio as gr
import requests
from PIL import Image
import io
import os
from fastapi import FastAPI
import base64
import asyncio
import httpx

app = FastAPI()

model_id=os.environ['MODEL_ID']

models = [
    {
        'name': '256x144',
        'host_env': 'FLUX_NEURON_256X144_MODEL_API_SERVICE_HOST',
        'port_env': 'FLUX_NEURON_256X144_MODEL_API_SERVICE_PORT',
        'height': 256,
        'width': 144
    },
    {
        'name': '1024x576',
        'host_env': 'FLUX_NEURON_1024X576_MODEL_API_SERVICE_HOST',
        'port_env': 'FLUX_NEURON_1024X576_MODEL_API_SERVICE_PORT',
        'height': 1024,
        'width': 576
    },
    {
        'name': '512x512',
        'host_env': 'FLUX_NEURON_512X512_MODEL_API_SERVICE_HOST',
        'port_env': 'FLUX_NEURON_512X512_MODEL_API_SERVICE_PORT',
        'height': 512,
        'width': 512
    }
]

for model in models:
    host = os.environ[model['host_env']]
    port = os.environ[model['port_env']]
    model['url'] = f"http://{host}:{port}/generate"

async def fetch_image(client, url, prompt, num_inference_steps):
    payload = {
        "prompt": prompt,
        "num_inference_steps": int(num_inference_steps)
    }
    try:
        response = await client.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        image_bytes = base64.b64decode(data['image']) 
        image = Image.open(io.BytesIO(image_bytes))
        execution_time = data.get('execution_time', 0)
        return image, f"{execution_time:.2f} seconds"
    except httpx.RequestError as e:
        return None, f"Request Error: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"

async def call_model_api(prompt, num_inference_steps):
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_image(client, model['url'], prompt, num_inference_steps)
            for model in models
        ]
        results = await asyncio.gather(*tasks)
    images = []
    exec_times = []
    for image, exec_time in results:
      images.append(image)
      exec_times.append(exec_time)
    return images + exec_times

@app.get("/health")
def healthy():
    return {"message": "Service is healthy"}

@app.get("/readiness")
def ready():
    return {"message": "Service is ready"}

with gr.Blocks() as interface:
    gr.Markdown(f"# {model_id} Image Generation App")
    gr.Markdown("Enter a prompt and specify the number of inference steps to generate images in different shapes.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=1, placeholder="Enter your prompt here...",elem_id="prompt-box")
            inference_steps = gr.Number(
                label="Inference Steps", 
                value=10, 
                precision=0, 
                info="Enter the number of inference steps; higher number takes more time but produces better image",
                elem_id="steps-number"
            )
            generate_button = gr.Button("Generate Images",variant="primary")
        
        with gr.Column(scale=2):
            image_components = []
            exec_time_components = []

            with gr.Row(equal_height=True):
              for idx, model in enumerate(models):
                 with gr.Column(scale=1,min_width=300):
                     img = gr.Image(label=f"{model['name']}",height=model['height'],width=model['width'],interactive=False)
                     exec_time = gr.Textbox(label=f"Execution Time ({model['name']})",interactive=False,lines=1,placeholder="Execution time will appear here...")
                     image_components.append(img)
                     exec_time_components.append(exec_time)

    # callback for the button
    generate_button.click(
        fn=call_model_api,
        inputs=[prompt, inference_steps],
        outputs=image_components + exec_time_components,
        api_name="generate_images"
    )
app = gr.mount_gradio_app(app, interface, path="/serve")
'''
interface = gr.Interface(
    fn=call_model_api,
    inputs=[
        gr.Textbox(label="Prompt", lines=1, placeholder="Enter your prompt here..."),
        gr.Number(
            label="Inference Steps", 
            value=10, 
            precision=0, 
            info="Enter the number of inference steps; higher number takes more time but produces better image"
        )
    ],
    outputs=[
        gr.Image(label=f"Image from {models[0]['name']}", height=models[0]['height'], width=models[0]['width']),
        gr.Textbox(label=f"Execution Time ({models[0]['name']})"),
        gr.Image(label=f"Image from {models[1]['name']}", height=models[1]['height'], width=models[1]['width']),
        gr.Textbox(label=f"Execution Time ({models[1]['name']})"),
        gr.Image(label=f"Image from {models[2]['name']}", height=models[2]['height'], width=models[2]['width']),
        gr.Textbox(label=f"Execution Time ({models[2]['name']})"),
    ],
    description="Enter a prompt and specify the number of inference steps to generate images using the model pipeline."
)
app = gr.mount_gradio_app(app,interface, path="/serve")
'''
