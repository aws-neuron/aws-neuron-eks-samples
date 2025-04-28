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

models = [
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
    gr.Markdown(f"# Image Generation App")
    gr.Markdown("Enter a prompt and specify the number of inference steps to generate images in different shapes.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                lines=1,
                placeholder="Enter your prompt here...",
                elem_id="prompt-box"
            )
            inference_steps = gr.Number(
                label="Inference Steps", 
                value=10, 
                precision=0, 
                info="Enter the number of inference steps; higher number takes more time but produces better image",
                elem_id="steps-number"
            )
            generate_button = gr.Button("Generate Images", variant="primary")
        
        with gr.Column(scale=2):
            image_components = []
            exec_time_components = []

            with gr.Row():
                for idx, model in enumerate(models):
                    with gr.Column():
                        # Title
                        gr.Markdown(f"**{model['name']}**")

                        # Scale down the image
                        preview_height = int(model['height'] / 2)
                        preview_width = int(model['width'] / 2)

                        img = gr.Image(
                            label="",
                            height=preview_height,
                            width=preview_width,
                            interactive=False
                        )
                        # Use Markdown for simpler smaller text
                        exec_time = gr.Markdown(value="")

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
