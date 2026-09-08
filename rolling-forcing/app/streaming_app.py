"""Gradio Streaming Video Generation App.

A web interface for generating videos with two modes:
1. Streaming Mode: See frames as they're generated (lower latency)
2. Quality Mode: Get encoded video chunks (better quality)

Usage:
    python streaming_app.py --config configs/rolling_forcing_dmd_small.yaml
    
    # With checkpoint
    python streaming_app.py --config configs/rolling_forcing_dmd_small.yaml \\
        --checkpoint checkpoints/rolling_forcing_dmd.pt --use_ema
"""
import argparse
import os
import sys
import time
import tempfile
import threading
from typing import List, Optional, Tuple
from collections import deque

import gradio as gr
import numpy as np
from PIL import Image

# Import streaming pipeline
from streaming_pipeline import StreamingInferencePipeline, StreamingConfig


# Global pipeline instance (loaded once)
_pipeline: Optional[StreamingInferencePipeline] = None
_generation_lock = threading.Lock()


def get_pipeline() -> StreamingInferencePipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Call init_pipeline() first.")
    return _pipeline


def init_pipeline(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    model_path: str = "wan_models/Wan2.1-T2V-1.3B",
    vae_path: str = "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    use_ema: bool = True,
    device: str = "neuron",
) -> None:
    """Initialize the global pipeline."""
    global _pipeline
    
    config = StreamingConfig(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        vae_path=vae_path,
        use_ema=use_ema,
        device=device,
    )
    _pipeline = StreamingInferencePipeline(config)
    print(f"[App] Pipeline initialized with config: {config_path}")


# =============================================================================
# Generation Functions for Gradio
# =============================================================================

def generate_streaming_mode(
    prompt: str,
    num_frames: int,
    seed: int,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[List[Image.Image], str]:
    """Generate video in streaming mode - yields frames progressively.
    
    Returns:
        Tuple of (list of frames as gallery, status message)
    """
    if not prompt.strip():
        return [], "⚠️ Please enter a prompt"
    
    with _generation_lock:
        try:
            import torch
            torch.manual_seed(seed)
            
            pipe = get_pipeline()
            frames = []
            start_time = time.time()
            
            progress(0, desc="Starting generation...")
            
            def progress_callback(current, total):
                progress(current / total, desc=f"Frame {current}/{total}")
            
            for frame_idx, frame in pipe.generate_streaming(
                prompt=prompt,
                num_frames=num_frames,
                progress_callback=progress_callback,
            ):
                frames.append(frame)
                # Yield intermediate results
                yield frames, f"🎬 Generated {len(frames)}/{num_frames} frames..."
            
            elapsed = time.time() - start_time
            fps = len(frames) / elapsed if elapsed > 0 else 0
            
            yield frames, f"✅ Generated {len(frames)} frames in {elapsed:.1f}s ({fps:.2f} fps)"
            
        except Exception as e:
            yield [], f"❌ Error: {str(e)}"


def generate_quality_mode(
    prompt: str,
    num_frames: int,
    seed: int,
    chunk_size: int,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], str]:
    """Generate video in quality mode - returns video file path.
    
    Returns:
        Tuple of (video file path, status message)
    """
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt"
    
    with _generation_lock:
        try:
            import torch
            import imageio
            
            torch.manual_seed(seed)
            
            pipe = get_pipeline()
            start_time = time.time()
            
            progress(0, desc="Starting generation...")
            
            # Collect all frames first
            all_frames = []
            
            def progress_callback(current, total):
                progress(current / total * 0.8, desc=f"Generating frame {current}/{total}")
            
            for frame_idx, frame in pipe.generate_streaming(
                prompt=prompt,
                num_frames=num_frames,
                progress_callback=progress_callback,
            ):
                all_frames.append(np.array(frame))
            
            # Encode as video
            progress(0.9, desc="Encoding video...")
            
            output_path = tempfile.mktemp(suffix=".mp4")
            imageio.mimwrite(output_path, all_frames, fps=pipe.config.fps)
            
            elapsed = time.time() - start_time
            progress(1.0, desc="Complete!")
            
            return output_path, f"✅ Generated {len(all_frames)} frames in {elapsed:.1f}s"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"


def generate_comparison_mode(
    prompt: str,
    num_frames: int,
    seed: int,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[List[Image.Image], Optional[str], str]:
    """Generate in both modes for comparison.
    
    Returns:
        Tuple of (gallery frames, video path, status)
    """
    if not prompt.strip():
        return [], None, "⚠️ Please enter a prompt"
    
    with _generation_lock:
        try:
            import torch
            import imageio
            
            torch.manual_seed(seed)
            
            pipe = get_pipeline()
            start_time = time.time()
            
            frames = []
            
            def progress_callback(current, total):
                progress(current / total * 0.8, desc=f"Frame {current}/{total}")
            
            # Generate frames
            for frame_idx, frame in pipe.generate_streaming(
                prompt=prompt,
                num_frames=num_frames,
                progress_callback=progress_callback,
            ):
                frames.append(frame)
            
            # Encode video
            progress(0.9, desc="Encoding video...")
            output_path = tempfile.mktemp(suffix=".mp4")
            imageio.mimwrite(
                output_path, 
                [np.array(f) for f in frames], 
                fps=pipe.config.fps
            )
            
            elapsed = time.time() - start_time
            progress(1.0)
            
            return frames, output_path, f"✅ Generated {len(frames)} frames in {elapsed:.1f}s"
            
        except Exception as e:
            return [], None, f"❌ Error: {str(e)}"


# =============================================================================
# Gradio UI
# =============================================================================

def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    
    css = """
    .streaming-gallery img {
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }
    .quality-video video {
        border: 2px solid #2196F3;
        border-radius: 8px;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
    }
    """
    
    with gr.Blocks(
        title="🎬 Streaming Video Generation",
        theme=gr.themes.Soft(),
        css=css,
    ) as demo:
        
        gr.Markdown("""
        # 🎬 Streaming Video Generation
        
        Generate videos with your text prompts! Choose between two modes:
        - **🚀 Streaming Mode**: See frames as they're generated (lower latency)
        - **🎥 Quality Mode**: Get a properly encoded video file (better quality)
        - **⚖️ Comparison Mode**: Run both and compare side-by-side
        """)
        
        # Common inputs
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="📝 Prompt",
                    placeholder="A cat walking on the beach at sunset...",
                    lines=2,
                )
            with gr.Column(scale=1):
                num_frames_input = gr.Slider(
                    minimum=9,
                    maximum=81,
                    value=21,
                    step=3,
                    label="🎞️ Number of Frames",
                )
                seed_input = gr.Number(
                    value=42,
                    label="🎲 Seed",
                    precision=0,
                )
        
        # Tabbed interface for different modes
        with gr.Tabs():
            
            # Tab 1: Streaming Mode
            with gr.TabItem("🚀 Streaming Mode"):
                gr.Markdown("""
                **Lower latency** - See frames appear as they're generated.
                Great for previewing and interactive exploration.
                """)
                
                with gr.Row():
                    stream_btn = gr.Button(
                        "🚀 Generate (Streaming)",
                        variant="primary",
                        size="lg",
                    )
                
                stream_gallery = gr.Gallery(
                    label="Generated Frames",
                    columns=7,
                    rows=3,
                    object_fit="contain",
                    height=400,
                    elem_classes=["streaming-gallery"],
                )
                stream_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-box"],
                )
                
                stream_btn.click(
                    fn=generate_streaming_mode,
                    inputs=[prompt_input, num_frames_input, seed_input],
                    outputs=[stream_gallery, stream_status],
                )
            
            # Tab 2: Quality Mode
            with gr.TabItem("🎥 Quality Mode"):
                gr.Markdown("""
                **Better quality** - Properly encoded video with compression.
                Better for final output and sharing.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        chunk_size_input = gr.Slider(
                            minimum=3,
                            maximum=21,
                            value=6,
                            step=3,
                            label="Chunk Size (frames per segment)",
                        )
                    with gr.Column(scale=2):
                        quality_btn = gr.Button(
                            "🎥 Generate (Quality)",
                            variant="primary",
                            size="lg",
                        )
                
                quality_video = gr.Video(
                    label="Generated Video",
                    height=400,
                    elem_classes=["quality-video"],
                )
                quality_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-box"],
                )
                
                quality_btn.click(
                    fn=generate_quality_mode,
                    inputs=[prompt_input, num_frames_input, seed_input, chunk_size_input],
                    outputs=[quality_video, quality_status],
                )
            
            # Tab 3: Comparison Mode
            with gr.TabItem("⚖️ Compare Both"):
                gr.Markdown("""
                **Compare side-by-side** - See both streaming frames and encoded video.
                Use this to evaluate quality vs latency tradeoffs.
                """)
                
                compare_btn = gr.Button(
                    "⚖️ Generate Both",
                    variant="primary",
                    size="lg",
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🚀 Streaming (Frame Gallery)")
                        compare_gallery = gr.Gallery(
                            label="Frames",
                            columns=5,
                            rows=2,
                            object_fit="contain",
                            height=300,
                        )
                    with gr.Column():
                        gr.Markdown("### 🎥 Quality (Encoded Video)")
                        compare_video = gr.Video(
                            label="Video",
                            height=300,
                        )
                
                compare_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-box"],
                )
                
                compare_btn.click(
                    fn=generate_comparison_mode,
                    inputs=[prompt_input, num_frames_input, seed_input],
                    outputs=[compare_gallery, compare_video, compare_status],
                )
        
        # Example prompts
        gr.Markdown("### 💡 Example Prompts")
        gr.Examples(
            examples=[
                ["A cat walking on the beach at sunset, cinematic"],
                ["A rocket launching into space with smoke trails"],
                ["Time-lapse of a flower blooming in a garden"],
                ["A robot dancing in a futuristic city"],
                ["Ocean waves crashing on rocks, slow motion"],
            ],
            inputs=prompt_input,
        )
        
        # Info section
        with gr.Accordion("ℹ️ About", open=False):
            gr.Markdown("""
            ## How it works
            
            This app uses a **Rolling Forcing** diffusion model to generate video
            autoregressively. The model generates frames in blocks, which enables
            streaming output before the full video is complete.
            
            ### Streaming Mode
            - Frames are decoded and displayed as soon as they're generated
            - Lower latency to first frame
            - Individual frames as PNG/JPEG
            
            ### Quality Mode  
            - Full video is encoded with proper video codec (H.264)
            - Better compression and quality
            - Playable in any video player
            
            ### Technical Details
            - Model: Wan2.1-T2V-1.3B with Rolling Forcing
            - Backend: AWS Neuron (Trainium/Inferentia)
            - VAE: 16-channel latent space
            """)
    
    return demo


# =============================================================================
# Mock Pipeline for Testing
# =============================================================================

class MockStreamingPipeline:
    """Mock pipeline for testing the UI without actual model."""
    
    def __init__(self):
        self.config = type('Config', (), {'fps': 16, 'num_frames': 21})()
    
    def generate_streaming(self, prompt, num_frames=21, progress_callback=None):
        """Generate mock frames."""
        import time
        
        for i in range(num_frames):
            # Create a gradient image with frame number
            img = np.zeros((480, 832, 3), dtype=np.uint8)
            
            # Gradient background
            for y in range(480):
                for x in range(832):
                    img[y, x, 0] = int(255 * (i / num_frames))  # Red increases
                    img[y, x, 1] = int(255 * (x / 832))  # Green gradient
                    img[y, x, 2] = int(255 * (y / 480))  # Blue gradient
            
            # Simulate generation time
            time.sleep(0.5)
            
            if progress_callback:
                progress_callback(i + 1, num_frames)
            
            yield i, Image.fromarray(img)


def init_mock_pipeline():
    """Initialize mock pipeline for testing."""
    global _pipeline
    _pipeline = MockStreamingPipeline()
    print("[App] Mock pipeline initialized for testing")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Streaming Video Generation App")
    parser.add_argument("--config", type=str, help="Path to model config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--model_path", type=str, default="wan_models/Wan2.1-T2V-1.3B")
    parser.add_argument("--vae_path", type=str, default="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights")
    parser.add_argument("--device", type=str, default="neuron", choices=["neuron", "cuda", "cpu"])
    parser.add_argument("--mock", action="store_true", help="Use mock pipeline for testing UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public share link")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    if args.mock:
        init_mock_pipeline()
    elif args.config:
        init_pipeline(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            model_path=args.model_path,
            vae_path=args.vae_path,
            use_ema=args.use_ema,
            device=args.device,
        )
    else:
        print("⚠️ No config provided, using mock pipeline for demo")
        init_mock_pipeline()
    
    # Create and launch demo
    demo = create_demo()
    demo.queue()  # Enable queuing for streaming
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
