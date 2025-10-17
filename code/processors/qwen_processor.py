#!/usr/bin/env python3
"""
Qwen3-VL processor for table extraction using Qwen3-VL-4B-Instruct model.
"""

import gc
import re
import threading
import time

# Default prompt from openrouter_processor
TABLE_EXTRACTION_PROMPT = """OCR the table and convert it to HTML. Use simple HTML tags, and specify colspan and rowspan where it is greater than 1. No additional styling required. Use simple HTML tags for formulas. The output should begin with <table> and end with </table>. Output your content in ```html ``` tags."""


class Qwen3VLProcessor:
    """
    Qwen3-VL processor using Qwen3-VL-4B-Instruct model for table extraction.
    """

    # Class-level shared models and lock for thread-safe singleton pattern
    _shared_models = {}  # {model_name: model_instance}
    _shared_processors = {}  # {model_name: processor_instance}
    _model_lock = threading.Lock()

    def __init__(self, model):
        # Map config model names to actual HuggingFace model IDs
        model_mapping = {
            "qwen3-vl-4b-instruct": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-vl-4b-thinking": "Qwen/Qwen3-VL-4B-Thinking",
        }

        # Use mapping if available, otherwise use the provided model name
        self.model = (
            model_mapping.get(model, model) if model else "Qwen/Qwen3-VL-4B-Instruct"
        )
        self.is_thinking_model = "thinking" in model.lower() if model else False
        # Point to shared instances (will be initialized lazily)
        self.processor = None
        self.qwen_model = None

    def parse_html(self, string):
        """Extract HTML content from markdown code blocks and thinking tags"""
        # Remove markdown code blocks
        result = string.replace("```html", "").replace("```", "").strip()

        # For thinking models, remove <think> reasoning tags if present
        # Thinking models output: <think>reasoning</think> <output>answer</output>
        if "<think>" in result and "</think>" in result:
            # Find and remove thinking tags and their content
            result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL)

        # Also handle <output> tags if present
        if "<output>" in result and "</output>" in result:
            match = re.search(r"<output>(.*?)</output>", result, re.DOTALL)
            if match:
                result = match.group(1).strip()

        # Extract only the HTML table content (handle thinking models that output plain text reasoning)
        # Look for <table> and extract everything from <table> to </table>
        if "<table>" in result and "</table>" in result:
            table_match = re.search(r"<table>.*?</table>", result, re.DOTALL)
            if table_match:
                result = table_match.group(0)

        return result.strip()

    def initialize_model(self):
        """Lazy load the model to avoid loading it if not needed (thread-safe singleton)"""
        # Check if already initialized for this instance
        if self.qwen_model is not None:
            return

        # Use lock to ensure thread-safe initialization
        with self._model_lock:
            # Double-check if another thread already initialized it
            if self.model in self._shared_models:
                self.qwen_model = self._shared_models[self.model]
                self.processor = self._shared_processors[self.model]
                print(f"♻️  Reusing already-loaded Qwen3-VL model: {self.model}")
                return

            print(f"📦 Loading Qwen3-VL model (first time): {self.model}")

            try:
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            except ImportError as e:
                error_msg = (
                    f"ERROR: transformers library not installed or incompatible: {e}"
                )
                print(error_msg)
                raise ImportError(error_msg)

            # Load model without device_map to avoid meta tensor issues
            import torch

            # Load model to CPU first, then move to device
            # This avoids the meta tensor issue with device_map="auto"
            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model,
                dtype=torch.bfloat16,
            )

            # Move model to CUDA if available
            if torch.cuda.is_available():
                qwen_model = qwen_model.to("cuda")
                print("Model loaded to CUDA device")
            else:
                print("Model loaded to CPU")

            # Load processor
            processor = AutoProcessor.from_pretrained(self.model)
            print("✅ Qwen3-VL model loaded successfully and cached for reuse")

            # Store in shared class variables
            self._shared_models[self.model] = qwen_model
            self._shared_processors[self.model] = processor

            # Also store in instance variables
            self.qwen_model = qwen_model
            self.processor = processor

    def process_document(self, file_path, prompt=None):
        print(f"Processing document with {self.model}: {file_path}")

        max_retries = 3
        retry_count = 0
        backoff_time = 2

        # Initialize variables for cleanup - moved to top level
        pages = None
        messages = None
        inputs = None
        generated_ids = None

        while retry_count < max_retries:
            try:
                # Lazy imports
                try:
                    from pdf2image import convert_from_path
                    from PIL import Image
                except ImportError as e:
                    error_msg = f"ERROR: Required library not installed: {e}"
                    print(error_msg)
                    return error_msg

                # Initialize model if not already loaded
                self.initialize_model()

                # Handle both PDF and image files
                pages_list = []
                if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    # For image files, load directly
                    pages_list = [Image.open(file_path)]
                else:
                    # For PDF files, convert to images first
                    pages_list = convert_from_path(
                        file_path, dpi=300, thread_count=4, fmt="jpg"
                    )

                pages = pages_list  # Assign to pages variable for cleanup

                if not pages:
                    print("No pages extracted from document")
                    return "<table></table>"

                # Use custom prompt if provided, otherwise use default
                prompt_text = prompt if prompt else TABLE_EXTRACTION_PROMPT

                # Build messages for each page
                # For multiple pages, we'll process them together
                content = [{"type": "text", "text": prompt_text}]

                for page in pages:
                    # Convert to RGB if needed
                    if page.mode in ("RGBA", "LA", "P"):
                        page = page.convert("RGB")

                    # Resize image if too large to reduce token count
                    # Qwen3-VL uses dynamic resolution, so large images = many vision tokens
                    max_dimension = 1280  # Balance between quality and speed
                    if max(page.size) > max_dimension:
                        ratio = max_dimension / max(page.size)
                        new_size = (int(page.width * ratio), int(page.height * ratio))
                        page = page.resize(new_size, Image.Resampling.LANCZOS)
                        print(
                            f"  Resized image from {pages_list[0].size} to {new_size} to reduce token count"
                        )

                    content.append({"type": "image", "image": page})

                messages = [{"role": "user", "content": content}]

                print(f"Preparing inputs for {self.model}...")
                # Preparation for inference
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.qwen_model.device)
                print(f"Input prepared with {inputs.input_ids.shape[1]} tokens")

                print(f"Generating output with {self.model}...")
                # Inference: Generation of the output with optimized parameters
                # Thinking models need more tokens for chain-of-thought reasoning
                max_tokens = 2048 if self.is_thinking_model else 1024

                generated_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Greedy decoding for consistency and speed
                    num_beams=1,  # No beam search for speed
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                )
                print(
                    f"Generation completed, decoding {generated_ids.shape[1]} tokens..."
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                print("Decoding completed")

                # Extract the first result
                if output_text and len(output_text) > 0:
                    table_html = output_text[0]
                    table_html = self.parse_html(table_html)

                    # Clean up before return
                    try:
                        if pages:
                            for page in pages:
                                if hasattr(page, "close"):
                                    page.close()
                        del pages, messages, inputs, generated_ids
                    except Exception:
                        pass
                    gc.collect()

                    return table_html
                else:
                    print(f"No output generated by {self.model}")
                    return "<table></table>"

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(
                        f"ERROR processing document with {self.model} after {max_retries} retries: {str(e)}"
                    )
                    import traceback

                    print(traceback.format_exc())

                    # Clean up before raising
                    try:
                        if pages is not None:
                            for page in pages:
                                if hasattr(page, "close"):
                                    page.close()
                        if messages is not None:
                            del messages
                        if inputs is not None:
                            del inputs
                        if generated_ids is not None:
                            del generated_ids
                        if pages is not None:
                            del pages
                    except Exception:
                        pass
                    gc.collect()

                    raise Exception(
                        f"{self.model} failed after {max_retries} retries: {str(e)}"
                    )
                else:
                    print(
                        f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds..."
                    )

                    # Clean up failed attempt objects
                    try:
                        if pages:
                            for page in pages:
                                if hasattr(page, "close"):
                                    page.close()
                        del pages, messages, inputs, generated_ids
                    except Exception:
                        pass
                    gc.collect()

                    time.sleep(backoff_time)
                    backoff_time *= 2

        # Final cleanup if all retries failed
        try:
            if pages:
                for page in pages:
                    if hasattr(page, "close"):
                        page.close()
            del pages, messages, inputs, generated_ids
        except Exception:
            pass
        gc.collect()
        return "<table></table>"
