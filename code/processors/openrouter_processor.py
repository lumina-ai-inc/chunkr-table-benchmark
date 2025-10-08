#!/usr/bin/env python3
"""
OpenRouter processor for table extraction using various LLM models.
"""

import base64
import gc
import io
import os
import time

# Default prompt
TABLE_EXTRACTION_PROMPT = """OCR the table and convert it to HTML. Use simple HTML tags, and specify colspan and rowspan where it is greater than 1. No additional styling required. Use simple HTML tags for formulas. The output should begin with <table> and end with </table>. Output your content in ```html ``` tags."""


class OpenRouterTableProcessor:
    def __init__(self, model):
        # Always use the OpenRouter key when routing through OpenRouter, even for openai/* models
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model

    def parse_html(self, string):
        return string.replace("```html", "").replace("```", "")

    def process_document(self, file_path, prompt=None):
        print(f"Processing document with OpenRouter: {file_path}")

        # Move imports outside the processing loop to avoid repeated import overhead
        try:
            from openai import OpenAI
            from pdf2image import convert_from_path
            from PIL import Image
            import io
        except ImportError as e:
            print(f"ERROR: Required library not installed: {e}")
            return None

        # Initialize variables to ensure they exist for cleanup
        pages = []
        base64_images = []
        messages = []
        response = None
        client = None

        try:
            # Handle both PDF and image files
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                # For image files, load directly
                pages = [Image.open(file_path)]
            else:
                # For PDF files, convert to images first
                pages = convert_from_path(file_path, dpi=300, thread_count=4, fmt="jpg")

            if not pages:
                return None

            max_retries = 3
            retry_count = 0
            backoff_time = 1
            
            while retry_count < max_retries:
                try:
                    print("Processing document with OpenRouter API...")

                    # Prepare OpenAI-style messages
                    # Use custom prompt if provided, otherwise use default table extraction
                    if prompt:
                        # Custom prompt provided - use it as the user message
                        messages = [
                            {
                                "role": "system",
                                "content": "You are an expert at analyzing documents and extracting table data. Extract all tables from the provided images and return them in clean HTML table format. If no tables are found, return an empty table."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ]
                    else:
                        # Default table extraction prompt
                        messages = [
                            {
                                "role": "system",
                                "content": "You are an expert at analyzing documents and extracting table data. Extract all tables from the provided images and return them in clean HTML table format. If no tables are found, return an empty table."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": TABLE_EXTRACTION_PROMPT
                                    }
                                ]
                            }
                        ]

                    # Add each page as a separate image
                    base64_images = []
                    for page in pages:
                        # Convert PIL image to bytes - ensure RGB mode for JPEG compatibility
                        if page.mode in ('RGBA', 'LA', 'P'):
                            # Convert RGBA, LA, or P mode to RGB for JPEG compatibility
                            page = page.convert('RGB')
                        
                        # Scale image by 2x for better quality, but cap max dimension to 8000px
                        original_size = page.size
                        target_w = original_size[0] * 2
                        target_h = original_size[1] * 2
                        max_dim = max(target_w, target_h)
                        if max_dim > 8000:
                            scale = 8000 / float(max_dim)
                            new_size = (int(target_w * scale), int(target_h * scale))
                        else:
                            new_size = (target_w, target_h)
                        # Handle PIL version compatibility for LANCZOS
                        try:
                            resample = Image.Resampling.LANCZOS  # Newer PIL versions
                        except AttributeError:
                            resample = Image.LANCZOS  # Older PIL versions
                        page = page.resize(new_size, resample)
                        print(f"  Scaled image from {original_size} to {new_size} for OpenRouter")
                        
                        # Convert to JPEG with iterative size reduction to stay under 5MB
                        img_byte_arr = io.BytesIO()
                        quality = 95
                        max_size_mb = 4.8  # Conservative limit
                        
                        while quality >= 70:
                            img_byte_arr.seek(0)
                            img_byte_arr.truncate(0)
                            
                            # Save with current quality
                            page.save(img_byte_arr, format="JPEG", quality=quality, optimize=True)
                            
                            # Check size
                            size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
                            
                            if size_mb <= max_size_mb:
                                break
                                
                            # Reduce quality for next iteration
                            quality -= 5
                        
                        # If still too large, reduce dimensions
                        if quality < 70:
                            current_size = page.size
                            reduction_factor = (max_size_mb / size_mb) ** 0.5
                            new_width = int(current_size[0] * reduction_factor)
                            new_height = int(current_size[1] * reduction_factor)
                            page = page.resize((new_width, new_height), resample)
                            
                            # Re-save with reduced dimensions
                            img_byte_arr = io.BytesIO()
                            page.save(img_byte_arr, format="JPEG", quality=85, optimize=True)
                            print(f"  Reduced to {new_width}x{new_height} to fit {max_size_mb}MB limit")

                        # Add image to messages
                        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                        base64_images.append(base64_image)
                        messages[1]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        )
                        # Clean up intermediate objects immediately
                        del img_byte_arr, base64_image

                    # Always route through OpenRouter endpoint
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=self.api_key,
                    )

                    # Always pass provider-prefixed model name for OpenRouter
                    completion_kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.0,
                    }

                    response = client.chat.completions.create(**completion_kwargs)

                    table_html = response.choices[0].message.content
                    table_html = self.parse_html(table_html)

                    return table_html

                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(
                            f"ERROR processing document with OpenRouter model {self.model} after {max_retries} retries: {str(e)}"
                        )
                        import traceback

                        print(traceback.format_exc())
                        raise Exception(f"OpenRouterProcessor API failed after {max_retries} retries: {str(e)}")
                    else:
                        print(
                            f"Error on attempt {retry_count}/{max_retries} with OpenRouter model {self.model}: {str(e)}. Retrying in {backoff_time} seconds..."
                        )
                        time.sleep(backoff_time)
                        backoff_time *= 2

        finally:
            # Enhanced cleanup to release ALL memory
            try:
                # Close all PIL Image objects
                if pages:
                    for page in pages:
                        if hasattr(page, 'close'):
                            page.close()
            except Exception:
                pass
            
            try:
                # Clear large data structures
                if base64_images:
                    base64_images.clear()
                if messages:
                    messages.clear()
                
                # Delete all variables to ensure they're freed
                del pages, messages, base64_images, response, client
            except Exception:
                pass
            
            # Force garbage collection
            gc.collect()