#!/usr/bin/env python3
"""
PaddleOCR processor for table extraction using vLLM server via OpenAI API.
"""

import base64
import gc
import os
import tempfile
import time

from openai import OpenAI
from PIL import Image


class PaddleOCRVLProcessor:
    """
    PaddleOCR processor using vLLM server via OpenAI API.
    Simple implementation with direct API calls.
    """

    def __init__(self, model):
        self.model = model or "PaddleOCR-VL-0.9B"
        self.client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key="",  # vLLM doesn't require a real API key
        )

    def process_document(self, file_path):
        print(f"Processing document with vLLM server via OpenAI API: {file_path}")

        max_retries = 3
        retry_count = 0
        backoff_time = 2

        # Initialize variables for cleanup
        response = None
        temp_file = None
        processing_path = file_path

        # Resize image if needed
        try:
            img = Image.open(file_path)
            original_size = img.size

            # Resize if too large to reduce processing load
            max_dimension = 1280  # Balance between quality and speed
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

                # Save to temporary file (will persist across retries)
                temp_fd, temp_file = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                img_resized.save(temp_file, format="PNG", optimize=False)

                print(
                    f"  Resized image from {original_size} to {new_size} to reduce processing load"
                )
                processing_path = temp_file
                img_resized.close()

            img.close()
        except Exception as e:
            print(f"Error resizing image: {e}")
            # Continue with original file if resize fails
            processing_path = file_path

        print(f"Processing image: {processing_path}")

        while retry_count < max_retries:
            try:
                # Encode image to base64 for OpenAI API
                with open(processing_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")

                # Determine image format
                image_format = "png"
                if processing_path.lower().endswith((".jpg", ".jpeg")):
                    image_format = "jpeg"
                elif processing_path.lower().endswith(".webp"):
                    image_format = "webp"

                # Run document parsing using OpenAI vision API
                print("🔄 Sending request to vLLM server...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{image_format};base64,{image_data}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "table",
                                },
                            ],
                        }
                    ],
                    max_tokens=10000,
                )
                print("✅ Prediction completed")

                if not response or not response.choices:
                    print("No output from vLLM server.")
                    # Clean up before return
                    try:
                        del response
                    except Exception:
                        pass
                    gc.collect()
                    return "<table></table>"

                # Extract HTML from OpenAI API response
                table_html = self.extract_html_from_response(response)

                # Clean up large objects and temp file before return
                try:
                    del response
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception:
                    pass
                gc.collect()

                return table_html

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(
                        f"ERROR processing document with vLLM after {max_retries} retries: {str(e)}"
                    )
                    import traceback

                    print(traceback.format_exc())

                    # Clean up before raising
                    try:
                        if response is not None:
                            del response
                        if temp_file and os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception:
                        pass
                    gc.collect()

                    raise Exception(
                        f"PaddleOCRVLProcessor failed after {max_retries} retries: {str(e)}"
                    )
                else:
                    print(
                        f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds..."
                    )

                    # Clean up failed attempt objects (but keep temp file for retry)
                    try:
                        if response is not None:
                            del response
                    except Exception:
                        pass
                    gc.collect()

                    time.sleep(backoff_time)
                    backoff_time *= 2

        # Final cleanup if all retries failed
        try:
            if response is not None:
                del response
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception:
            pass
        gc.collect()
        return "<table></table>"

    def extract_html_from_response(self, response):
        """
        Extract HTML table from OpenAI API response.
        The response contains the table in the message content.
        """
        if not response or not response.choices:
            return "<table></table>"

        try:
            # Get the content from the response
            content = response.choices[0].message.content

            if not content:
                return "<table></table>"

            # The model should return HTML directly based on our prompt
            # Look for <table> tags in the content
            if "<table>" in content.lower():
                # Extract just the table content
                import re

                # Find all table tags (case insensitive)
                tables = re.findall(
                    r"<table>.*?</table>", content, re.IGNORECASE | re.DOTALL
                )
                if tables:
                    return "\n".join(tables)

            # If no HTML table found, check for custom PaddleOCR format
            # Format uses tags like <fcel>, <xcel>, <nl>, <ecel>, etc.
            if "<fcel>" in content or "<xcel>" in content or "<nl>" in content:
                print("⚠️  Detected custom PaddleOCR format, converting to HTML...")
                final_content = self.convert_paddle_format_to_html(content)
                print(final_content)
                if final_content:
                    return final_content
                else:
                    return "<table></table>"

            print("⚠️  Could not extract table from response content")
            return "<table></table>"

        except Exception as e:
            print(f"Error extracting HTML from response: {e}")
            import traceback

            print(traceback.format_exc())
            return "<table></table>"

    def convert_paddle_format_to_html(self, content):
        """
        Convert PaddleOCR's OTSL (Open Table Structure Language) format to HTML.

        Uses the official PaddleX converter which properly handles:
        - Cell spans (rowspan, colspan)
        - OTSL tags: <fcel>, <xcel>, <lcel>, <ucel>, <ecel>, <nl>
        - Complex table structures

        This ensures identical output to PaddleOCR CLI.
        """
        try:
            from paddlex.inference.pipelines.paddleocr_vl.uilts import (
                convert_otsl_to_html,
            )

            # Use the official OTSL to HTML converter
            html = convert_otsl_to_html(content)

            # Return the converted HTML or fallback if empty
            return html if html else "<table></table>"

        except ImportError as e:
            print(f"⚠️  Failed to import PaddleX OTSL converter: {e}")
            print("Falling back to basic conversion...")
            return self._fallback_otsl_conversion(content)
        except Exception as e:
            print(f"⚠️  Error converting OTSL to HTML: {e}")
            import traceback

            print(traceback.format_exc())
            print("Falling back to basic conversion...")
            return self._fallback_otsl_conversion(content)

    def _fallback_otsl_conversion(self, content):
        """
        Fallback OTSL to HTML conversion if official converter fails.
        Basic conversion without rowspan/colspan support.
        """
        import re

        html = "<table>"

        # Split by newline markers to get rows
        rows = content.split("<nl>")

        for row in rows:
            row = row.strip()
            if not row:
                continue

            html += "<tr>"

            # Pattern: Find content between cell tags
            cell_pattern = r"<(?:fcel|xcel|lcel|ucel|ecel)>(.*?)(?=<(?:fcel|xcel|lcel|ucel|ecel)>|$)"
            matches = re.findall(cell_pattern, row, re.DOTALL)

            for cell_content in matches:
                cell_text = cell_content.strip()
                if cell_text:
                    # Escape HTML special characters
                    cell_text = (
                        cell_text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                    )
                    html += f"<td>{cell_text}</td>"

            html += "</tr>"

        html += "</table>"
        return html
