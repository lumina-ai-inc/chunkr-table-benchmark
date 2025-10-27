#!/usr/bin/env python3
"""
Chandra processor supporting vLLM inference via OpenAI API.
Standard HTML table extraction without OTSL conversion.
"""

import base64
import gc
import os
import tempfile
import time
import re
from openai import OpenAI
from PIL import Image


class ChandraProcessor:
    def __init__(self, model="chandra"):
        self.model = model
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="",
        )
        print("✅ Initialized ChandraProcessor in vLLM mode.")

    def process_document(self, file_path):
        print(f"Processing document with vLLM server via OpenAI API: {file_path}")

        max_retries = 3
        retry_count = 0
        backoff_time = 2
        response = None
        temp_file = None
        processing_path = file_path

        try:
            img = Image.open(file_path)
            original_size = img.size
            max_dimension = 1280
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                temp_fd, temp_file = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                img_resized.save(temp_file, format="PNG", optimize=False)
                print(f"  Resized {original_size} → {new_size}")
                processing_path = temp_file
                img_resized.close()
            img.close()
        except Exception as e:
            print(f"⚠️ Error resizing image: {e}")
            processing_path = file_path

        while retry_count < max_retries:
            try:
                with open(processing_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")

                image_format = "png"
                if processing_path.lower().endswith((".jpg", ".jpeg")):
                    image_format = "jpeg"

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
                                {"type": "text", "text": "table"},
                            ],
                        }
                    ],
                    max_tokens=8192,
                )
                print("✅ Prediction completed")

                if not response or not response.choices:
                    return "<table></table>"

                html = self.extract_html_from_response(response)
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
                gc.collect()
                return html

            except Exception as e:
                retry_count += 1
                print(f"Error {retry_count}/{max_retries}: {e}")
                if retry_count >= max_retries:
                    raise
                time.sleep(backoff_time)
                backoff_time *= 2

        return "<table></table>"

    def extract_html_from_response(self, response):
        """
        Extract HTML table from model response.
        """
        try:
            content = response.choices[0].message.content
            if not content:
                return "<table></table>"

            if "<table" in content.lower():
                tables = re.findall(r"<table.*?</table>", content, re.IGNORECASE | re.DOTALL)
                if tables:
                    return "\n".join(tables)

            print("⚠️ No table content detected.")
            return "<table></table>"

        except Exception as e:
            print(f"Error extracting HTML from response: {e}")
            return "<table></table>"

