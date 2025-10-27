#!/usr/bin/env python3
"""
DeepseekOCR processor for table extraction using direct vLLM engine.
Based on the DeepSeek-OCR reference implementation.
"""

import gc
import os
import sys
import time
import re

from PIL import Image, ImageOps

# Add DeepSeek-OCR path to system path for imports
DEEPSEEK_OCR_PATH = "/home/ubuntu/projects/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
if DEEPSEEK_OCR_PATH not in sys.path:
    sys.path.insert(0, DEEPSEEK_OCR_PATH)

# user older version of vllm
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_PLUGINS'] = ''  # Disable all plugins to avoid PaddleX conflicts
os.environ['TORCH_COMPILE_DISABLE'] = '1'  # Avoid torch compile issues

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Import DeepSeek-OCR components
try:
    from deepseek_ocr import DeepseekOCRForCausalLM
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from process.image_process import DeepseekOCRProcessor as ImageProcessor
except ImportError as e:
    print(f"Error importing DeepSeek-OCR modules: {e}")
    print(f"Make sure DeepSeek-OCR is installed at: {DEEPSEEK_OCR_PATH}")
    raise

# Register the custom model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


class DeepseekOCRProcessor:
    """
    DeepseekOCR processor using direct vLLM engine.
    Uses the official DeepSeek-OCR implementation with custom processors.
    """

    def __init__(self, model):
        self.model_path = model or "deepseek-ai/DeepSeek-OCR"
        self.llm = None
        self.image_processor = None
        self.sampling_params = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the vLLM engine and processors."""
        print(f"🔄 Initializing DeepSeek-OCR model from: {self.model_path}")
        
        try:
            # Initialize vLLM engine with DeepSeek-OCR model
            self.llm = LLM(
                model=self.model_path,
                hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
                block_size=256,
                max_model_len=8192,
                enforce_eager=False,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,
                swap_space=0,
                max_num_seqs=1,  # Process one at a time for benchmarking
            )
            
            # Initialize image processor
            self.image_processor = ImageProcessor()
            
            # Create logits processors for better table handling
            # NoRepeatNGramLogitsProcessor prevents repetitive patterns
            # whitelist_token_ids: {128821, 128822} are <td> and </td> tokens
            logits_processors = [
                NoRepeatNGramLogitsProcessor(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822}
                )
            ]
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=12288,  # Increased for longer tables
                logits_processors=logits_processors,
                skip_special_tokens=False,
                stop_token_ids=None,  # Don't stop early
            )
            
            print("✅ DeepSeek-OCR model initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing DeepSeek-OCR model: {e}")
            import traceback
            print(traceback.format_exc())
            raise

    def _load_image(self, image_path):
        """Load and preprocess image with EXIF orientation correction."""
        try:
            image = Image.open(image_path)
            # Correct image orientation based on EXIF data
            corrected_image = ImageOps.exif_transpose(image)
            return corrected_image
        except Exception as e:
            print(f"Error loading image with EXIF correction: {e}")
            try:
                return Image.open(image_path)
            except Exception as e2:
                print(f"Error loading image: {e2}")
                return None

    def process_document(self, file_path):
        print(f"Processing document with DeepSeek-OCR direct vLLM: {file_path}")

        max_retries = 3
        retry_count = 0
        backoff_time = 2

        # Load and optionally resize image
        try:
            img = self._load_image(file_path)
            if img is None:
                return "<table></table>"
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_size = img.size

            # Resize if too large to reduce processing load
            # DeepSeek-OCR handles cropping internally, but we can reduce initial size
            max_dimension = 1280
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"  Resized image from {original_size} to {new_size}")

        except Exception as e:
            print(f"Error loading/resizing image: {e}")
            import traceback
            print(traceback.format_exc())
            return "<table></table>"

        print(f"Processing image with size: {img.size}")

        while retry_count < max_retries:
            try:
                # Tokenize image using DeepSeek-OCR's image processor
                # cropping=True enables dynamic cropping for better OCR quality
                print("🔄 Tokenizing image with DeepSeek-OCR processor...")
                image_features = self.image_processor.tokenize_with_images(
                    images=[img],
                    bos=True,
                    eos=True,
                    cropping=True  # Enable dynamic cropping for better quality
                )
                
                # Prepare the prompt for table extraction
                prompt = '<image>\n<|grounding|>Convert this table to HTML.'
                
                # Create the request
                request = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_features}
                }
                
                # Generate output using vLLM
                print("🔄 Generating output with vLLM...")
                outputs = self.llm.generate(
                    request,
                    self.sampling_params
                )
                
                print("✅ Prediction completed")

                if not outputs or len(outputs) == 0:
                    print("No output from vLLM.")
                    gc.collect()
                    return "<table></table>"

                # Extract the generated text
                generated_text = outputs[0].outputs[0].text
                
                # Extract HTML table from the generated text
                table_html = self._extract_html_from_text(generated_text)

                # Clean up
                del outputs
                del image_features
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

                    gc.collect()
                    raise Exception(
                        f"DeepseekOCRProcessor failed after {max_retries} retries: {str(e)}"
                    )
                else:
                    print(
                        f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds..."
                    )
                    gc.collect()
                    time.sleep(backoff_time)
                    backoff_time *= 2

        # Final cleanup if all retries failed
        gc.collect()
        return "<table></table>"

    def _re_match(self, text):
        """
        Official re_match function from DeepSeek-OCR to identify and extract special tokens.
        Pattern matches: <|ref|>label<|/ref|><|det|>coordinates<|/det|>
        """
        pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        matches_image = []
        matches_other = []
        
        for a_match in matches:
            if '<|ref|>image<|/ref|>' in a_match[0]:
                matches_image.append(a_match[0])
            else:
                matches_other.append(a_match[0])
        
        return matches, matches_image, matches_other

    def _extract_html_from_text(self, text):
        """
        Extract and clean HTML table from generated text using official DeepSeek-OCR logic.
        This follows the same cleaning process as run_dpsk_ocr_image.py
        """
        if not text:
            return "<table></table>"

        try:
            # Use official DeepSeek-OCR cleaning logic
            _, matches_image, matches_other = self._re_match(text)
            
            # Start with the raw text
            cleaned_output = text
            
            # Remove image references (these are bounding box annotations)
            for idx, a_match_image in enumerate(matches_image):
                cleaned_output = cleaned_output.replace(a_match_image, '')
            
            # Remove other special token references (table, title, etc. with bounding boxes)
            for idx, a_match_other in enumerate(matches_other):
                cleaned_output = cleaned_output.replace(a_match_other, '')
            
            # Additional cleaning from official script
            cleaned_output = cleaned_output.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
            
            # Remove any remaining special tokens
            cleaned_output = re.sub(r'<\|grounding\|>', '', cleaned_output, flags=re.IGNORECASE)
            cleaned_output = re.sub(r'<\|[^|]*\|>', '', cleaned_output, flags=re.IGNORECASE)
            
            # Strip whitespace
            cleaned_output = cleaned_output.strip()
            
            print(f"✅ Cleaned output preview (first 500 chars): {cleaned_output[:500]}")
            
            # Check if we have valid table content
            if '<table' in cleaned_output.lower() and '</table>' in cleaned_output.lower():
                # Extract the table(s)
                tables = re.findall(
                    r'<table[^>]*>.*?</table>', cleaned_output, re.IGNORECASE | re.DOTALL
                )
                if tables:
                    print(f"✅ Extracted {len(tables)} complete table(s)")
                    return "\n".join(tables)
            
            # If no complete tables found but there's table content, return what we have
            if '<table' in cleaned_output.lower():
                print("⚠️  Found incomplete table content, returning as-is")
                return cleaned_output
            
            # No table content found
            print("⚠️  No table content found in cleaned output")
            return "<table></table>"

        except Exception as e:
            print(f"❌ Error in official cleaning logic: {e}")
            import traceback
            print(traceback.format_exc())
            return "<table></table>"
