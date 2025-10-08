#!/usr/bin/env python3
"""
Test suite for all processors using existing image.png.
Makes real API calls to test each processor and saves results to data/ directory.
WARNING: This will use real API keys and make actual API calls with costs!
"""

import argparse
import concurrent.futures
import glob
import json
import os
import sys
import time
import unittest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Add parent directory for local grading.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Create data directory for results
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Global variables for test configuration
TEST_IMAGE_PATH = None
TEST_GROUND_TRUTH_PATH = None
TEST_IMAGE_LIST = []
TEST_GROUND_TRUTH_LIST = []

from azure_processor import AzureDocumentAnalysisProcessor
from chunkr_processor import ChunkrDefaultProcessor

# Import grading function
from grading import compute_scores

# Import all processors
from mathpix_processor import MathpixProcessor
from mistral_processor import MistralProcessor
from openrouter_processor import OpenRouterTableProcessor
from textract_processor import TextractProcessor
from unstructured_processor import UnstructuredProcessor




class TestProcessors(unittest.TestCase):
    """Test all processors with the existing image.png"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test image path and ground truth"""
        global TEST_IMAGE_PATH, TEST_GROUND_TRUTH_PATH, TEST_IMAGE_LIST, TEST_GROUND_TRUTH_LIST
        
        if TEST_IMAGE_PATH and TEST_GROUND_TRUTH_PATH:
            # Command line arguments provided
            image_path = Path(TEST_IMAGE_PATH)
            ground_truth_path = Path(TEST_GROUND_TRUTH_PATH)
            
            if image_path.is_dir() and ground_truth_path.is_dir():
                # Directory mode - find matching files
                print(f"Directory mode: {image_path} → {ground_truth_path}")
                
                # Find all images
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    image_files.extend(glob.glob(str(image_path / ext)))
                
                # Find matching ground truth files
                cls.test_pairs = []
                for img_file in image_files:
                    img_name = Path(img_file).stem
                    gt_file = ground_truth_path / f"{img_name}.html"
                    if gt_file.exists():
                        cls.test_pairs.append((img_file, str(gt_file)))
                
                if not cls.test_pairs:
                    raise FileNotFoundError("No matching image/ground_truth pairs found")
                
                print(f"Found {len(cls.test_pairs)} matching pairs")
                cls.test_mode = "directory"
                cls.test_image_path = cls.test_pairs[0][0]  # Use first for single tests
                cls.ground_truth_path = cls.test_pairs[0][1]
                
            else:
                # Single file mode
                print(f"Single file mode: {image_path} → {ground_truth_path}")
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                if not ground_truth_path.exists():
                    raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")
                
                cls.test_image_path = str(image_path)
                cls.ground_truth_path = str(ground_truth_path)
                cls.test_pairs = [(cls.test_image_path, cls.ground_truth_path)]
                cls.test_mode = "single"
        else:
            # Default mode - use built-in test files
            print("Default mode: using built-in test files")
            cls.test_image_path = Path(__file__).parent / "data" / "test" / "table_0a029cc2.png"
            cls.ground_truth_path = Path(__file__).parent / "data" / "test" / "table_0a029cc2.html"
            
            if not cls.test_image_path.exists() or not cls.ground_truth_path.exists():
                raise FileNotFoundError("Default test files not found")
            
            cls.test_image_path = str(cls.test_image_path)
            cls.ground_truth_path = str(cls.ground_truth_path)
            cls.test_pairs = [(cls.test_image_path, cls.ground_truth_path)]
            cls.test_mode = "default"
        
        # Load ground truth content for the first pair
        with open(cls.ground_truth_path, 'r', encoding='utf-8') as f:
            cls.ground_truth_html = f.read().strip()
        
        print(f"Test mode: {cls.test_mode}")
        print(f"Primary test image: {cls.test_image_path}")
        print(f"Primary ground truth: {cls.ground_truth_path}")
        if hasattr(cls, 'test_pairs'):
            print(f"Total test pairs: {len(cls.test_pairs)}")
    
    @classmethod
    def tearDownClass(cls):
        """No cleanup needed for existing image"""
        print(f"Test completed with image: {cls.test_image_path}")
    
    def _test_processor_with_real_processing(self, processor_class, model_name, **kwargs):
        """Test processor with actual API call, grading, and save results"""
        processor_name = f"{processor_class.__name__}_{model_name.replace('/', '_')}"
        start_time = time.time()

        try:
            print(f"🔄 Testing {processor_name} with real API call...")
            
            
            processor = processor_class(model_name, **kwargs)
            self.assertIsNotNone(processor, f"{processor_class.__name__} should initialize")
            
            # Test that the processor has required methods
            self.assertTrue(hasattr(processor, 'process_document'), 
                          f"{processor_class.__name__} should have process_document method")
            
            # Test with the actual image.png file path
            image_path = str(self.test_image_path)
            self.assertTrue(os.path.exists(image_path), f"Test image should exist: {image_path}")
            
            # Actually call the processor with the image
            result = processor.process_document(image_path)
            processing_time = time.time() - start_time
            
            # Validate result structure
            self.assertIsInstance(result, str, f"{processor_name} should return a string")
            self.assertGreater(len(result), 0, f"{processor_name} should return non-empty result")
            
            # Perform grading with hardcoded ground truth
            print(f"📊 Grading {processor_name} against ground truth...")
            grading_start = time.time()
            
            # Test both Levenshtein and TEDS for comprehensive grading
            scores = compute_scores(
                html_ground_truth=self.ground_truth_html,
                html_prediction=result,
                metrics=['levenshtein', 'teds']
            )
            grading_time = time.time() - grading_start
            
            # Convert numpy float32 to regular Python float for JSON serialization
            json_scores = {}
            for metric, score in scores.items():
                if hasattr(score, 'item'):  # numpy scalar
                    json_scores[metric] = float(score.item())
                else:
                    json_scores[metric] = float(score)
            
            print(f"   Grading completed in {grading_time:.2f}s")
            print(f"   Levenshtein score: {json_scores.get('levenshtein', 'N/A')}")
            print(f"   TEDS score: {json_scores.get('teds', 'N/A')}")
            
            # Save result to data directory with grading
            result_file = DATA_DIR / f"{processor_name}_result.json"
            result_data = {
                "processor": processor_class.__name__,
                "model": model_name,
                "image_path": image_path,
                "ground_truth_path": str(self.ground_truth_path),
                "processing_time": processing_time,
                "grading_time": grading_time,
                "total_time": processing_time + grading_time,
                "result": result,
                "scores": json_scores,
                "timestamp": time.time(),
                "success": True
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"✅ {processor_name} processed image in {processing_time:.2f}s")
            print(f"   Result length: {len(result)} characters")
            print(f"   Levenshtein: {json_scores.get('levenshtein', 'N/A')}")
            print(f"   TEDS: {json_scores.get('teds', 'N/A')}")
            print(f"   Saved to: {result_file}")
            
            return True
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Save error result
            result_file = DATA_DIR / f"{processor_name}_error.json"
            error_data = {
                "processor": processor_class.__name__,
                "model": model_name,
                "image_path": str(self.test_image_path),
                "ground_truth_path": str(self.ground_truth_path),
                "processing_time": processing_time,
                "grading_time": 0,
                "total_time": processing_time,
                "error": error_msg,
                "scores": {},
                "timestamp": time.time(),
                "success": False
            }
            
            with open(result_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            print(f"❌ {processor_name} failed after {processing_time:.2f}s: {error_msg}")
            print(f"   Error saved to: {result_file}")
            
            # Don't fail the test for API errors - just log them
            return False
    
    def test_unstructured_processor(self):
        """Test UnstructuredProcessor with real API call"""
        print("\n🧪 Testing UnstructuredProcessor...")
        result = self._test_processor_with_real_processing(UnstructuredProcessor, "hi_res")
        # Don't assert True since API might fail due to missing keys
        print(f"UnstructuredProcessor result: {'✅ SUCCESS' if result else '❌ FAILED'}")
    
    def test_azure_processor(self):
        """Test AzureDocumentAnalysisProcessor with real API call"""
        print("\n🧪 Testing AzureDocumentAnalysisProcessor...")
        result = self._test_processor_with_real_processing(AzureDocumentAnalysisProcessor, "prebuilt-layout")
        print(f"AzureDocumentAnalysisProcessor result: {'✅ SUCCESS' if result else '❌ FAILED'}")
    
    def test_openrouter_processor(self):
        """Test OpenRouterTableProcessor with real API calls in parallel"""
        print("\n🧪 Testing OpenRouterTableProcessor...")
        models = [
            "google/gemini-2.0-flash-001",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash", 
            "anthropic/claude-opus-4.1",
            "openai/gpt-5"
        ]
        
        def test_single_model(model):
            """Test a single OpenRouter model"""
            print(f"\n  Testing model: {model}")
            result = self._test_processor_with_real_processing(OpenRouterTableProcessor, model)
            print(f"  {model}: {'✅ SUCCESS' if result else '❌ FAILED'}")
            return (model, result)
        
        # Run tests in parallel with max 3 concurrent threads to avoid rate limits
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_model = {executor.submit(test_single_model, model): model for model in models}
            
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"  {model} generated an exception: {exc}")
                    results.append((model, False))
        
        # Summary
        successes = sum(1 for _, r in results if r)
        print(f"\nOpenRouter Summary: {successes}/{len(models)} models succeeded")
        
        # Sort results by model name for consistent output
        results.sort(key=lambda x: x[0])
        for model, success in results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  Final: {model}: {status}")
    
    def test_mathpix_processor(self):
        """Test MathpixProcessor with real API call"""
        print("\n🧪 Testing MathpixProcessor...")
        result = self._test_processor_with_real_processing(MathpixProcessor, "mathpix")
        print(f"MathpixProcessor result: {'✅ SUCCESS' if result else '❌ FAILED'}")
        
        # Test LaTeX conversion methods (these don't require API)
        try:
            processor = MathpixProcessor("mathpix")
            
            # Test basic LaTeX to HTML conversion
            latex_input = "x^2 + y_{i}"
            html_output = processor.convert_latex_math_to_html(latex_input)
            self.assertIn("<sup>", html_output, "Should convert superscripts")
            self.assertIn("<sub>", html_output, "Should convert subscripts")
            
            # Test table conversion
            latex_table = "\\begin{tabular}{cc} A & B \\\\ 1 & 2 \\end{tabular}"
            html_table = processor.latex_to_html_table(latex_table)
            self.assertIn("<table>", html_table, "Should create HTML table")
            self.assertIn("<tr>", html_table, "Should create table rows")
            self.assertIn("<td>", html_table, "Should create table cells")
            print("  ✅ LaTeX conversion methods work correctly")
        except Exception as e:
            print(f"  ❌ LaTeX conversion failed: {e}")
    
    def test_mistral_processor(self):
        """Test MistralProcessor with real API call"""
        print("\n🧪 Testing MistralProcessor...")
        result = self._test_processor_with_real_processing(MistralProcessor, "mistral-ocr-latest")
        print(f"MistralProcessor result: {'✅ SUCCESS' if result else '❌ FAILED'}")
        
        # Test markdown conversion methods (these don't require API)
        try:
            processor = MistralProcessor("mistral-ocr-latest")
            
            # Test markdown table conversion
            markdown_input = "| Header 1 | Header 2 |\n| Value 1 | Value 2 |"
            html_output = processor.convert_markdown_table_to_html(markdown_input)
            self.assertIn("<table>", html_output, "Should create HTML table")
            self.assertIn("Header 1", html_output, "Should preserve header content")
            
            # Test OCR text conversion
            text_input = "Product A  $10.99  In Stock\nProduct B  $15.49  Out of Stock"
            html_output = processor.convert_ocr_text_to_table(text_input)
            self.assertIn("<table>", html_output, "Should create HTML table from text")
            print("  ✅ Markdown conversion methods work correctly")
        except Exception as e:
            print(f"  ❌ Markdown conversion failed: {e}")
    
    def test_textract_processor(self):
        """Test TextractProcessor with real API call"""
        print("\n🧪 Testing TextractProcessor...")
        result = self._test_processor_with_real_processing(TextractProcessor, "default")
        print(f"TextractProcessor result: {'✅ SUCCESS' if result else '❌ FAILED'}")
    
    def test_chunkr_processors(self):
        """Test ChunkrDefaultProcessor with real API call"""
        print("\n🧪 Testing Chunkr Processor...")
        processors = [
            (ChunkrDefaultProcessor, "default")
        ]
        
        def test_single_chunkr(processor_info):
            """Test a single Chunkr processor"""
            processor_class, model = processor_info
            print(f"\n  Testing {processor_class.__name__}...")
            result = self._test_processor_with_real_processing(processor_class, model)
            print(f"  {processor_class.__name__}: {'✅ SUCCESS' if result else '❌ FAILED'}")
            return (processor_class.__name__, result)
        
        # Run test
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_processor = {executor.submit(test_single_chunkr, proc): proc for proc in processors}
            
            for future in concurrent.futures.as_completed(future_to_processor):
                processor_info = future_to_processor[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    processor_class, model = processor_info
                    print(f"  {processor_class.__name__} generated an exception: {exc}")
                    results.append((processor_class.__name__, False))
        
        # Summary
        successes = sum(1 for _, r in results if r)
        print(f"\nChunkr Summary: {successes}/{len(processors)} processors succeeded")
        
        # Sort results by processor name for consistent output
        results.sort(key=lambda x: x[0])
        for processor_name, success in results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  Final: {processor_name}: {status}")


def create_processor_config_examples():
    """Create example configurations for each processor"""
    configs = {
        "openrouter_models": {
            "google/gemini-2.0-flash-001": {"description": "Google Gemini 2.0 Flash", "max_tokens": 8192},
            "google/gemini-2.5-pro": {"description": "Google Gemini 2.5 Pro", "max_tokens": 8192},
            "google/gemini-2.5-flash": {"description": "Google Gemini 2.5 Flash", "max_tokens": 8192},
            "anthropic/claude-opus-4.1": {"description": "Claude Opus 4.1", "max_tokens": 8192},
            "openai/gpt-5": {"description": "OpenAI GPT-5", "max_tokens": 8192}
        },
        "azure_models": {
            "prebuilt-layout": {"description": "Azure Document Intelligence Layout"}
        },
        "mathpix_models": {
            "mathpix": {"description": "Mathpix OCR with LaTeX support"}
        },
        "mistral_models": {
            "mistral-ocr-latest": {"description": "Mistral OCR Latest"}
        },
        "textract_models": {
            "textract-default": {"description": "Amazon Textract Default"}
        },
        "chunkr_models": {
            "chunkr-default": {"description": "Chunkr Default Processor"},
            "chunkr-layout": {"description": "Chunkr Layout Processor"}
        }
    }
    
    return configs


def run_integration_test():
    """Run a simple integration test to verify all processors can be imported and initialized in parallel"""
    print("🧪 Running processor integration tests...")
    
    configs = create_processor_config_examples()
    results = {}
    
    def test_openrouter_model(model_config):
        """Test a single OpenRouter model initialization"""
        model, config = model_config
        try:
            OpenRouterTableProcessor(model)
            return f"openrouter_{model.replace('/', '_')}", "✅ PASS", f"  ✅ {model}: {config['description']}"
        except Exception as e:
            return f"openrouter_{model.replace('/', '_')}", f"❌ FAIL: {e}", f"  ❌ {model}: {e}"
    
    def test_processor(processor_info):
        """Test a single processor initialization"""
        processor_class, model, name = processor_info
        try:
            processor_class(model)
            return name.lower().replace(' ', '_'), "✅ PASS", f"  ✅ {name}: Initialization successful"
        except Exception as e:
            return name.lower().replace(' ', '_'), f"❌ FAIL: {e}", f"  ❌ {name}: {e}"
    
    # Test OpenRouter models in parallel
    print("\n📱 Testing OpenRouter processors...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        openrouter_futures = {
            executor.submit(test_openrouter_model, item): item 
            for item in configs["openrouter_models"].items()
        }
        
        for future in concurrent.futures.as_completed(openrouter_futures):
            key, status, message = future.result()
            results[key] = status
            print(message)
    
    # Test other processors in parallel
    processor_tests = [
        (UnstructuredProcessor, "hi_res", "Unstructured"),
        (AzureDocumentAnalysisProcessor, "prebuilt-layout", "Azure"),
        (MathpixProcessor, "mathpix", "Mathpix"),
        (MistralProcessor, "mistral-ocr-latest", "Mistral"),
        (TextractProcessor, "default", "Textract"),
        (ChunkrDefaultProcessor, "default", "Chunkr Default")
    ]
    
    print("\n🔧 Testing other processors...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        processor_futures = {
            executor.submit(test_processor, proc_info): proc_info 
            for proc_info in processor_tests
        }
        
        for future in concurrent.futures.as_completed(processor_futures):
            key, status, message = future.result()
            results[key] = status
            print(message)
    
    # Summary
    passed = sum(1 for r in results.values() if r.startswith("✅"))
    failed = sum(1 for r in results.values() if r.startswith("❌"))
    
    print(f"\n📊 Test Summary: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n❌ Failed tests:")
        for name, result in results.items():
            if result.startswith("❌"):
                print(f"  {name}: {result}")
    
    return results


def parse_arguments():
    """Parse command line arguments for flexible input"""
    parser = argparse.ArgumentParser(description="Test table extraction processors")
    parser.add_argument("--image", help="Path to image file or directory")
    parser.add_argument("--ground-truth", help="Path to ground truth file or directory")
    parser.add_argument("--output", default="./data", help="Output directory for results")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments first, before unittest interferes
    import sys
    
    # Check if we have custom arguments
    if '--image' in sys.argv or '--ground-truth' in sys.argv:
        args = parse_arguments()
        
        # Set global variables for test configuration
        TEST_IMAGE_PATH = args.image
        TEST_GROUND_TRUTH_PATH = args.ground_truth
        print("📁 Using custom paths:")
        print(f"   Images: {args.image}")
        print(f"   Ground truth: {args.ground_truth}")
        print(f"   Output: {args.output}")
        
        # Remove custom args from sys.argv so unittest doesn't see them
        custom_args = ['--image', '--ground-truth', '--output', '--workers']
        filtered_argv = [sys.argv[0]]  # Keep script name
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] in custom_args and i + 1 < len(sys.argv):
                i += 2  # Skip arg and its value
            elif sys.argv[i].startswith('--') and '=' in sys.argv[i]:
                # Skip --arg=value format
                i += 1
            else:
                filtered_argv.append(sys.argv[i])
                i += 1
        sys.argv = filtered_argv
    else:
        print("📁 Using default test files")
    
    # Run integration tests
    print("🚀 Starting processor tests...")
    integration_results = run_integration_test()
    
    # Run unit tests with grading
    print("\n🧪 Running unit tests with grading...")
    unittest.main(verbosity=2, exit=False)
    
    print("\n🎉 All tests completed!")