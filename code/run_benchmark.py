import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from grading import compute_scores  # noqa: E402
from processors import PROCESSOR_CONFIGS  # noqa: E402


def load_config_file(config_path: Union[str, Path]) -> dict:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {config_path.suffix}. Use .yaml, .yml, or .json"
            )


class BenchmarkRunner:
    """Orchestrates benchmark execution with flexible configuration"""

    def __init__(
        self,
        processors: List[str],
        models: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        ground_truth_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        workers: int = 4,
        parallel: bool = False,
        max_images: Optional[int] = None,
    ):
        self.processors = processors
        self.models = models or []
        self.metrics = metrics or ["levenshtein"]
        self.image_dir = (
            Path(image_dir) if image_dir else Path(__file__).parent / "data" / "images"
        )
        self.ground_truth_dir = (
            Path(ground_truth_dir)
            if ground_truth_dir
            else Path(__file__).parent / "data" / "ground_truth"
        )
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(__file__).parent / "data" / "results"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.parallel = parallel
        self.max_images = max_images
        self.results = []

        # Create timestamped run directory
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(__file__).parent / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Run directory: {self.run_dir}")

    def get_processor_configs(self) -> List[Tuple[str, Any, str]]:
        """Generate list of (processor_name, processor_class, model) tuples to test"""
        configs = []

        for proc_name in self.processors:
            if proc_name == "all":
                # Test all processors
                for p in PROCESSOR_CONFIGS.keys():
                    configs.extend(self._get_configs_for_processor(p))
            else:
                configs.extend(self._get_configs_for_processor(proc_name))

        return configs

    def _get_configs_for_processor(self, proc_name: str) -> List[Tuple[str, Any, str]]:
        """Get processor configurations for a specific processor"""
        configs = []

        if proc_name not in PROCESSOR_CONFIGS:
            print(f"⚠️  Unknown processor: {proc_name}")
            return configs

        proc_config = PROCESSOR_CONFIGS[proc_name]

        # Handle processors with multiple classes (like chunkr, paddle)
        if "processor_classes" in proc_config:
            for model_name, processor_class in proc_config["processor_classes"].items():
                # Filter by specified models if provided
                if not self.models or model_name in self.models:
                    configs.append(
                        (f"{proc_name}_{model_name}", processor_class, model_name)
                    )
        else:
            # Single processor class with multiple models
            processor_class = proc_config["processor_class"]
            models = proc_config.get("models", ["default"])

            for model in models:
                # Filter by specified models if provided
                if (
                    not self.models
                    or model in self.models
                    or any(m in model for m in self.models)
                ):
                    configs.append((f"{proc_name}_{model}", processor_class, model))

        return configs

    def get_test_pairs(
        self, specific_images: Optional[List[str]] = None
    ) -> List[Tuple[Path, Path]]:
        """Get list of (image_path, ground_truth_path) pairs to test"""
        test_pairs = []

        if specific_images:
            # Test only specific images
            for img_name in specific_images:
                img_path = self.image_dir / img_name
                gt_name = Path(img_name).stem + ".html"
                gt_path = self.ground_truth_dir / gt_name

                if img_path.exists() and gt_path.exists():
                    test_pairs.append((img_path, gt_path))
                else:
                    print(f"⚠️  Missing pair for {img_name}")
        else:
            # Test all available images
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_path in self.image_dir.glob(ext):
                    gt_path = self.ground_truth_dir / f"{img_path.stem}.html"
                    if gt_path.exists():
                        test_pairs.append((img_path, gt_path))

        # Apply max_images limit if specified
        if self.max_images and len(test_pairs) > self.max_images:
            print(
                f"⚠️  Limiting to first {self.max_images} images (out of {len(test_pairs)} available)"
            )
            test_pairs = test_pairs[: self.max_images]

        return test_pairs

    def run_single_test(
        self,
        processor_name: str,
        processor_class: Any,
        model: str,
        image_path: Path,
        ground_truth_path: Path,
    ) -> Dict[str, Any]:
        """Run a single processor test on a single image"""
        test_id = f"{processor_name}_{image_path.stem}"
        print(f"🔄 Testing {test_id}...")

        start_time = time.time()
        result = {
            "processor": processor_name,
            "model": model,
            "image": str(image_path),
            "ground_truth": str(ground_truth_path),
            "success": False,
            "error": None,
        }

        try:
            # Initialize processor
            processor = processor_class(model)

            # Process document
            processing_start = time.time()
            html_output = processor.process_document(str(image_path))
            processing_time = time.time() - processing_start

            # Save HTML output to structured directory
            processor_output_dir = self.run_dir / processor_name
            processor_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = processor_output_dir / f"{image_path.stem}.html"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_output)

            # Load ground truth
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                ground_truth_html = f.read().strip()

            # Compute scores
            grading_start = time.time()
            scores = compute_scores(
                html_ground_truth=ground_truth_html,
                html_prediction=html_output,
                metrics=self.metrics,
            )
            grading_time = time.time() - grading_start

            # Convert numpy types to Python types for JSON serialization
            json_scores = {}
            for metric, score in scores.items():
                if hasattr(score, "item"):
                    json_scores[metric] = float(score.item())
                else:
                    json_scores[metric] = float(score)

            result.update(
                {
                    "success": True,
                    "processing_time": processing_time,
                    "grading_time": grading_time,
                    "total_time": time.time() - start_time,
                    "scores": json_scores,
                    "output_length": len(html_output),
                    "output_file": str(output_file),
                }
            )

            # Print summary
            scores_str = ", ".join([f"{k}={v:.4f}" for k, v in json_scores.items()])
            print(f"✅ {test_id}: {scores_str} ({processing_time:.2f}s)")

        except Exception as e:
            result.update(
                {
                    "success": False,
                    "error": str(e),
                    "total_time": time.time() - start_time,
                }
            )
            print(f"❌ {test_id}: {str(e)}")

        return result

    def run_benchmark(self, specific_images: Optional[List[str]] = None):
        """Run full benchmark with all specified configurations"""
        print("🚀 Starting benchmark...")
        print(f"   Processors: {', '.join(self.processors)}")
        print(f"   Metrics: {', '.join(self.metrics)}")
        print(f"   Workers: {self.workers}")
        print(f"   Parallel: {self.parallel}")
        print()

        # Get configurations
        processor_configs = self.get_processor_configs()
        test_pairs = self.get_test_pairs(specific_images)

        print("📊 Configuration:")
        print(f"   {len(processor_configs)} processor configurations")
        print(f"   {len(test_pairs)} test images")
        print(f"   {len(processor_configs) * len(test_pairs)} total tests")
        print()

        if not processor_configs:
            print("❌ No valid processor configurations found")
            return

        if not test_pairs:
            print("❌ No valid test pairs found")
            return

        # Initialize results file at the beginning
        self._initialize_results_file()

        # Run tests
        all_results = []

        if self.parallel:
            # Parallel execution with interleaved tasks
            # Interleave by image first, then processor to ensure all processors
            # start working immediately instead of sequentially
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []
                for img_path, gt_path in test_pairs:
                    for proc_name, proc_class, model in processor_configs:
                        future = executor.submit(
                            self.run_single_test,
                            proc_name,
                            proc_class,
                            model,
                            img_path,
                            gt_path,
                        )
                        futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                        # Update results file after each test completes
                        self._update_results_file(result)
                    except Exception as e:
                        print(f"❌ Test execution failed: {e}")
        else:
            # Sequential execution
            for proc_name, proc_class, model in processor_configs:
                for img_path, gt_path in test_pairs:
                    result = self.run_single_test(
                        proc_name, proc_class, model, img_path, gt_path
                    )
                    all_results.append(result)
                    # Update results file after each test completes
                    self._update_results_file(result)

        # Save results
        self._save_results(all_results)
        self._print_summary(all_results)

    def _initialize_results_file(self):
        """Initialize results file at the beginning of the benchmark"""
        output_data = {
            "run_id": self.run_id,
            "timestamp": self.run_id,
            "config": {
                "processors": self.processors,
                "models": self.models,
                "metrics": self.metrics,
                "workers": self.workers,
                "parallel": self.parallel,
                "max_images": self.max_images,
            },
            "results": [],  # Start with empty results
            "processor_stats": {},  # Per-processor statistics (updated in real-time)
            "status": "running",
        }

        # Save initial file
        self.run_results_file = self.run_dir / "benchmark_results.json"
        with open(self.run_results_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"📝 Results file initialized: {self.run_results_file}")

    def _calculate_processor_stats(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics per processor"""
        processor_stats = {}

        for result in results:
            if not result.get("success"):
                continue

            proc = result["processor"]

            if proc not in processor_stats:
                processor_stats[proc] = {
                    "processor": proc,
                    "total_tests": 0,
                    "successful_tests": 0,
                    "failed_tests": 0,
                    "avg_processing_time": 0.0,
                    "avg_grading_time": 0.0,
                    "metrics": {},
                }

            stats = processor_stats[proc]
            stats["total_tests"] += 1
            stats["successful_tests"] += 1

            # Accumulate times
            stats["avg_processing_time"] += result.get("processing_time", 0)
            stats["avg_grading_time"] += result.get("grading_time", 0)

            # Accumulate metric scores
            for metric, score in result.get("scores", {}).items():
                if metric not in stats["metrics"]:
                    stats["metrics"][metric] = {"sum": 0.0, "count": 0, "avg": 0.0}
                stats["metrics"][metric]["sum"] += score
                stats["metrics"][metric]["count"] += 1

        # Calculate averages
        for proc, stats in processor_stats.items():
            if stats["successful_tests"] > 0:
                stats["avg_processing_time"] /= stats["successful_tests"]
                stats["avg_grading_time"] /= stats["successful_tests"]

                for metric in stats["metrics"].values():
                    metric["avg"] = metric["sum"] / metric["count"]
                    # Remove sum and count, keep only avg
                    del metric["sum"]
                    del metric["count"]

        return processor_stats

    def _update_results_file(self, new_result: Dict[str, Any]):
        """Append a new result to the results file and update processor stats"""
        # Use a lock to prevent race conditions in parallel execution
        if not hasattr(self, "_results_lock"):
            self._results_lock = threading.Lock()

        with self._results_lock:
            # Read current results
            with open(self.run_results_file, "r") as f:
                output_data = json.load(f)

            # Append new result
            output_data["results"].append(new_result)

            # Calculate and update processor statistics
            output_data["processor_stats"] = self._calculate_processor_stats(
                output_data["results"]
            )

            # Write back to file
            with open(self.run_results_file, "w") as f:
                json.dump(output_data, f, indent=2)

    def _save_results(self, results: List[Dict[str, Any]]):
        """Finalize results file with completion status"""
        # Read current results (should already have all results from updates)
        with open(self.run_results_file, "r") as f:
            output_data = json.load(f)

        # Update status to completed
        output_data["status"] = "completed"

        # Write final version
        with open(self.run_results_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results finalized: {self.run_results_file}")

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics"""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total tests: {len(results)}")
        print(
            f"Successful: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)"
        )
        print(f"Failed: {len(failed)} ({len(failed) / len(results) * 100:.1f}%)")

        if successful:
            # Calculate per-processor statistics
            processor_stats = self._calculate_processor_stats(successful)

            # Print per-processor averages
            print("\n📊 Per-Processor Statistics:")
            print("-" * 80)

            for proc_name in sorted(processor_stats.keys()):
                stats = processor_stats[proc_name]
                print(f"\n🔹 {proc_name}")
                print(f"   Tests: {stats['successful_tests']}")
                print(f"   Avg Processing Time: {stats['avg_processing_time']:.2f}s")
                print(f"   Avg Grading Time: {stats['avg_grading_time']:.2f}s")
                print("   Metrics:")
                for metric, metric_stats in sorted(stats["metrics"].items()):
                    print(f"      {metric}: {metric_stats['avg']:.4f}")

            # Global averages
            print("\n" + "-" * 80)
            print("📈 Global Average Scores (all processors):")
            metric_sums = {}
            metric_counts = {}

            for result in successful:
                for metric, score in result.get("scores", {}).items():
                    metric_sums[metric] = metric_sums.get(metric, 0) + score
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1

            for metric in sorted(metric_sums.keys()):
                avg = metric_sums[metric] / metric_counts[metric]
                print(f"   {metric}: {avg:.4f}")

            # Average times
            avg_processing = sum(r["processing_time"] for r in successful) / len(
                successful
            )
            avg_grading = sum(r["grading_time"] for r in successful) / len(successful)
            print("\n⏱️  Global Average Times:")
            print(f"   Processing: {avg_processing:.2f}s")
            print(f"   Grading: {avg_grading:.2f}s")

            # Top performers per metric
            print("\n🏆 Top Performers (single best score):")
            for metric in sorted(metric_sums.keys()):
                top_result = max(
                    successful, key=lambda r: r.get("scores", {}).get(metric, 0)
                )
                score = top_result["scores"][metric]
                processor = top_result["processor"]
                print(f"   {metric}: {processor} ({score:.4f})")

            # Best average performer per metric
            print("\n🥇 Best Average Performer (by processor avg):")
            for metric in sorted(metric_sums.keys()):
                best_proc = None
                best_avg = -1
                for proc_name, stats in processor_stats.items():
                    if metric in stats["metrics"]:
                        avg = stats["metrics"][metric]["avg"]
                        if avg > best_avg:
                            best_avg = avg
                            best_proc = proc_name
                if best_proc:
                    print(f"   {metric}: {best_proc} ({best_avg:.4f})")

        if failed:
            print(f"\n❌ Failed Tests ({len(failed)}):")
            error_summary = {}
            for result in failed:
                error = result.get("error", "Unknown error")
                error_summary[error] = error_summary.get(error, 0) + 1

            for error, count in sorted(error_summary.items(), key=lambda x: -x[1]):
                print(f"   {count}x: {error[:80]}")

        print("=" * 80)
        print("\n📂 Output Structure:")
        print(f"   {self.run_dir}/")
        print("   ├── benchmark_results.json")
        for i, proc in enumerate(self.processors):
            prefix = "├──" if i < len(self.processors) - 1 else "└──"
            print(f"   {prefix} {proc}_*/")
            print("       └── table_*.html")


def main():
    parser = argparse.ArgumentParser(
        description="Flexible benchmark for table extraction processors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file option
    parser.add_argument(
        "--config",
        help="Path to YAML/JSON config file (overrides other arguments)",
    )

    # Processor selection
    parser.add_argument(
        "--processors",
        nargs="+",
        choices=[
            "all",
            "openrouter",
            "azure",
            "chunkr",
            "mathpix",
            "mistral",
            "textract",
            "unstructured",
            "paddle",
        ],
        default=["all"],
        help="Processors to test (default: all)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (filters the models list)",
    )

    # Metric selection
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["levenshtein", "teds"],
        default=["levenshtein"],
        help="Grading metrics to compute (default: levenshtein)",
    )

    # Data selection
    parser.add_argument(
        "--images",
        nargs="+",
        help="Specific images to test (e.g., table_0a029cc2.png)",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to test (useful for quick tests)",
    )

    parser.add_argument(
        "--image-dir",
        help="Directory containing test images (default: code/data/images)",
    )

    parser.add_argument(
        "--ground-truth-dir",
        help="Directory containing ground truth HTML files (default: code/data/ground_truth)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="./data/results",
        help="Output directory for results (default: ./data/results)",
    )

    # Execution
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution (default: sequential)",
    )

    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        print(f"📄 Loading configuration from: {args.config}")
        config = load_config_file(args.config)

        # Create runner from config file
        runner = BenchmarkRunner(
            processors=config.get("processors", ["all"]),
            models=config.get("models"),
            metrics=config.get("metrics", ["levenshtein"]),
            image_dir=config.get("image_dir"),
            ground_truth_dir=config.get("ground_truth_dir"),
            output_dir=config.get("output_dir", "./data/results"),
            workers=config.get("workers", 4),
            parallel=config.get("parallel", False),
            max_images=config.get("max_images"),
        )
        runner.run_benchmark(specific_images=config.get("images"))
    else:
        # Create runner from command-line arguments
        runner = BenchmarkRunner(
            processors=args.processors,
            models=args.models,
            metrics=args.metrics,
            image_dir=args.image_dir,
            ground_truth_dir=args.ground_truth_dir,
            output_dir=args.output_dir,
            workers=args.workers,
            parallel=args.parallel,
            max_images=args.max_images,
        )
        runner.run_benchmark(specific_images=args.images)


if __name__ == "__main__":
    main()
