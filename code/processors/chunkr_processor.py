#!/usr/bin/env python3
"""
Chunkr API processors for table extraction.
Includes ChunkrTableProcessor, ChunkrDefaultProcessor, and ChunkrLayoutProcessor.
"""

import asyncio
import gc
import os
import time


class ChunkrDefaultProcessor:
    def __init__(self, model):
        self.api_key = os.getenv("CHUNKR_API_KEY")
        self.model = model

    def process_document(self, file_path):
        print(f"Processing document with Chunkr Default: {file_path}")

        # Initialize variables for cleanup
        client = None
        table_html = None
        loop = None

        max_retries = 3
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                print("Uploading document to Chunkr using default configuration...")
                # Create a new Chunkr client for each request to avoid asyncio issues
                from chunkr_ai import AsyncChunkr

                client = AsyncChunkr(api_key=self.api_key)

                # Run the async function in a new event loop
                async def run_upload():
                    if client and file_path:
                        with open(file_path, "rb") as f:
                            uploaded_file = await client.files.create(file=f)
                            task = await client.tasks.parse.create(
                                file=uploaded_file.url,
                                ocr_strategy="All",
                                segmentation_strategy="Page",
                                segment_processing={
                                    "page": {
                                        "strategy": "LLM",
                                        "format": "Html",
                                    }
                                },
                            )

                        # Poll until task is completed
                        while not task.completed:
                            await asyncio.sleep(1)  # Wait 1 second
                            task = await client.tasks.parse.get(
                                task.task_id
                            )  # Refresh task status

                        if task.status == "Succeeded" and task.output:
                            result = ""
                            for chunk in task.output.chunks:
                                result += chunk.content or ""
                        else:
                            raise Exception(
                                f"Task failed with status: {task.status}, message: {task.message}"
                            )

                        await client.close()
                        return result

                # Create a new event loop and run the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                table_html = loop.run_until_complete(run_upload())
                loop.close()
                loop = None  # Set to None after closing

                return table_html
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(
                        f"ERROR processing document after {max_retries} retries: {str(e)}"
                    )
                    import traceback

                    print(traceback.format_exc())
                    raise Exception(
                        f"ChunkrDefaultProcessor API failed after {max_retries} retries: {str(e)}"
                    )
                else:
                    print(
                        f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds..."
                    )
                    time.sleep(backoff_time)
                    backoff_time *= 2

            finally:
                try:
                    # Clean up variables
                    if client is not None:
                        del client
                    if table_html is not None:
                        del table_html
                    if loop is not None:
                        loop.close()
                        del loop
                except Exception:
                    pass
                gc.collect()
