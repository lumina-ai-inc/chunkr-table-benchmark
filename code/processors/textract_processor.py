#!/usr/bin/env python3
"""
Amazon Textract processor for table extraction.
"""

import os
import gc
import time


class TextractProcessor:
    def __init__(self, model):
        self.model = model or "default"  # Textract doesn't use models like LLMs
        self.region = os.getenv("AWS_REGION", "us-east-1")
        
    def process_document(self, file_path):
        print(f"Processing document with Amazon Textract: {file_path}")
        
        max_retries = 3
        retry_count = 0
        backoff_time = 2
        
        # Initialize variables for cleanup
        extractor = None
        document = None
        tables = None
        
        while retry_count < max_retries:
            try:
                from textractor import Textractor
                from textractor.data.constants import TextractFeatures
                import boto3
                from botocore.exceptions import ClientError, NoCredentialsError
                
                print("Processing document with Amazon Textract API...")
                
                # Initialize Textractor with explicit boto3 Session if AWS creds are present
                # Support multiple env var aliases commonly used
                aws_access_key_id = (
                    os.getenv("AWS_ACCESS_KEY_ID")
                    or os.getenv("AWS_ACCESS_KEY")
                    or os.getenv("AWS__ACCESS_KEY")
                )
                aws_secret_access_key = (
                    os.getenv("AWS_SECRET_ACCESS_KEY")
                    or os.getenv("AWS_SECRET_KEY")
                    or os.getenv("AWS__SECRET_KEY")
                )
                aws_session_token = os.getenv("AWS_SESSION_TOKEN")
                region_name = (
                    self.region
                    or os.getenv("AWS_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or os.getenv("AWS__REGION")
                    or "us-east-1"
                )

                session_kwargs = {"region_name": region_name}
                if aws_access_key_id and aws_secret_access_key:
                    session_kwargs.update({
                        "aws_access_key_id": aws_access_key_id,
                        "aws_secret_access_key": aws_secret_access_key,
                    })
                    if aws_session_token:
                        session_kwargs["aws_session_token"] = aws_session_token

                session = boto3.Session(**session_kwargs)

                try:
                    extractor = Textractor(boto3_session=session)
                except TypeError:
                    # Fallback for older Textractor versions that might not accept boto3_session
                    extractor = Textractor(region_name=self.region)
                
                # Process the document with table extraction only
                document = extractor.analyze_document(
                    file_source=file_path,
                    features=[TextractFeatures.TABLES]
                )
                
                print(f"Document processed successfully! Found {len(document.tables)} table(s).")
                
                # Extract and convert tables to HTML
                if document.tables:
                    # Collect all table HTML
                    all_tables_html = []
                    
                    for i, table in enumerate(document.tables):
                        try:
                            # Convert table to HTML using Textractor's built-in method
                            table_html = table.to_html()
                            all_tables_html.append(table_html)
                            print(f"Table {i+1} processed - Confidence: {table.confidence:.2f}%")
                        except Exception as table_error:
                            print(f"Warning: Failed to process table {i+1}: {table_error}")
                            # Create fallback table structure
                            fallback_html = "<table></table>"
                            all_tables_html.append(fallback_html)
                    
                    # Join all tables into a single HTML string
                    result_html = "\n".join(all_tables_html)
                    
                    # Clean up large objects before return
                    try:
                        del document, tables, extractor
                    except Exception:
                        pass
                    gc.collect()
                    
                    return result_html
                else:
                    print("No tables found in the document.")
                    # Clean up before return
                    try:
                        del document, tables, extractor
                    except Exception:
                        pass
                    gc.collect()
                    
                    return "<table></table>"
                    
            except ImportError as e:
                error_msg = f"ERROR: Required library not installed: {e}"
                print(error_msg)
                print("Please install with: pip install amazon-textract-textractor boto3")
                return error_msg
            except (NoCredentialsError, ClientError) as e:
                error_msg = f"ERROR: AWS authentication or permission issue: {e}"
                print(error_msg)
                print("Make sure you have:")
                print("1. Valid AWS credentials configured (AWS CLI, environment variables, or IAM role)")
                print("2. Textract permissions in your AWS account")
                print("3. Correct AWS region specified")
                return error_msg
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"ERROR processing document with Textract after {max_retries} retries: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    
                    # Clean up before raising
                    try:
                        del document, tables, extractor
                    except Exception:
                        pass
                    gc.collect()
                    
                    raise Exception(f"TextractProcessor API failed after {max_retries} retries: {str(e)}")
                else:
                    print(f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds...")
                    
                    # Clean up failed attempt objects
                    try:
                        del document, tables, extractor
                    except Exception:
                        pass
                    gc.collect()
                    
                    time.sleep(backoff_time)
                    backoff_time *= 2
        
        # Final cleanup if all retries failed
        try:
            del document, tables, extractor
        except Exception:
            pass
        gc.collect()