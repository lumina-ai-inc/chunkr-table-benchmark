#!/usr/bin/env python3
"""
Unstructured API processor for table extraction.
"""

import os
import gc
import time


class UnstructuredProcessor:
    def __init__(self, model):
        self.api_key = os.getenv("UNSTRUCTURED_API_KEY")
        self.server_url = os.getenv("UNSTRUCTURED_URL", "https://api.unstructuredapp.io")
        self.model = model or "hi_res"  # Default strategy

    def process_document(self, file_path):
        print(f"Processing document with Unstructured: {file_path}")

        if not self.api_key:
            error_msg = "ERROR: UNSTRUCTURED_API_KEY environment variable not set"
            print(error_msg)
            return error_msg

        max_retries = 20
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            res = None
            elements = None
            uc_client = None
            try:
                from unstructured_client import UnstructuredClient
                from unstructured_client.models import shared

                print("Processing document with Unstructured API...")

                # Create client using the modern SDK approach
                with UnstructuredClient(
                    api_key_auth=self.api_key,
                    server_url=self.server_url,
                ) as uc_client:

                    with open(file_path, "rb") as f:
                        res = uc_client.general.partition(request={
                            "partition_parameters": {
                                "files": {
                                    "content": f,
                                    "file_name": os.path.basename(file_path),
                                },
                                "strategy": shared.Strategy.HI_RES,
                            },
                        })
                
                # Extract table elements OUTSIDE the with block
                elements = res.elements
                table_html = self.extract_tables_as_html(elements)

                # Explicitly release heavy objects BEFORE return
                try:
                    del elements, res, uc_client
                except Exception:
                    pass
                gc.collect()

                return table_html

            except ImportError:
                error_msg = "ERROR: unstructured-client not installed. Please install with: pip install unstructured-client"
                print(error_msg)
                return error_msg
            except Exception as e:
                # Clean up heavy objects from failed attempt
                try:
                    del elements, res, uc_client
                except Exception:
                    pass
                gc.collect()
                
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"ERROR processing document with Unstructured after {max_retries} retries: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    raise Exception(f"UnstructuredProcessor API failed after {max_retries} retries: {str(e)}")
                else:
                    print(f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 2

    def extract_tables_as_html(self, elements):
        """Extract table elements and convert to HTML"""
        tables = []
        
        print(f"DEBUG: Processing {len(elements) if elements else 0} elements from Unstructured API")
        
        if not elements:
            print("DEBUG: No elements returned from API")
            return "<table></table>"
        
        # Debug: Print first few elements to understand structure
        for i, element in enumerate(elements[:3]):
            print(f"DEBUG: Element {i}: type={getattr(element, 'type', 'NO_TYPE')}")
            print(f"DEBUG: Element {i} attributes: {dir(element)}")
            if hasattr(element, 'text'):
                text_content = getattr(element, 'text', '')
                print(f"DEBUG: Element {i} text: {text_content[:200]}...")
            if hasattr(element, 'metadata'):
                print(f"DEBUG: Element {i} has metadata: {element.metadata is not None}")
            # Check if it's a dictionary-like object
            if hasattr(element, 'keys'):
                print(f"DEBUG: Element {i} is dict-like with keys: {list(element.keys())}")
                if 'type' in element:
                    print(f"DEBUG: Element {i} dict type: {element['type']}")
                if 'text' in element:
                    print(f"DEBUG: Element {i} dict text: {element['text'][:200]}...")
            print(f"DEBUG: Element {i} full object: {element}")
            print("---")
        
        for element in elements:
            # Handle both dictionary-style and attribute-style access
            if isinstance(element, dict):
                element_type = element.get('type')
                element_text = element.get('text', '')
                element_metadata = element.get('metadata', {})
            else:
                element_type = getattr(element, 'type', None)
                element_text = getattr(element, 'text', '')
                element_metadata = getattr(element, 'metadata', None)
            
            # Look for table elements
            if element_type == 'Table':
                print(f"DEBUG: Found Table element with text: {element_text[:100]}...")
                
                # Try to get HTML representation if available
                if element_metadata:
                    if isinstance(element_metadata, dict) and 'text_as_html' in element_metadata:
                        # Dictionary-style metadata
                        html_content = element_metadata['text_as_html']
                        if html_content:
                            tables.append(html_content)
                            print(f"DEBUG: Using text_as_html from dict metadata")
                        elif element_text:
                            tables.append(f"<table><tr><td>{element_text}</td></tr></table>")
                            print(f"DEBUG: Using text fallback from dict")
                    elif hasattr(element_metadata, 'text_as_html') and element_metadata.text_as_html:
                        # Attribute-style metadata
                        tables.append(element_metadata.text_as_html)
                        print(f"DEBUG: Using text_as_html from attr metadata")
                    elif element_text:
                        tables.append(f"<table><tr><td>{element_text}</td></tr></table>")
                        print(f"DEBUG: Using text fallback from attr")
                elif element_text:
                    # Fallback: wrap plain text in basic table structure
                    tables.append(f"<table><tr><td>{element_text}</td></tr></table>")
                    print(f"DEBUG: Using text only fallback")
            
            # Also look for elements that might contain tabular data
            elif element_type in ['Header', 'NarrativeText', 'ListItem'] and element_text:
                # Check if text looks like it might be tabular data
                if any(indicator in element_text.lower() for indicator in ['%', '増加', '減少', '不変', '業', '売上']) or \
                   '\t' in element_text or '|' in element_text:
                    print(f"DEBUG: Found potential table data in {element_type}: {element_text[:100]}...")
                    # Convert to simple table format
                    lines = element_text.split('\n')
                    if len(lines) > 1:
                        html = "<table>"
                        for line in lines:
                            if line.strip():
                                # Split by tabs or multiple spaces
                                cells = [cell.strip() for cell in line.replace('\t', '|').split('|') if cell.strip()]
                                if len(cells) > 1:
                                    html += "<tr>"
                                    for cell in cells:
                                        html += f"<td>{cell}</td>"
                                    html += "</tr>"
                                else:
                                    html += f"<tr><td>{line.strip()}</td></tr>"
                        html += "</table>"
                        tables.append(html)
                        print(f"DEBUG: Created table from {element_type} content")

        print(f"DEBUG: Found {len(tables)} table(s)")
        
        if tables:
            result = "\n".join(tables)
            print(f"DEBUG: Final result length: {len(result)} characters")
            return result
        else:
            print("DEBUG: No tables found, returning empty table")
            return "<table></table>"