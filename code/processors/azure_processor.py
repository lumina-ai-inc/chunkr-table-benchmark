#!/usr/bin/env python3
"""
Azure Document Intelligence processor for table extraction.
"""

import os
import gc
import time


class AzureDocumentAnalysisProcessor:
    def __init__(self, model):
        self.endpoint = os.getenv("AZURE_URL")
        self.api_key = os.getenv("AZURE_KEY")
        self.model = "prebuilt-layout"  # Hardcoded to prebuilt-layout

    def process_document(self, file_path):
        print(f"Processing document with Azure Document Analysis: {file_path}")

        if not self.api_key or not self.endpoint:
            error_msg = "ERROR: AZURE_KEY or AZURE_URL environment variables not set"
            print(error_msg)
            return error_msg

        max_retries = 3
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                from azure.core.credentials import AzureKeyCredential
                from azure.ai.documentintelligence import DocumentIntelligenceClient
                from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
                from azure.core.exceptions import HttpResponseError

                print("Analyzing document with Azure Document Intelligence...")

                # Create client
                client = DocumentIntelligenceClient(
                    endpoint=self.endpoint, 
                    credential=AzureKeyCredential(self.api_key)
                )

                # Read file
                with open(file_path, "rb") as f:
                    file_content = f.read()

                # Analyze document with rate limiting awareness
                poller = client.begin_analyze_document(
                    self.model, 
                    AnalyzeDocumentRequest(bytes_source=file_content)
                )
                
                # Clean up file content immediately after sending
                del file_content
                gc.collect()
                
                result = poller.result()
                
                # Extract tables and convert to HTML
                tables = result.tables if result.tables else []
                table_html = self.convert_tables_to_html(tables)
                
                # Clean up large Azure response objects
                try:
                    del result, poller, tables
                except Exception:
                    pass
                gc.collect()
                
                return table_html

            except ImportError:
                error_msg = "ERROR: azure-ai-documentintelligence not installed. Please install with: pip install azure-ai-documentintelligence"
                print(error_msg)
                return error_msg
            except Exception as e:
                # Clean up on exception
                try:
                    del client, file_content
                except Exception:
                    pass
                gc.collect()
                
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"ERROR processing document with Azure Document Analysis after {max_retries} retries: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    raise Exception(f"AzureDocumentAnalysisProcessor API failed after {max_retries} retries: {str(e)}")
                else:
                    print(f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 2

            finally:
                try:
                    del client, result, table_html, tables
                except Exception:
                    pass
                gc.collect()

    def convert_tables_to_html(self, tables):
        """Convert Azure Document Analysis tables to HTML"""
        if not tables:
            return "<table></table>"
        
        html_tables = []
        
        for table in tables:
            html = "<table>"
            
            # Get table dimensions
            row_count = table.row_count
            column_count = table.column_count
            
            # Create a grid to track cell placement
            grid = [[None for _ in range(column_count)] for _ in range(row_count)]
            
            # Place cells in the grid
            for cell in table.cells:
                row_index = cell.row_index
                column_index = cell.column_index
                row_span = getattr(cell, 'row_span', 1) or 1
                column_span = getattr(cell, 'column_span', 1) or 1
                content = cell.content or ""
                kind = getattr(cell, 'kind', 'content')
                
                # Place the cell in the grid
                if row_index < row_count and column_index < column_count:
                    grid[row_index][column_index] = {
                        "content": content,
                        "row_span": row_span,
                        "column_span": column_span,
                        "kind": kind
                    }
                    
                    # Mark spanned cells as occupied
                    for r in range(row_index, min(row_index + row_span, row_count)):
                        for c in range(column_index, min(column_index + column_span, column_count)):
                            if r != row_index or c != column_index:
                                grid[r][c] = "spanned"
            
            # Generate HTML rows
            for row_index in range(row_count):
                html += "<tr>"
                for column_index in range(column_count):
                    cell = grid[row_index][column_index]
                    
                    if cell == "spanned":
                        # Skip cells that are part of a span
                        continue
                    elif cell is None:
                        # Empty cell
                        html += "<td></td>"
                    else:
                        # Regular cell
                        tag = "th" if cell["kind"] in ["columnHeader", "rowHeader"] else "td"
                        
                        # Add span attributes if needed
                        attrs = ""
                        if cell["row_span"] > 1:
                            attrs += f' rowspan="{cell["row_span"]}"'
                        if cell["column_span"] > 1:
                            attrs += f' colspan="{cell["column_span"]}"'
                        
                        html += f"<{tag}{attrs}>{cell['content']}</{tag}>"
                
                html += "</tr>"
            
            html += "</table>"
            html_tables.append(html)
        
        return "\n".join(html_tables)