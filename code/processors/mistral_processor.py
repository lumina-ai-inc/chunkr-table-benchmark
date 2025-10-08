#!/usr/bin/env python3
"""
Mistral OCR API processor for table extraction.
"""

import os
import base64
import gc
import time


class MistralProcessor:
    def __init__(self, model):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = model or "mistral-ocr-latest"

    def encode_image(self, image_path):
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            error_msg = f"Error: The file {image_path} was not found."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: {e}"
            print(error_msg)
            return error_msg

    def process_document(self, file_path):
        print(f"Processing document with Mistral OCR: {file_path}")

        if not self.api_key:
            error_msg = "ERROR: MISTRAL_API_KEY environment variable not set"
            print(error_msg)
            return error_msg

        max_retries = 3
        retry_count = 0
        backoff_time = 1

        # Encode image to base64 ONCE outside retry loop
        base64_image = self.encode_image(file_path)
        if not base64_image or base64_image.startswith("Error:"):
            return base64_image if base64_image else "ERROR: Failed to encode image to base64"

        while retry_count < max_retries:
            ocr_response = None
            client = None
            try:
                from mistralai import Mistral

                print("Processing document with Mistral OCR API...")

                # Create client
                client = Mistral(api_key=self.api_key)

                # Process document with OCR
                ocr_response = client.ocr.process(
                    model=self.model,
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    include_image_base64=False  # Don't duplicate image data!
                )
 
                # Extract markdown content and convert to HTML table
                result_html = "<table></table>"  # Default
                
                if hasattr(ocr_response, 'pages') and ocr_response.pages:
                    # Mistral OCR returns pages with markdown content
                    for page in ocr_response.pages:
                        if hasattr(page, 'markdown') and page.markdown:
                            markdown_content = page.markdown
                            print(f"DEBUG: Found markdown content (full): {markdown_content}")
                            result_html = self.convert_markdown_table_to_html(markdown_content)
                            break
                    
                    if result_html == "<table></table>":
                        print("No markdown content found in pages")
                
                # Fallback: check for text attributes
                elif hasattr(ocr_response, 'text') and ocr_response.text:
                    # Convert OCR text to HTML table format
                    text_content = ocr_response.text
                    result_html = self.convert_ocr_text_to_table(text_content)
                elif hasattr(ocr_response, 'content') and ocr_response.content:
                    # Alternative attribute name
                    text_content = ocr_response.content
                    result_html = self.convert_ocr_text_to_table(text_content)
                else:
                    # Fallback: check response attributes
                    print(f"DEBUG: OCR response attributes: {dir(ocr_response)}")
                    print(f"DEBUG: OCR response: {ocr_response}")
                    # Try to extract any text-like content
                    for attr in ['text', 'content', 'result', 'output']:
                        if hasattr(ocr_response, attr):
                            content = getattr(ocr_response, attr)
                            if content:
                                result_html = self.convert_ocr_text_to_table(str(content))
                                break
                    
                    if result_html == "<table></table>":
                        print("No text content found in OCR response")
                
                # Clean up large objects before return
                try:
                    del ocr_response, client
                except Exception:
                    pass
                gc.collect()
                
                return result_html

            except ImportError:
                error_msg = "ERROR: mistralai not installed. Please install with: pip install mistralai"
                print(error_msg)
                return error_msg
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"ERROR processing document with Mistral OCR after {max_retries} retries: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    # Clean up before raising
                    try:
                        del base64_image
                    except Exception:
                        pass
                    gc.collect()
                    raise Exception(f"MistralProcessor API failed after {max_retries} retries: {str(e)}")
                else:
                    print(f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds...")
                    # Clean up failed attempt objects
                    try:
                        del ocr_response, client
                    except Exception:
                        pass
                    gc.collect()
                    time.sleep(backoff_time)
                    backoff_time *= 2
        
        # Clean up base64_image after all retries
        try:
            del base64_image
        except Exception:
            pass
        gc.collect()

    def convert_markdown_table_to_html(self, markdown_content):
        """Convert Mistral OCR markdown table to HTML table format"""
        if not markdown_content:
            return "<table></table>"
        
        print(f"DEBUG: Converting markdown: {markdown_content}")
        
        # Split by lines and find table rows (lines that start and end with |)
        lines = markdown_content.strip().split('\n')
        table_rows = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                # Remove leading and trailing pipes and split by |
                cells = [cell.strip() for cell in line[1:-1].split('|')]
                # Filter out empty cells but keep cells with meaningful content
                filtered_cells = []
                for cell in cells:
                    if cell and cell not in ['', ' ']:
                        filtered_cells.append(cell)
                
                if filtered_cells:  # Only add rows with actual content
                    table_rows.append(filtered_cells)
        
        if not table_rows:
            # No valid table found, treat as plain text
            return self.convert_ocr_text_to_table(markdown_content)
        
        # Convert to HTML
        html = "<table>"
        
        for i, row in enumerate(table_rows):
            html += "<tr>"
            
            for cell in row:
                # Use th for first row if it contains table headers
                is_header = i == 0 and any(indicator in cell for indicator in ['売上', '増加', '減少', '不変', '業'])
                tag = "th" if is_header else "td"
                
                # Clean up cell content
                cell_content = cell.strip()
                if cell_content:
                    html += f"<{tag}>{cell_content}</{tag}>"
            
            html += "</tr>"
        
        html += "</table>"
        
        print(f"DEBUG: Generated HTML: {html}")
        return html

    def convert_ocr_text_to_table(self, text_content):
        """Convert OCR extracted text to HTML table format"""
        if not text_content:
            return "<table></table>"
        
        # Look for tabular patterns in the text
        lines = text_content.strip().split('\n')
        
        # Try to detect table-like content
        table_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Check if line looks like table data (contains percentages, numbers, etc.)
                if any(indicator in line for indicator in ['%', '増加', '減少', '不変', '業', '売上']) or \
                   len(line.split()) > 2:  # Multiple words/values
                    table_lines.append(line)
        
        if not table_lines:
            # Fallback: wrap all content in a simple table
            return f"<table><tr><td>{text_content}</td></tr></table>"
        
        # Convert to HTML table
        html = "<table>"
        
        for i, line in enumerate(table_lines):
            html += "<tr>"
            
            # Split line into cells by common separators
            cells = []
            
            # Try different splitting strategies
            if '\t' in line:
                cells = line.split('\t')
            elif '  ' in line:  # Multiple spaces
                cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
            else:
                # Split by single space and group consecutive similar tokens
                words = line.split()
                cells = words
            
            # Ensure we have reasonable cells
            if not cells:
                cells = [line]
            
            # Add cells to row
            for cell in cells:
                cell = cell.strip()
                if cell:
                    # Use th for first row if it looks like headers
                    tag = "th" if i == 0 and any(header in cell for header in ['売上', '増加', '減少', '不変']) else "td"
                    html += f"<{tag}>{cell}</{tag}>"
            
            html += "</tr>"
        
        html += "</table>"
        return html