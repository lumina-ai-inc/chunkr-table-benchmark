#!/usr/bin/env python3
"""
Mathpix API processor for table extraction with LaTeX to HTML conversion.
"""

import gc
import json
import os
import re
import time

import requests


class MathpixProcessor:
    def __init__(self, model):
        self.app_id = os.getenv("MATHPIX_APP_ID")
        self.app_key = os.getenv("MATHPIX_APP_KEY")
        self.model = model
        self.api_url = "https://api.mathpix.com/v3/text"

    def clean_mathpix_html(self, html_content):
        """Keep all Mathpix HTML content, only remove hidden LaTeX elements"""
        try:
            from bs4 import BeautifulSoup, Tag
            
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            if not table or not isinstance(table, Tag):
                return "<table></table>"
            
            # Only remove hidden LaTeX elements but keep all visible content and styling
            latex_elements = table.find_all('latex')
            for latex_elem in latex_elements:
                if isinstance(latex_elem, Tag):
                    style_attr = latex_elem.get('style', '')
                    if style_attr and 'display: none' in str(style_attr):
                        latex_elem.decompose()
            
            # Return the complete table with all styling preserved
            return str(table)
            
        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            # Fallback: return original if it contains table
            if '<table' in html_content:
                return html_content
            return "<table></table>"

    def process_document(self, file_path):
        print(f"Processing document with Mathpix API: {file_path}")

        if not self.app_id or not self.app_key:
            print("ERROR: MATHPIX_APP_ID or MATHPIX_APP_KEY environment variables not set")
            return None

        max_retries = 3
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                print("Processing document with Mathpix API...")

                # Read file content
                with open(file_path, "rb") as f:
                    file_content = f.read()
                
                files = {"file": ("image.png", file_content, "image/png")}
                
                # Use your proven working request format
                data = {
                    "options_json": json.dumps({
                        "formats": ["text", "data", "html"],
                        "data_options": {
                            "include_tsv": True,
                            "include_table_html": True,
                            "include_latex": True
                        },
                        "math_inline_delimiters": ["$", "$"],
                        "rm_spaces": True
                    })
                }
                
                headers = {
                    "app_id": self.app_id,
                    "app_key": self.app_key
                }

                response = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=60
                )
                
                # Clean up file content immediately
                del file_content, files

                if response.status_code != 200:
                    error_message = f"Mathpix API Error - Status: {response.status_code}, Response: {response.text}"
                    print(f"MATHPIX API ERROR: {error_message}")
                    raise Exception(error_message)

                # Parse the response
                result = response.json()
                
                # Method 1: Extract HTML from data array (your working example)
                if "data" in result and isinstance(result["data"], list):
                    for item in result["data"]:
                        if item.get("type") == "html" and item.get("value"):
                            html_content = item["value"]
                            table_html = self.clean_mathpix_html(html_content)
                            print("Using HTML from data array")
                            print(f"Mathpix extraction completed. Confidence: {result.get('confidence', 'unknown')}")
                            return table_html
                
                # Method 2: Extract HTML from top-level html field  
                if "html" in result and result["html"]:
                    html_content = result["html"]
                    table_html = self.clean_mathpix_html(html_content)
                    print("Using HTML from top-level field")
                    print(f"Mathpix extraction completed. Confidence: {result.get('confidence', 'unknown')}")
                    return table_html
                
                # If no HTML found, return empty table
                print("No HTML found in Mathpix response")
                return "<table></table>"

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"ERROR processing document with Mathpix API after {max_retries} retries: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    raise Exception(f"MathpixProcessor API failed after {max_retries} retries: {str(e)}")
                else:
                    print(f"Error on attempt {retry_count}/{max_retries}: {str(e)}. Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 2

            finally:
                try:
                    # Clean up variables if they exist
                    result = locals().get('result')
                    table_html = locals().get('table_html')
                    if result is not None:
                        del result
                    if table_html is not None:
                        del table_html
                except Exception:
                    pass
                gc.collect()

    def extract_table_from_html(self, html_content):
        """Extract table from full HTML document returned by API"""
        return self.clean_mathpix_html(html_content)

    def convert_latex_math_to_html(self, latex_input):
        """Convert LaTeX math expressions to HTML"""
        # Simple conversion for basic LaTeX math
        html_output = latex_input
        
        # Convert superscripts (^{})
        html_output = re.sub(r'\^{([^}]+)}', r'<sup>\1</sup>', html_output)
        html_output = re.sub(r'\^(\w)', r'<sup>\1</sup>', html_output)
        
        # Convert subscripts (_{})
        html_output = re.sub(r'_{([^}]+)}', r'<sub>\1</sub>', html_output)
        html_output = re.sub(r'_(\w)', r'<sub>\1</sub>', html_output)
        
        return html_output

    def latex_to_html_table(self, latex_table):
        """Convert LaTeX table to HTML table"""
        try:
            # Simple LaTeX table parser for testing
            lines = latex_table.strip().split('\n')
            html_lines = ['<table>']
            
            for line in lines:
                line = line.strip()
                if line.startswith('\\begin{tabular}'):
                    continue
                elif line.startswith('\\end{tabular}'):
                    continue
                elif '&' in line:
                    # Process table row
                    cells = [cell.strip() for cell in line.replace('\\\\', '').split('&')]
                    html_lines.append('<tr>')
                    for cell in cells:
                        html_lines.append(f'<td>{cell}</td>')
                    html_lines.append('</tr>')
            
            html_lines.append('</table>')
            return '\n'.join(html_lines)
        except Exception as e:
            print(f"Error converting LaTeX table: {e}")
            return "<table></table>"
