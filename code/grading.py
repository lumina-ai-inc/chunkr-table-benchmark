import os
import numpy as np
import numpy.typing as npt
from Levenshtein import distance as levenshtein_distance
import html
from bs4 import BeautifulSoup
import re
import unicodedata
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import numpy.typing as npt
from lxml import etree

TEDS_EMPTY_PENALTY = float(os.getenv("TEDS_EMPTY_PENALTY", "1.0"))


def is_blank_cell_label(label: str) -> bool:
    if not label:
        return True
    parts = label.split('|')
    if len(parts) < 1:
        return True
    text_part = parts[0].strip()
    return text_part == "" or text_part.isspace()

def canonicalize_table(html: str) -> str:
    soup = None
    try:
        soup = BeautifulSoup(html, "lxml")
        
        tbl = soup.find("table")
        if not tbl:
            for tag in soup.find_all(["b", "i", "strong", "em", "span"]):
                tag.unwrap()
            body = soup.find("body")
            if body:
                content = ''.join(str(child) for child in body.children)
                return re.sub(r"\s+", " ", content).strip()
            return re.sub(r"\s+", " ", str(soup)).strip()
        
        # Remove empty math spans that Mathpix generates
        for span in tbl.find_all("span", class_="math-inline"):
            if not span.get_text().strip():  # Empty math span
                span.decompose()  # Remove completely
            else:
                span.unwrap()  # Keep content, remove span
        
        # Remove TSV elements that Mathpix includes
        for tsv in tbl.find_all("tsv"):
            tsv.decompose()
        
        # Remove nested table divs that Mathpix creates
        for div in tbl.find_all("div", class_="inline-tabular"):
            div.unwrap()  # Keep nested table, remove wrapper
        
        # Standard canonicalization
        for tag in tbl.find_all(["thead", "tbody", "tfoot"]):
            tag.unwrap()
        for th in tbl.find_all("th"):
            th.name = "td"
        for tag in tbl.find_all(["b", "i", "strong", "em", "span"]):
            tag.unwrap()
        
        # Remove ALL attributes except rowspan/colspan to ensure clean comparison
        # Process the table element itself first
        if tbl.attrs:
            tbl.attrs.clear()
        
        # Then process all child elements
        for tag in tbl.find_all():
            if tag.attrs:
                # Keep only essential structural attributes
                preserved_attrs = {}
                for attr in ['rowspan', 'colspan']:
                    if attr in tag.attrs and tag.attrs[attr] != '1':  # Only keep if > 1
                        preserved_attrs[attr] = tag.attrs[attr]
                # Clear all attributes first, then set preserved ones
                tag.attrs.clear()
                tag.attrs.update(preserved_attrs)
        
        body = soup.find("body")
        if body:
            content = ''.join(str(child) for child in body.children)
            out = re.sub(r"\s+", " ", content).strip()
        else:
            out = re.sub(r"\s+", " ", str(soup)).strip()
        
        return out
    finally:
        import gc

        try:
            if soup is not None:
                soup.clear()
                del soup
        except Exception:
            pass
        gc.collect()

def compute_cell_alignment_score(first_cell: str | None, second_cell: str | None) -> float:
    if first_cell is None or second_cell is None:
        return -1.0
    if first_cell == second_cell:
        return 1.0
    edit_dist = levenshtein_distance(first_cell, second_cell)
    max_length = max(len(first_cell), len(second_cell))
    if max_length == 0:
        return 0.0
    similarity_ratio = 1.0 - (edit_dist / max_length)
    weighted_score = -1.0 + similarity_ratio * (1.0 - (-1.0))
    return weighted_score

def html_to_numpy(html_string: str) -> npt.NDArray[np.str_]:
    dom_tree = etree.HTML(html_string, parser=etree.HTMLParser())
    if dom_tree is None:
        return np.array([], dtype=str)

    table_rows: list[list[str]] = []
    span_info: dict[int, tuple[str, int]] = {}

    tr_elements = dom_tree.xpath("//tr")
    if not tr_elements:
        return np.array([], dtype=str)

    for table_row in dom_tree.xpath("//tr"):
        current_row: list[str] = []
        column_index = 0

        while span_info.get(column_index, (None, 0))[1] > 0:
            current_row.append(span_info[column_index][0])
            span_info[column_index] = (
                span_info[column_index][0],
                span_info[column_index][1] - 1,
            )
            if span_info[column_index][1] == 0:
                del span_info[column_index]
            column_index += 1

        for table_cell in table_row.xpath("td|th"):
            while span_info.get(column_index, (None, 0))[1] > 0:
                current_row.append(span_info[column_index][0])
                span_info[column_index] = (
                    span_info[column_index][0],
                    span_info[column_index][1] - 1,
                )
                if span_info[column_index][1] == 0:
                    del span_info[column_index]
                column_index += 1

            row_span = safe_parse_span(table_cell.get("rowspan", "1"))
            col_span = safe_parse_span(table_cell.get("colspan", "1"))
            cell_text = "".join(table_cell.itertext()).strip()

            if row_span > 1:
                for i in range(col_span):
                    span_info[column_index + i] = (cell_text, row_span - 1)

            for _ in range(col_span):
                current_row.append(cell_text)
            column_index += col_span

        while span_info.get(column_index, (None, 0))[1] > 0:
            current_row.append(span_info[column_index][0])
            span_info[column_index] = (
                span_info[column_index][0],
                span_info[column_index][1] - 1,
            )
            if span_info[column_index][1] == 0:
                del span_info[column_index]
            column_index += 1

        table_rows.append(current_row)

    max_columns = max(map(len, table_rows)) if table_rows else 0
    for row in table_rows:
        row.extend([""] * (max_columns - len(row)))

    additional_elements = dom_tree.xpath("//p")
    for elem in additional_elements:
        text_content = "".join(elem.itertext()).strip()
        if text_content:
            content_row = [text_content] + [""] * (max_columns - 1)
            table_rows.append(content_row)

    return np.array(table_rows)


def safe_parse_span(value: str, default: int = 1) -> int:
    try:
        return round(float(value))
    except (ValueError, TypeError):
        return default


def perform_sequence_alignment(
    first_seq: list[str], second_seq: list[str], gap_cost: int
) -> tuple[list[str | None], list[str | None], float]:
    dp_matrix = None
    path_matrix = None
    final_column = None
    final_row = None
    result_seq1 = None
    result_seq2 = None
    try:
        seq1_len = len(first_seq)
        seq2_len = len(second_seq)
        dp_matrix = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.float32)
        path_matrix = np.full((seq1_len + 1, seq2_len + 1), None)
        for row_idx in range(1, seq1_len + 1):
            path_matrix[row_idx, 0] = "vertical"
        for col_idx in range(1, seq2_len + 1):
            path_matrix[0, col_idx] = "horizontal"
        for i in range(1, seq1_len + 1):
            curr_elem1 = first_seq[i - 1]
            for j in range(1, seq2_len + 1):
                curr_elem2 = second_seq[j - 1]
                diagonal_score = dp_matrix[i - 1, j - 1] + compute_cell_alignment_score(curr_elem1, curr_elem2)
                vertical_score = dp_matrix[i - 1, j] + gap_cost
                horizontal_score = dp_matrix[i, j - 1] + gap_cost
                optimal_score = max(diagonal_score, vertical_score, horizontal_score)
                dp_matrix[i, j] = optimal_score
                if optimal_score == diagonal_score:
                    path_matrix[i, j] = "diagonal"
                elif optimal_score == vertical_score:
                    path_matrix[i, j] = "vertical"
                else:
                    path_matrix[i, j] = "horizontal"
        end_i, end_j = seq1_len, seq2_len
        best_score = dp_matrix[end_i, end_j]
        optimal_i, optimal_j = end_i, end_j
        final_column = dp_matrix[:, seq2_len]
        if final_column.max() > best_score:
            optimal_i = final_column.argmax()
            optimal_j = seq2_len
            best_score = final_column[optimal_i]
        final_row = dp_matrix[seq1_len, :]
        if final_row.max() > best_score:
            optimal_i = seq1_len
            optimal_j = final_row.argmax()
            best_score = final_row[optimal_j]
        result_seq1 = []
        result_seq2 = []
        current_i, current_j = optimal_i, optimal_j
        while current_i > 0 or current_j > 0:
            direction = path_matrix[current_i, current_j]
            if current_i > 0 and current_j > 0 and direction == "diagonal":
                result_seq1.insert(0, first_seq[current_i - 1])
                result_seq2.insert(0, second_seq[current_j - 1])
                current_i -= 1
                current_j -= 1
            elif current_i > 0 and (current_j == 0 or direction == "vertical"):
                result_seq1.insert(0, first_seq[current_i - 1])
                result_seq2.insert(0, None)
                current_i -= 1
            elif current_j > 0 and (current_i == 0 or direction == "horizontal"):
                result_seq1.insert(0, None)
                result_seq2.insert(0, second_seq[current_j - 1])
                current_j -= 1
            else:
                break
        return result_seq1, result_seq2, best_score
    finally:
        import gc

        try:
            if dp_matrix is not None:
                del dp_matrix
            if path_matrix is not None:
                del path_matrix
            if final_column is not None:
                del final_column
            if final_row is not None:
                del final_row
        except Exception:
            pass
        gc.collect()

def evaluate_table_correspondence(
    reference_table: npt.NDArray[np.str_], candidate_table: npt.NDArray[np.str_]
) -> float:
    ref_normalized = None
    cand_normalized = None
    reference_rows = None
    candidate_rows = None
    pairwise_scores = None
    row_dp = None
    row_traceback = None
    alignment_cache = None
    try:
        if len(reference_table) == 0 or len(candidate_table) == 0:
            return 0.0
        if reference_table.shape[1] != candidate_table.shape[1]:
            target_cols = max(reference_table.shape[1], candidate_table.shape[1])
            if reference_table.shape[1] < target_cols:
                ref_padding = np.full(
                    (reference_table.shape[0], target_cols - reference_table.shape[1]),
                    "",
                    dtype=np.str_,
                )
                reference_table = np.hstack((reference_table, ref_padding))
            if candidate_table.shape[1] < target_cols:
                cand_padding = np.full(
                    (candidate_table.shape[0], target_cols - candidate_table.shape[1]),
                    "",
                    dtype=np.str_,
                )
                candidate_table = np.hstack((candidate_table, cand_padding))
        def standardize_cell_content(content: str) -> str:
            if not isinstance(content, str):
                return ""
            decoded_content = html.unescape(content)
            cleaned = "".join(decoded_content.replace("\n", " ").replace("-", "").split()).replace(" ", "")
            return cleaned
        normalize_func = np.vectorize(standardize_cell_content)
        ref_normalized = normalize_func(reference_table)
        cand_normalized = normalize_func(candidate_table)
        reference_rows = [list(row) for row in ref_normalized]
        candidate_rows = [list(row) for row in cand_normalized]
        ref_count = len(reference_rows)
        cand_count = len(candidate_rows)
        total_comparisons = ref_count * cand_count
        pairwise_scores = np.zeros((ref_count, cand_count), dtype=np.float32)
        alignment_cache = {}
        MAX_CACHE_SIZE = 100
        for ref_idx in range(ref_count):
            ref_row_data = reference_rows[ref_idx]
            for cand_idx in range(cand_count):
                cand_row_data = candidate_rows[cand_idx]
                cache_key = (tuple(ref_row_data), tuple(cand_row_data))
                if cache_key in alignment_cache:
                    column_alignment_score = alignment_cache[cache_key]
                else:
                    _, _, column_alignment_score = perform_sequence_alignment(ref_row_data, cand_row_data, -1)
                    if len(alignment_cache) < MAX_CACHE_SIZE:
                        alignment_cache[cache_key] = column_alignment_score
                pairwise_scores[ref_idx, cand_idx] = column_alignment_score + 5
        row_dp = np.zeros((ref_count + 1, cand_count + 1), dtype=np.float32)
        row_traceback = np.full((ref_count + 1, cand_count + 1), None)
        for ref_idx in range(1, ref_count + 1):
            row_traceback[ref_idx, 0] = "up"
        for cand_idx in range(1, cand_count + 1):
            row_traceback[0, cand_idx] = "left"
        for ref_idx in range(1, ref_count + 1):
            for cand_idx in range(1, cand_count + 1):
                match_score = row_dp[ref_idx - 1, cand_idx - 1] + pairwise_scores[ref_idx - 1, cand_idx - 1]
                delete_score = row_dp[ref_idx - 1, cand_idx] + (-3)
                insert_score = row_dp[ref_idx, cand_idx - 1] + (-3)
                best_score = max(match_score, delete_score, insert_score)
                row_dp[ref_idx, cand_idx] = best_score
                if best_score == match_score:
                    row_traceback[ref_idx, cand_idx] = "diag"
                elif best_score == delete_score:
                    row_traceback[ref_idx, cand_idx] = "up"
                else:
                    row_traceback[ref_idx, cand_idx] = "left"
        end_ref, end_cand = ref_count, cand_count
        optimal_score = row_dp[end_ref, end_cand]
        best_ref, best_cand = end_ref, end_cand
        last_column_scores = row_dp[:, cand_count]
        if last_column_scores.max() > optimal_score:
            best_ref = last_column_scores.argmax()
            best_cand = cand_count
            optimal_score = last_column_scores[best_ref]
        last_row_scores = row_dp[ref_count, :]
        if last_row_scores.max() > optimal_score:
            best_ref = ref_count
            best_cand = last_row_scores.argmax()
            optimal_score = last_row_scores[best_cand]
        actual_score = optimal_score
        if ref_count == 0 or cand_count == 0:
            return 0.0
        max_rows = max(ref_count, cand_count)
        max_cols = max(reference_table.shape[1], candidate_table.shape[1])
        theoretical_max = max_rows * (5 + max_cols * 1)
        total_possible_score = theoretical_max
        final_similarity = actual_score / total_possible_score
        return max(0.0, min(final_similarity, 1.0))
    finally:
        import gc

        try:
            if ref_normalized is not None:
                del ref_normalized
            if cand_normalized is not None:
                del cand_normalized
            if reference_rows is not None:
                reference_rows.clear()
                del reference_rows
            if candidate_rows is not None:
                candidate_rows.clear()
                del candidate_rows
            if pairwise_scores is not None:
                del pairwise_scores
            if row_dp is not None:
                del row_dp
            if row_traceback is not None:
                del row_traceback
            if alignment_cache is not None:
                alignment_cache.clear()
                del alignment_cache
        except Exception:
            pass
        gc.collect()


def compare_html_tables_from_canonicalized(
    normalized_gt: str, normalized_pred: str
) -> float:
    """Compare already-canonicalized HTML tables."""

    gt_array = html_to_numpy(normalized_gt)
    pred_array = html_to_numpy(normalized_pred)
    return evaluate_table_correspondence(gt_array, pred_array)


def compare_html_tables(html_ground_truth: str, html_prediction: str) -> float:
    normalized_gt = canonicalize_table(html_ground_truth)
    normalized_pred = canonicalize_table(html_prediction)
    return compare_html_tables_from_canonicalized(normalized_gt, normalized_pred)

try:
    from zss import Node

    ZSS_AVAILABLE = True
except ImportError:
    ZSS_AVAILABLE = False


def safe_parse_span(value: str, default: int = 1) -> int:
    try:
        return round(float(value))
    except (ValueError, TypeError):
        return default




def teds_score(html_pred: str, html_gt: str) -> float:
    try:
        from table_recognition_metric import TEDS
        
        def is_empty_table(html_str):
            import re
            # Remove whitespace and check if it's just <table></table>
            cleaned = re.sub(r'\s+', '', html_str.strip())
            return cleaned == '<table></table>' or '<tr>' not in html_str
        
        gt_empty = is_empty_table(html_gt)
        pred_empty = is_empty_table(html_pred)
        
        # Handle empty table cases correctly
        if gt_empty and pred_empty:
            return 0.0  # Both empty - handle division by zero
        elif gt_empty or pred_empty:
            return 0.0  # One empty, one not - should be 0.0
        
        teds = TEDS()
        
        def wrap_html(html_str):
            if not html_str.strip().startswith('<html>'):
                return f'<html><body>{html_str}</body></html>'
            return html_str
        
        # Use the full canonicalized HTML (includes p tags and other content)
        wrapped_gt = wrap_html(html_gt)
        wrapped_pred = wrap_html(html_pred)
        
        score = teds(wrapped_gt, wrapped_pred)
        
        # TEDs library can return negative scores - we cap at 0.0
        if score < 0:
            print(f"WARNING: Negative TEDS score detected ({score}), capping to 0.0")
            score = 0.0
        
        return float(score)
        
    except ImportError:
        print("ERROR: table_recognition_metric not installed. Install with: pip install table_recognition_metric")
        return 0.0
    except Exception as e:
        print(f"TEDS computation failed: {e}")
        return 0.0


def compute_scores(html_ground_truth: str, html_prediction: str, metrics: list = None) -> dict:
    import gc
    if metrics is None:
        metrics = ['levenshtein']
    
    scores = {}
    
    try:
        # Canonicalize ONCE for both metrics to ensure consistency
        normalized_gt = canonicalize_table(html_ground_truth)
        normalized_pred = canonicalize_table(html_prediction)
        
        if 'levenshtein' in metrics:
            try:
                scores['levenshtein'] = compare_html_tables_from_canonicalized(normalized_gt, normalized_pred)
            except Exception as e:
                scores['levenshtein'] = 0.0
                print(f"Levenshtein computation failed: {e}")
        
        if 'teds' in metrics:
            try:
                teds_result = teds_score(normalized_pred, normalized_gt)
                scores['teds'] = teds_result
            except Exception as e:
                scores['teds'] = 0.0
                print(f"TEDS computation failed: {e}")
                
    finally:
        # Clean up
        try:
            if 'normalized_gt' in locals():
                del normalized_gt
            if 'normalized_pred' in locals():
                del normalized_pred
        except Exception:
            pass
        gc.collect()
            
    return scores

