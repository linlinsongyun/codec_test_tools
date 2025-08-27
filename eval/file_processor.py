




import os
from typing import Iterator, Tuple

class FileProcessingError(Exception):
    """Custom exception for file processing issues."""
    pass

def process_input_file(gt_base_path: str, pred_base_path: str, input_txt_path: str) -> Iterator[Tuple[str, str]]:

    skipped_files_count = 0
    total_lines = 0

    if not os.path.isfile(input_txt_path):
        raise FileProcessingError(f"Error: Input file '{input_txt_path}' not found.")

    with open(input_txt_path, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')
            
            if len(parts) < 3: # 确保至少有三列
                print(f"Warning: Line {line_num + 1} has invalid format: '{line}'. Skipping.")
                skipped_files_count += 1
                continue
            gt_relative_path = parts[2].strip()
            # parts[0]+.wav是合成音频名，parts[2]是原始GT文件的相对路径
            if not gt_relative_path.endswith('.wav'):
                print(f"Warning: GT file path '{gt_relative_path}' in line {line_num + 1} does not end with '.wav'. Skipping.")
                skipped_files_count += 1
                continue
            
            gt_file_path = os.path.join(gt_base_path, gt_relative_path)

            #pred_file_name = parts[0].strip() + '.wav'
            pred_file_name = parts[0]
            pred_file_path = os.path.join(pred_base_path, pred_file_name)


            if not os.path.isfile(gt_file_path):
                print(f"Warning: Ground truth file not found: {gt_file_path}. Skipping.")
                skipped_files_count += 1
                continue
            
            if not os.path.isfile(pred_file_path):
                print(f"Warning: Predicted file missing: {pred_file_path}. Skipping.")
                skipped_files_count += 1
                continue

            yield gt_file_path, pred_file_path
            
    if skipped_files_count > 0:
        print(f"Finished processing. {skipped_files_count} out of {total_lines} lines were skipped due to errors.")
    elif total_lines == 0:
        print("Warning: Input file is empty or contains no valid lines.")
