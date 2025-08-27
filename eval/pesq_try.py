import numpy as np
import librosa
import sys
import os
from tqdm import tqdm

# --- 导入文件处理模块 ---
# 确保 file_processor.py 和 pesq_eval.py 在同一目录下
try:
    from file_processor import process_input_file, FileProcessingError
except ImportError:
    print("Error: Could not import 'file_processor.py'. Make sure it's in the same directory.")
    sys.exit(1)

# --- 导入 PESQ 库 ---
try:
    from pesq import pesq as pesq_func # 避免与自定义函数名冲突
except ImportError:
    raise ImportError("请先安装 pesq: pip install pesq")

# --- PESQ 计算核心函数 ---
def compute_pesq(reference_path, estimation_path, sample_rate=16000, mode=None):
    """
    读取两个 wav 路径，计算 PESQ 分数
    """
    try:
        reference, sr_ref = librosa.load(reference_path, sr=sample_rate)
        estimation, sr_est = librosa.load(estimation_path, sr=sample_rate)
    except Exception as e:
        raise ValueError(f"Failed to load audio files: {reference_path} or {estimation_path}. Error: {e}")

    if sr_ref != sample_rate or sr_est != sample_rate:
        raise ValueError(f"Sampling rate mismatch for PESQ: Expected {sample_rate}, got ref:{sr_ref}, est:{sr_est}")

    # 对齐长度
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]

    if mode is None:
        if sample_rate == 8000:
            mode = 'nb'
        elif sample_rate == 16000:
            mode = 'wb'
        else:
            raise ValueError(f"Unsupported sample rate for PESQ mode auto-detection: {sample_rate}. Please specify mode manually.")

    try:
        score = pesq_func(sample_rate, reference, estimation, mode)
        return score
    except Exception as e:
        raise RuntimeError(f"PESQ calculation failed for {os.path.basename(reference_path)}: {e}")

# --- PESQ 批量处理和结果保存函数 ---
# 注意：这个函数替换了你之前 pesq_eval.py 中的 process_files
def calculate_and_save_pesq(input_txt_path: str, pred_audio_base_dir: str, result_file_path: str):
    """
    使用 file_processor 提供的文件对，计算 PESQ 并保存结果。
    
    Args:
        input_txt_path (str): 包含 GT 文件相对路径和前缀的文本文件。
        pred_audio_base_dir (str): 预测音频文件的根目录。
        result_file_path (str): PESQ 结果将要保存的完整文件路径。
    """
    # **关键修改点1: 在文件写入前确保目录存在**
    # 获取结果文件所在的目录
    result_dir = os.path.dirname(result_file_path)
    # 如果目录不存在，则创建它，包括所有中间目录
    os.makedirs(result_dir, exist_ok=True) 

    all_pesq_scores = []
    
    print(f"Starting PESQ calculation. Input list: {input_txt_path}, Pred base dir: {pred_audio_base_dir}")

    try:
        

        for gt_file, pred_file in tqdm(
            process_input_file("", pred_audio_base_dir, input_txt_path), # 第一个参数设为""，因为gt_file_path似乎在txt里就是完整的
            desc="Calculating PESQ"
        ):
            try:
                pesq_score = compute_pesq(gt_file, pred_file, sample_rate=16000)
                all_pesq_scores.append(pesq_score)
            except (ValueError, RuntimeError) as e: # 捕获 compute_pesq 内部可能抛出的异常
                print(f"⚠️ Failed to compute PESQ for {os.path.basename(gt_file)} vs {os.path.basename(pred_file)}: {e}")
                continue
                
    except FileProcessingError as e:
        print(f"Error during file processing: {e}")
        sys.exit(1) # 如果文件列表本身有问题，则退出

    # 写入结果
    with open(result_file_path, 'w', newline='') as out:
        if all_pesq_scores:
            for score in all_pesq_scores:
                out.write(f"{score:.4f}\n")
            
            avg_pesq = sum(all_pesq_scores) / len(all_pesq_scores)
            out.write(f"avg:{avg_pesq:.4f}\n")
            print(f"PESQ scores saved to {result_file_path}")
            print(f"Average PESQ: {avg_pesq:.4f}")
        else:
            out.write("No valid PESQ scores computed.\n")
            print("No valid PESQ scores could be computed.")

# --- 主函数入口 ---
def main():
    if len(sys.argv) != 3:
        print("Usage: python pesq_eval.py <input_txt_path> <pred_audio_base_dir>")
        print("  <input_txt_path>: Path to the text file listing GT/Pred file pairs.")
        print("  <pred_audio_base_dir>: Base directory where predicted audio files are located.")
        sys.exit(1)

    input_txt_path = sys.argv[1]     # 列表文件路径 (如: /path/to/my_list.txt)
    pred_audio_base_dir = sys.argv[2] # 预测音频的根目录 (如: /path/to/predicted_wavs)

    # 构造结果保存路径，并确保目录存在
    # 这是最终的结果文件名，例如：/path/to/predicted_wavs/result/pesq_result.txt
    result_save_path = os.path.join(pred_audio_base_dir, "result", "pesq_result")
    result_dir = os.path.dirname(result_save_path)

    os.makedirs(result_dir, exist_ok=True)
    
    # 调用新的计算和保存函数
    calculate_and_save_pesq(input_txt_path, pred_audio_base_dir, result_save_path)

if __name__ == "__main__":
    main()
