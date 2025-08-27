import numpy as np
import librosa
import sys
import os
from tqdm import tqdm

# --- 导入文件处理模块 ---
# 确保 file_processor.py 和 stoi_eval.py 在同一目录下
try:
    from file_processor import process_input_file, FileProcessingError
except ImportError:
    print("Error: Could not import 'file_processor.py'. Make sure it's in the same directory.")
    sys.exit(1)

def stoi_wrapper(reference: np.ndarray, estimation: np.ndarray, sr: int = 24000) -> float:
    """Wrapper to allow independent axis for STOI.

    Args:
        reference: numpy.ndarray, shape [..., num_samples] (干净信号)
        estimation: numpy.ndarray, shape [..., num_samples] (估计信号)
        sr: Sample rate.

    Returns:
        STOI score (float)
    """
    try:
        from pystoi.stoi import stoi as pystoi_stoi
    except ImportError:
        raise ImportError("请先安装 pystoi 库: pip install pystoi")

    # 确保输入是浮点数类型，且可广播
    estimation, reference = np.broadcast_arrays(estimation, reference)

    # pystoi_stoi 函数期望的是一维数组
    if reference.ndim >= 2:
        # 如果输入是多维的（例如多声道或批处理），则按行迭代计算
        # 注意：STOI通常用于单声道语音，多声道可能需要特殊处理或平均
        return np.array([
            pystoi_stoi(x_entry, y_entry, fs_sig=sr)
            for x_entry, y_entry in zip(reference, estimation)
        ])
    else:
        # 对于单声道一维数组直接计算
        return pystoi_stoi(reference, estimation, fs_sig=sr)
    
# --- STOI 批量处理和结果保存函数 ---
# 替换原有的 process_files 函数
def calculate_and_save_stoi(input_txt_path: str, pred_audio_base_dir: str, result_file_path: str):
    """
    使用 file_processor 提供的文件对，计算 STOI 并保存结果。
    
    Args:
        input_txt_path (str): 包含 GT 文件相对路径和前缀的文本文件。
        pred_audio_base_dir (str): 预测音频文件的根目录。
        result_file_path (str): STOI 结果将要保存的完整文件路径。
    """
    # **确保结果目录存在**
    result_dir = os.path.dirname(result_file_path)
    os.makedirs(result_dir, exist_ok=True) 

    all_stoi_scores = []
    
    print(f"Starting STOI calculation. Input list: {input_txt_path}, Pred base dir: {pred_audio_base_dir}")

    # **打开结果文件，并准备写入**
    with open(result_file_path, 'w', newline='') as out:
        # **调用 file_processor.process_input_file 来获取文件对**
        try:
            # 同样，假设 `parts[2]` (即 `wav_info`) 在 input_txt_path 中是 GT 文件的完整路径
            # 所以 `gt_base_path` 参数传入空字符串 `""`
            for gt_file, pred_file in tqdm(
                process_input_file("", pred_audio_base_dir, input_txt_path),
                desc="Calculating STOI"
            ):
                try:
                    # 加载音频文件
                    # STOI 对采样率敏感，确保 `sr` 参数与你的 STOI 算法期望的采样率一致
                    # 你的原始代码中是 sr=24000
                    target_sr = 24000 
                    ref_audio, sr_ref = librosa.load(gt_file, sr=target_sr)
                    deg_audio, sr_deg = librosa.load(pred_file, sr=target_sr)

                    if sr_ref != target_sr or sr_deg != target_sr:
                        print(f"Warning: Audio for {os.path.basename(gt_file)} was resampled to {target_sr}Hz. Original SR was {sr_ref}/{sr_deg}. Skipping if actual loaded SR differs.")
                        # 再次检查，确保librosa真的加载到了目标采样率
                        # 在大多数情况下，librosa.load(..., sr=target_sr) 会自动处理重采样
                        # 这里的assert可以确保最终用于计算的sr是目标sr
                        # assert sr_ref == target_sr and sr_deg == target_sr, "Loaded audio not at target sampling rate!"
                        # 如果assert触发，说明librosa加载有问题，或者你希望严格控制重采样过程
                        # 在这里，我们假设librosa.load(sr=target_sr)能够确保最终采样率
                    
                    # 对齐长度
                    min_len = min(len(ref_audio), len(deg_audio))
                    ref_audio = ref_audio[:min_len]
                    deg_audio = deg_audio[:min_len]

                    # 计算 STOI
                    stoi_score = stoi_wrapper(ref_audio, deg_audio, sr=target_sr)
                    # pystoi 返回的是一个浮点数，直接写入
                    out.write(f"{stoi_score:.4f}\n")
                    all_stoi_scores.append(stoi_score)

                except Exception as e:
                    print(f"⚠️ Error processing file pair ({gt_file}, {pred_file}): {e}. Skipping.")
                    out.write("N/A\n") # 写入 N/A 表示该样本计算失败
                    continue
                    
        except FileProcessingError as e:
            print(f"Error during file list processing for STOI: {e}")
            sys.exit(1) # 如果文件列表本身有问题，则退出

        # 计算并写入平均值
        if all_stoi_scores:
            avg_stoi = sum(all_stoi_scores) / len(all_stoi_scores)
            out.write(f"avg:{avg_stoi:.4f}\n")
            print(f"✅ STOI comparison complete. Results saved to {result_file_path}")
            print(f"Average STOI: {avg_stoi:.4f}")
        else:
            out.write("No valid STOI scores computed.\n")
            print("No valid STOI scores could be computed.")

# --- 主函数入口 ---
def main():
    """主函数，获取命令行参数并执行处理"""
    if len(sys.argv) != 3:
        print("Usage: python stoi_eval.py <input_txt_path> <pred_audio_base_dir>")
        print("  <input_txt_path>: Path to the text file listing GT/Pred file pairs.")
        print("  <pred_audio_base_dir>: Base directory where predicted audio files are located.")
        sys.exit(1)

    input_txt_path = sys.argv[1]     # 列表文件路径 (e.g., /path/to/my_list.txt)
    pred_audio_base_dir = sys.argv[2] # 预测音频的根目录 (e.g., /path/to/predicted_wavs)

    # 构造结果保存路径，这将确保在 pred_audio_base_dir 下有一个 'result' 文件夹
    result_save_path = os.path.join(pred_audio_base_dir, "result", "stoi_result")
    
    # 调用新的计算和保存函数
    calculate_and_save_stoi(input_txt_path, pred_audio_base_dir, result_save_path)

if __name__ == "__main__":
    main()
