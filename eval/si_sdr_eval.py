import numpy as np
import librosa
import sys
import os
from tqdm import tqdm

# --- 导入文件处理模块 ---
# 确保 file_processor.py 和 si_sdr_eval.py 在同一目录下
try:
    from file_processor import process_input_file, FileProcessingError
except ImportError:
    print("Error: Could not import 'file_processor.py'. Make sure it's in the same directory.")
    sys.exit(1)

def compute_si_sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, shape [..., T] (干净信号)
        estimation: numpy.ndarray, shape [..., T] (估计信号)

    Returns:
        SI-SDR score (float) in dB
    """
    # 确保输入是浮点数类型
    reference = reference.astype(np.float64)
    estimation = estimation.astype(np.float64)

    # 广播数组以确保它们兼容 (如果维度不同，但可广播)
    estimation, reference = np.broadcast_arrays(estimation, reference)

    # 计算参考信号的能量
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # 防止除以零
    epsilon = 1e-8

    # 计算最优尺度因子 alpha
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / (reference_energy + epsilon)
    
    # 投影：估计信号在参考信号上的最佳线性投影
    projection = optimal_scaling * reference
    
    # 噪声：估计信号与投影信号之间的差异
    noise = estimation - projection

    # 计算信噪比 (SNR) 的平方
    # (投影信号的能量) / (噪声信号的能量)
    ratio = np.sum(projection ** 2, axis=-1) / (np.sum(noise ** 2, axis=-1) + epsilon)
    
    # 转换为 dB
    return 10 * np.log10(ratio + epsilon)

# --- SI-SDR 批量处理和结果保存函数 ---
# 替换原有的 process_files 函数
def calculate_and_save_si_sdr(input_txt_path: str, pred_audio_base_dir: str, result_file_path: str):
    """
    使用 file_processor 提供的文件对，计算 SI-SDR 并保存结果。
    
    Args:
        input_txt_path (str): 包含 GT 文件相对路径和前缀的文本文件。
        pred_audio_base_dir (str): 预测音频文件的根目录。
        result_file_path (str): SI-SDR 结果将要保存的完整文件路径。
    """
    # **确保结果目录存在**
    result_dir = os.path.dirname(result_file_path)
    os.makedirs(result_dir, exist_ok=True) 

    all_si_sdr_scores = []
    
    print(f"Starting SI-SDR calculation. Input list: {input_txt_path}, Pred base dir: {pred_audio_base_dir}")

    # **打开结果文件，并准备写入**
    with open(result_file_path, 'w', newline='') as out:
        # **调用 file_processor.process_input_file 来获取文件对**
        try:
            # 同样，假设 `parts[2]` (即 `wav_info`) 在 input_txt_path 中是 GT 文件的完整路径
            # 所以 `gt_base_path` 参数传入空字符串 `""`
            for gt_file, pred_file in tqdm(
                process_input_file("", pred_audio_base_dir, input_txt_path),
                desc="Calculating SI-SDR"
            ):
                try:
                    # 加载音频文件
                    # 注意：SI-SDR通常对采样率不敏感，但一致性很重要。
                    # 这里默认使用 16000 Hz，如果你的音频实际采样率不同，请调整。
                    ref_audio, sr_ref = librosa.load(gt_file, sr=16000)
                    deg_audio, sr_deg = librosa.load(pred_file, sr=16000)

                    if sr_ref != sr_deg:
                        # 这种情况在 librosa.load(sr=16000) 时通常不会发生，
                        # 但如果将来参数改变或文件损坏可能出现。
                        print(f"Warning: Sampling rates mismatch for {os.path.basename(gt_file)}. Skipping.")
                        out.write("N/A\n")
                        continue
                    
                    # 对齐长度
                    min_len = min(len(ref_audio), len(deg_audio))
                    ref_audio = ref_audio[:min_len]
                    deg_audio = deg_audio[:min_len]

                    # 计算 SI-SDR
                    si_sdr_score = compute_si_sdr(ref_audio, deg_audio)
                    out.write(f"{si_sdr_score:.4f}\n")
                    all_si_sdr_scores.append(si_sdr_score)

                except Exception as e:
                    print(f"⚠️ Error processing file pair ({gt_file}, {pred_file}): {e}. Skipping.")
                    out.write("N/A\n") # 写入 N/A 表示该样本计算失败
                    continue
                    
        except FileProcessingError as e:
            print(f"Error during file list processing for SI-SDR: {e}")
            sys.exit(1) # 如果文件列表本身有问题，则退出

        # 计算并写入平均值
        if all_si_sdr_scores:
            avg_si_sdr = sum(all_si_sdr_scores) / len(all_si_sdr_scores)
            out.write(f"avg:{avg_si_sdr:.4f}\n")
            print(f"✅ SI-SDR comparison complete. Results saved to {result_file_path}")
            print(f"Average SI-SDR: {avg_si_sdr:.4f}")
        else:
            out.write("No valid SI-SDR scores computed.\n")
            print("No valid SI-SDR scores could be computed.")

# --- 主函数入口 ---
def main():
    """主函数，获取命令行参数并执行处理"""
    if len(sys.argv) != 3:
        print("Usage: python si_sdr_eval.py <input_txt_path> <pred_audio_base_dir>")
        print("  <input_txt_path>: Path to the text file listing GT/Pred file pairs.")
        print("  <pred_audio_base_dir>: Base directory where predicted audio files are located.")
        sys.exit(1)

    input_txt_path = sys.argv[1]     # 列表文件路径 (e.g., /path/to/my_list.txt)
    pred_audio_base_dir = sys.argv[2] # 预测音频的根目录 (e.g., /path/to/predicted_wavs)

    # 构造结果保存路径，这将确保在 pred_audio_base_dir 下有一个 'result' 文件夹
    result_save_path = os.path.join(pred_audio_base_dir, "result", "si_sdr_result")
    
    # 调用新的计算和保存函数
    calculate_and_save_si_sdr(input_txt_path, pred_audio_base_dir, result_save_path)

if __name__ == "__main__":
    main()
