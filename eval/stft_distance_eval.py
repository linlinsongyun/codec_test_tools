import numpy as np
import librosa
import soundfile as sf
import sys
import os
from tqdm import tqdm

# --- 导入文件处理模块 ---
# 确保 file_processor.py 和 stft_distance_eval.py 在同一目录下
try:
    from file_processor import process_input_file, FileProcessingError
except ImportError:
    print("Error: Could not import 'file_processor.py'. Make sure it's in the same directory.")
    sys.exit(1)

def compute_stft_distance(wav_path_1, wav_path_2, sr=16000, n_fft=1024, hop_length=256):
    """
    计算两个 wav 文件的 STFT 幅度谱的 L2 范数距离。
    
    Args:
        wav_path_1 (str): 第一个 wav 文件的路径 (通常是 Ground Truth)。
        wav_path_2 (str): 第二个 wav 文件的路径 (通常是 Predicted)。
        sr (int): 目标采样率。音频将被重采样到此采样率。
        n_fft (int): FFT 窗口大小，应该是 2 的幂次方以提高效率。
        hop_length (int): 帧之间的跳跃长度。

    Returns:
        float: STFT 幅度谱的 L2 范数距离。
    """
    try:
        # 使用 soundfile.read 加载音频，返回数据和原始采样率
        wav1, orig_sr1 = sf.read(wav_path_1)
        wav2, orig_sr2 = sf.read(wav_path_2)

        # 处理多声道音频（转换为单声道）
        if len(wav1.shape) > 1:
            wav1 = wav1.mean(axis=1)
        if len(wav2.shape) > 1:
            wav2 = wav2.mean(axis=1)
        
        # 重采样到目标采样率
        if orig_sr1 != sr:
            wav1 = librosa.resample(wav1, orig_sr=orig_sr1, target_sr=sr)
        if orig_sr2 != sr:
            wav2 = librosa.resample(wav2, orig_sr=orig_sr2, target_sr=sr)

        # 对齐长度
        min_len = min(len(wav1), len(wav2))
        wav1 = wav1[:min_len]
        wav2 = wav2[:min_len]

        # 计算 STFT 幅度谱
        stft1 = librosa.stft(wav1, n_fft=n_fft, hop_length=hop_length)
        stft2 = librosa.stft(wav2, n_fft=n_fft, hop_length=hop_length)

        mag1 = np.abs(stft1)
        mag2 = np.abs(stft2)

        # 对齐 shape (时间帧数)
        min_t = min(mag1.shape[1], mag2.shape[1])
        mag1 = mag1[:, :min_t]
        mag2 = mag2[:, :min_t]

        # 计算 L2 范数距离
        l2_dist = np.mean((mag1 - mag2) ** 2)
        return l2_dist
    except Exception as e:
        print(f"⚠️ Error computing STFT distance for {wav_path_1} and {wav_path_2}: {e}")
        return None

# --- STFT 距离批量处理和结果保存函数 ---
# 替换原有的 process_files 函数
def calculate_and_save_stft_distance(input_txt_path: str, pred_audio_base_dir: str, result_file_path: str):
    """
    使用 file_processor 提供的文件对，计算 STFT L2 距离并保存结果。
    
    Args:
        input_txt_path (str): 包含 GT 文件相对路径和前缀的文本文件。
        pred_audio_base_dir (str): 预测音频文件的根目录。
        result_file_path (str): STFT 距离结果将要保存的完整文件路径。
    """
    # **确保结果目录存在**
    result_dir = os.path.dirname(result_file_path)
    os.makedirs(result_dir, exist_ok=True) 

    all_l2_dist = []
    
    print(f"Starting STFT L2 Distance calculation. Input list: {input_txt_path}, Pred base dir: {pred_audio_base_dir}")

    # **打开结果文件，并准备写入**
    with open(result_file_path, 'w', newline='') as out:
        # **调用 file_processor.process_input_file 来获取文件对**
        try:
            # 同样，假设 `parts[2]` 在 input_txt_path 中是 GT 文件的完整路径
            # 所以 `gt_base_path` 参数传入空字符串 `""`
            for gt_file, pred_file in tqdm(
                process_input_file("", pred_audio_base_dir, input_txt_path),
                desc="Calculating STFT Distance"
            ):
                # 计算 STFT L2 距离
                l2_dist = compute_stft_distance(gt_file, pred_file, sr=16000, n_fft=1024, hop_length=256)
                
                if l2_dist is not None:
                    out.write(f"{l2_dist:.4f}\n")
                    all_l2_dist.append(l2_dist)
                else:
                    # 如果计算失败，可以在结果文件中标记或跳过
                    out.write("N/A\n") 
                    
        except FileProcessingError as e:
            print(f"Error during file processing for STFT Distance: {e}")
            sys.exit(1) # 如果文件列表本身有问题，则退出

        # 计算并写入平均值
        if all_l2_dist:
            avg_l2 = sum(all_l2_dist) / len(all_l2_dist)
            out.write(f"avg:{avg_l2:.4f}\n")
            print(f"✅ STFT L2 Distance comparison complete. Results saved to {result_file_path}")
            print(f"Average STFT L2 Distance: {avg_l2:.4f}")
        else:
            out.write("No valid STFT L2 Distances computed.\n")
            print("No valid STFT L2 Distances could be computed.")

# --- 主函数入口 ---
def main():
    """主函数，获取命令行参数并执行处理"""
    if len(sys.argv) != 3:
        print("Usage: python stft_distance_eval.py <input_txt_path> <pred_audio_base_dir>")
        print("  <input_txt_path>: Path to the text file listing GT/Pred file pairs.")
        print("  <pred_audio_base_dir>: Base directory where predicted audio files are located.")
        sys.exit(1)

    input_txt_path = sys.argv[1]     # 列表文件路径 (e.g., /path/to/my_list.txt)
    pred_audio_base_dir = sys.argv[2] # 预测音频的根目录 (e.g., /path/to/predicted_wavs)

    # 构造结果保存路径，这将确保在 pred_audio_base_dir 下有一个 'result' 文件夹
    # 增加 .txt 扩展名以便清晰识别
    result_save_path = os.path.join(pred_audio_base_dir, "result", "stft_distance_result")
    result_dir = os.path.dirname(result_save_path)

    os.makedirs(result_dir, exist_ok=True)
    
    # 调用新的计算和保存函数
    calculate_and_save_stft_distance(input_txt_path, pred_audio_base_dir, result_save_path)

if __name__ == "__main__":
    main()
