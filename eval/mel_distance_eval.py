import sys
import os
import csv
from tqdm import tqdm

# --- 导入文件处理模块 ---
# 确保 file_processor.py 和 mcd_eval.py 在同一目录下
try:
    from file_processor import process_input_file, FileProcessingError
except ImportError:
    print("Error: Could not import 'file_processor.py'. Make sure it's in the same directory.")
    sys.exit(1)

# --- 导入 MCD 计算库 ---
# 假设 mel_cepstral_distance 库已安装，并且 compare_audio_files 函数可用
try:
    from mel_cepstral_distance import compare_audio_files
except ImportError:
    raise ImportError("请先安装 mel_cepstral_distance 库或确保其在PYTHONPATH中。")

def calculate_mcd_for_pair(gt_path, pred_path):
    """计算MCD和Penalty并返回结果"""
    try:
        # 假设 compare_audio_files 接受 GT 和 Pred 路径并返回 MCD 和 Penalty
        mcd, penalty = compare_audio_files(gt_path, pred_path)
        return mcd, penalty
    except Exception as e:
        print(f"⚠️ Error comparing {gt_path} and {pred_path} for MCD: {e}")
        return None, None

# --- MCD 批量处理和结果保存函数 ---
# 替换原有的 process_files 函数
def calculate_and_save_mcd(input_txt_path: str, pred_audio_base_dir: str, result_file_path: str):
    """
    使用 file_processor 提供的文件对，计算 MCD 并保存结果到 CSV 文件。
    
    Args:
        input_txt_path (str): 包含 GT 文件相对路径和前缀的文本文件。
        pred_audio_base_dir (str): 预测音频文件的根目录。
        result_file_path (str): MCD 结果将要保存的完整 CSV 文件路径。
    """
    # **确保结果目录存在**
    result_dir = os.path.dirname(result_file_path)
    os.makedirs(result_dir, exist_ok=True) 

    all_mcd = []
    all_penalty = []
    
    print(f"Starting MCD calculation. Input list: {input_txt_path}, Pred base dir: {pred_audio_base_dir}")

    # **打开 CSV 文件，并准备写入**
    with open(result_file_path, 'w', newline='') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(['filename', 'MCD', 'Penalty'])  # CSV 表头
        
        # **调用 file_processor.process_input_file 来获取文件对**
        try:
            # 同样，假设 `parts[2]` (即 `wav_info`) 在 input_txt_path 中是 GT 文件的完整路径
            # 所以 `gt_base_path` 参数传入空字符串 `""`
            for gt_file, pred_file in tqdm(
                process_input_file("", pred_audio_base_dir, input_txt_path),
                desc="Calculating MCD"
            ):
                # 提取文件名用于 CSV 记录
                filename_for_csv = os.path.basename(gt_file)
                
                # 计算 MCD 和 Penalty
                mcd, penalty = calculate_mcd_for_pair(gt_file, pred_file)
                
                if mcd is not None and penalty is not None:
                    writer.writerow([filename_for_csv, f"{mcd:.4f}", f"{penalty:.4f}"]) # 格式化输出到CSV
                    all_mcd.append(mcd)
                    all_penalty.append(penalty)
                else:
                    # 如果计算失败，可以在CSV中标记或跳过，这里选择跳过
                    writer.writerow([filename_for_csv, "N/A", "N/A"]) # 失败的样本也记录下来
                    
        except FileProcessingError as e:
            print(f"Error during file processing for MCD: {e}")
            sys.exit(1) # 如果文件列表本身有问题，则退出

        # 计算并写入平均值
        if all_mcd and all_penalty: # 确保有计算成功的样本
            avg_mcd = sum(all_mcd) / len(all_mcd)
            avg_penalty = sum(all_penalty) / len(all_penalty)
            writer.writerow(['Average', f"{avg_mcd:.4f}", f"{avg_penalty:.4f}"])
            print(f"✅ MCD comparison complete. Results saved to {result_file_path}")
            print(f"Average MCD: {avg_mcd:.4f}, Average Penalty: {avg_penalty:.4f}")
        else:
            writer.writerow(['Average', 'N/A', 'N/A']) # 没有成功计算的样本
            print("No valid MCD scores could be computed.")

# --- 主函数入口 ---
def main():
    """主函数，获取命令行参数并执行处理"""
    if len(sys.argv) != 3:
        print("Usage: python mcd_eval.py <input_txt_path> <pred_audio_base_dir>")
        print("  <input_txt_path>: Path to the text file listing GT/Pred file pairs.")
        print("  <pred_audio_base_dir>: Base directory where predicted audio files are located.")
        sys.exit(1)

    input_txt_path = sys.argv[1]     # 列表文件路径 (e.g., /path/to/my_list.txt)
    pred_audio_base_dir = sys.argv[2] # 预测音频的根目录 (e.g., /path/to/predicted_wavs)

    # 构造结果保存路径，这将确保在 pred_audio_base_dir 下有一个 'result' 文件夹
    result_save_path = os.path.join(pred_audio_base_dir, "result", "mcd_result")
    result_dir = os.path.dirname(result_save_path)

    os.makedirs(result_dir, exist_ok=True)
    
    # 调用新的计算和保存函数
    calculate_and_save_mcd(input_txt_path, pred_audio_base_dir, result_save_path)

if __name__ == "__main__":
    main()
