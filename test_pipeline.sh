. /mnt/nas1/zhangruonan/anaconda3/bin/activate dptts_z

wavdir=$1

if [ -z "$1" ]; then
    echo "错误: 请提供一个路径作为命令行参数。"
    exit 1
fi

wavdir=$1

echo "原始路径: $wavdir"

# 判断路径是否以 "/" 开头
if [[ "$wavdir" == /* ]]; then
    echo "路径是绝对路径，无需转换。"
else
    echo "路径是相对路径，正在转换为绝对路径..."
    wavdir=$(realpath -m "$wavdir")
    echo "转换后的路径: $wavdir"
fi

python /mnt/nas1/zhangying/git_tools/DAC/stable-audio-tools-main/test_codec_tools/prepare_test_meta.py $wavdir

meta_file=$wavdir/test_meta.txt
cd eval/

bash eval_other.sh $meta_file $wavdir
