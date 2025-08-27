import sys
import os

def check_abs_path(test_wavdir):
    if os.path.isabs(test_wavdir):
        print("路径是绝对路径，无需转换。")
    else:
        print("路径是相对路径，正在转换为绝对路径...")
        # 如果是相对路径，则转换为绝对路径
        # os.path.abspath() 会根据当前工作目录来解析相对路径
        original_path = test_wavdir
        test_wavdir = os.path.abspath(test_wavdir)
    return test_wavdir

if __name__=='__main__':
    
    orig_set = '/mnt/nas1/zhangying/cosy_data_25hz/test_set_fm/pgc_1spk1utt.txt'
    orig_data = open(orig_set).readlines()

    test_wavdir = sys.argv[1]
    
    test_wavdir = check_abs_path(test_wavdir)
    
    test_set = os.path.join(test_wavdir, 'test_meta.txt')
    fo = open(test_set, 'w')
    for line in orig_data:
        orig_wav_path = line.strip().split('|')[0]
        spk, wav_name = orig_wav_path.split('/')[-2:]
        syn_wav_path = os.path.join(test_wavdir, spk, wav_name)
        if os.path.isfile(syn_wav_path) and os.path.isfile(orig_wav_path):
            fo.write('%s|None|%s|None\n'%(syn_wav_path, orig_wav_path))
    fo.close()
    print('saved ', test_set)
        
        
