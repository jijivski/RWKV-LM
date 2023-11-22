# builder = MMapIndexedDatasetBuilder(f'{out_name}.bin')
#     out.append(0)
#     builder.add_item(np.array(out, dtype=np.uint16))
#     builder.end_document()
# builder.finalize((f'{out_name}.idx'))

# import MMapIndexedDataset
from binidx import MMapIndexedDataset


original_dict={}
original_dict.update({chr(0):0})

        # note the space and the dot
original_dict.update({chr(i): i for i in range(256) if (chr(i)>='A' and chr(i)<='Z')or
                (chr(i) in'1234567890. ,' or (chr(i)>='a' and chr(i)<='z') or chr(i) in '()=#+-*/^')})
sorted_items = sorted(original_dict.items(), key=lambda x: x[1])

vocab={char: i for i, (char, origial_id) in enumerate(sorted_items)}
inv_vocab = {str(i) : char for char, i in vocab.items()}


# ddd=[r'/workspace2/jijivski/RWKV-LM/RWKV-v4neo/src/data_1114_test/1114_100_text_document',r'/workspace2/jijivski/RWKV_calc/data/1114_1e6_text_document']
# ddd=[r'/workspace2/jijivski/RWKV-LM/RWKV-v4neo/src/data_1114_test/1118_1e6_text_document']
# ddd=[r'/workspace2/jijivski/RWKV-LM/RWKV-v4neo/src/data_1114_test/__73_text_document']# can not append suffix like .bin and .idx
ddd=[r'/workspace2/jijivski/RWKV_calc/data/1118_1e6_text_document']# can not append suffix like .bin and .idx

for data_file in ddd:
    print('\n\n\n\nloading', data_file, '\n\n\n\n')
    data = MMapIndexedDataset(data_file)
    data_len = len(data)
    data_size = len(data._bin_buffer)# // 2
    print(f"Data has {data_size} tokens, {data_len} items.")

    for idx in [2]:
        ptr, size = data._index[idx]
        dix = data.get(idx=idx, offset=0, length=size).astype(int)
        print('-'*70 + f'[{data_file} idx {idx} sz {size}]')
        
        # if dix[-1] != 0:
        print(f'dix[-1]=={dix[-1]},{dix}')


        # 对于这一组测试， 规则上来说是可以对应出来的，说明做错了，不可以正常实验，
        for i in dix:
            # print(chr(i),end='')
            print(inv_vocab[str(i)],end='')

            
            # (-0.859790+-0.164435)= (+ (- 0 0.859790) (- 0 0.164435))= (+-0.859790 -0.164435)= (e-6 (+-0859790 -0164435))= (e-6 -1024225)= -1.024225


# ----------------------------------------------------------------------[/workspace2/jijivski/RWKV-LM/RWKV-v4neo/src/data_1114_test/1114_100_text_document idx 0 sz 1002]
# dix[-1]==53,[40 45 57 ... 55 49 53]




# loading /workspace2/jijivski/RWKV_calc/data/1114_1e6_text_document 




# Data has 6087085858 tokens, 30000000 items.
# ----------------------------------------------------------------------[/workspace2/jijivski/RWKV_calc/data/1114_1e6_text_document idx 0 sz 136]
# dix[-1]==0,[ 40  45  48  46  56  53  57  55  57  48  43  45  48  46  49  54  52  52
#   51  53  41  61  32  40  43  32  40  45  32  48  32  48  46  56  53  57
#   55  57  48  41  32  40  45  32  48  32  48  46  49  54  52  52  51  53
#   41  41  61  32  40  43  45  48  46  56  53  57  55  57  48  32  45  48
#   46  49  54  52  52  51  53  41  61  32  40 101  45  54  32  40  43  45
#   48  56  53  57  55  57  48  32  45  48  49  54  52  52  51  53  41  41
#   61  32  40 101  45  54  32  45  49  48  50  52  50  50  53  41  61  32
#   45  49  46  48  50  52  50  50  53   0]