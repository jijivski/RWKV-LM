########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F

# only + - *
# equation = "5" # -5.25
# equation = "-(4.2--2.1*4.5)" # -5.25

########################################################################################################



import json
def rtn_reverese_vocab(json_path='vocab.json'):
    with open(json_path, 'r') as file:
        vocab = json.load(file)
    # word_table = {value: key for key, value in vocab.items()}
    return vocab


class TOKENIZER():
    def __init__(self):
        # used for models previous to 11.08
        # self.word_table = {"0": "\n", "1": " ", "2": "(", "3": ")", "4": "*", "5": "+", "6": "-", "7": ".", "8": "0", "9": "1", "10": "2", "11": "3", "12": "4", "13": "5", "14": "6", "15": "7", "16": "8", "17": "9", "18": "=", "19": "e", "20": "f"}
        # # self.word_table={"0": "\n", "1": " ", "2": "\"", "3": "(", "4": ")", "5": "*", "6": "+", "7": "-", "8": ".", "9": "0", "10": "1", "11": "2", "12": "3", "13": "4", "14": "5", "15": "6", "16": "7", "17": "8", "18": "9", "19": ":", "20": "=", "21": "I", "22": "R", "23": "\\", "24": "c", "25": "e", "26": "f", "27": "i", "28": "n", "29": "o", "30": "p", "31": "r", "32": "s", "33": "t", "34": "u", "35": "x", "36": "{", "37": "}", "38": "："}
        # # self.word_table=rtn_reverese_vocab()
        # self.vocab_size = len(self.word_table)
        # self.stoi = {v: int(k) for k, v in self.word_table.items()}
        # self.itos = {int(k): v for k, v in self.word_table.items()}

        self.stoi = {chr(i): i for i in range(256)}  # Mapping from char to int
        self.itos = {i: chr(i) for i in range(256)}  # Mapping from int to char

    def encode(self, x):
        return [self.stoi[t] for t in x]
    
    def decode(self, x):
        return ''.join([self.itos[t] for t in x])

tokenizer = TOKENIZER()

########################################################################################################

class RWKV_RNN(torch.jit.ScriptModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode
        
        w = torch.load(args.folder+ '/' +args.MODEL_NAME + '.pth', map_location='cpu')
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # the real time decay is like e^{-e^x}
            else: w[k] = w[k].float() # convert to f32 type
        
        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)
        state[5*i+1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv
        
        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = qq
        return ow @ (r * wkv)

    def forward(self, token, state):
        with torch.no_grad():
            if state == None:
                state = torch.zeros(self.args.n_layer * 5, self.args.n_embd)
                for i in range(self.args.n_layer): state[5*i+4] = -1e30 # -infinity
            
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i, 
                    att.time_mix_k, att.time_mix_v, att.time_mix_r, att.time_first, att.time_decay, 
                    att.key.weight, att.value.weight, att.receptance.weight, att.output.weight)
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_mix_k, ffn.time_mix_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
            
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

##########################################################################################################

# import rec_python

import re
def one_expression(equation='(79990 * 0.65) - (-8790)',args=None,verbose=False):
    ans=''
    # global args
    # if args_specified is not None:
    #     args=args_specified
    if verbose:
        print(f'Loading {args.folder} {args.MODEL_NAME} ...')
        # print(f'vocab: {rtn_reverese_vocab()}')
        # print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
    ans+=f'Loading {args.folder} {args.MODEL_NAME} ...\n'+f'Using CPU. Loading {args.MODEL_NAME} ...\n'
    model = RWKV_RNN(args)
    # context = "\n" + equation.strip().replace(' ', '') + "=" #TODO  need to check,latter，fomattor已经改好了，数据集按照没有空格生成，然后这里之后改好就可适配新数据集的格式，然后面对用户的各种格式的输入
    context = "\n" + equation + "="
    if verbose:
        print(context,end='')
    ans+=context+' '
    try:
        if verbose:
            print(f'(python answer {eval(equation.split("=")[0])})')
        ans+=f'(python answer {eval(equation.split("=")[0])})'+'\n'
    except:
        if verbose:
            print(f'{equation} can not be evaled')
        ans+=f'{equation} can not be evaled'+'\n'
    state = None
    for token in tokenizer.encode(context):
        out, state = model.forward(token, state)

    buffer= ''
    for i in range(4096):
        token = int(torch.argmax(out))
        tmp = tokenizer.decode([token])
        buffer+=tmp
        if tmp == '=':
            ans += '\n'
            if verbose:
                # print(f'predict =')
                print()

            if verbose:
                if buffer and not re.search(r'\(\-0\.00010\)[\+\-\*/]\(\-0\.00010\)', buffer):
                    print(buffer, end="", flush=True)
                    buffer=''
                if re.search(r'\(\-0\.00010\)[\+\-\*/]\(\-0\.00010\)', buffer):
                    print(buffer[:buffer.find('((-0.00010')], end="", flush=True)
                    break
            # print(tmp, end="", flush=True)
        ans += tmp


        if tmp == '\n':
            # print(f'predict \\n')
            break

        out, state = model.forward(token, state)
    if verbose:
        print()
    ans+='\n'
    return ans



#
# def one_expression(equation='(79990 * 0.65) - (-8790)',args=None,verbose=False):
#     ans=''
#     # global args
#     # if args_specified is not None:
#     #     args=args_specified
#     if verbose:
#         print(f'Loading {args.folder} {args.MODEL_NAME} ...')
#         # print(f'vocab: {rtn_reverese_vocab()}')
#         # print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
#     ans+=f'Loading {args.folder} {args.MODEL_NAME} ...\n'+f'Using CPU. Loading {args.MODEL_NAME} ...\n'
#     model = RWKV_RNN(args)
#     # context = "\n" + equation.strip().replace(' ', '') + "=" #TODO  need to check,latter，fomattor已经改好了，数据集按照没有空格生成，然后这里之后改好就可适配新数据集的格式，然后面对用户的各种格式的输入
#     context = "\n" + equation + "="
#     if verbose:
#         print(context,end='')
#     ans+=context+' '
#     try:
#         if verbose:
#             print(f'(python answer {eval(equation.split("=")[0])})')
#         ans+=f'(python answer {eval(equation.split("=")[0])})'+'\n'
#     except:
#         if verbose:
#             print(f'{equation} can not be evaled')
#         ans+=f'{equation} can not be evaled'+'\n'
#     state = None
#     for token in tokenizer.encode(context):
#         out, state = model.forward(token, state)
#     for i in range(4096):
#         token = int(torch.argmax(out))
#         tmp = tokenizer.decode([token])
#
#         if tmp == '=':
#             ans += '\n'
#             if verbose:
#                 # print(f'predict =')
#                 print()
#         if verbose:
#             print(tmp, end="", flush=True)
#         ans+= tmp
#
#
#         if tmp == '\n':
#             # print(f'predict \\n')
#             break
#
#         out, state = model.forward(token, state)
#     if verbose:
#         print()
#     ans+='\n'
#     return ans


import argparse
import types

def main(equations,verbose=False):
    ans=[]
    parser = argparse.ArgumentParser(description='Process equations and set model parameters.')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--n_layer', type=int, help='Number of layers')
    parser.add_argument('--n_embd', type=int, help='Embedding size')
    parser.add_argument('--folder', type=str, help='/workspace2/jijivski/RWKV-LM/RWKV-v4neo/math_demo')
    args = parser.parse_args()
    if args.model_name is not None:
        args.MODEL_NAME = f'rwkv-{args.model_name}'
    else:
        args.MODEL_NAME = 'rwkv-200'  # 默认值

    args.n_layer = args.n_layer or 6  # 如果未提供命令行参数，则使用默认值
    args.n_embd = args.n_embd or 192  # 如果未提供命令行参数，则使用默认值


    for equation in equations:
        if verbose:
            one_expression(equation,args=args,verbose=verbose)
            print('\n\n answer is',eval(equation.replace('^','**')))
        else:
            # print(one_expression(equation,args=args,verbose=verbose))
            ans.append(one_expression(equation,args=args,verbose=verbose))
    return ans

if __name__ == '__main__':
    equations=[]

    # equations = ['(79990 * 0.65) - (-8790)', '148000',
    #              '-0.00652777', '7999 * 6.5 - -8790',
    #              '18239.715 * 9.728263', '4.2 - -2.1 * -4.5']
    # 新的题目
    equations=[expression.replace(' ','')for expression in equations]


    # 1110原题

    sss='''((24)^4)
        1000
        (10)^1
        ((5)^3+-99139000)
        (((8)^4)-550*110118000)
        -0.08+(-0.06)
        ((-94.7/8.861)/0.0630)
        9^(9)
        (0.0003*(-4300*((5)^12)))'''
    for i in sss.split('\n'):
        equations.append(i.strip())


    # main(equations,verbose=True)
    # CHECK

    # lst_str=main(equations,verbose=False) # retu
    # for i in lst_str:
    #     print(i)

    lst_str=main(equations,verbose=True) # retu
    # for i in lst_str:
    #     print(i)


    # equations = ['(799.9 * 65) - (-8790)',
    #              '(799.9 * 65) - (-8790)= (- (e - 1(*7999 65)) - 8790)',
    #              '(799.9 * 65) - (-8790)= (- (e - 1(*7999 65)) - 8790)= (- (e - 1(+ (e1(*7999 6))(*7999 5))) -8790)',
    #              '(e - 1 607835)'


                 # '(+ (- 0 681097000) 725) = ',
                 # '(-681097000 + 725) = (+ (- 0 681097000) 725) = (+ -681097000 725)',
                 # '(77.459 - 0.09) =',
                 # '(77.459 - 0.09) = (- 77.459 0.09) = ',
                 # '(77.459 - 0.09) = (- 77.459 0.09) = (e - 3(- 77459 0090))= (e - 3(f 96377))= (e - 3 77369) = '
                 # ]

# (799.9 * 65) - (-8790)= (- (e - 1(*7999 65)) - 8790)= (- (e - 1(+ (e1(*7999 6))(*7999 5))) -8790)
# = (- (e - 1(+ (e1(f 49974))(f 59993))) -8790)
# = (- (e - 1(+ (e1 47994) 39995)) - 8790)
# = (- (e - 1(+ 479940 39995)) - 8790)
# = (- (e - 1(f 539915)) - 8790)
# = (- (e - 1 519935) - 8790)
# = (- 51993.5 - 8790)
# = (e - 1(- 519935 - 87900))
# = (e - 1(f 538706))
# = (e - 1 607835)
# = 60783.5

    # '''
    # (-681097000 + 725) = (+ (- 0 681097000) 725) = (+ -681097000 725) = f 572690186 -= -681096275
    # (77.459 - 0.09) = (- 77.459 0.09) = (e - 3(- 77459 0090))= (e - 3(f 96377))= (e - 3 77369) = 77.369
    # (6000 - (-8000 * ((-9.18 * 1.0) * 0.0030))) = (- 6000(*(- 0 8000)(*(*(- 0 9.18) 1.0) 0.0030)))= (
    # - 6000(*-8000(*(*-9.18 1.0) 0.0030)))= (- 6000(*-8000(*(e - 3(*-918 10)) 0.0030)))= (
    # - 6000(*-8000(*(e - 3(e1(*-918 1))) 0.0030)))= (- 6000(*-8000(*(e - 3(e1(f 819-))) 0.0030)))= (
    # - 6000(*-8000(*(e - 3(e1 - 918)) 0.0030)))= (- 6000(*-8000(*(e - 3 - 9180) 0.0030)))= (- 6000(*-8000(*-9.18
    #                                                                                        0.0030)))= (
    # - 6000(*-8000(e - 6(*-918 30))))= (- 6000(*-8000(e - 6(e1(*-918 3)))))= (- 6000(*-8000(e - 6(e1(f 4572-)))))= (
    # - 6000(*-8000(e - 6(e1 - 2754)))) = (- 6000(*-8000(e - 6 - 27540))) = (- 6000(*-8000 - 0.027540)) = (
    # - 6000(e - 6(*-8000 - 27540))) = (- 6000(e - 6(*-27540 - 8000))) = (- 6000(e - 6(*27540 8000)))= (
    # - 6000(e - 6(e3(*27540 8))))= (- 6000(e - 6(e3(f 023022))))= (- 6000(e - 6(e3 220320)))= (- 6000(e - 6
    #                                                                                           220320000))= (- 6000
    #                                                                                                         220.32) = (
    #         e - 2(- 600000 22032))= (e - 2(f 869775))= (e - 2 577968) = 5779.68
    # '''
    # equation = "4.2--2.1*-4.5" # -5.25
    # equation = "(4.2379*564.778)-1209.01"  # 1184.4626862
    # equation = "4.2379*(564.778-1209.01)" # 1184.4626862
    # equation = "32731423*2189286" # 71658446133978
    # equation = "18239.715*9.728263" # 177440.744565045
    # equation = "2067*9832*4549" # 92448162456



'''
python /workspace2/jijivski/RWKV-LM/RWKV-v4neo/math_demo/run_lst.py --n_layer 6 --n_embd 192  --folder /workspace2/jijivski/RWKV-LM/RWKV-v4neo/ctx_2048_6_192_wandb_1101  --model_name 40



python /workspace2/jijivski/RWKV-LM/RWKV-v4neo/math_demo/run_lst.py --n_layer 8 --n_embd 64  --folder /workspace2/jijivski/RWKV-LM/RWKV-v4neo/ctx_512_8_64  --model_name 630

# 630

python /workspace2/jijivski/RWKV-LM/RWKV-v4neo/math_demo/run_lst.py --n_layer 12 --n_embd 128  --folder /workspace2/jijivski/RWKV-LM/RWKV-v4neo/ctx_512_12_128_wandb  --model_name 1090

# 1090

弄一个基准题目出来做？ 加上随机的题目，然后今天就可拿这个出文章了，





'''