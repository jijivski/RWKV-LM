########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import re
import torch
import types
import argparse
import json
from torch.nn import functional as F
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# only + - *
# equation = "5" # -5.25
# equation = "-(4.2--2.1*4.5)" # -5.25

########################################################################################################


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

        # self.stoi = {chr(i): i for i in range(256)}  # Mapping from char to int
        # self.itos = {i: chr(i) for i in range(256)}  # Mapping from int to char
        original_dict = {}
        original_dict.update({chr(0): 0})
        original_dict.update({chr(i): i for i in range(256) if (chr(i) >= 'A' and chr(i) <= 'Z') or
                              (chr(i) in '1234567890. ,' or (chr(i) >= 'a' and chr(i) <= 'z') or chr(i) in '()=#+-*/^')})
        sorted_items = sorted(original_dict.items(), key=lambda x: x[1])

        self.stoi = {char: i for i,
                     (char, origial_id) in enumerate(sorted_items)}
        self.itos = {i: char for char, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

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
        self.eval()  # set torch to inference mode

        w = torch.load(args.folder + '/' + args.MODEL_NAME +
                       '.pth', map_location='cpu')
        for k in w.keys():
            if '.time_' in k:
                w[k] = w[k].squeeze()
            if '.time_decay' in k:
                # the real time decay is like e^{-e^x}
                w[k] = -torch.exp(w[k].float())
            else:
                w[k] = w[k].float()  # convert to f32 type

        self.w = types.SimpleNamespace()  # set self.w from w
        self.w.blocks = {}
        # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
        for k in w.keys():
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here:
                        here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p):
                        setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
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
                for i in range(self.args.n_layer):
                    state[5*i+4] = -1e30  # -infinity

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


def one_expression(equation='(79990 * 0.65) - (-8790)', args=None, verbose=False):
    ans = ''
    # global args
    # if args_specified is not None:
    #     args=args_specified
    if verbose:
        print(f'Loading {args.folder} {args.MODEL_NAME} ...')
        # print(f'vocab: {rtn_reverese_vocab()}')
        # print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
    ans += f'Loading {args.folder} {args.MODEL_NAME} ...\n' + \
        f'Using CPU. Loading {args.MODEL_NAME} ...\n'
    model = RWKV_RNN(args)
    # TODO  need to check,latter，fomattor已经改好了，数据集按照没有空格生成，然后这里之后改好就可适配新数据集的格式，然后面对用户的各种格式的输入
    context = chr(0) + equation.strip().replace(' ', '') + "="
    # context = equation+'='
    # context = equation
    if verbose:
        print(context, end='')
    ans += context+' '
    try:
        if verbose:
            question = equation.split("=")[0].replace('^', '**')
            print(f'(python answer {eval(question)})')
        ans += f'(python answer {eval(question)})'+'\n'
    except:
        if verbose:
            # pass
            print(f'   ###### {equation} can not be evaled')
        # ans+=f'{equation} can not be evaled'+'\n'
    state = None
    for token in tokenizer.encode(context):
        out, state = model.forward(token, state)

    tmp = 1
    buffer = ''
    for i in range(4000):
        token = int(torch.argmax(out))
        buffer = tmp
        tmp = tokenizer.decode([token])
        if tmp == '#':
            if verbose:
                print()
            ans += '\n'
        if tmp == '=':
            if not buffer == '#':  # buffer.isalpha():
                if verbose:
                    print()
                ans += '\n'

        # buffer+=tmp
        # if tmp == '=':
        #     ans += '\n'
        #     if verbose:
        #         # print(f'predict =')
        #         print()

        #     if verbose:
        #         if buffer and not re.search(r'\(\-0\.00010\)[\+\-\*/]\(\-0\.00010\)', buffer):
        #             print(buffer, end="", flush=True)
        #             buffer=''
        #         if re.search(r'\(\-0\.00010\)[\+\-\*/]\(\-0\.00010\)', buffer):
        #             print(buffer[:buffer.find('((-0.00010')], end="", flush=True)
        #             break
        #     # print(tmp, end="", flush=True)
        # ans += tmp
        if verbose:
            print(tmp, end="", flush=True)
        ans += tmp

        if tmp == chr(0):
            break

        out, state = model.forward(token, state)
    if verbose:
        print()
    ans += '\n'
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


def main(equations, verbose=False, args=None):
    ans = []
    if args is None:
        parser = argparse.ArgumentParser(
            description='Process equations and set model parameters.')
        parser.add_argument('--model_name', type=str, help='Model name')
        parser.add_argument('--n_layer', type=int, help='Number of layers')
        parser.add_argument('--n_embd', type=int, help='Embedding size')
        parser.add_argument(
            '--folder', type=str, help='/workspace2/jijivski/RWKV-LM/RWKV-v4neo/math_demo')
        args = parser.parse_args()
    if args.model_name is not None:
        args.MODEL_NAME = f'rwkv-{args.model_name}'
    else:
        args.MODEL_NAME = 'rwkv-200'  # 默认值

    args.n_layer = args.n_layer or 6  # 如果未提供命令行参数，则使用默认值
    args.n_embd = args.n_embd or 192  # 如果未提供命令行参数，则使用默认值

    for equation in equations:
        if verbose:
            one_expression(equation, args=args, verbose=verbose)
            try:
                print('\n\n answer is', eval(equation.replace('^', '**')))
            except:
                print('can not be evaluated')
        else:
            # print(one_expression(equation,args=args,verbose=verbose))
            ans.append(one_expression(equation, args=args, verbose=verbose))
    return ans


if __name__ == '__main__':
    equations = []

    # equations = ['(79990 * 0.65) - (-8790)', '148000',
    #              '-0.00652777', '7999 * 6.5 - -8790',
    #              '18239.715 * 9.728263', '4.2 - -2.1 * -4.5']
    # 新的题目
    equations = [expression.replace(' ', '')for expression in equations]

    # 1110原题

    sss = '''((24)^4)
        1000
        (10)^1
        ((5)^3+-99139000)
        (((8)^4)-550*110118000)
        -0.08+(-0.06)
        ((-94.7/8.861)/0.0630)
        9^(9)
        (0.0003*(-4300*((5)^12)))
        10
        4+53'''

    # sss='''f=6.00,K=7.07,7^1-(f*9.04)+K*1.33-4.88
    # (0^0-4^7*(1^2))
    # 5^3'''

    sss = '''-9400*6.92*0.13
        V=9.18,H=8.49,(V*9.00)*2.34*7.76*H
        E=7.03,i=8.26,Y=9.33,(((E+E)-4.88+2.98)+i-Y)
        7^1
        (((6^2)/7.52/1.36)-3.59+1.58)'''

    sss = '''(-0.00086+10400)'''  # 10399.99914
    # sss='''(300+-4500)='''  # -4200
    sss = '''-0.00086+10400='''  # 10399.99914
    sss = '''300+-4500='''
#  (+ (- 0 300) (- 0 4500))
# = (+-300 -4500)
# = -4800
    sss = '''(-0.859790+-0.164435)'''  # -1.024225
    sss = '''-0.859790+-0.164435='''  # -1.024225
#     sss='''300+-4500= (+ 300 (- 0 4500))= (+300 -4500)
# (-0.859790+-0.164435)
# (-0.859790+-0.164435)= (+ (- 0 0.859790) (- 0 0.164435))= (+-0.859790 -0.164435)= (e-6 (+-0859790 -0164435))
# 300+-4500
# 300+-4500= (+ 300 (- 0 4500))'''

#     sss='''300+-4500
# (-0.859790+-0.164435)
# '''
#


#     sss='''6+3
# 8+142343
# 7.234+5.653'''

#         {"text": "(-0.859790+-0.164435)= (+ (- 0 0.859790) (- 0 0.164435))= (+-0.859790 -0.164435)= (e-6 (+-0859790 -0164435))= (e-6 -1024225)= -1.024225"}
# {"text": "(-0.00086+10400)= (+ (- 0 0.00086) 10400)= (+-0.00086 10400)= (e-5 (+-000086 1040000000))= (e-5 1039999914)= 10399.99914"}
# {"text": "(-5000+98641.6)= (+ (- 0 5000) 98641.6)= (+-5000 98641.6)= (e-1 (+-50000 986416))= (e-1 936416)= 93641.6"}
# {"text": "4640+0.737= (+ 4640 0.737)= (e-3 (+4640000 0737))= (e-3 4640737)= 4640.737"}
# {"text": "(0.0018)+-3872.0= (+ 0.0018 (- 0 3872.0))= (+0.0018 -3872.0)= (e-4 (+00018 -38720000))= (e-4 -38719982)= -3871.9982"}
# {"text": "(-587200+1904000)= (+ (- 0 587200) 1904000)= (+-587200 1904000)= 1316800"}
# {"text": "-0.834450+0.000923= (+ (- 0 0.834450) 0.000923)= (+-0.834450 0.000923)= (e-6 (+-0834450 0000923))= (e-6 -833527)= -0.833527"}
# {"text": "(6063.2+-670167000)= (+ 6063.2 (- 0 670167000))= (+6063.2 -670167000)= (e-1 (+60632 -6701670000))= (e-1 -6701609368)= -670160936.8"}
# {"text": "(58705200+424600)= (+ 58705200 424600)= 59129800"}
# {"text": "300+-4500= (+ 300 (- 0 4500))= (+300 -4500)= -4200"}


#     sss='''(58705200+424600)
# 58705200+424600=
# 58705200+424600'''

# = (+ 58705200 424600)
# = 59129800

#  (+ (- 0 58705200) 424600)
# = (+-58705200 424600)
# = -58280600

# 0
# = (+ (- 0 58705200) 4246000)
# = (+-58705200 4246000)
# = -54459200

    # sss = '''(6+3)
    # 6+3
    # (6+3.4444444)
    # 6+3.4444444
    # (6.5563636+3)
    # 6.5563636+3
    # (134324+4312324.4232)
    # 134324+4312324.4232'''
    # 上面是去掉了=的版本
    # sss='''4.6^6
    # 4.6^(6)
    # (4.6)^6
    # (4.6^6)
    # 7.7^3
    # 7.7^(3)
    # (7.7)^3
    # (7.7^3)
    # 9.3^4
    # (9.3^4)
    # 9.3^(4)'''
    # sss='''70^2
    # 72^4
    # 16^12'''


#     sss='''97270 - 18 ^ (28)
# ((7) ^ 3 / -810.122)
# -9.6311 * 20 - (99300) - -785.484 + -0.0002150
# 7420.03 + (-9319.0)
# (92.2 * (16) ^ 30)
# (11) ^ 22`
# (0.4 * (-0.0071))
# ((0) ^ 9)
# 0.1
# 0.000131 * (-600 * (-0.036620) * 0.09677 * (0.001))'''

#     sss='''h=4.50, u=7.19, P=8.91, (9.09 * 8.34 / h * u) * P * 7.93 + 1.06
# F=5.16, M=4.80, ((4.41 / 2.00) + 7.51 + F) + M
# i=8.76, Z=3.94, ((i + 1.36) / (3 ^ 8)) - ((Z * 8.80) * Z)'''
# 8559.5510692996
# 19.675
# -136.6061375522

#     input_formular_answer='''g=8.63, T=1.70, V=2.06, E=8.76, g * T * 4 ^ 2 / ((7.75 * V) / E)
#     8.63 * 1.70 * 4 ^ 2 / ((7.75 * 2.06) / 8.76)
#     128.79970936423427
#     ((1 ^ 1) - 6 ^ 8)
#     ((1 ^ 1) - 6 ^ 8)
#     -1679615
#     i=8.76, Z=3.94, ((i + 1.36) / (3 ^ 8)) - ((Z * 8.80) * Z)
#     ((8.76 + 1.36) / (3 ^ 8)) - ((3.94 * 8.80) * 3.94)
#     -136.60613755220243'''

#     input_formular_answer='''z=1.67, q=2.67, ((1 ^ 3) / ((6 ^ 4) / z * q))
# ((1 ^ 3) / ((6 ^ 4) / 1.67 * 2.67))
# 0.0004826143246867341
# s=3.66, x=6.86, y=8.42, I=7.30, 1.27 - s - x / y - 0 ^ 8 / (2.91 - I)
# 1.27 - 3.66 - 6.86 / 8.42 - 0 ^ 8 / (2.91 - 7.30)
# -3.2047268408551073
# 5 ^ 7
# 5 ^ 7
# 78125
# m=3.21, u=1.48, N=1.94, 5.14 - m * u - N * 5 ^ 7
# 5.14 - 3.21 * 1.48 - 1.94 * 5 ^ 7
# -151562.1108
# s=6.98, G=6.25, ((2.52 - 1.47) + (4.49 * s) - G)
# ((2.52 - 1.47) + (4.49 * 6.98) - 6.25)
# 26.1402'''
    input_formular_answer='''M=6.61, ((7 ^ 7) / (7 ^ 2) - 2.28 / M)
((7 ^ 7) / (7 ^ 2) - 2.28 / 6.61)
16806.655068078668
(1 ^ 0)
(1 ^ 0)
1
H=1.27, O=6.07, 0 ^ 7 - H / O - ((6.89 + 9.98) + 3.84)
0 ^ 7 - 1.27 / 6.07 - ((6.89 + 9.98) + 3.84)
-20.919225700164745
R=9.68, d=7.82, t=2.06, ((4.48 + R) / d - 4.38 + 3 ^ 4 + t)
((4.48 + 9.68) / 7.82 - 4.38 + 3 ^ 4 + 2.06)
80.49074168797954
D=9.96, U=6.42, A=6.19, a=1.38, n=2.32, ((D + 2.60) + U / A) - (a / 8.78 / n)
((9.96 + 2.60) + 6.42 / 6.19) - (1.38 / 8.78 / 2.32)
13.529408687709608
a=8.80, (((8.47 + 1.66) * a - 4.35) / 8.49)
(((8.47 + 1.66) * 8.80 - 4.35) / 8.49)
9.987514723203772
N=5.68, U=9.36, R=9.79, c=2.14, (N * U) - (R - 8.94) - c + 3.15 + 4.79
(5.68 * 9.36) - (9.79 - 8.94) - 2.14 + 3.15 + 4.79
58.11479999999999
r=2.25, s=5.05, (1 ^ 4) / ((r + 5.01) - (s + 5.78))
(1 ^ 4) / ((2.25 + 5.01) - (5.05 + 5.78))
-0.2801120448179272
M=5.91, U=7.63, (2 ^ 0) * ((9.66 * M) / (U / 3.87))
(2 ^ 0) * ((9.66 * 5.91) / (7.63 / 3.87))
28.95683119266055
R=9.94, f=8.92, ((R * 9.11 * (6.16 + 4.41)) - f)
((9.94 * 9.11 * (6.16 + 4.41)) - 8.92)
948.2294380000001'''
    sss=''
    ans=[]
    for cnt,ss in enumerate(input_formular_answer.split('\n')):
        if cnt%3==0:
            sss+=ss.strip()+'\n'
        if cnt%3==2:
            ans.append(float(ss))


# '''h=4.50, u=7.19, P=8.91, (9.09 * 8.34 / h * u) * P * 7.93 + 1.06
# (9.09 * 8.34 / 4.50 * 7.19) * 8.91 * 7.93 + 1.06
# 8559.5510692996

# F=5.16, M=4.80, ((4.41 / 2.00) + 7.51 + F) + M
# ((4.41 / 2.00) + 7.51 + 5.16) + 4.80
# 19.675

# i=8.76, Z=3.94, ((i + 1.36) / (3 ^ 8)) - ((Z * 8.80) * Z)
# ((8.76 + 1.36) / (3 ^ 8)) - ((3.94 * 8.80) * 3.94)
# -136.60613755220243'''


# content+='='


#     sss='''59000 + -5207300
# -60660 + (-73709500)
# (-0.00483619) + (6)
# ((-6.37) + 0.0097)
# (55.12) + 20
# -686600 + -0.0002150
# (-2711.39 + (-0.000355632))
# -53200 + 889.5
# -30 + -550
# (0.0010 + (-0.0071))'''
# # content+='='


# (6+3.4444444)=(python answer 9.4444444)
#  (+ 6 3.44444)
# = (e-5 (+600000 344444))
# = (e-5 944444)
# = 9.44444


# (6.5563636+3)=(python answer 9.556363600000001)
#  (+ 6.55636 3)
# = (e-5 (+655636 300000))
# = (e-5 955636)
# = 9.55636

    for i in sss.split('\n'):
        if len(i) == 0:
            continue
        # equations.append(f'({i.strip()})')
        equations.append(f'{i.strip()}')

        print(f'the input is {i.strip()}')

    # main(equations,verbose=True)
    # CHECK

    # lst_str=main(equations,verbose=False) # retu
    # for i in lst_str:
    #     print(i)
    verbose = True
    lst_str = main(equations, verbose=verbose)  # retu
    if verbose is False:
        for i in lst_str:
            print(i)

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
