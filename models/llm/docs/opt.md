# OPT (Open Pretraining Transformers)

OPT (Open Pre-trained Transformer) class of language models are released by Meta and are available through [Huggingface](https://huggingface.co/facebook/opt-1.3b)

This example shows how to deploy OPT models on AIE and custom AIE+CPU backend in **eager mode**.

# Model Structure
This image shows model structure after quantizing and replacing Linear node with AIE custom linear node.
![Model Structure](../figures/opt1.3b.png)

# Model Variants
OPT has several variants: **opt-125m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b, etc.**. User can select any model variant for this example but the upper limit is dependent on the system memory capacity.

Max. sequence length of model is 2048  (max. position embeddings), i.e. maximum number of tokens in input prompt.

# Supported features
1. State-of-the-art [SmoothQuant](https://arxiv.org/pdf/2211.10438.pdf) to condition weights for 8-bit quantization
2. State of the art [AWQ](https://arxiv.org/pdf/2306.00978.pdf) for 3-bit and 4-bit quantization
3. Flash Attention v2
4. Optimized attention for prefill and token phases
5. Ahead of time parameter caching and memory management
6. Model evaluation : benchmark, decode, perplexity, compute complexity
7. **w8a8 and w8a16** with **SmoothQuant**; **w4abf16** with **AWQ**, **AWQplus**, **PerGrp** quantization recipes

# Perplexity scores
Perplexity is measured using negative log likelihood scores.
***Perplexity measurement takes 3-4hrs to coimplete on wikitext2-raw test set***
***Lower value is better***
* Max. sequence length of model = 2048  (max. position embeddings of opt-1.3b)

Baseline: V100 (FP32) : **14.6240**

The following numbers are on RyzenAI
| **Precision+Config**                               | **opt-1.3b CPU** | **opt-1.3b AIE**
|----------------------------------------------------|------------------|-----------------
FP32                                                 |  14.2637         | na
FP32, Flash Attention v2                             |  14.9346         | na
BF16, Flash Attention v2                             |  14.9772         | na
int8 GEMM (PTDQ), other ops FP32                     | 231.6443         | 16.1019
int8 GEMM (SmoothQuant + PTDQ),                      |  15.5526         | 15.0677
int8 GEMM (SmoothQuant + PTDQ), Flash Attention v2   |  15.4157         | 15.2020
int8 GEMM (SmoothQuant + PTDQ),                      |    na            | 14.9346
int8 GEMM (SmoothQuant + PTDQ), Flash Attention v2   |    na            | 15.0854
int4 AWQ (Group size:128)

# [README for w8a8 with SmoothQuant](./opt_w8a8.md)

# [OPT-1.3b for w4abf16 with AWQ, PerGrp Quant](./opt-1.3b_w4abf16.md)

# [OPT-6.7b for w4abf16 with AWQ, PerGrp Quant](./opt-6.7b_w4abf16.md)


# OPT Model demo using opt_demo.py (to be updated)
This script gives user option to run the model on any set or prompts with 3 search strategies
```
python opt_demo.py --help
usage: opt_demo.py [-h] [--model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}] [--target {cpu,aie}] [--quant_mode {none,ptdq}] [--load]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}
                        Different OPT model sizes
  --target {cpu,aie}    cpu, aie
  --quant_mode {none,ptdq}
                        Quantization mode - none, or smoothquant+pytorch dynamic-quant
  --load                Load quantized weights from checkpoint
```

The demo gives user flexibility to provide any prompts, with different search options and output token lengths.
Three search options are provided: ***greedy, stochastic and contrastive***. These search options provide different level of quality to text generation process. User can modify the strengths of parameters in this file.

***This feature is available without --load option only.***

This is optional, to see individual tokens as they print to the screen in greedy search mode, open the

```installationfolder\anaconda3\envs\ryzenai-transformers\lib\site-packages\transformers\generation\utils.py```

In ```def greedy_search(...)``` function,

after ```next_tokens = torch.argmax(next_tokens_scores, dim=-1)```,

add this new line: ```print(self.tokenizer.decode(next_tokens)) ```

This prints each new token to the screen as the text-generation process unfolds.

```
python opt_demo.py  --quant_mode w8a8 --target aie --load
python opt_demo.py  --quant_mode w8a8 --target cpu --load
python opt_demo.py  --quant_mode none --target cpu
```

Here are examples for 3 search options for the same prompt and token length on AIE with SmoothQuant + Pytorch Dynamic Quantization:

```
********************
Enter prompt or 'exit': San Francisco is a city of
Enter response length (1-1000): 30
Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 0
Setting search to:  Greedy search
San Francisco is a city of contrasts. It’s a city of the arts, of the food, of the people, of the history
********************
Enter prompt or 'exit': San Francisco is a city of
Enter response length (1-1000): 30
Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 1
Setting search to:  Stochastic search
San Francisco is a city of incredible contrasts. It has the highest concentration of Jews of any city in the world, and it is known as a
********************
Enter prompt or 'exit': San Francisco is a city of
Enter response length (1-1000): 30
Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 2
Setting search to:  Contrastive search
San Francisco is a city of many cultures.

The city has a long history of immigration and is home to the largest number of immigrants in
********************
```

# Extract shapes of MatMuls
```
python run_smoothquant.py --model_name facebook/opt-1.3b --target cpu --precision bf16 --task infershapes
```

# Computation complexity analysis

```
python run_smoothquant.py --model_name facebook/opt-1.3b --target cpu --precision bf16 --task countgops
...

Seqlen: 4  GOPs: 10.491088924
Seqlen: 8  GOPs: 20.988493912
Seqlen: 16  GOPs: 42.002252080
Seqlen: 32  GOPs: 84.105561184
Seqlen: 64  GOPs: 168.615350464
Seqlen: 128  GOPs: 338.847613312
Seqlen: 256  GOPs: 684.162876160
Seqlen: 512  GOPs: 1394.196350464
Seqlen: 1024  GOPs: 2891.875093504
Seqlen: 2000  GOPs: 6033.473446000
```

# [Calibration and PTSQ - outdated](./calibration.md)
