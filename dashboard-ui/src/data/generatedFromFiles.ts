/* eslint-disable */
// 本文件由 npm run generate:data 自动生成，请勿手改。源文件：仓库根目录 data/operator、data/infer 下 CSV。

export const OP_TABLE_FROM_FILES = {
  "ascend": {
    "Add": [
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.020143,
        "pt": 0.010594
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.019813,
        "pt": 0.010622
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.020082,
        "pt": 0.010738
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0.020493,
        "pt": 0.010674
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0.020438,
        "pt": 0.010607
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0.020103,
        "pt": 0.010752
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "FP32",
        "ic": 0.020317,
        "pt": 0.010637
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "FP16",
        "ic": 0.020076,
        "pt": 0.010677
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "BF16",
        "ic": 0.019797,
        "pt": 0.010604
      }
    ],
    "Cast": [
      {
        "shape": "bs=16, hidden=5632, fp16->fp32",
        "dtype": "FP16",
        "ic": 0.015751,
        "pt": 0.042433
      },
      {
        "shape": "bs=16, hidden=5632, fp32->fp16",
        "dtype": "FP32",
        "ic": 0.015609,
        "pt": 0.042591
      },
      {
        "shape": "bs=16, hidden=5632, fp32->bf16",
        "dtype": "FP32",
        "ic": 0.015492,
        "pt": 0.042662
      }
    ],
    "Cat": [
      {
        "shape": "2x(4, 64), dim=0",
        "dtype": "FP32",
        "ic": 0.020879,
        "pt": 0.044789
      },
      {
        "shape": "2x(4, 64), dim=0",
        "dtype": "FP16",
        "ic": 0.020631,
        "pt": 0.044248
      },
      {
        "shape": "2x(4, 64), dim=0",
        "dtype": "BF16",
        "ic": 0.020278,
        "pt": 0.044411
      },
      {
        "shape": "(4, 32)+(4, 64), dim=1",
        "dtype": "FP32",
        "ic": 0.020582,
        "pt": 0.044269
      },
      {
        "shape": "(4, 32)+(4, 64), dim=1",
        "dtype": "FP16",
        "ic": 0.020308,
        "pt": 0.044446
      },
      {
        "shape": "(4, 32)+(4, 64), dim=1",
        "dtype": "BF16",
        "ic": 0.020331,
        "pt": 0.044463
      },
      {
        "shape": "4x(1, 1024), dim=1",
        "dtype": "FP32",
        "ic": 0.029818,
        "pt": 0.045154
      },
      {
        "shape": "4x(1, 1024), dim=1",
        "dtype": "FP16",
        "ic": 0.029523,
        "pt": 0.045078
      },
      {
        "shape": "4x(1, 1024), dim=1",
        "dtype": "BF16",
        "ic": 0.029674,
        "pt": 0.046211
      }
    ],
    "Gemm": [
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.02305,
        "pt": 0.019564
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.022962,
        "pt": 0.012916
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.022306,
        "pt": 0.012943
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP32",
        "ic": 0.022549,
        "pt": 0.019589
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP16",
        "ic": 0.022619,
        "pt": 0.013052
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "BF16",
        "ic": 0.022635,
        "pt": 0.012873
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP32",
        "ic": 0.025228,
        "pt": 0.014741
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP16",
        "ic": 0.027776,
        "pt": 0.011733
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "BF16",
        "ic": 0.028154,
        "pt": 0.011487
      }
    ],
    "Linear": [
      {
        "shape": "M=4, K=64, N=32",
        "dtype": "FP32",
        "ic": 0.025708,
        "pt": 0.088799
      },
      {
        "shape": "M=4, K=64, N=32",
        "dtype": "FP16",
        "ic": 0.025249,
        "pt": 0.142709
      },
      {
        "shape": "M=4, K=64, N=32",
        "dtype": "BF16",
        "ic": 0.027908,
        "pt": 0.143725
      },
      {
        "shape": "M=1, K=4096, N=4096",
        "dtype": "FP32",
        "ic": 0.056449,
        "pt": 0.089664
      },
      {
        "shape": "M=1, K=4096, N=4096",
        "dtype": "FP16",
        "ic": 0.026193,
        "pt": 0.349535
      },
      {
        "shape": "M=1, K=4096, N=4096",
        "dtype": "BF16",
        "ic": 0.027529,
        "pt": 0.34877
      },
      {
        "shape": "bs=4, M=8, K=128, N=64",
        "dtype": "FP32",
        "ic": 0.073665,
        "pt": 0.090359
      },
      {
        "shape": "bs=4, M=8, K=128, N=64",
        "dtype": "FP16",
        "ic": 0.073199,
        "pt": 0.144602
      },
      {
        "shape": "bs=4, M=8, K=128, N=64",
        "dtype": "BF16",
        "ic": 0.070734,
        "pt": 0.137466
      }
    ],
    "MatMul": [
      {
        "shape": "M=4, K=64, N=32",
        "dtype": "FP32",
        "ic": 0.020581,
        "pt": 0.048252
      },
      {
        "shape": "M=4, K=64, N=32",
        "dtype": "FP16",
        "ic": 0.020429,
        "pt": 0.083963
      },
      {
        "shape": "M=4, K=64, N=32",
        "dtype": "BF16",
        "ic": 0.020579,
        "pt": 0.084223
      },
      {
        "shape": "M=2, K=128, N=256",
        "dtype": "FP32",
        "ic": 0.020582,
        "pt": 0.048543
      },
      {
        "shape": "M=2, K=128, N=256",
        "dtype": "FP16",
        "ic": 0.020816,
        "pt": 0.083821
      },
      {
        "shape": "M=2, K=128, N=256",
        "dtype": "BF16",
        "ic": 0.021628,
        "pt": 0.084331
      },
      {
        "shape": "bs=4, M=8, K=128, N=64",
        "dtype": "FP32",
        "ic": 0.021014,
        "pt": 0.051063
      },
      {
        "shape": "bs=4, M=8, K=128, N=64",
        "dtype": "FP16",
        "ic": 0.020892,
        "pt": 0.08676
      },
      {
        "shape": "bs=4, M=8, K=128, N=64",
        "dtype": "BF16",
        "ic": 0.020875,
        "pt": 0.087777
      }
    ],
    "Mul": [
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.020653,
        "pt": 0.00855
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.020579,
        "pt": 0.008521
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.020414,
        "pt": 0.00854
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0.020695,
        "pt": 0.008566
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0.020905,
        "pt": 0.008736
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0.020744,
        "pt": 0.008648
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "FP32",
        "ic": 0.020643,
        "pt": 0.008617
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "FP16",
        "ic": 0.020471,
        "pt": 0.008805
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "BF16",
        "ic": 0.020515,
        "pt": 0.00853
      }
    ]
  },
  "cambricon": {
    "CausalSoftmax": [
      {
        "shape": "M=3, N=3",
        "dtype": "FP16",
        "ic": 0.094429,
        "pt": 1.045267
      },
      {
        "shape": "M=3, N=3",
        "dtype": "BF16",
        "ic": 0.102903,
        "pt": 0.978345
      },
      {
        "shape": "M=3, N=3",
        "dtype": "FP32",
        "ic": 0.099518,
        "pt": 0.790179
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.108272,
        "pt": 0.982462
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.10851,
        "pt": 0.98072
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.105074,
        "pt": 0.790314
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.109852,
        "pt": 1.075256
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.105422,
        "pt": 1.072285
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.101235,
        "pt": 0.876319
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP16",
        "ic": 0.153843,
        "pt": 0.980796
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "BF16",
        "ic": 0.158734,
        "pt": 0.979013
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP32",
        "ic": 0.134516,
        "pt": 0.786985
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.265022,
        "pt": 0.98348
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.263519,
        "pt": 0.980056
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.224654,
        "pt": 0.786791
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.279279,
        "pt": 1.087633
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.277733,
        "pt": 1.09094
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.231732,
        "pt": 0.883058
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP16",
        "ic": 0.236904,
        "pt": 0.993053
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "BF16",
        "ic": 0.236797,
        "pt": 0.991576
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP32",
        "ic": 0.169307,
        "pt": 0.792383
      }
    ],
    "RMSNorm": [
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.194202,
        "pt": 1.05337
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.194576,
        "pt": 1.015319
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.186848,
        "pt": 0.950878
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.198418,
        "pt": 1.022772
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.198685,
        "pt": 1.008092
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.200894,
        "pt": 0.944005
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.195431,
        "pt": 1.018403
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.19537,
        "pt": 1.017246
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.193088,
        "pt": 0.954126
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.194541,
        "pt": 1.011864
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.201191,
        "pt": 1.01803
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.192439,
        "pt": 0.9573
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.196974,
        "pt": 1.023106
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.202419,
        "pt": 1.023811
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.200134,
        "pt": 0.953282
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.198379,
        "pt": 1.012315
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.203767,
        "pt": 1.019173
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.205844,
        "pt": 0.961053
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.198177,
        "pt": 1.134726
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.201133,
        "pt": 1.126289
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.196603,
        "pt": 1.054488
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.201989,
        "pt": 1.1301
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.200315,
        "pt": 1.113056
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.212954,
        "pt": 1.056347
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.213964,
        "pt": 1.154997
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.211114,
        "pt": 1.029214
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.210607,
        "pt": 0.959183
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.215586,
        "pt": 1.017423
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.217096,
        "pt": 1.01523
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.211961,
        "pt": 0.943123
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.211906,
        "pt": 1.117603
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.214481,
        "pt": 1.130273
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.22067,
        "pt": 1.061092
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.216525,
        "pt": 1.184753
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.212086,
        "pt": 1.123815
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.214153,
        "pt": 1.046613
      }
    ],
    "TopK": [
      {
        "shape": "M=6, N=8",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=6, N=8",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=6, N=8",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=8, N=4",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=5, N=5",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=3, N=7",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=10, N=3",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=2, N=16",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      }
    ],
    "Add": [
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.107916,
        "pt": 0.073867
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.112475,
        "pt": 0.073362
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.103615,
        "pt": 0.068311
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.121133,
        "pt": 0.198053
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.124782,
        "pt": 0.194088
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.113975,
        "pt": 0.196644
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.102594,
        "pt": 0.067558
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.10523,
        "pt": 0.066251
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.103428,
        "pt": 0.066199
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.122928,
        "pt": 0.196748
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.115665,
        "pt": 0.194487
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.114011,
        "pt": 0.195584
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.105502,
        "pt": 0.067015
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.107749,
        "pt": 0.06761
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.101249,
        "pt": 0.066859
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.133207,
        "pt": 0.198717
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.132268,
        "pt": 0.196243
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.130743,
        "pt": 0.197279
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.138316,
        "pt": 0.18128
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.133596,
        "pt": 0.178602
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.135946,
        "pt": 0.179757
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.103942,
        "pt": 0.069112
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.105709,
        "pt": 0.070874
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.095456,
        "pt": 0.07154
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.130534,
        "pt": 0.203192
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.127029,
        "pt": 0.200655
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.129589,
        "pt": 0.202823
      }
    ],
    "MatMul": [
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP16",
        "ic": 0.114612,
        "pt": 0.080441
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "BF16",
        "ic": 0.144409,
        "pt": 0.078137
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP32",
        "ic": 0.114104,
        "pt": 0.083652
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP16",
        "ic": 0.119618,
        "pt": 0.088419
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "BF16",
        "ic": 0.115179,
        "pt": 0.085378
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP32",
        "ic": 0.1171,
        "pt": 0.085042
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.110301,
        "pt": 0.14275
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.110761,
        "pt": 0.145421
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.131711,
        "pt": 0.143833
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP16",
        "ic": 0.104144,
        "pt": 0.136173
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "BF16",
        "ic": 0.100773,
        "pt": 0.145287
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP32",
        "ic": 0.102083,
        "pt": 0.146328
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.13372,
        "pt": 0.102344
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.131087,
        "pt": 0.09766
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.134051,
        "pt": 0.101302
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP16",
        "ic": 0.117345,
        "pt": 0.097629
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "BF16",
        "ic": 0.113424,
        "pt": 0.099967
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP32",
        "ic": 0.116632,
        "pt": 0.111626
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP16",
        "ic": 0.100875,
        "pt": 0.137883
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "BF16",
        "ic": 0.100769,
        "pt": 0.140401
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP32",
        "ic": 0.103727,
        "pt": 0.139695
      }
    ],
    "SiLU": [
      {
        "shape": "M=2, N=4",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=2, N=4",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=2, N=4",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=128, N=64",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      }
    ]
  },
  "hygon": {
    "CausalSoftmax": [
      {
        "shape": "M=3, N=3",
        "dtype": "FP16",
        "ic": 0.009787,
        "pt": 0.129153
      },
      {
        "shape": "M=3, N=3",
        "dtype": "BF16",
        "ic": 0.009956,
        "pt": 0.127809
      },
      {
        "shape": "M=3, N=3",
        "dtype": "FP32",
        "ic": 0.010025,
        "pt": 0.103338
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.009882,
        "pt": 0.129674
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.010166,
        "pt": 0.129008
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.010146,
        "pt": 0.102473
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.010096,
        "pt": 0.132463
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.010067,
        "pt": 0.132972
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.0101,
        "pt": 0.103579
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP16",
        "ic": 0.010386,
        "pt": 0.130492
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "BF16",
        "ic": 0.009926,
        "pt": 0.130201
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP32",
        "ic": 0.010338,
        "pt": 0.103961
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.018776,
        "pt": 0.132994
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.023624,
        "pt": 0.133352
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.019198,
        "pt": 0.105249
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.018776,
        "pt": 0.13603
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.023753,
        "pt": 0.137159
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.019198,
        "pt": 0.106402
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP16",
        "ic": 0.012845,
        "pt": 0.131947
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "BF16",
        "ic": 0.015126,
        "pt": 0.130327
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP32",
        "ic": 0.012797,
        "pt": 0.103811
      }
    ],
    "RMSNorm": [
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.011271,
        "pt": 0.111328
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.011458,
        "pt": 0.110337
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.011395,
        "pt": 0.09899
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.011513,
        "pt": 0.110699
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.011443,
        "pt": 0.110204
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.011534,
        "pt": 0.10041
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.011416,
        "pt": 0.111249
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.011415,
        "pt": 0.11134
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.011388,
        "pt": 0.100242
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.011324,
        "pt": 0.114
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.011887,
        "pt": 0.114306
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.01266,
        "pt": 0.112938
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.012602,
        "pt": 0.12759
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.01181,
        "pt": 0.135993
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.012592,
        "pt": 0.123379
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.012703,
        "pt": 0.125937
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.012671,
        "pt": 0.136462
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.012664,
        "pt": 0.12494
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.011879,
        "pt": 0.140525
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.012519,
        "pt": 0.129116
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.012685,
        "pt": 0.119069
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.012626,
        "pt": 0.129367
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.012661,
        "pt": 0.128851
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.012606,
        "pt": 0.118872
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.012373,
        "pt": 0.138384
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.012464,
        "pt": 0.137587
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.012375,
        "pt": 0.113523
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.012405,
        "pt": 0.136883
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.012391,
        "pt": 0.13773
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.01237,
        "pt": 0.116364
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.012403,
        "pt": 0.140968
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.012451,
        "pt": 0.142111
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.01246,
        "pt": 0.118252
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.012348,
        "pt": 0.140708
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.012305,
        "pt": 0.140645
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.012406,
        "pt": 0.126882
      }
    ],
    "Embedding": [
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP16",
        "ic": 0.009938,
        "pt": 0.015114
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "BF16",
        "ic": 0.00965,
        "pt": 0.014426
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP32",
        "ic": 0.009753,
        "pt": 0.014327
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP16",
        "ic": 0.138897,
        "pt": 0.014345
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "BF16",
        "ic": 0.139898,
        "pt": 0.014609
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP32",
        "ic": 0.054246,
        "pt": 0.014694
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP16",
        "ic": 0.009833,
        "pt": 0.014471
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "BF16",
        "ic": 0.009714,
        "pt": 0.014641
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP32",
        "ic": 0.009692,
        "pt": 0.014611
      }
    ],
    "TopK": [
      {
        "shape": "M=6, N=8",
        "dtype": "FP16",
        "ic": 0.752856,
        "pt": 0.015139
      },
      {
        "shape": "M=6, N=8",
        "dtype": "BF16",
        "ic": 0.754373,
        "pt": 0.017277
      },
      {
        "shape": "M=6, N=8",
        "dtype": "FP32",
        "ic": 0.758689,
        "pt": 0.0143
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP16",
        "ic": 0.500202,
        "pt": 0.023292
      },
      {
        "shape": "M=8, N=4",
        "dtype": "BF16",
        "ic": 0.498938,
        "pt": 0.023969
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP32",
        "ic": 0.499198,
        "pt": 0.023629
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP16",
        "ic": 0.75521,
        "pt": 0.022224
      },
      {
        "shape": "M=5, N=5",
        "dtype": "BF16",
        "ic": 0.832933,
        "pt": 0.024798
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP32",
        "ic": 0.923767,
        "pt": 0.024589
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP16",
        "ic": 0.956161,
        "pt": 0.035582
      },
      {
        "shape": "M=3, N=7",
        "dtype": "BF16",
        "ic": 0.969089,
        "pt": 0.035545
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP32",
        "ic": 0.965043,
        "pt": 0.035829
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP16",
        "ic": 0.502966,
        "pt": 0.014571
      },
      {
        "shape": "M=10, N=3",
        "dtype": "BF16",
        "ic": 0.503783,
        "pt": 0.016071
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP32",
        "ic": 0.504451,
        "pt": 0.016271
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP16",
        "ic": 0.971573,
        "pt": 0.038934
      },
      {
        "shape": "M=2, N=16",
        "dtype": "BF16",
        "ic": 0.966229,
        "pt": 0.039889
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP32",
        "ic": 0.96785,
        "pt": 0.039818
      }
    ],
    "Add": [
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.041482,
        "pt": 0.009111
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.041571,
        "pt": 0.008854
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.041936,
        "pt": 0.008827
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.041367,
        "pt": 0.009919
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.041575,
        "pt": 0.009895
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.04157,
        "pt": 0.00984
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.04142,
        "pt": 0.009182
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.041392,
        "pt": 0.009189
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.042087,
        "pt": 0.009092
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.041375,
        "pt": 0.009983
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.041503,
        "pt": 0.009956
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.041572,
        "pt": 0.009954
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.041569,
        "pt": 0.009154
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.041479,
        "pt": 0.009175
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.041318,
        "pt": 0.009205
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.041265,
        "pt": 0.009971
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.041716,
        "pt": 0.009931
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.041485,
        "pt": 0.00998
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.041643,
        "pt": 0.009976
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.042646,
        "pt": 0.011307
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.041576,
        "pt": 0.011242
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.042938,
        "pt": 0.009793
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.042949,
        "pt": 0.00998
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.041968,
        "pt": 0.010183
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.04198,
        "pt": 0.011427
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.042363,
        "pt": 0.011104
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.042101,
        "pt": 0.011665
      }
    ],
    "MatMul": [
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP16",
        "ic": 0.017647,
        "pt": 0.018309
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "BF16",
        "ic": 0.017627,
        "pt": 0.01848
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP32",
        "ic": 0.016959,
        "pt": 0.016546
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP16",
        "ic": 0.014391,
        "pt": 0.015255
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "BF16",
        "ic": 0.014221,
        "pt": 0.015169
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP32",
        "ic": 0.013069,
        "pt": 0.013427
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.019134,
        "pt": 0.02903
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.024418,
        "pt": 0.028864
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.041652,
        "pt": 0.041604
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP16",
        "ic": 0.017331,
        "pt": 0.028172
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "BF16",
        "ic": 0.017387,
        "pt": 0.027995
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP32",
        "ic": 0.017087,
        "pt": 0.026161
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.013768,
        "pt": 0.014654
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.013628,
        "pt": 0.014272
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.020477,
        "pt": 0.020313
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP16",
        "ic": 0.013445,
        "pt": 0.014092
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "BF16",
        "ic": 0.013259,
        "pt": 0.013864
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP32",
        "ic": 0.02771,
        "pt": 0.027872
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP16",
        "ic": 0.017604,
        "pt": 0.028415
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "BF16",
        "ic": 0.017764,
        "pt": 0.028245
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP32",
        "ic": 0.01688,
        "pt": 0.025773
      }
    ],
    "SiLU": [
      {
        "shape": "M=2, N=4",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=2, N=4",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=2, N=4",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=128, N=64",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0,
        "pt": 0
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0,
        "pt": 0
      }
    ]
  },
  "iluvatar": {
    "Add": [
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.018356,
        "pt": 0.013768
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.019648,
        "pt": 0.015195
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.021223,
        "pt": 0.015999
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0.022378,
        "pt": 0.017322
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0.023909,
        "pt": 0.017336
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0.023916,
        "pt": 0.017336
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "FP32",
        "ic": 0.019963,
        "pt": 0.017281
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "FP16",
        "ic": 0.019959,
        "pt": 0.017326
      },
      {
        "shape": "M=16, N=4096",
        "dtype": "BF16",
        "ic": 0.019854,
        "pt": 0.01733
      }
    ],
    "Causal_softmax": [
      {
        "shape": "bs=32, seq=512",
        "dtype": "FP32",
        "ic": 0.01495,
        "pt": 0.058623
      },
      {
        "shape": "bs=32, seq=512",
        "dtype": "FP16",
        "ic": 0.016247,
        "pt": 0.088117
      },
      {
        "shape": "bs=32, seq=512",
        "dtype": "BF16",
        "ic": 0.017703,
        "pt": 0.090513
      },
      {
        "shape": "bs=4, heads=20, seq=512",
        "dtype": "FP32",
        "ic": 0.022238,
        "pt": 0.079136
      },
      {
        "shape": "bs=4, heads=20, seq=512",
        "dtype": "FP16",
        "ic": 0.022633,
        "pt": 0.114115
      },
      {
        "shape": "bs=4, heads=20, seq=512",
        "dtype": "BF16",
        "ic": 0.022641,
        "pt": 0.112362
      }
    ],
    "Gemm": [
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.106689,
        "pt": 0.108511
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.034229,
        "pt": 0.050382
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.033671,
        "pt": 0.050376
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP32",
        "ic": 0.110243,
        "pt": 0.110435
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP16",
        "ic": 0.03837,
        "pt": 0.037306
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "BF16",
        "ic": 0.036873,
        "pt": 0.03703
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP32",
        "ic": 0.017378,
        "pt": 0.01734
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP16",
        "ic": 0.020597,
        "pt": 0.020612
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "BF16",
        "ic": 0.020587,
        "pt": 0.020606
      }
    ],
    "Rms_norm": [
      {
        "shape": "bs=2, seq=4, hidden=2048",
        "dtype": "FP32",
        "ic": 0.010282,
        "pt": 0.058665
      },
      {
        "shape": "bs=2, seq=4, hidden=2048",
        "dtype": "FP16",
        "ic": 0.011049,
        "pt": 0.103154
      },
      {
        "shape": "bs=2, seq=4, hidden=2048",
        "dtype": "BF16",
        "ic": 0.0113,
        "pt": 0.104387
      },
      {
        "shape": "bs=16, hidden=5632",
        "dtype": "FP32",
        "ic": 0.01386,
        "pt": 0.077777
      },
      {
        "shape": "bs=16, hidden=5632",
        "dtype": "FP16",
        "ic": 0.011182,
        "pt": 0.120675
      },
      {
        "shape": "bs=16, hidden=5632",
        "dtype": "BF16",
        "ic": 0.011151,
        "pt": 0.118685
      }
    ],
    "Swiglu": [
      {
        "shape": "bs=16, hidden=5632",
        "dtype": "FP32",
        "ic": 0.018707,
        "pt": 0.05884
      },
      {
        "shape": "bs=16, hidden=5632",
        "dtype": "FP16",
        "ic": 0.021242,
        "pt": 0.16574
      },
      {
        "shape": "bs=16, hidden=5632",
        "dtype": "BF16",
        "ic": 0.024096,
        "pt": 0.168865
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0.022965,
        "pt": 0.070715
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0.025448,
        "pt": 0.168191
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0.024097,
        "pt": 0.167276
      }
    ]
  },
  "metax": {
    "CausalSoftmax": [
      {
        "shape": "M=3, N=3",
        "dtype": "FP16",
        "ic": 0.013635,
        "pt": 0.110871
      },
      {
        "shape": "M=3, N=3",
        "dtype": "BF16",
        "ic": 0.01155,
        "pt": 0.111493
      },
      {
        "shape": "M=3, N=3",
        "dtype": "FP32",
        "ic": 0.016652,
        "pt": 0.087707
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.05365,
        "pt": 0.085022
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.039246,
        "pt": 0.148124
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.053587,
        "pt": 0.164711
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.022535,
        "pt": 0.160622
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.028964,
        "pt": 0.10713
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.02251,
        "pt": 0.096489
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP16",
        "ic": 0.035145,
        "pt": 0.13513
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "BF16",
        "ic": 0.032604,
        "pt": 0.121022
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP32",
        "ic": 0.040895,
        "pt": 0.080091
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.116487,
        "pt": 0.204376
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.05753,
        "pt": 0.211073
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.0867,
        "pt": 0.137035
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.077953,
        "pt": 0.251325
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.094521,
        "pt": 0.214623
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.090993,
        "pt": 0.230679
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP16",
        "ic": 0.043283,
        "pt": 0.130034
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "BF16",
        "ic": 0.079785,
        "pt": 0.126761
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP32",
        "ic": 0.032143,
        "pt": 0.121325
      }
    ],
    "RMSNorm": [
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.016996,
        "pt": 0.113902
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.028262,
        "pt": 0.098644
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.02594,
        "pt": 0.087757
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.01925,
        "pt": 0.089169
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.024112,
        "pt": 0.106743
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.015421,
        "pt": 0.113128
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.010957,
        "pt": 0.133982
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.011461,
        "pt": 0.11209
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.014231,
        "pt": 0.1104
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.012487,
        "pt": 0.092292
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.025135,
        "pt": 0.101656
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.014933,
        "pt": 0.077524
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.030123,
        "pt": 0.105386
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.027753,
        "pt": 0.125731
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.011717,
        "pt": 0.075296
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.029602,
        "pt": 0.085077
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.025103,
        "pt": 0.103111
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.012006,
        "pt": 0.081179
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.028189,
        "pt": 0.088881
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.021345,
        "pt": 0.115345
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.022438,
        "pt": 0.076853
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.027529,
        "pt": 0.086333
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.029357,
        "pt": 0.116251
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.03911,
        "pt": 0.09701
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.011731,
        "pt": 0.166145
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.055241,
        "pt": 0.250735
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.052253,
        "pt": 0.129109
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.025846,
        "pt": 0.195656
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.011057,
        "pt": 0.150595
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.01677,
        "pt": 0.146829
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.036359,
        "pt": 0.170067
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.01528,
        "pt": 0.174609
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.038336,
        "pt": 0.133928
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.032136,
        "pt": 0.13557
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.028555,
        "pt": 0.131683
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.036448,
        "pt": 0.132855
      }
    ],
    "Embedding": [
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP16",
        "ic": 0.023533,
        "pt": 0.029413
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "BF16",
        "ic": 0.010668,
        "pt": 0.024552
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP32",
        "ic": 0.020556,
        "pt": 0.018767
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP16",
        "ic": 0.104279,
        "pt": 0.035665
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "BF16",
        "ic": 0.101971,
        "pt": 0.026004
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP32",
        "ic": 0.122889,
        "pt": 0.029138
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP16",
        "ic": 0.008241,
        "pt": 0.024338
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "BF16",
        "ic": 0.030502,
        "pt": 0.012038
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP32",
        "ic": 0.021877,
        "pt": 0.028943
      }
    ],
    "TopK": [
      {
        "shape": "M=6, N=8",
        "dtype": "FP16",
        "ic": 2.861134,
        "pt": 0.026136
      },
      {
        "shape": "M=6, N=8",
        "dtype": "BF16",
        "ic": 2.628399,
        "pt": 0.041775
      },
      {
        "shape": "M=6, N=8",
        "dtype": "FP32",
        "ic": 2.843801,
        "pt": 0.026716
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP16",
        "ic": 1.912624,
        "pt": 0.086622
      },
      {
        "shape": "M=8, N=4",
        "dtype": "BF16",
        "ic": 1.800785,
        "pt": 0.063622
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP32",
        "ic": 1.577265,
        "pt": 0.043003
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP16",
        "ic": 2.623598,
        "pt": 0.075198
      },
      {
        "shape": "M=5, N=5",
        "dtype": "BF16",
        "ic": 2.534146,
        "pt": 0.075602
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP32",
        "ic": 2.792694,
        "pt": 0.077191
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP16",
        "ic": 2.869623,
        "pt": 0.065922
      },
      {
        "shape": "M=3, N=7",
        "dtype": "BF16",
        "ic": 2.61598,
        "pt": 0.078159
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP32",
        "ic": 2.61206,
        "pt": 0.06452
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP16",
        "ic": 1.951078,
        "pt": 0.036654
      },
      {
        "shape": "M=10, N=3",
        "dtype": "BF16",
        "ic": 1.963748,
        "pt": 0.032727
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP32",
        "ic": 2.233065,
        "pt": 0.034673
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP16",
        "ic": 2.551683,
        "pt": 0.068988
      },
      {
        "shape": "M=2, N=16",
        "dtype": "BF16",
        "ic": 2.521023,
        "pt": 0.077074
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP32",
        "ic": 2.505571,
        "pt": 0.08076
      }
    ],
    "Add": [
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.090582,
        "pt": 0.011058
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.07378,
        "pt": 0.011875
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.09095,
        "pt": 0.013693
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.074648,
        "pt": 0.008409
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.084558,
        "pt": 0.020535
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.072711,
        "pt": 0.018874
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.094695,
        "pt": 0.017127
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.104813,
        "pt": 0.007081
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.106044,
        "pt": 0.013038
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.082896,
        "pt": 0.012263
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.143787,
        "pt": 0.010816
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.095978,
        "pt": 0.012027
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.083723,
        "pt": 0.016617
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.144439,
        "pt": 0.01166
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.076188,
        "pt": 0.010515
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.112394,
        "pt": 0.02782
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.081711,
        "pt": 0.022087
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.14024,
        "pt": 0.017798
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.081727,
        "pt": 0.015465
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.090931,
        "pt": 0.019829
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.100524,
        "pt": 0.013294
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.077266,
        "pt": 0.012078
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.1476,
        "pt": 0.013557
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.127949,
        "pt": 0.007134
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.09635,
        "pt": 0.032625
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.139831,
        "pt": 0.050474
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.142207,
        "pt": 0.049469
      }
    ],
    "MatMul": [
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP16",
        "ic": 0.024642,
        "pt": 0.019783
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "BF16",
        "ic": 0.020176,
        "pt": 0.020349
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP32",
        "ic": 0.021615,
        "pt": 0.028188
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP16",
        "ic": 0.01283,
        "pt": 0.016909
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "BF16",
        "ic": 0.013011,
        "pt": 0.028065
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP32",
        "ic": 0.0177,
        "pt": 0.011163
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.083885,
        "pt": 0.087879
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.07928,
        "pt": 0.062126
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.077383,
        "pt": 0.094954
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP16",
        "ic": 0.019872,
        "pt": 0.028064
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "BF16",
        "ic": 0.018415,
        "pt": 0.035138
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP32",
        "ic": 0.013184,
        "pt": 0.019668
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.079638,
        "pt": 0.045003
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.083457,
        "pt": 0.067896
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.103102,
        "pt": 0.079031
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP16",
        "ic": 0.115237,
        "pt": 0.160425
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "BF16",
        "ic": 0.116744,
        "pt": 0.133648
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP32",
        "ic": 0.09235,
        "pt": 0.083693
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP16",
        "ic": 0.024869,
        "pt": 0.028519
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "BF16",
        "ic": 0.030141,
        "pt": 0.020819
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP32",
        "ic": 0.025517,
        "pt": 0.022534
      }
    ],
    "SiLU": [
      {
        "shape": "M=2, N=4",
        "dtype": "FP16",
        "ic": 0.135339,
        "pt": 0.015925
      },
      {
        "shape": "M=2, N=4",
        "dtype": "BF16",
        "ic": 0.074622,
        "pt": 0.021634
      },
      {
        "shape": "M=2, N=4",
        "dtype": "FP32",
        "ic": 0.115891,
        "pt": 0.028268
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP16",
        "ic": 0.093413,
        "pt": 0.019769
      },
      {
        "shape": "M=128, N=64",
        "dtype": "BF16",
        "ic": 0.080463,
        "pt": 0.031497
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP32",
        "ic": 0.096212,
        "pt": 0.037899
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP16",
        "ic": 0.068982,
        "pt": 0.036639
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "BF16",
        "ic": 0.086803,
        "pt": 0.020481
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP32",
        "ic": 0.063301,
        "pt": 0.029843
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP16",
        "ic": 0.090702,
        "pt": 0.014832
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "BF16",
        "ic": 0.089,
        "pt": 0.017586
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP32",
        "ic": 0.063291,
        "pt": 0.024054
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP16",
        "ic": 0.103683,
        "pt": 0.014591
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "BF16",
        "ic": 0.080669,
        "pt": 0.023987
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP32",
        "ic": 0.077627,
        "pt": 0.026801
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP16",
        "ic": 0.083478,
        "pt": 0.023449
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "BF16",
        "ic": 0.068692,
        "pt": 0.02535
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP32",
        "ic": 0.083297,
        "pt": 0.025245
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.092367,
        "pt": 0.035966
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.144934,
        "pt": 0.030738
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.12544,
        "pt": 0.043191
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0.128792,
        "pt": 0.024609
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0.110109,
        "pt": 0.031422
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0.117647,
        "pt": 0.019975
      }
    ]
  },
  "mthreads": {
    "CausalSoftmax": [
      {
        "shape": "M=3, N=3",
        "dtype": "FP16",
        "ic": 0.017228,
        "pt": 0.219861
      },
      {
        "shape": "M=3, N=3",
        "dtype": "BF16",
        "ic": 0.015092,
        "pt": 0.209454
      },
      {
        "shape": "M=3, N=3",
        "dtype": "FP32",
        "ic": 0.01519,
        "pt": 0.167179
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.015205,
        "pt": 0.22162
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.015456,
        "pt": 0.217761
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.015671,
        "pt": 0.163798
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.015377,
        "pt": 0.238748
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.015393,
        "pt": 0.239843
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.01507,
        "pt": 0.167277
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP16",
        "ic": 0.015997,
        "pt": 0.215992
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "BF16",
        "ic": 0.015931,
        "pt": 0.216364
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP32",
        "ic": 0.016869,
        "pt": 0.171688
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.048507,
        "pt": 0.213543
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.049158,
        "pt": 0.208482
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.050372,
        "pt": 0.131486
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.048696,
        "pt": 0.193585
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.048907,
        "pt": 0.20369
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.050393,
        "pt": 0.134409
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP16",
        "ic": 0.033525,
        "pt": 0.169996
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "BF16",
        "ic": 0.033693,
        "pt": 0.170425
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP32",
        "ic": 0.032915,
        "pt": 0.143978
      }
    ],
    "RMSNorm": [
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.017637,
        "pt": 0.225277
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.018175,
        "pt": 0.228036
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.018076,
        "pt": 0.205846
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.017378,
        "pt": 0.224018
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.018035,
        "pt": 0.223188
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.017973,
        "pt": 0.200604
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.018034,
        "pt": 0.208318
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.018115,
        "pt": 0.209767
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.019044,
        "pt": 0.189358
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.017793,
        "pt": 0.203682
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.018069,
        "pt": 0.198326
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.019678,
        "pt": 0.190303
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.018337,
        "pt": 0.210373
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.018044,
        "pt": 0.204073
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.017943,
        "pt": 0.182264
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.017962,
        "pt": 0.210684
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.018107,
        "pt": 0.199334
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.01737,
        "pt": 0.190342
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.018119,
        "pt": 0.236141
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.017992,
        "pt": 0.235205
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.017929,
        "pt": 0.216622
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.01812,
        "pt": 0.235991
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.017636,
        "pt": 0.234407
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.018076,
        "pt": 0.216596
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.017379,
        "pt": 0.213134
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.017111,
        "pt": 0.213004
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.017285,
        "pt": 0.192931
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.017654,
        "pt": 0.213408
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.017781,
        "pt": 0.212899
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.017918,
        "pt": 0.192917
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.015368,
        "pt": 0.239487
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.015486,
        "pt": 0.239988
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.015685,
        "pt": 0.220399
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.015607,
        "pt": 0.241268
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.015536,
        "pt": 0.239942
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.011016,
        "pt": 0.220343
      }
    ],
    "Embedding": [
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP16",
        "ic": 0.016681,
        "pt": 0.018365
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "BF16",
        "ic": 0.016991,
        "pt": 0.017793
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP32",
        "ic": 0.016143,
        "pt": 0.017794
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP16",
        "ic": 0.142939,
        "pt": 0.020498
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "BF16",
        "ic": 0.142868,
        "pt": 0.017291
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP32",
        "ic": 0.075153,
        "pt": 0.018163
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP16",
        "ic": 0.016105,
        "pt": 0.017392
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "BF16",
        "ic": 0.016626,
        "pt": 0.018154
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP32",
        "ic": 0.017951,
        "pt": 0.017484
      }
    ],
    "TopK": [
      {
        "shape": "M=6, N=8",
        "dtype": "FP16",
        "ic": 1.375818,
        "pt": 0.024335
      },
      {
        "shape": "M=6, N=8",
        "dtype": "BF16",
        "ic": 1.303775,
        "pt": 0.024645
      },
      {
        "shape": "M=6, N=8",
        "dtype": "FP32",
        "ic": 1.370844,
        "pt": 0.02433
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP16",
        "ic": 1.058127,
        "pt": 0.046262
      },
      {
        "shape": "M=8, N=4",
        "dtype": "BF16",
        "ic": 1.065943,
        "pt": 0.046093
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP32",
        "ic": 1.065595,
        "pt": 0.04708
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP16",
        "ic": 1.348646,
        "pt": 0.022548
      },
      {
        "shape": "M=5, N=5",
        "dtype": "BF16",
        "ic": 1.30082,
        "pt": 0.02261
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP32",
        "ic": 1.288613,
        "pt": 0.022158
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP16",
        "ic": 1.297174,
        "pt": 0.04453
      },
      {
        "shape": "M=3, N=7",
        "dtype": "BF16",
        "ic": 1.306557,
        "pt": 0.044575
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP32",
        "ic": 1.280816,
        "pt": 0.043555
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP16",
        "ic": 1.075252,
        "pt": 0.022638
      },
      {
        "shape": "M=10, N=3",
        "dtype": "BF16",
        "ic": 1.08652,
        "pt": 0.023542
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP32",
        "ic": 1.058583,
        "pt": 0.022423
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP16",
        "ic": 1.299758,
        "pt": 0.044755
      },
      {
        "shape": "M=2, N=16",
        "dtype": "BF16",
        "ic": 1.289314,
        "pt": 0.043364
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP32",
        "ic": 1.29073,
        "pt": 0.043699
      }
    ],
    "Add": [
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.111088,
        "pt": 0.01462
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.111325,
        "pt": 0.016996
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.109417,
        "pt": 0.016618
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.112784,
        "pt": 0.040525
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.11275,
        "pt": 0.04246
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.113186,
        "pt": 0.042262
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.10551,
        "pt": 0.015171
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.107646,
        "pt": 0.014909
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.108669,
        "pt": 0.014934
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.111347,
        "pt": 0.040156
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.111791,
        "pt": 0.037089
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.115375,
        "pt": 0.036897
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.111439,
        "pt": 0.015545
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.110396,
        "pt": 0.014793
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.109146,
        "pt": 0.015489
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.119996,
        "pt": 0.037826
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.115443,
        "pt": 0.038811
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.115671,
        "pt": 0.040904
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.119261,
        "pt": 0.016896
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.118248,
        "pt": 0.01723
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.120715,
        "pt": 0.014483
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.113191,
        "pt": 0.015329
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.107413,
        "pt": 0.010696
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.107926,
        "pt": 0.011041
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.117225,
        "pt": 0.035338
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.116098,
        "pt": 0.027592
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.121227,
        "pt": 0.024027
      }
    ],
    "MatMul": [
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP16",
        "ic": 0.02107,
        "pt": 0.021651
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "BF16",
        "ic": 0.021047,
        "pt": 0.019704
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP32",
        "ic": 0.025477,
        "pt": 0.01905
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP16",
        "ic": 0.035695,
        "pt": 0.03225
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "BF16",
        "ic": 0.032085,
        "pt": 0.028286
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP32",
        "ic": 0.031207,
        "pt": 0.028055
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.12532,
        "pt": 0.124846
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.12747,
        "pt": 0.127111
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.117789,
        "pt": 0.117516
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP16",
        "ic": 0.032355,
        "pt": 0.037081
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "BF16",
        "ic": 0.032162,
        "pt": 0.037491
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP32",
        "ic": 0.031787,
        "pt": 0.027638
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.022965,
        "pt": 0.039889
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.023147,
        "pt": 0.040546
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.021244,
        "pt": 0.039297
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP16",
        "ic": 0.033927,
        "pt": 0.031868
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "BF16",
        "ic": 0.033232,
        "pt": 0.032401
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP32",
        "ic": 0.029396,
        "pt": 0.198981
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP16",
        "ic": 0.032265,
        "pt": 0.037094
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "BF16",
        "ic": 0.032255,
        "pt": 0.037111
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP32",
        "ic": 0.032307,
        "pt": 0.026901
      }
    ],
    "SiLU": [
      {
        "shape": "M=2, N=4",
        "dtype": "FP16",
        "ic": 0.078161,
        "pt": 0.024105
      },
      {
        "shape": "M=2, N=4",
        "dtype": "BF16",
        "ic": 0.078775,
        "pt": 0.025042
      },
      {
        "shape": "M=2, N=4",
        "dtype": "FP32",
        "ic": 0.077434,
        "pt": 0.024207
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP16",
        "ic": 0.078958,
        "pt": 0.024178
      },
      {
        "shape": "M=128, N=64",
        "dtype": "BF16",
        "ic": 0.076872,
        "pt": 0.024562
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP32",
        "ic": 0.07569,
        "pt": 0.024201
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP16",
        "ic": 0.075893,
        "pt": 0.024043
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "BF16",
        "ic": 0.075367,
        "pt": 0.023515
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP32",
        "ic": 0.074924,
        "pt": 0.024378
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP16",
        "ic": 0.075529,
        "pt": 0.024238
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "BF16",
        "ic": 0.07616,
        "pt": 0.024325
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP32",
        "ic": 0.075783,
        "pt": 0.023982
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP16",
        "ic": 0.075838,
        "pt": 0.024408
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "BF16",
        "ic": 0.075351,
        "pt": 0.024348
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP32",
        "ic": 0.076345,
        "pt": 0.0244
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP16",
        "ic": 0.075938,
        "pt": 0.024193
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "BF16",
        "ic": 0.075659,
        "pt": 0.024477
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP32",
        "ic": 0.075764,
        "pt": 0.024426
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.076272,
        "pt": 0.021713
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.082506,
        "pt": 0.027981
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.076894,
        "pt": 0.023089
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0.075871,
        "pt": 0.024066
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0.07583,
        "pt": 0.026273
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0.075938,
        "pt": 0.025661
      }
    ]
  },
  "nvidia": {
    "CausalSoftmax": [
      {
        "shape": "M=3, N=3",
        "dtype": "FP16",
        "ic": 0.006581,
        "pt": 0.065546
      },
      {
        "shape": "M=3, N=3",
        "dtype": "BF16",
        "ic": 0.006427,
        "pt": 0.065766
      },
      {
        "shape": "M=3, N=3",
        "dtype": "FP32",
        "ic": 0.00642,
        "pt": 0.052705
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.006341,
        "pt": 0.066423
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.006445,
        "pt": 0.065648
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.006353,
        "pt": 0.053571
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP16",
        "ic": 0.006411,
        "pt": 0.066575
      },
      {
        "shape": "M=32, N=512",
        "dtype": "BF16",
        "ic": 0.006438,
        "pt": 0.066439
      },
      {
        "shape": "M=32, N=512",
        "dtype": "FP32",
        "ic": 0.006366,
        "pt": 0.05236
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP16",
        "ic": 0.006439,
        "pt": 0.066963
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "BF16",
        "ic": 0.006422,
        "pt": 0.068901
      },
      {
        "shape": "bs=32, seq=5, hidden=5",
        "dtype": "FP32",
        "ic": 0.006403,
        "pt": 0.051177
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.013257,
        "pt": 0.067941
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.014781,
        "pt": 0.065739
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.012442,
        "pt": 0.051679
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP16",
        "ic": 0.013264,
        "pt": 0.06753
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "BF16",
        "ic": 0.014781,
        "pt": 0.069248
      },
      {
        "shape": "bs=32, seq=20, hidden=512",
        "dtype": "FP32",
        "ic": 0.01244,
        "pt": 0.053494
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP16",
        "ic": 0.008815,
        "pt": 0.066391
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "BF16",
        "ic": 0.009098,
        "pt": 0.065727
      },
      {
        "shape": "bs=28, seq=15, hidden=15",
        "dtype": "FP32",
        "ic": 0.008009,
        "pt": 0.052025
      }
    ],
    "RMSNorm": [
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.007772,
        "pt": 0.066294
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.007711,
        "pt": 0.066075
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.007571,
        "pt": 0.059403
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.007525,
        "pt": 0.06549
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.007536,
        "pt": 0.065962
      },
      {
        "shape": "M=1, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.007473,
        "pt": 0.058896
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.007565,
        "pt": 0.066551
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.007669,
        "pt": 0.066727
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "FP16",
        "ic": 0.00742,
        "pt": 0.059555
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.008089,
        "pt": 0.066936
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.007731,
        "pt": 0.067592
      },
      {
        "shape": "M=2, N=4, size=4",
        "dtype": "BF16",
        "ic": 0.007506,
        "pt": 0.060345
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.007659,
        "pt": 0.066879
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.00748,
        "pt": 0.06695
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.007592,
        "pt": 0.059371
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.007568,
        "pt": 0.066634
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.007536,
        "pt": 0.066984
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.008507,
        "pt": 0.060159
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.007501,
        "pt": 0.06728
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.00755,
        "pt": 0.067362
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "FP16",
        "ic": 0.007601,
        "pt": 0.060296
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.007541,
        "pt": 0.067262
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.007407,
        "pt": 0.067176
      },
      {
        "shape": "bs=2, seq=2, hidden=4, size=4",
        "dtype": "BF16",
        "ic": 0.007436,
        "pt": 0.060715
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.007583,
        "pt": 0.068343
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.007477,
        "pt": 0.066544
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.007474,
        "pt": 0.060109
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.007698,
        "pt": 0.067291
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.007478,
        "pt": 0.066561
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.007642,
        "pt": 0.059776
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.007481,
        "pt": 0.068058
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.011073,
        "pt": 0.067961
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "FP16",
        "ic": 0.007521,
        "pt": 0.060984
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.007535,
        "pt": 0.067811
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.007534,
        "pt": 0.069071
      },
      {
        "shape": "M=16, N=2048, size=2048",
        "dtype": "BF16",
        "ic": 0.007573,
        "pt": 0.060545
      }
    ],
    "Embedding": [
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP16",
        "ic": 0.006692,
        "pt": 0.010756
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "BF16",
        "ic": 0.006589,
        "pt": 0.010815
      },
      {
        "shape": "M=1, N=5, M=32000, N=4",
        "dtype": "FP32",
        "ic": 0.006602,
        "pt": 0.010664
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP16",
        "ic": 0.060203,
        "pt": 0.010698
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "BF16",
        "ic": 0.058017,
        "pt": 0.010739
      },
      {
        "shape": "M=2, N=10, M=32000, N=2048",
        "dtype": "FP32",
        "ic": 0.044,
        "pt": 0.010351
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP16",
        "ic": 0.006483,
        "pt": 0.010639
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "BF16",
        "ic": 0.00648,
        "pt": 0.010505
      },
      {
        "shape": "M=1, N=5, M=10, N=10",
        "dtype": "FP32",
        "ic": 0.00653,
        "pt": 0.010638
      }
    ],
    "TopK": [
      {
        "shape": "M=6, N=8",
        "dtype": "FP16",
        "ic": 0.435852,
        "pt": 0.009919
      },
      {
        "shape": "M=6, N=8",
        "dtype": "BF16",
        "ic": 0.396934,
        "pt": 0.009802
      },
      {
        "shape": "M=6, N=8",
        "dtype": "FP32",
        "ic": 0.398159,
        "pt": 0.009831
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP16",
        "ic": 0.313976,
        "pt": 0.018095
      },
      {
        "shape": "M=8, N=4",
        "dtype": "BF16",
        "ic": 0.313508,
        "pt": 0.017035
      },
      {
        "shape": "M=8, N=4",
        "dtype": "FP32",
        "ic": 0.314095,
        "pt": 0.017271
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP16",
        "ic": 0.40708,
        "pt": 0.016082
      },
      {
        "shape": "M=5, N=5",
        "dtype": "BF16",
        "ic": 0.401918,
        "pt": 0.015949
      },
      {
        "shape": "M=5, N=5",
        "dtype": "FP32",
        "ic": 0.401479,
        "pt": 0.016336
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP16",
        "ic": 0.400816,
        "pt": 0.023797
      },
      {
        "shape": "M=3, N=7",
        "dtype": "BF16",
        "ic": 0.399128,
        "pt": 0.023646
      },
      {
        "shape": "M=3, N=7",
        "dtype": "FP32",
        "ic": 0.400229,
        "pt": 0.023884
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP16",
        "ic": 0.312614,
        "pt": 0.00966
      },
      {
        "shape": "M=10, N=3",
        "dtype": "BF16",
        "ic": 0.313277,
        "pt": 0.009818
      },
      {
        "shape": "M=10, N=3",
        "dtype": "FP32",
        "ic": 0.313715,
        "pt": 0.009832
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP16",
        "ic": 0.392805,
        "pt": 0.023587
      },
      {
        "shape": "M=2, N=16",
        "dtype": "BF16",
        "ic": 0.390576,
        "pt": 0.023604
      },
      {
        "shape": "M=2, N=16",
        "dtype": "FP32",
        "ic": 0.391126,
        "pt": 0.023953
      }
    ],
    "Add": [
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.010488,
        "pt": 0.006039
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.010353,
        "pt": 0.005824
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.010126,
        "pt": 0.00581
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.010413,
        "pt": 0.006287
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.012363,
        "pt": 0.007117
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.010035,
        "pt": 0.006731
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.010386,
        "pt": 0.006375
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.009984,
        "pt": 0.00553
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.009961,
        "pt": 0.00564
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP16",
        "ic": 0.010268,
        "pt": 0.006258
      },
      {
        "shape": "M=13, N=4",
        "dtype": "BF16",
        "ic": 0.010044,
        "pt": 0.006174
      },
      {
        "shape": "M=13, N=4",
        "dtype": "FP32",
        "ic": 0.010184,
        "pt": 0.006073
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.010142,
        "pt": 0.00573
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.010101,
        "pt": 0.005613
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.010053,
        "pt": 0.005602
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.010286,
        "pt": 0.006293
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.010918,
        "pt": 0.006186
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.010087,
        "pt": 0.00616
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP16",
        "ic": 0.010052,
        "pt": 0.006292
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "BF16",
        "ic": 0.010036,
        "pt": 0.006221
      },
      {
        "shape": "bs=13, seq=4, hidden=4",
        "dtype": "FP32",
        "ic": 0.010306,
        "pt": 0.006148
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.010129,
        "pt": 0.00566
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.010076,
        "pt": 0.005555
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.010152,
        "pt": 0.005737
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.010126,
        "pt": 0.006586
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.010031,
        "pt": 0.006303
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.010184,
        "pt": 0.006231
      }
    ],
    "MatMul": [
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP16",
        "ic": 0.009274,
        "pt": 0.009507
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "BF16",
        "ic": 0.00852,
        "pt": 0.008788
      },
      {
        "shape": "M=2, N=4, K=3",
        "dtype": "FP32",
        "ic": 0.009414,
        "pt": 0.008625
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP16",
        "ic": 0.009034,
        "pt": 0.009463
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "BF16",
        "ic": 0.008769,
        "pt": 0.00954
      },
      {
        "shape": "M=128, N=64, K=256",
        "dtype": "FP32",
        "ic": 0.012313,
        "pt": 0.011335
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.012341,
        "pt": 0.01369
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.012334,
        "pt": 0.013719
      },
      {
        "shape": "bs=2, M=4, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.053796,
        "pt": 0.05379
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP16",
        "ic": 0.008951,
        "pt": 0.013832
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "BF16",
        "ic": 0.008917,
        "pt": 0.013896
      },
      {
        "shape": "bs=4, M=48, N=6, K=64",
        "dtype": "FP32",
        "ic": 0.009149,
        "pt": 0.013595
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP16",
        "ic": 0.011734,
        "pt": 0.011365
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "BF16",
        "ic": 0.011772,
        "pt": 0.011466
      },
      {
        "shape": "M=1, N=2048, K=2048",
        "dtype": "FP32",
        "ic": 0.013413,
        "pt": 0.013417
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP16",
        "ic": 0.012549,
        "pt": 0.013626
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "BF16",
        "ic": 0.012076,
        "pt": 0.012249
      },
      {
        "shape": "M=6, N=2560, K=2048",
        "dtype": "FP32",
        "ic": 0.020502,
        "pt": 0.029066
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP16",
        "ic": 0.008955,
        "pt": 0.013909
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "BF16",
        "ic": 0.008818,
        "pt": 0.013816
      },
      {
        "shape": "bs=8, M=16, N=32, K=16",
        "dtype": "FP32",
        "ic": 0.008363,
        "pt": 0.013188
      }
    ],
    "SiLU": [
      {
        "shape": "M=2, N=4",
        "dtype": "FP16",
        "ic": 0.010881,
        "pt": 0.011481
      },
      {
        "shape": "M=2, N=4",
        "dtype": "BF16",
        "ic": 0.009398,
        "pt": 0.011645
      },
      {
        "shape": "M=2, N=4",
        "dtype": "FP32",
        "ic": 0.009562,
        "pt": 0.011288
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP16",
        "ic": 0.01044,
        "pt": 0.012252
      },
      {
        "shape": "M=128, N=64",
        "dtype": "BF16",
        "ic": 0.010436,
        "pt": 0.012047
      },
      {
        "shape": "M=128, N=64",
        "dtype": "FP32",
        "ic": 0.00949,
        "pt": 0.011416
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP16",
        "ic": 0.009481,
        "pt": 0.011653
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "BF16",
        "ic": 0.011238,
        "pt": 0.011662
      },
      {
        "shape": "bs=2, seq=4, hidden=8",
        "dtype": "FP32",
        "ic": 0.009473,
        "pt": 0.011632
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP16",
        "ic": 0.009418,
        "pt": 0.011602
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "BF16",
        "ic": 0.009446,
        "pt": 0.011493
      },
      {
        "shape": "bs=4, seq=48, hidden=6",
        "dtype": "FP32",
        "ic": 0.009469,
        "pt": 0.011437
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP16",
        "ic": 0.009468,
        "pt": 0.011714
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "BF16",
        "ic": 0.009489,
        "pt": 0.011796
      },
      {
        "shape": "M=1, N=2048",
        "dtype": "FP32",
        "ic": 0.009537,
        "pt": 0.011561
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP16",
        "ic": 0.009508,
        "pt": 0.011426
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "BF16",
        "ic": 0.00938,
        "pt": 0.011274
      },
      {
        "shape": "bs=8, seq=16, hidden=32",
        "dtype": "FP32",
        "ic": 0.009418,
        "pt": 0.011553
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP16",
        "ic": 0.009443,
        "pt": 0.011568
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "BF16",
        "ic": 0.009458,
        "pt": 0.011558
      },
      {
        "shape": "M=16, N=5632",
        "dtype": "FP32",
        "ic": 0.009919,
        "pt": 0.011506
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP16",
        "ic": 0.009438,
        "pt": 0.011578
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "BF16",
        "ic": 0.009418,
        "pt": 0.011628
      },
      {
        "shape": "bs=4, seq=4, hidden=5632",
        "dtype": "FP32",
        "ic": 0.009442,
        "pt": 0.011406
      }
    ]
  }
} as Record<string, Record<string, { shape: string; dtype: string; ic: number; pt: number }[]>>

export const INFER_TABLE_FROM_FILES = {
  "generic": {
    "prefill": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 2472,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 2488,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 2497,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5350,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5387,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5380,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 4720,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 4738,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 4739,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 4402,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 4409,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 4414,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5968,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5965,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5968,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 4756,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 4758,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 4777,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5672,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5672,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5674,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 6131,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 6130,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 6139,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 6234,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 6221,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 6237,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 6220,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 6225,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 6224,
        "framework": "InfiniLM"
      }
    ],
    "decode": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 82,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 83,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 82,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 84,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 84,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 81,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 78,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 77,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 75,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 330,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 329,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 316,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 329,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 329,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 315,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 297,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 295,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 283,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 1275,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 1240,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1142,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 1254,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 1224,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1131,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 3252,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 3100,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 2657,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 3154,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 3019,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 2601,
        "framework": "InfiniLM"
      }
    ]
  },
  "cambricon": {
    "prefill": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 296,
        "framework": "InfiniLM",
        "ttft": 108.13
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 294,
        "framework": "InfiniLM",
        "ttft": 108.77
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 294,
        "framework": "InfiniLM",
        "ttft": 108.8
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 529,
        "framework": "InfiniLM",
        "ttft": 484.2
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 529,
        "framework": "InfiniLM",
        "ttft": 484.2
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 527,
        "framework": "InfiniLM",
        "ttft": 485.41
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 757,
        "framework": "InfiniLM",
        "ttft": 169.09
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 740,
        "framework": "InfiniLM",
        "ttft": 173.03
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 750,
        "framework": "InfiniLM",
        "ttft": 170.66
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 662,
        "framework": "InfiniLM",
        "ttft": 1545.53
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 659,
        "framework": "InfiniLM",
        "ttft": 1554.14
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 657,
        "framework": "InfiniLM",
        "ttft": 1558.12
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 1232,
        "framework": "InfiniLM",
        "ttft": 415.43
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 1239,
        "framework": "InfiniLM",
        "ttft": 413.39
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1239,
        "framework": "InfiniLM",
        "ttft": 413.16
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 710,
        "framework": "InfiniLM",
        "ttft": 5768.25
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 711,
        "framework": "InfiniLM",
        "ttft": 5757.43
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 711,
        "framework": "InfiniLM",
        "ttft": 5758.43
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 1398,
        "framework": "InfiniLM",
        "ttft": 1464.54
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 1394,
        "framework": "InfiniLM",
        "ttft": 1473.69
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1396,
        "framework": "InfiniLM",
        "ttft": 1467.53
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 718,
        "framework": "InfiniLM",
        "ttft": 22822.43
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 719,
        "framework": "InfiniLM",
        "ttft": 22796.38
      }
    ],
    "decode": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 12,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 12,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 12,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 12,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 12,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 12,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 51,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 49,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 50,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 51,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 50,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 50,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 223,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 225,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 222,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 222,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 225,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 220,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 551,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 521,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 519,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 555,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 544,
        "framework": "InfiniLM"
      }
    ]
  },
  "hygon": {
    "prefill": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 315,
        "framework": "InfiniLM",
        "ttft": 101.67
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 314,
        "framework": "InfiniLM",
        "ttft": 101.98
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 308,
        "framework": "InfiniLM",
        "ttft": 103.96
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 387,
        "framework": "InfiniLM",
        "ttft": 660.66
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 386,
        "framework": "InfiniLM",
        "ttft": 663.39
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 385,
        "framework": "InfiniLM",
        "ttft": 665.38
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 11785,
        "framework": "InfiniLM",
        "ttft": 347.55
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 11801,
        "framework": "InfiniLM",
        "ttft": 347.09
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 11739,
        "framework": "InfiniLM",
        "ttft": 348.92
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 366,
        "framework": "InfiniLM",
        "ttft": 349.47
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 365,
        "framework": "InfiniLM",
        "ttft": 350.88
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 367,
        "framework": "InfiniLM",
        "ttft": 348.6
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 10174,
        "framework": "InfiniLM",
        "ttft": 100.65
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 10120,
        "framework": "InfiniLM",
        "ttft": 101.18
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 10253,
        "framework": "InfiniLM",
        "ttft": 99.87
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 13157,
        "framework": "InfiniLM",
        "ttft": 1245.23
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 13152,
        "framework": "InfiniLM",
        "ttft": 1245.77
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 13151,
        "framework": "InfiniLM",
        "ttft": 1245.88
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5664,
        "framework": "InfiniLM",
        "ttft": 90.4
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5658,
        "framework": "InfiniLM",
        "ttft": 90.49
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5617,
        "framework": "InfiniLM",
        "ttft": 91.15
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 11984,
        "framework": "InfiniLM",
        "ttft": 341.8
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 11999,
        "framework": "InfiniLM",
        "ttft": 341.36
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 11964,
        "framework": "InfiniLM",
        "ttft": 342.37
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 8323,
        "framework": "InfiniLM",
        "ttft": 246.07
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 8301,
        "framework": "InfiniLM",
        "ttft": 246.73
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 8362,
        "framework": "InfiniLM",
        "ttft": 244.92
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 13307,
        "framework": "InfiniLM",
        "ttft": 1231.24
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 13324,
        "framework": "InfiniLM",
        "ttft": 1229.66
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 13302,
        "framework": "InfiniLM",
        "ttft": 1231.72
      }
    ],
    "decode": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 34,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 34,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 33,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 34,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 34,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 33,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 33,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 33,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 33,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 85,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 84,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 197,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 197,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 195,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 196,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 197,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 195,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 238,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 238,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 236,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 238,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 238,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 234,
        "framework": "InfiniLM"
      }
    ]
  },
  "metax": {
    "prefill": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 1442,
        "framework": "InfiniLM",
        "ttft": 22.18
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 1429,
        "framework": "InfiniLM",
        "ttft": 22.39
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1252,
        "framework": "InfiniLM",
        "ttft": 25.55
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 6883,
        "framework": "InfiniLM",
        "ttft": 37.19
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 6813,
        "framework": "InfiniLM",
        "ttft": 37.57
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 6469,
        "framework": "InfiniLM",
        "ttft": 39.57
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 10262,
        "framework": "InfiniLM",
        "ttft": 399.13
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 10260,
        "framework": "InfiniLM",
        "ttft": 399.22
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 10189,
        "framework": "InfiniLM",
        "ttft": 401.99
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 2980,
        "framework": "InfiniLM",
        "ttft": 42.95
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 2957,
        "framework": "InfiniLM",
        "ttft": 43.28
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 2807,
        "framework": "InfiniLM",
        "ttft": 45.59
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 8818,
        "framework": "InfiniLM",
        "ttft": 116.11
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 8804,
        "framework": "InfiniLM",
        "ttft": 116.3
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 8588,
        "framework": "InfiniLM",
        "ttft": 119.23
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 9515,
        "framework": "InfiniLM",
        "ttft": 1721.8
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 9508,
        "framework": "InfiniLM",
        "ttft": 1723.09
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 9484,
        "framework": "InfiniLM",
        "ttft": 1727.52
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 3720,
        "framework": "InfiniLM",
        "ttft": 137.62
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 3709,
        "framework": "InfiniLM",
        "ttft": 138.04
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 3658,
        "framework": "InfiniLM",
        "ttft": 139.96
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 9350,
        "framework": "InfiniLM",
        "ttft": 438.05
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 9341,
        "framework": "InfiniLM",
        "ttft": 438.47
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 9295,
        "framework": "InfiniLM",
        "ttft": 440.65
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 4058,
        "framework": "InfiniLM",
        "ttft": 504.57
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 4063,
        "framework": "InfiniLM",
        "ttft": 504.84
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 4043,
        "framework": "InfiniLM",
        "ttft": 504.39
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 8740,
        "framework": "InfiniLM",
        "ttft": 1874.57
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 8709,
        "framework": "InfiniLM",
        "ttft": 1881.1
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 8705,
        "framework": "InfiniLM",
        "ttft": 1881.93
      }
    ],
    "decode": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 66,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 67,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 235,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 234,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 227,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 235,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 234,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 227,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 217,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 221,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 219,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 883,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 873,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 852,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 876,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 869,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 850,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 2959,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 2856,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 2595,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 2948,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 2825,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 2546,
        "framework": "InfiniLM"
      }
    ]
  },
  "mthreads": {
    "prefill": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 967,
        "framework": "InfiniLM",
        "ttft": 33.1
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 972,
        "framework": "InfiniLM",
        "ttft": 32.93
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1075,
        "framework": "InfiniLM",
        "ttft": 29.77
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 4493,
        "framework": "InfiniLM",
        "ttft": 56.97
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 4690,
        "framework": "InfiniLM",
        "ttft": 54.59
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 4314,
        "framework": "InfiniLM",
        "ttft": 59.34
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5227,
        "framework": "InfiniLM",
        "ttft": 783.57
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5222,
        "framework": "InfiniLM",
        "ttft": 784.32
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5192,
        "framework": "InfiniLM",
        "ttft": 788.96
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 2720,
        "framework": "InfiniLM",
        "ttft": 47.06
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 2946,
        "framework": "InfiniLM",
        "ttft": 43.46
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 2929,
        "framework": "InfiniLM",
        "ttft": 43.7
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 6771,
        "framework": "InfiniLM",
        "ttft": 151.22
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 6899,
        "framework": "InfiniLM",
        "ttft": 148.43
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 6862,
        "framework": "InfiniLM",
        "ttft": 149.22
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5333,
        "framework": "InfiniLM",
        "ttft": 3072.29
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5353,
        "framework": "InfiniLM",
        "ttft": 3060.53
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5376,
        "framework": "InfiniLM",
        "ttft": 3047.59
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5732,
        "framework": "InfiniLM",
        "ttft": 89.32
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5846,
        "framework": "InfiniLM",
        "ttft": 87.58
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5976,
        "framework": "InfiniLM",
        "ttft": 85.68
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 7118,
        "framework": "InfiniLM",
        "ttft": 575.42
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 7117,
        "framework": "InfiniLM",
        "ttft": 575.51
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 7169,
        "framework": "InfiniLM",
        "ttft": 571.39
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 7212,
        "framework": "InfiniLM",
        "ttft": 283.97
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 7258,
        "framework": "InfiniLM",
        "ttft": 282.17
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 7245,
        "framework": "InfiniLM",
        "ttft": 282.66
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 7100,
        "framework": "InfiniLM",
        "ttft": 2307.55
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 7129,
        "framework": "InfiniLM",
        "ttft": 2298.1
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 7146,
        "framework": "InfiniLM",
        "ttft": 2292.73
      }
    ],
    "decode": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 34,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 37,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 38,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 34,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 38,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 38,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 37,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 36,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 35,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 113,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 117,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 117,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 115,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 119,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 117,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 116,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 115,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 111,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 534,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 588,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 568,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 526,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 602,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 564,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 1722,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 1806,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1633,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 1694,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 1831,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 1619,
        "framework": "InfiniLM"
      }
    ]
  },
  "nvidia": {
    "prefill": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 580,
        "framework": "InfiniLM",
        "ttft": 55.15
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 555,
        "framework": "InfiniLM",
        "ttft": 57.69
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 549,
        "framework": "InfiniLM",
        "ttft": 58.24
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 3831,
        "framework": "InfiniLM",
        "ttft": 66.83
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 4129,
        "framework": "InfiniLM",
        "ttft": 62
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 4099,
        "framework": "InfiniLM",
        "ttft": 62.45
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 7295,
        "framework": "InfiniLM",
        "ttft": 561.51
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 7308,
        "framework": "InfiniLM",
        "ttft": 560.5
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 7343,
        "framework": "InfiniLM",
        "ttft": 557.84
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5090,
        "framework": "InfiniLM",
        "ttft": 25.15
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5205,
        "framework": "InfiniLM",
        "ttft": 24.59
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5303,
        "framework": "InfiniLM",
        "ttft": 24.14
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 11748,
        "framework": "InfiniLM",
        "ttft": 87.16
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 11792,
        "framework": "InfiniLM",
        "ttft": 86.84
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 11906,
        "framework": "InfiniLM",
        "ttft": 86.01
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 12821,
        "framework": "InfiniLM",
        "ttft": 1277.86
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 12860,
        "framework": "InfiniLM",
        "ttft": 1274.01
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 12891,
        "framework": "InfiniLM",
        "ttft": 1270.99
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 5120,
        "framework": "InfiniLM",
        "ttft": 99.99
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 5136,
        "framework": "InfiniLM",
        "ttft": 99.68
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 5129,
        "framework": "InfiniLM",
        "ttft": 99.82
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 7584,
        "framework": "InfiniLM",
        "ttft": 540.08
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 7562,
        "framework": "InfiniLM",
        "ttft": 541.64
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 7533,
        "framework": "InfiniLM",
        "ttft": 543.76
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 8113,
        "framework": "InfiniLM",
        "ttft": 252.44
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 8141,
        "framework": "InfiniLM",
        "ttft": 251.56
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 8151,
        "framework": "InfiniLM",
        "ttft": 251.25
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 12804,
        "framework": "InfiniLM",
        "ttft": 1279.57
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 12838,
        "framework": "InfiniLM",
        "ttft": 1276.17
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 12857,
        "framework": "InfiniLM",
        "ttft": 1274.32
      }
    ],
    "decode": [
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 40,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 39,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 39,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 40,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 39,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 39,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 39,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 39,
        "framework": "InfiniLM"
      },
      {
        "batch": 1,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 38,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 309,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 304,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 295,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 308,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 300,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 293,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 256,
        "model": "9g8b",
        "tps": 292,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 290,
        "framework": "InfiniLM"
      },
      {
        "batch": 4,
        "inLen": 4096,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 288,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 579,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 554,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 475,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 565,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 541,
        "framework": "InfiniLM"
      },
      {
        "batch": 16,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 465,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 256,
        "model": "9g8b",
        "tps": 3698,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 3582,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 32,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 3243,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 256,
        "model": "9g8b",
        "tps": 3635,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 1024,
        "model": "9g8b",
        "tps": 3525,
        "framework": "InfiniLM"
      },
      {
        "batch": 64,
        "inLen": 256,
        "outLen": 4096,
        "model": "9g8b",
        "tps": 3196,
        "framework": "InfiniLM"
      }
    ]
  }
} as Record<string, { prefill: Array<Record<string, unknown>>; decode: Array<Record<string, unknown>> }>

export const BENCHMARK_DATA_META = {
  "generatedAt": "2026-05-08T08:38:30.934Z",
  "operatorSources": [
    "data\\operator\\ascend_operator_20260430.csv",
    "data\\operator\\cambricon_operator_20260506.csv",
    "data\\operator\\hygon_operator_20260429.csv",
    "data\\operator\\iluvatar_operator_20260430.csv",
    "data\\operator\\metax_operator_20260429.csv",
    "data\\operator\\moore_operator_20260429.csv",
    "data\\operator\\ops_data.csv"
  ],
  "inferSources": [
    "data\\infer\\ali_infer_20260429.csv",
    "data\\infer\\cambricon_infer_20260428.csv",
    "data\\infer\\hygon_infer_20260429.csv",
    "data\\infer\\metax_infer_20260428.csv",
    "data\\infer\\mthreads_infer_20260429.csv",
    "data\\infer\\nvidia_infer_20260506.csv"
  ]
}
