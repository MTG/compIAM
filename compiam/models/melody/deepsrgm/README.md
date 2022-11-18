The model summaries of the LSTM and GRU based models, along with details of output shapes after every layer, and number of trainable parameters, are as follows:


**LSTM-based model:**

```

=====================================================================
                  Kernel Shape     Output Shape     Params  Mult-Adds
Layer                                                                
0_embeddings        [128, 209]  [40, 5000, 128]    26.752k    26.752k
1_rnn                        -  [40, 5000, 768]  2.758656M  2.752512M
2_attention_layer     [1, 768]        [40, 768]     5.768k      768.0
3_fc1               [768, 384]        [40, 384]   295.296k   294.912k
4_relu                       -        [40, 384]          -          -
5_dropout                    -        [40, 384]          -          -
6_fc2                [384, 10]         [40, 10]      3.85k      3.84k
---------------------------------------------------------------------
                         Totals
Total params          3.090322M
Trainable params      3.090322M
Non-trainable params        0.0
Mult-Adds             3.078784M
=====================================================================

```
**GRU-based model:**

```

=====================================================================
                  Kernel Shape     Output Shape     Params  Mult-Adds
Layer                                                                
0_embeddings        [128, 209]  [40, 5000, 128]    26.752k    26.752k
1_rnn                        -  [40, 5000, 768]  2.068992M  2.064384M
2_attention_layer     [1, 768]        [40, 768]     5.768k      768.0
3_fc1               [768, 384]        [40, 384]   295.296k   294.912k
4_relu                       -        [40, 384]          -          -
5_dropout                    -        [40, 384]          -          -
6_fc2                [384, 10]         [40, 10]      3.85k      3.84k
---------------------------------------------------------------------
                         Totals
Total params          2.400658M
Trainable params      2.400658M
Non-trainable params        0.0
Mult-Adds             2.390656M
=====================================================================

```

Thus, switching from LSTM to GRU leads to a very significant reduction in number of trainable parameters, which leads to faster training. We also empirically observe that the GRU based model learns faster. 