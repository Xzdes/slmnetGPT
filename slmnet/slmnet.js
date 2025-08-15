/**
 * @file slmnet/slmnet.js
 * @description slmnetGPT v2.0 - Главный экспортный файл библиотеки.
 */

import { Tensor } from './Tensor.js';
import { Ops } from './Ops.js';
import { Layer, DenseLayer, Sequential, ReLU, Sigmoid, EmbeddingLayer, LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock } from './Layers.js';
// ИЗМЕНЕНИЕ: Импортируем Adam
import { SGD, Adam } from './Optimizers.js'; 
import { cross_entropy_loss } from './Losses.js';
import { CharacterTokenizer } from './Tokenizer.js';

const slmnet = {
    Tensor,
    Ops,
    layers: {
        Layer,
        Dense: DenseLayer,
        Sequential,
        ReLU,
        Sigmoid,
        Embedding: EmbeddingLayer,
        LayerNorm,
        MultiHeadAttention,
        FeedForward,
        TransformerBlock
    },
    optimizers: {
        SGD,
        // ИЗМЕНЕНИЕ: Добавляем Adam
        Adam 
    },
    losses: {
        cross_entropy_loss
    },
    tokenizers: {
        CharacterTokenizer
    }
};

export default slmnet;