/**
 * @file slmnet/Layers.js
 * @description slmnetGPT v2.0 - Набор строительных блоков (слоев) для нейросетей.
 */

import { Tensor } from './Tensor.js';
import { Ops } from './Ops.js';

class Layer {
    constructor() {}

    forward(inputs) {
        throw new Error("Метод forward() должен быть реализован в дочернем классе.");
    }
    
    parameters() {
        const params = [];
        for (const key in this) {
            const prop = this[key];
            if (prop instanceof Tensor && prop.requires_grad) {
                params.push(prop);
            }
            else if (prop instanceof Layer) {
                params.push(...prop.parameters());
            }
            else if (Array.isArray(prop)) {
                for(const item of prop) {
                    if (item instanceof Layer) {
                        params.push(...item.parameters());
                    }
                }
            }
        }
        return params;
    }

    __call__(inputs) {
        return this.forward(inputs);
    }
}

class DenseLayer extends Layer {
    constructor(in_features, out_features, use_bias = true) {
        super();
        const limit = Math.sqrt(2 / in_features); 
        this.weights = new Tensor(
            Float32Array.from({ length: in_features * out_features }, () => (Math.random() * 2 - 1) * limit),
            [in_features, out_features],
            true
        );
        this.use_bias = use_bias;
        this.bias = this.use_bias ? Tensor.zeros([1, out_features], true) : null;
    }

    forward(inputs) {
        const matmul_result = inputs.dot(this.weights);
        return this.use_bias ? matmul_result.add(this.bias) : matmul_result;
    }
}

class Sequential extends Layer {
    constructor(layers) {
        super();
        this.layers = layers;
    }
    
    forward(inputs) {
        let current_output = inputs;
        for (const layer of this.layers) {
            current_output = layer.forward(current_output);
        }
        return current_output;
    }
}

class ReLU extends Layer {
    forward(inputs) {
        return inputs.relu();
    }
}

class Sigmoid extends Layer {
    forward(inputs) {
        return inputs.sigmoid();
    }
}

class EmbeddingLayer extends Layer {
    constructor(vocab_size, embedding_dim) {
        super();
        this.embedding_dim = embedding_dim;
        this.weights = Tensor.random([vocab_size, embedding_dim], true);
    }

    forward(ids_tensor) {
        const [batch_size, seq_len] = ids_tensor.shape;
        const result_data = new Float32Array(batch_size * seq_len * this.embedding_dim);

        for (let i = 0; i < ids_tensor.size; i++) {
            const id = ids_tensor.data[i];
            const weight_offset = id * this.embedding_dim;
            const result_offset = i * this.embedding_dim;
            // Копируем вектор
            for (let j = 0; j < this.embedding_dim; j++) {
                result_data[result_offset + j] = this.weights.data[weight_offset + j];
            }
        }
        
        const result = new Tensor(result_data, [batch_size * seq_len, this.embedding_dim], this.weights.requires_grad);
        
        if (this.weights.requires_grad) {
            result._ctx = {
                inputs: [ids_tensor, this.weights],
                backward: (upstream_grad) => {
                    if (this.weights.requires_grad) {
                        for (let i = 0; i < ids_tensor.size; i++) {
                            const id = ids_tensor.data[i];
                            const weight_offset = id * this.embedding_dim;
                            const grad_offset = i * this.embedding_dim;
                            // Накапливаем градиенты
                            for (let j = 0; j < this.embedding_dim; j++) {
                                this.weights.grad.data[weight_offset + j] += upstream_grad.data[grad_offset + j];
                            }
                        }
                    }
                }
            };
        }
        return result;
    }
}

/**
 * Слой нормализации (Layer Normalization) с ПОЛНОЙ реализацией backward pass.
 */
class LayerNorm extends Layer {
    constructor(feature_dim, eps = 1e-5) {
        super();
        this.feature_dim = feature_dim;
        this.eps = eps;
        this.gamma = Tensor.ones([1, feature_dim], true); // scale
        this.beta = Tensor.zeros([1, feature_dim], true); // shift
    }

    forward(x) {
        const [rows, cols] = x.shape;
        const requires_grad = x.requires_grad || this.gamma.requires_grad || this.beta.requires_grad;

        const result_data = new Float32Array(x.size);
        // Сохраняем промежуточные значения для backward pass
        const mean = new Float32Array(rows);
        const variance = new Float32Array(rows);
        const x_normalized = new Float32Array(x.size);

        for(let i = 0; i < rows; i++) {
            const offset = i * cols;
            let sum = 0;
            for(let j = 0; j < cols; j++) sum += x.data[offset + j];
            mean[i] = sum / cols;

            let sum_sq_diff = 0;
            for(let j = 0; j < cols; j++) sum_sq_diff += Math.pow(x.data[offset + j] - mean[i], 2);
            variance[i] = sum_sq_diff / cols;

            const std_dev_inv = 1.0 / Math.sqrt(variance[i] + this.eps);
            for(let j = 0; j < cols; j++) {
                const normalized = (x.data[offset + j] - mean[i]) * std_dev_inv;
                x_normalized[offset + j] = normalized;
                result_data[offset + j] = normalized * this.gamma.data[j] + this.beta.data[j];
            }
        }
        
        const result = new Tensor(result_data, x.shape, requires_grad);
        
        if (requires_grad) {
            result._ctx = {
                inputs: [x, this.gamma, this.beta],
                backward: (upstream_grad) => {
                    const C = this.feature_dim;

                    for (let i = 0; i < rows; i++) {
                        const offset = i * C;
                        const std_inv = 1.0 / Math.sqrt(variance[i] + this.eps);
                        
                        let d_norm_sum = 0;
                        let d_norm_x_norm_sum = 0;

                        for (let j = 0; j < C; j++) {
                            const grad_idx = offset + j;
                            const upstream = upstream_grad.data[grad_idx];
                            const x_norm_val = x_normalized[grad_idx];
                            
                            // Градиент для gamma
                            if (this.gamma.requires_grad) {
                                this.gamma.grad.data[j] += upstream * x_norm_val;
                            }
                            // Градиент для beta
                            if (this.beta.requires_grad) {
                                this.beta.grad.data[j] += upstream;
                            }

                            // Промежуточные суммы для градиента x
                            const d_norm = this.gamma.data[j] * upstream;
                            d_norm_sum += d_norm;
                            d_norm_x_norm_sum += d_norm * x_norm_val;
                        }

                        // Градиент для x
                        if (x.requires_grad) {
                            for (let j = 0; j < C; j++) {
                                const grad_idx = offset + j;
                                const d_norm = this.gamma.data[j] * upstream_grad.data[grad_idx];
                                const x_norm_val = x_normalized[grad_idx];
                                
                                let dx = C * d_norm - d_norm_sum - x_norm_val * d_norm_x_norm_sum;
                                x.grad.data[grad_idx] += (std_inv / C) * dx;
                            }
                        }
                    }
                }
            };
        }
        return result;
    }
}

class MultiHeadAttention extends Layer {
    constructor(embedding_dim, num_heads) {
        super();
        if (embedding_dim % num_heads !== 0) throw new Error("embedding_dim должен делиться на num_heads.");
        this.embedding_dim = embedding_dim;
        this.num_heads = num_heads;
        this.head_dim = embedding_dim / num_heads;

        this.wq = new DenseLayer(embedding_dim, embedding_dim, false);
        this.wk = new DenseLayer(embedding_dim, embedding_dim, false);
        this.wv = new DenseLayer(embedding_dim, embedding_dim, false);
        this.wo = new DenseLayer(embedding_dim, embedding_dim, false);
    }

    forward(x) {
        const [seq_len, _] = x.shape;
        
        const Q = this.wq.forward(x);
        const K = this.wk.forward(x);
        const V = this.wv.forward(x);

        // Манипуляции с формами для разделения на головы
        const q_heads = Q.reshape([seq_len, this.num_heads, this.head_dim]);
        const k_heads = K.reshape([seq_len, this.num_heads, this.head_dim]);
        const v_heads = V.reshape([seq_len, this.num_heads, this.head_dim]);

        const attention_outputs = [];
        for (let h = 0; h < this.num_heads; h++) {
            // "Вырезаем" данные для одной головы. Это "глупые" операции без графа.
            const q = this._get_head(q_heads, h);
            const k = this._get_head(k_heads, h);
            const v = this._get_head(v_heads, h);

            let scores = q.dot(k.transpose());
            scores = scores.mul(new Tensor([1.0 / Math.sqrt(this.head_dim)]));

            for(let i = 0; i < seq_len; i++) {
                for(let j = i + 1; j < seq_len; j++) {
                    scores.data[i * seq_len + j] = -Infinity;
                }
            }
            
            const attention_weights = Ops.softmax(scores);
            attention_outputs.push(attention_weights.dot(v));
        }

        // ИСПРАВЛЕНИЕ: Правильно объединяем головы для корректного графа
        const combined = this._combine_heads(attention_outputs, seq_len);
        
        return this.wo.forward(combined);
    }
    
    _get_head(tensor_3d, head_index) {
        const [seq_len, num_heads, head_dim] = tensor_3d.shape;
        const head_data = new Float32Array(seq_len * head_dim);
        for (let i = 0; i < seq_len; i++) {
            for (let j = 0; j < head_dim; j++) {
                head_data[i * head_dim + j] = tensor_3d.data[i * num_heads * head_dim + head_index * head_dim + j];
            }
        }
        return new Tensor(head_data, [seq_len, head_dim]);
    }

    _combine_heads(heads_list, seq_len) {
        const combined_data = new Float32Array(seq_len * this.embedding_dim);
        for (let i = 0; i < seq_len; i++) {
            for (let h = 0; h < this.num_heads; h++) {
                const head_data = heads_list[h].data;
                for (let j = 0; j < this.head_dim; j++) {
                    combined_data[i * this.embedding_dim + h * this.head_dim + j] = head_data[i * this.head_dim + j];
                }
            }
        }
        
        const requires_grad = heads_list.some(h => h.requires_grad);
        const result = new Tensor(combined_data, [seq_len, this.embedding_dim], requires_grad);
        
        if (requires_grad) {
            result._ctx = {
                inputs: heads_list,
                backward: (upstream_grad) => {
                    for (let h = 0; h < this.num_heads; h++) {
                        const head_grad_tensor = heads_list[h].grad;
                        for (let i = 0; i < seq_len; i++) {
                            for (let j = 0; j < this.head_dim; j++) {
                                head_grad_tensor.data[i * this.head_dim + j] += upstream_grad.data[i * this.embedding_dim + h * this.head_dim + j];
                            }
                        }
                    }
                }
            };
        }
        return result;
    }
}

class FeedForward extends Layer {
    constructor(embedding_dim, hidden_dim) {
        super();
        this.net = new Sequential([
            new DenseLayer(embedding_dim, hidden_dim),
            new ReLU(),
            new DenseLayer(hidden_dim, embedding_dim)
        ]);
    }

    forward(x) {
        return this.net.forward(x);
    }
}

class TransformerBlock extends Layer {
    constructor(embedding_dim, num_heads) {
        super();
        this.attention = new MultiHeadAttention(embedding_dim, num_heads);
        this.ffn = new FeedForward(embedding_dim, embedding_dim * 4);
        this.ln1 = new LayerNorm(embedding_dim);
        this.ln2 = new LayerNorm(embedding_dim);
    }

    forward(x) {
        const norm_x1 = this.ln1.forward(x);
        const attention_output = this.attention.forward(norm_x1);
        const x1 = x.add(attention_output);

        const norm_x2 = this.ln2.forward(x1);
        const ffn_output = this.ffn.forward(norm_x2);
        const x2 = x1.add(ffn_output);
        
        return x2;
    }
}

export { Layer, DenseLayer, Sequential, ReLU, Sigmoid, EmbeddingLayer, LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock };