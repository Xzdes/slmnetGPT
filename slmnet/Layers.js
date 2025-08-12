/**
 * @file slmnet/Layers.js
 * @description slmnetGPT v1.0 - Набор строительных блоков (слоев) для нейросетей.
 */

import { Tensor } from './Tensor.js';

/**
 * Базовый класс для всех слоев.
 */
class Layer {
    constructor() {}

    forward(inputs) {
        throw new Error("Метод forward() должен быть реализован в дочернем классе.");
    }

    /**
     * Собирает обучаемые параметры (тензоры с requires_grad=true).
     * Этот базовый метод ищет параметры прямо в свойствах объекта.
     */
    parameters() {
        const params = [];
        for (const key in this) {
            const prop = this[key];
            if (prop instanceof Tensor && prop.requires_grad) {
                params.push(prop);
            }
            // Эта часть нужна для вложенных слоев (не используется в текущей версии, но полезна для будущего)
            else if (prop instanceof Layer) {
                params.push(...prop.parameters());
            }
        }
        return params;
    }

    // Позволяет вызывать слой как функцию: model(inputs)
    __call__(inputs) {
        return this.forward(inputs);
    }
}

/**
 * Полносвязный слой (y = xW + b).
 */
class DenseLayer extends Layer {
    constructor(in_features, out_features) {
        super();
        // Инициализация весов по методу He для ReLU
        const limit = Math.sqrt(2 / in_features); 
        this.weights = new Tensor(
            Float32Array.from({ length: in_features * out_features }, () => (Math.random() * 2 - 1) * limit),
            [in_features, out_features],
            true // Этот тензор нужно обучать
        );
        this.bias = Tensor.zeros([1, out_features], true); // Этот тензор тоже нужно обучать
    }

    forward(inputs) {
        // inputs.dot(this.weights) вернет тензор с _ctx
        const matmul_result = inputs.dot(this.weights);
        // matmul_result.add(this.bias) тоже вернет тензор с _ctx
        return matmul_result.add(this.bias);
    }
}

/**
 * Контейнер для последовательного соединения слоев.
 */
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

    // --- НАЧАЛО ИСПРАВЛЕНИЯ ---
    /**
     * ПРАВИЛЬНЫЙ метод для сбора параметров из вложенных слоев.
     * Он переопределяет базовый метод Layer.parameters().
     */
    parameters() {
        const params = [];
        // Проходим по каждому слою в массиве this.layers
        for (const layer of this.layers) {
            // Рекурсивно вызываем parameters() для каждого слоя и добавляем его параметры в общий список
            params.push(...layer.parameters());
        }
        return params;
    }
    // --- КОНЕЦ ИСПРАВЛЕНИЯ ---
}

/**
 * Слой активации ReLU.
 */
class ReLU extends Layer {
    forward(inputs) {
        return inputs.relu();
    }
    // У этого слоя нет обучаемых параметров, поэтому его метод parameters() (унаследованный) вернет [].
}

/**
 * Слой активации Sigmoid.
 */
class Sigmoid extends Layer {
    forward(inputs) {
        return inputs.sigmoid();
    }
    // У этого слоя тоже нет обучаемых параметров.
}

export { Layer, DenseLayer, Sequential, ReLU, Sigmoid };