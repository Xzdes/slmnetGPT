/**
 * @file slmnet/slmnet.js
 * @description slmnetGPT v1.0 - Главный экспортный файл библиотеки.
 * Предоставляет доступ ко всем основным компонентам фреймворка.
 */

import { Tensor } from './Tensor.js';
import { Ops } from './Ops.js';
// **ИСПРАВЛЕНИЕ: Добавляем Sigmoid в импорт**
import { Layer, DenseLayer, Sequential, ReLU, Sigmoid } from './Layers.js'; 
import { SGD } from './Optimizers.js';

const slmnet = {
    Tensor,
    Ops,
    layers: {
        Layer,
        Dense: DenseLayer,
        Sequential,
        ReLU,
        // **ИСПРАВЛЕНИЕ: Добавляем Sigmoid в экспортируемый объект**
        Sigmoid 
    },
    optimizers: {
        SGD
    },
};

export default slmnet;