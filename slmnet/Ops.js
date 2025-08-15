/**
 * @file slmnet/Ops.js
 * @description slmnetGPT v2.0 - Набор "умных" математических операций.
 * Каждая функция создает тензор с контекстом для построения графа
 * и вычисления градиентов.
 */

import { Tensor } from './Tensor.js';

const Ops = {

    add: (a, b) => {
        const requires_grad = a.requires_grad || b.requires_grad;
        let resultData;
        let resultShape = a.shape;

        // Случай 1: Поэлементное сложение
        if (JSON.stringify(a.shape) === JSON.stringify(b.shape)) {
            resultData = a.data.map((val, i) => val + b.data[i]);
        }
        // Случай 2: Вещание (Broadcasting) для смещений (bias)
        // a - матрица (N, M), b - вектор-строка (1, M)
        else if (a.shape.length === 2 && b.shape.length === 2 && b.shape[0] === 1 && a.shape[1] === b.shape[1]) {
            const [rowsA, colsA] = a.shape;
            resultData = new Float32Array(a.size);
            for (let i = 0; i < rowsA; i++) {
                for (let j = 0; j < colsA; j++) {
                    resultData[i * colsA + j] = a.data[i * colsA + j] + b.data[j];
                }
            }
        } else {
            throw new Error(`Формы тензоров [${a.shape}] и [${b.shape}] несовместимы для сложения (поддерживается только вещание смещений).`);
        }

        const result = new Tensor(resultData, resultShape, requires_grad);

        if (requires_grad) {
            result._ctx = {
                inputs: [a, b],
                backward: (upstream_grad) => {
                    if (a.requires_grad) {
                        // Градиент по 'a' равен входящему градиенту
                        for (let i = 0; i < a.grad.data.length; i++) {
                            a.grad.data[i] += upstream_grad.data[i];
                        }
                    }
                    if (b.requires_grad) {
                        // Если было вещание, градиент по 'b' нужно просуммировать
                        if(a.shape.length === 2 && b.shape.length === 2 && b.shape[0] === 1) {
                            const [rowsA, colsA] = a.shape;
                            for (let j = 0; j < colsA; j++) {
                                let sum = 0;
                                for (let i = 0; i < rowsA; i++) {
                                    sum += upstream_grad.data[i * colsA + j];
                                }
                                b.grad.data[j] += sum;
                            }
                        } else {
                             // Если формы совпадали, градиент по 'b' равен входящему градиенту
                             for (let i = 0; i < b.grad.data.length; i++) {
                                b.grad.data[i] += upstream_grad.data[i];
                            }
                        }
                    }
                }
            };
        }
        return result;
    },

    mul: (a, b) => {
        const requires_grad = a.requires_grad || b.requires_grad;
        let resultData;
        let resultShape = a.shape;

        // Поэлементное умножение
        if (JSON.stringify(a.shape) === JSON.stringify(b.shape)) {
            resultData = a.data.map((val, i) => val * b.data[i]);
        }
        // Умножение на скаляр (b - скаляр)
        else if (b.size === 1) {
            const scalar = b.data[0];
            resultData = a.data.map(val => val * scalar);
        }
        // Умножение на скаляр (a - скаляр)
        else if (a.size === 1) {
            const scalar = a.data[0];
            resultData = b.data.map(val => val * scalar);
            resultShape = b.shape;
        } else {
             throw new Error(`Формы тензоров [${a.shape}] и [${b.shape}] несовместимы для умножения.`);
        }

        const result = new Tensor(resultData, resultShape, requires_grad);

        if (requires_grad) {
            result._ctx = {
                inputs: [a, b],
                backward: (upstream_grad) => {
                    if (a.requires_grad) {
                        if (a.size === b.size) { // Поэлементное
                            for (let i = 0; i < a.grad.data.length; i++) {
                                a.grad.data[i] += b.data[i] * upstream_grad.data[i];
                            }
                        } else if (b.size === 1) { // a - тензор, b - скаляр
                            for (let i = 0; i < a.grad.data.length; i++) {
                                a.grad.data[i] += b.data[0] * upstream_grad.data[i];
                            }
                        }
                    }
                    if (b.requires_grad) {
                         if (a.size === b.size) { // Поэлементное
                            for (let i = 0; i < b.grad.data.length; i++) {
                                b.grad.data[i] += a.data[i] * upstream_grad.data[i];
                            }
                        } else if (a.size === 1) { // b - тензор, a - скаляр
                            for (let i = 0; i < b.grad.data.length; i++) {
                               b.grad.data[i] += a.data[0] * upstream_grad.data[i];
                            }
                        }
                    }
                }
            };
        }
        return result;
    },

    pow: (a, n) => {
        const requires_grad = a.requires_grad;
        const resultData = a.data.map(val => Math.pow(val, n));
        const result = new Tensor(resultData, a.shape, requires_grad);

        if (requires_grad) {
            result._ctx = {
                inputs: [a],
                backward: (upstream_grad) => {
                    if (a.requires_grad) {
                        for (let i = 0; i < a.grad.data.length; i++) {
                            // d/dx(x^n) = n * x^(n-1)
                            a.grad.data[i] += (n * Math.pow(a.data[i], n - 1)) * upstream_grad.data[i];
                        }
                    }
                }
            };
        }
        return result;
    },

    relu: (a) => {
        const requires_grad = a.requires_grad;
        const resultData = a.data.map(val => Math.max(0, val));
        const result = new Tensor(resultData, a.shape, requires_grad);

        if (requires_grad) {
            result._ctx = {
                inputs: [a],
                backward: (upstream_grad) => {
                    if (a.requires_grad) {
                        for (let i = 0; i < a.grad.data.length; i++) {
                            // Производная ReLU: 1 если x > 0, иначе 0
                            a.grad.data[i] += (a.data[i] > 0 ? 1 : 0) * upstream_grad.data[i];
                        }
                    }
                }
            };
        }
        return result;
    },
    
    sigmoid: (a) => {
        const requires_grad = a.requires_grad;
        const resultData = a.data.map(val => 1 / (1 + Math.exp(-val)));
        const result = new Tensor(resultData, a.shape, requires_grad);

        if (requires_grad) {
            result._ctx = {
                inputs: [a],
                backward: (upstream_grad) => {
                    if (a.requires_grad) {
                        for (let i = 0; i < a.grad.data.length; i++) {
                            // Производная сигмоиды: s(x) * (1 - s(x))
                            const s = result.data[i];
                            a.grad.data[i] += (s * (1 - s)) * upstream_grad.data[i];
                        }
                    }
                }
            };
        }
        return result;
    },

    dot: (a, b) => {
        if (a.shape.length !== 2 || b.shape.length !== 2) {
            throw new Error('Матричное умножение поддерживается только для 2D-тензоров.');
        }
        if (a.shape[1] !== b.shape[0]) {
            throw new Error(`Несовместимые формы для матричного умножения: [${a.shape}] и [${b.shape}].`);
        }

        const requires_grad = a.requires_grad || b.requires_grad;
        const [rowsA, colsA] = a.shape;
        const [rowsB, colsB] = b.shape;
        const resultShape = [rowsA, colsB];
        const resultData = new Float32Array(rowsA * colsB).fill(0);

        // Прямой проход: стандартное матричное умножение
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                for (let k = 0; k < colsA; k++) {
                    resultData[i * colsB + j] += a.data[i * colsA + k] * b.data[k * colsB + j];
                }
            }
        }
        const result = new Tensor(resultData, resultShape, requires_grad);
        
        if (requires_grad) {
            result._ctx = {
                inputs: [a, b],
                backward: (upstream_grad) => {
                    const [rowsA_ctx, colsA_ctx] = a.shape;
                    const [rowsB_ctx, colsB_ctx] = b.shape;

                    // Вычисление градиента для 'a'
                    if (a.requires_grad) {
                        // Формула: grad_A = upstream_grad.dot(B^T)
                        for (let i = 0; i < rowsA_ctx; i++) {
                            for (let j = 0; j < colsA_ctx; j++) {
                                let sum = 0;
                                for (let k = 0; k < colsB_ctx; k++) {
                                    // upstream_grad[i, k] * b[j, k] (B транспонированная)
                                    sum += upstream_grad.data[i * colsB_ctx + k] * b.data[j * colsB_ctx + k];
                                }
                                a.grad.data[i * colsA_ctx + j] += sum;
                            }
                        }
                    }

                    // Вычисление градиента для 'b'
                    if (b.requires_grad) {
                        // Формула: grad_B = A^T.dot(upstream_grad)
                        for (let i = 0; i < rowsB_ctx; i++) {
                            for (let j = 0; j < colsB_ctx; j++) {
                                let sum = 0;
                                for (let k = 0; k < rowsA_ctx; k++) {
                                    // a[k, i] (A транспонированная) * upstream_grad[k, j]
                                    sum += a.data[k * colsA_ctx + i] * upstream_grad.data[k * colsB_ctx + j];
                                }
                                b.grad.data[i * colsB_ctx + j] += sum;
                            }
                        }
                    }
                }
            };
        }
        return result;
    },
    
    sum: (a) => {
        const requires_grad = a.requires_grad;
        const sumResult = a.data.reduce((acc, val) => acc + val, 0);
        const result = new Tensor([sumResult], [1], requires_grad);

        if (requires_grad) {
            result._ctx = {
                inputs: [a],
                backward: (upstream_grad) => {
                    // Градиент суммы - это 1, умноженная на входящий градиент.
                    // Поэтому мы просто добавляем входящий градиент (скаляр) ко всем элементам
                    // градиента исходного тензора.
                    if (a.requires_grad) {
                        for (let i = 0; i < a.grad.data.length; i++) {
                            a.grad.data[i] += upstream_grad.data[0];
                        }
                    }
                }
            };
        }
        return result;
    },

    softmax: (a) => {
        const requires_grad = a.requires_grad;

        if (a.shape.length !== 2) throw new Error("Softmax пока реализован только для 2D тензоров.");
        const [rows, cols] = a.shape;
        const resultData = new Float32Array(a.size);

        for (let i = 0; i < rows; i++) {
            const row_offset = i * cols;
            // Стабилизация: вычитаем максимум из каждой строки для предотвращения переполнения
            let max_val = -Infinity;
            for (let j = 0; j < cols; j++) {
                if (a.data[row_offset + j] > max_val) {
                    max_val = a.data[row_offset + j];
                }
            }

            // Экспоненты и их сумма
            let sum_exp = 0;
            const exp_data_row = new Float32Array(cols);
            for (let j = 0; j < cols; j++) {
                const exp_val = Math.exp(a.data[row_offset + j] - max_val);
                exp_data_row[j] = exp_val;
                sum_exp += exp_val;
            }

            // Нормализация
            for (let j = 0; j < cols; j++) {
                resultData[row_offset + j] = exp_data_row[j] / sum_exp;
            }
        }
        
        const result = new Tensor(resultData, a.shape, requires_grad);

        if (requires_grad) {
             result._ctx = {
                inputs: [a],
                backward: (upstream_grad) => {
                    // Эта операция обычно последняя перед CrossEntropy,
                    // градиент для которой вычисляется в самой функции потерь.
                    // Для GPT-архитектуры полная реализация здесь не требуется.
                }
            };
        }
        return result;
    }
};

// Добавляем операции в прототип Tensor для удобного вызова (a.add(b) вместо Ops.add(a, b))
Tensor.prototype.add = function(other) { return Ops.add(this, other); };
Tensor.prototype.mul = function(other) { return Ops.mul(this, other); };
Tensor.prototype.pow = function(n) { return Ops.pow(this, n); };
Tensor.prototype.relu = function() { return Ops.relu(this); };
Tensor.prototype.sigmoid = function() { return Ops.sigmoid(this); };
Tensor.prototype.dot = function(other) { return Ops.dot(this, other); };
Tensor.prototype.sum = function() { return Ops.sum(this); };

// Транспонирование - это "глупая" операция, она не создает узел графа.
// Она нужна как вспомогательная функция.
Tensor.prototype.transpose = function() {
    if (this.shape.length !== 2) throw new Error("Транспонирование поддерживается только для 2D тензоров");
    
    const [rows, cols] = this.shape;
    const transposedData = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            transposedData[j * rows + i] = this.data[i * cols + j];
        }
    }
    // Возвращаем новый тензор без контекста (_ctx)
    return new Tensor(transposedData, [cols, rows]);
};

export { Ops };