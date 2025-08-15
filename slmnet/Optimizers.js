/**
 * @file slmnet/Optimizers.js
 * @description slmnetGPT v2.0 - Алгоритмы оптимизации для обучения моделей.
 */

class Optimizer {
    constructor(parameters, learning_rate) {
        if (parameters === undefined || learning_rate === undefined) {
            throw new Error("Необходимо предоставить 'parameters' и 'learning_rate'.");
        }
        this.parameters = parameters;
        this.lr = learning_rate;
    }

    step() {
        throw new Error("Метод step() должен быть реализован в дочернем классе.");
    }

    zero_grad() {
        for (const p of this.parameters) {
            if (p.grad) {
                p.grad.data.fill(0);
            }
        }
    }
}

class SGD extends Optimizer {
    constructor(parameters, learning_rate = 0.01) {
        super(parameters, learning_rate);
    }

    step() {
        for (const p of this.parameters) {
            if (p.grad) {
                for (let i = 0; i < p.data.length; i++) {
                    p.data[i] -= this.lr * p.grad.data[i];
                }
            }
        }
    }
}


/**
 * НОВЫЙ ОПТИМИЗАТОР: Adam (Adaptive Moment Estimation)
 */
class Adam extends Optimizer {
    constructor(parameters, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        super(parameters, learning_rate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0; // Счетчик шагов

        // Инициализируем буферы для каждого параметра
        this.m = new Map(); // Первый момент (скользящее среднее градиентов)
        this.v = new Map(); // Второй момент (скользящее среднее квадратов градиентов)

        for (const p of this.parameters) {
            this.m.set(p, new Float32Array(p.size).fill(0));
            this.v.set(p, new Float32Array(p.size).fill(0));
        }
    }

    step() {
        this.t++;

        for (const p of this.parameters) {
            if (p.grad) {
                const m_prev = this.m.get(p);
                const v_prev = this.v.get(p);
                const grad_data = p.grad.data;

                for (let i = 0; i < p.data.length; i++) {
                    const g = grad_data[i];

                    // Обновляем первый момент (m)
                    const m_t = this.beta1 * m_prev[i] + (1 - this.beta1) * g;
                    m_prev[i] = m_t;

                    // Обновляем второй момент (v)
                    const v_t = this.beta2 * v_prev[i] + (1 - this.beta2) * (g * g);
                    v_prev[i] = v_t;
                    
                    // Коррекция смещения (bias correction)
                    const m_hat = m_t / (1 - Math.pow(this.beta1, this.t));
                    const v_hat = v_t / (1 - Math.pow(this.beta2, this.t));

                    // Обновляем параметр
                    p.data[i] -= this.lr * m_hat / (Math.sqrt(v_hat) + this.epsilon);
                }
            }
        }
    }
}


// Экспортируем оба класса
export { SGD, Adam };