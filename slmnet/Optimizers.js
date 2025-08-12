/**
 * @file slmnet/Optimizers.js
 * @description slmnetGPT v1.0 - Алгоритмы оптимизации для обучения моделей.
 */

/**
 * Базовый класс для всех оптимизаторов.
 * В этой версии он не делает ничего, но служит для единства архитектуры.
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
        // Проходимся по всем обучаемым параметрам модели...
        for (const p of this.parameters) {
            // ...и заполняем их тензоры градиентов нулями.
            // Это необходимо делать перед каждым новым вычислением градиентов (backward pass),
            // так как градиенты по умолчанию накапливаются (суммируются).
            if (p.grad) {
                p.grad.data.fill(0);
            }
        }
    }
}


/**
 * Оптимизатор по алгоритму Стохастического Градиентного Спуска (SGD).
 */
class SGD extends Optimizer {
    /**
     * @param {Tensor[]} parameters - Массив обучаемых тензоров, обычно получается из model.parameters().
     * @param {number} learning_rate - Скорость обучения (learning rate).
     */
    constructor(parameters, learning_rate = 0.01) {
        super(parameters, learning_rate);
    }

    /**
     * Выполняет один шаг оптимизации.
     * Обновляет каждый параметр по формуле: param = param - learning_rate * param.grad
     */
    step() {
        // Проходимся по всем обучаемым параметрам модели (весам, смещениям).
        for (const p of this.parameters) {
            if (p.grad) {
                // Обновляем данные самого тензора, вычитая из них градиент,
                // умноженный на скорость обучения.
                for (let i = 0; i < p.data.length; i++) {
                    p.data[i] -= this.lr * p.grad.data[i];
                }
            }
        }
    }
}

// В будущем здесь можно будет добавить и другие оптимизаторы, например, Adam.
// class Adam extends Optimizer { ... }


// Экспортируем классы оптимизаторов
export { SGD };