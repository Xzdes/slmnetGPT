/**
 * @file slmnet/Tensor.js
 * @description slmnetGPT v1.0 - Фундаментальный N-мерный контейнер данных.
 * Это "умный" узел графа, способный хранить градиент и запускать
 * обратное распространение ошибки.
 */

class Tensor {
    /**
     * @param {Array|Float32Array} data - Данные тензора.
     * @param {number[]} shape - Форма тензора.
     * @param {boolean} [requires_grad=false] - Флаг, указывающий, нужно ли вычислять градиент для этого тензора.
     * @param {object} [ctx=null] - Контекст (узел графа), который создал этот тензор.
     */
    constructor(data, shape, requires_grad = false, ctx = null) {
        this.data = data instanceof Float32Array ? data : new Float32Array(data);
        this.shape = shape;
        this.size = this.data.length;

        // --- Ключевые свойства для обучения ---

        // Флаг, указывающий, что для этого тензора нужно вычислять и накапливать градиент.
        // Обычно true для весов и смещений модели.
        this.requires_grad = requires_grad;
        
        // Градиент - это тоже тензор, такой же формы, изначально заполненный нулями.
        // Он создается только для тех тензоров, которые этого требуют.
        this.grad = this.requires_grad ? Tensor.zeros(this.shape) : null;
        
        // Контекст для построения графа вычислений в обратном порядке.
        // Хранит информацию об операции, которая создала этот тензор,
        // и функцию для вычисления градиента (backward).
        this._ctx = ctx;
    }

    /**
     * Запускает обратное распространение ошибки, начиная с этого тензора.
     * Обычно вызывается для тензора ошибки (loss), который является скаляром.
     */
    backward() {
        if (!this.requires_grad) {
            throw new Error("Нельзя вызывать backward() для тензора, у которого requires_grad=false.");
        }
        if (this.size !== 1) {
            throw new Error("Backward() можно вызывать только для скалярного тензора (например, loss-тензора с одним элементом).");
        }
        
        // --- Этап 1: Построение графа и его топологическая сортировка ---
        // Мы должны обойти все "предки" этого тензора, чтобы вычислить их градиенты
        // в правильном порядке: от конца к началу.
        const buildGraph = (tensor, visited, sortedGraph) => {
            if (visited.has(tensor)) return;
            visited.add(tensor);

            if (tensor._ctx) {
                tensor._ctx.inputs.forEach(input => buildGraph(input, visited, sortedGraph));
                sortedGraph.push(tensor);
            }
        };

        const visited = new Set();
        const sortedGraph = [];
        buildGraph(this, visited, sortedGraph);

        // --- Этап 2: Обратный проход по графу ---
        
        // Градиент самого себя (dLoss/dLoss) равен 1.
        this.grad = Tensor.ones(this.shape);
        
        // Идем по отсортированному графу в обратном порядке.
        for (let i = sortedGraph.length - 1; i >= 0; i--) {
            const tensor = sortedGraph[i];
            
            // Если у тензора есть контекст и функция backward, вызываем ее.
            // Эта функция вычислит градиенты для "родителей" (inputs) этого тензора
            // и добавит их к свойству .grad родителей.
            if (tensor._ctx && typeof tensor._ctx.backward === 'function') {
                tensor._ctx.backward(tensor.grad);
            }
        }
    }

    // --- Вспомогательные статические методы для удобства создания тензоров ---

    static from(arr, requires_grad = false) {
        const { flatData, inferredShape } = Tensor._inferShapeAndFlatten(arr);
        return new Tensor(flatData, inferredShape, requires_grad);
    }
    
    static zeros(shape, requires_grad = false) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Tensor(new Float32Array(size), shape, requires_grad);
    }

    static ones(shape, requires_grad = false) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size).fill(1);
        return new Tensor(data, shape, requires_grad);
    }

    static random(shape, requires_grad = false) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = Float32Array.from({ length: size }, () => Math.random() * 2 - 1);
        return new Tensor(data, shape, requires_grad);
    }

    static _inferShapeAndFlatten(arr) {
        const flatData = [];
        const inferredShape = [];
        let currentLevel = arr;
        while (Array.isArray(currentLevel)) {
            if (currentLevel.length === 0) break;
            inferredShape.push(currentLevel.length);
            if (Array.isArray(currentLevel[0])) {
                const firstLen = currentLevel[0].length;
                for (let i = 1; i < currentLevel.length; i++) {
                    if (!Array.isArray(currentLevel[i]) || currentLevel[i].length !== firstLen) {
                        throw new Error("Вложенные массивы имеют разную длину. Невозможно создать тензор.");
                    }
                }
            }
            currentLevel = currentLevel[0];
        }
        const flatten = (subArr) => {
            for (const el of subArr) {
                if (Array.isArray(el)) flatten(el);
                else flatData.push(el);
            }
        };
        flatten(arr);
        return { flatData, inferredShape };
    }

    // --- Методы для отладки ---
    print() {
        console.log('Tensor {');
        console.log('  shape:', this.shape);
        console.log('  data:', this.data);
        if (this.grad) {
            console.log('  grad:', this.grad.data);
        }
        console.log('}');
    }
}

export { Tensor };