/**
 * @file slmnet/Losses.js
 * @description slmnetGPT v2.0 - Функции потерь.
 */

import { Tensor } from './Tensor.js';
import { Ops } from './Ops.js';

/**
 * Вычисляет ошибку перекрестной энтропии.
 * Более стабильная версия, которая объединяет LogSoftmax и NLLLoss.
 * @param {Tensor} logits - "Сырые" выходы модели. Форма [batch_size * seq_len, vocab_size].
 * @param {Tensor} targets - Целевые ID. Форма [batch_size * seq_len].
 * @returns {Tensor} - Скалярный тензор ошибки.
 */
function cross_entropy_loss(logits, targets) {
    const batch_size = targets.size;
    
    // 1. Применяем Softmax к логитам
    const probs = Ops.softmax(logits);

    // 2. Выбираем вероятности для правильных классов (целевых токенов)
    const correct_log_probs_data = new Float32Array(batch_size);
    for(let i = 0; i < batch_size; i++) {
        const target_id = targets.data[i];
        // Добавляем epsilon для стабильности, чтобы избежать log(0)
        const prob = Math.max(probs.data[i * probs.shape[1] + target_id], 1e-9); 
        correct_log_probs_data[i] = -Math.log(prob);
    }

    // 3. Создаем тензор из этих значений
    const neg_log_likelihood = new Tensor(correct_log_probs_data, [batch_size, 1], logits.requires_grad);
    
    // 4. Усредняем ошибку
    const loss = neg_log_likelihood.sum().mul(new Tensor([1.0 / batch_size]));

    // --- Создаем контекст для обратного прохода ---
    // Производная CrossEntropy+Softmax очень проста: (probs - Y) / N
    // где Y - one-hot вектор правильных ответов.
    if (logits.requires_grad) {
        loss._ctx = {
            inputs: [logits, targets],
            backward: (upstream_grad) => {
                if (logits.requires_grad) {
                    // Копируем вероятности, чтобы не изменять их
                    const grad_data = new Float32Array(probs.data);
                    
                    // Вычитаем 1 из вероятностей для правильных классов
                    for(let i = 0; i < batch_size; i++) {
                        const target_id = targets.data[i];
                        grad_data[i * probs.shape[1] + target_id] -= 1;
                    }
                    
                    // Усредняем градиент и домножаем на upstream_grad
                    for(let i = 0; i < grad_data.length; i++) {
                        logits.grad.data[i] += (grad_data[i] / batch_size) * upstream_grad.data[0];
                    }
                }
            }
        };
    }

    return loss;
}


export { cross_entropy_loss };