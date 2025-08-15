/**
 * @file slmnet/Tokenizer.js
 * @description slmnetGPT v2.0 - Простой символьный токенизатор.
 */

class CharacterTokenizer {
    /**
     * @param {string} text - Полный текст для построения словаря.
     */
    constructor(text) {
        const char_set = new Set(text.split(''));
        this.vocab = [...char_set].sort();
        this.vocab_size = this.vocab.length;

        this.char_to_id = new Map();
        this.id_to_char = new Map();

        this.vocab.forEach((char, i) => {
            this.char_to_id.set(char, i);
            this.id_to_char.set(i, char);
        });
    }

    /**
     * Преобразует строку в массив ID токенов.
     * @param {string} text 
     * @returns {number[]}
     */
    encode(text) {
        const encoded = [];
        for (const char of text) {
            if (this.char_to_id.has(char)) {
                encoded.push(this.char_to_id.get(char));
            }
        }
        return encoded;
    }

    /**
     * Преобразует массив ID токенов обратно в строку.
     * @param {number[]} ids 
     * @returns {string}
     */
    decode(ids) {
        let text = '';
        for (const id of ids) {
            if (this.id_to_char.has(id)) {
                text += this.id_to_char.get(id);
            }
        }
        return text;
    }
}

export { CharacterTokenizer };