# slmnetGPT v2.0: A Client-Side Generative Transformer

**slmnetGPT** is a lightweight, zero-dependency neural network framework and a proof-of-concept Generative Pre-trained Transformer (GPT) model, written entirely in vanilla JavaScript. It demonstrates that the core principles of modern deep learning architectures can be implemented and trained from scratch, directly in the browser.

The project successfully builds and trains a character-level transformer model that learns to generate text in the style of a given sample, persisting its "brain" in `localStorage`. It serves as a powerful educational tool for understanding the inner workings of technologies like GPT and as a tangible demonstration of client-side AI.


---

## The Journey: From a Simple Idea to a Working GPT

This project evolved significantly from its initial concept. The primary goal was to explore the feasibility of building a GPT-like model in the browser without any external libraries, which required building an entire deep learning framework from the ground up.

The development process was a meticulous, step-by-step debug cycle that addressed fundamental challenges in training deep neural networks:

1.  **Exploding Gradients:** Early training attempts were highly unstable, with the loss function quickly becoming `NaN`. This was diagnosed as exploding gradients, a common problem in deep networks. The solution was to implement **gradient clipping**, a crucial technique that rescales gradients if their norm exceeds a certain threshold, ensuring training stability.
2.  **Unstable Learning & Stagnation:** Even with gradient clipping, the model struggled to converge. The loss would oscillate and fail to decrease consistently. This was traced back to an incomplete implementation of the **Layer Normalization** backward pass, a critical component for stabilizing transformers. A mathematically correct, full backward pass for `LayerNorm` was implemented, which was the key to unlocking stable training.
3.  **Inefficient Optimization:** The initial Stochastic Gradient Descent (SGD) optimizer was too simplistic for the complex loss landscape of a transformer. The introduction of the **Adam optimizer**, with its adaptive learning rates and momentum, provided the robustness needed for the model to navigate this landscape and find a deep minimum.
4.  **Low-Quality Generation:** Early models produced repetitive and nonsensical text. This was overcome by **scaling the model** (increasing embedding dimensions, adding more transformer layers) and introducing **Temperature Sampling** during generation. This allowed the model to be more creative and produce text that was coherent and stylistically similar to the training data.

This iterative process of diagnosing and solving core deep learning problems is what makes this project a powerful and realistic demonstration of building AI systems.

---

## Core Features & What It Can Do

### The `slmnet` Framework
*   **`Tensor.js`**: A multi-dimensional data container that forms the backbone of the framework. Each Tensor can track its computational history, enabling automatic differentiation (autograd).
*   **`Ops.js`**: A library of "smart" mathematical operations (`dot`, `add`, `softmax`, etc.) that operate on Tensors. Each function builds a node in the computation graph and knows how to compute its own gradients during backpropagation.
*   **`Layers.js`**: High-level, object-oriented building blocks for neural networks. This includes not only basic layers like `DenseLayer` and `ReLU` but also the complex components of a transformer:
    *   `EmbeddingLayer`: Converts token IDs into dense vectors.
    *   `MultiHeadAttention`: The core mechanism allowing the model to weigh the importance of different tokens in a sequence.
    *   `LayerNorm`: A vital normalization layer with a complete, stable backward pass implementation.
    *   `TransformerBlock`: A complete decoder block combining multi-head attention, feed-forward networks, and residual connections.
*   **`Optimizers.js`**: Advanced optimization algorithms. The final version includes both `SGD` and the powerful `Adam` optimizer.
*   **`Losses.js`**: A dedicated `cross_entropy_loss` function, which is the standard for training language models.

### The `index.html` Application
*   **An End-to-End GPT Implementation**: A complete, working character-level language model.
*   **In-Browser Training**: The entire training process—from tokenizing the text to running thousands of backpropagation steps—happens live in the browser.
*   **Dynamic UI**: A simple interface allows you to provide training text, monitor the loss in real-time, and see detailed logs in the developer console.
*   **Text Generation**: Once trained, the model can generate new text from a given prompt, mimicking the style of the training data.
*   **Creative Control**: The generation process uses temperature sampling, allowing for more diverse and interesting outputs than simple greedy decoding.

---

## Project Structure

```
slmnetGPT/
├── slmnet/
│   ├── Layers.js        # Foundational and Transformer-specific layers
│   ├── Ops.js           # Mathematical operations and their gradients
│   ├── Optimizers.js    # SGD and Adam optimizers
│   ├── Losses.js        # Cross-entropy loss function
│   ├── Tokenizer.js     # A simple character-level tokenizer
│   ├── slmnet.js        # Main export file for the framework
│   └── Tensor.js        # The core data structure with autograd
└── index.html           # The runnable GPT application and UI
```

---

## How to Run

This project is designed for maximum simplicity. **There are no build steps or external dependencies.**

1.  **Clone or Download:** Get the project files onto your local machine.
2.  **Start a Local Server:** Because the project uses ES Modules (`import`/`export`), you cannot simply open `index.html` from the filesystem. You need to serve it via a local web server. The easiest way is with a code editor extension like **"Live Server"** for VS Code.
3.  **Open in Browser:** Navigate to the local address provided by your server (e.g., `http://localhost:8080`).

**Using the Application:**
1.  **Train:** Click the "Начать обучение" (Start Training) button. The process will take several minutes as the model is quite deep. You can monitor the progress in the log window and in the browser's developer console (F12).
2.  **Generate:** Once training is complete, type a starting prompt into the input box and click "Сгенерировать" (Generate). The model will generate new text based on your prompt.

---

## How It Works: A Look Under the Hood

The model in `index.html` is a decoder-only Transformer, the same fundamental architecture as the original GPT.

1.  **Tokenization:** The training text is broken down into a sequence of characters (tokens), each mapped to a unique integer ID.
2.  **Embedding:** Each token ID is converted into a high-dimensional vector via a trainable `EmbeddingLayer`. A separate `EmbeddingLayer` learns positional encodings to give the model a sense of sequence order.
3.  **Transformer Blocks:** The sequence of vectors is processed by a stack of `TransformerBlock`s. Each block performs two main computations:
    *   **Multi-Head Self-Attention:** Allows each token to look at and integrate information from all other tokens in the context. Causal masking is used to ensure that a token can only attend to previous tokens, not future ones.
    *   **Feed-Forward Network:** A simple neural network that processes each token's vector independently, adding computational depth.
4.  **Prediction:** After passing through all the blocks, a final `DenseLayer` acts as a prediction head. It converts the final processed vector for each token into a set of `logits`—a raw score for every possible next character in the vocabulary.
5.  **Loss Calculation:** The `softmax` function converts these logits into probabilities. The `cross_entropy_loss` function then compares the model's predicted probabilities with the actual next character in the text, calculating the error.
6.  **Backpropagation & Optimization:** The error is backpropagated through the entire network, from the loss function all the way back to the embedding layers. The `Adam` optimizer uses these gradients to update every single trainable parameter in the model, nudging it closer to making better predictions. This cycle is repeated thousands of times.

This project is a testament to what can be achieved with foundational principles and persistent debugging. It successfully demystifies the magic behind large language models, making their core concepts accessible and interactive.