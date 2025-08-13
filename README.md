# slmnetGPT

*A lightweight, zero-dependency neural network framework in vanilla JavaScript for building and training simple models directly in the browser.*

slmnetGPT is a proof-of-concept project demonstrating that the fundamental components of a deep learning framework can be built from scratch in vanilla JavaScript. It is designed to be educational, transparent, and a launchpad for experimenting with client-side AI. The included example is a simple chatbot that lives, learns, and evolves entirely within the user's browser, persisting its knowledge in `localStorage`.

---

## The Vision: Smarter, Faster, Hybrid AI in Your Browser

This project was born from a simple but powerful idea: not all AI processing needs to happen on a powerful server. Many user interactions are simple, repetitive, or can be significantly enhanced by a small, fast, local model. slmnetGPT serves as a tangible exploration of this "Hybrid AI" architecture.

#### The Problem with Purely Server-Side LLMs
- **Latency:** Every single interaction, even a simple "hello," requires a round-trip to a server, creating noticeable delays.
- **Cost:** Server-side models, especially large language models (LLMs), are resource-intensive and can be expensive to operate at scale. API calls to services like OpenAI have associated costs.
- **Generic Interactions:** Without complex session management, interactions can feel stateless and repetitive, with the LLM often re-introducing itself.

#### The slmnetGPT Solution: A Hybrid Approach
Imagine a small, nimble model running in the browser that acts as a "frontend" for a larger, more powerful model on the server. This is the core philosophy.

**1. Instantaneous Query Handling & Optimization**
A user writes many simple requests like "Thank you!", "How are you?", or "Okay." Why send these to a massive server-side LLM? The client-side model can handle these instantly, providing a snappy, responsive feel and saving server resources and API credits for more complex queries.

**2. The Context Manager for Seamless Dialogues**
This is where the hybrid model shines. Consider this flow:
*   **User:** "Hi! Tell me about hybrid AI."
*   **slmnetGPT (in browser, instantly):** "Hello there! Of course, I'll be happy to tell you."
*   **"Behind the Scenes":** At the same moment, the small model sends a structured request to the main LLM on the server:
    ```json
    {
      "main_request": "Tell me about hybrid AI.",
      "context_package": {
        "status": "dialog_started",
        "user_greeting": "Hi!",
        "bot_reply": "Hello there! Of course, I'll be happy to tell you.",
        "tone": "friendly"
      }
    }
    ```
The large LLM receives this rich context and understands the entire picture. It doesn't need to repeat greetings; it can continue the dialogue organically, as if it were part of the conversation from the very beginning. This solves key problems:
*   **No Perceptible Switching:** The user experiences a single, seamless conversation.
*   **Resource Efficiency:** The LLM doesn't waste energy on redundant analysis; it gets straight to the point.
*   **Deep Context:** The dialogue becomes more natural and personalized.

**3. Instant Feedback for a Better User Experience**
Even for complex queries, the local model can provide immediate feedback.
*   **User:** "Can you write me a script to analyze this data...?"
*   **slmnetGPT (in browser, instantly):** "Certainly! I'm working on that script for you now..."
*   **Server LLM:** While the user reads this, the powerful server-side model is already generating the detailed response.

For the user, the interaction feels instantaneous and more engaging, eliminating the "dead air" of waiting for a server to respond.

---

## For Whom is This Project?

*   **For JavaScript Developers** curious about the inner workings of machine learning.
*   **For Students and Educators** looking for a simple, readable implementation of core deep learning concepts (Tensors, Autograd, Layers, Optimizers).
*   **For Hobbyists** who want to experiment with client-side AI in their web projects.
*   **For Frontend Developers** building interfaces for large LLMs and looking for innovative ways to improve UX and optimize backend resource usage.

---

## What It Can Do: Key Features

### The Core Framework (`slmnet/`)
*   **`Tensor.js`**: A fundamental N-dimensional data container. It tracks computational history and holds gradients, forming the backbone of the automatic differentiation (autograd) system.
*   **`Ops.js`**: A library of mathematical operations (`add`, `mul`, `dot`, `relu`, `sigmoid`, etc.) that operate on Tensors. Each operation is "aware" of the computation graph and knows how to calculate its gradients during backpropagation.
*   **`Layers.js`**: High-level, reusable "building blocks" for neural networks, such as `DenseLayer` (fully-connected), `ReLU`, `Sigmoid`, and a `Sequential` container to stack them.
*   **`Optimizers.js`**: Algorithms like Stochastic Gradient Descent (`SGD`) that use the computed gradients to update the model's parameters (weights and biases).

### The Example Application (`index.html`)
*   **Fully In-Browser Chatbot**: A demonstration of the framework in action.
*   **Real-Time Learning**: You can teach the bot. If it makes a mistake, provide the correct answer, and the model will retrain itself on its accumulated knowledge.
*   **Dynamic Model Adaptation**: When a new word or a new type of response is introduced, the model's architecture (input/output layers) is dynamically rebuilt, and the learned knowledge from the old model is transferred to the new, larger one.
*   **Persistence**: The bot's "brain"—its learned weights, vocabulary, and conversational memory—is saved to the browser's `localStorage`, so it remembers what it learned across sessions.

---

## Project Structure

```
slmnetGPT/
├── slmnet/
│   ├── Layers.js        # Neural network layers (Dense, ReLU, etc.)
│   ├── Ops.js           # Mathematical operations for Tensors (add, dot, etc.)
│   ├── Optimizers.js    # Optimization algorithms (SGD)
│   ├── slmnet.js        # Main export file for the framework
│   └── Tensor.js        # The core data structure with autograd capabilities
└── index.html           # The runnable chatbot application and its UI
```

---

## How to Deploy and Run

This project is designed for simplicity. There are **no build steps or external dependencies**.

1.  **Clone or Download:** Get the files onto your local machine.
2.  **Start a Local Server:** Because the project uses ES Modules (`import`/`export`), you cannot simply open `index.html` from the filesystem. You need to serve it via a local web server.
    There are many ways to do this:
    *   **Using a Code Editor Extension:** The easiest way is to use an extension like **"Live Server"** for Visual Studio Code. It starts a server with a single click.
    *   **Using Node.js:** If you have Node.js installed, you can use the popular `http-server` package.
        ```bash
        # Install the server (once)
        npm install -g http-server
        # Run it in the project folder
        http-server .
        ```
    Any other static web server of your choice will also work.
3.  **Open in Browser:** Navigate to the address provided by your local server (usually something like `http://localhost:8080` or `http://127.0.0.1:5500`). The chatbot interface should load.

---

## How to Configure and Customize

All the logic for the chatbot example is contained within the `<script type="module">` tag in `index.html`.

*   **Changing the Model Architecture**:
    Locate the `createModel` function. You can easily add more layers, change the number of neurons in the hidden layer, or experiment with different activation functions.
    ```javascript
    function createModel(vocabSize, responseCount) {
         return new slmnet.layers.Sequential([
            new slmnet.layers.Dense(vocabSize, 32), // Input -> 32 neurons
            new slmnet.layers.ReLU(),
            new slmnet.layers.Dense(32, 64), // Add another hidden layer
            new slmnet.layers.ReLU(),
            new slmnet.layers.Dense(64, responseCount), // 64 -> Output
            new slmnet.layers.Sigmoid()
        ]);
    }
    ```

*   **Adjusting Training Parameters**:
    In the `runTrainingSession` function, you can modify the learning rate and the number of training epochs.
    ```javascript
    const optimizer = new slmnet.optimizers.SGD(model.parameters(), 0.05); // Adjust learning rate (e.g., to 0.01)
    const epochs = isInitial ? 100 : 50; // Adjust number of training cycles
    ```

*   **Defining the Initial "Personality"**:
    The bot starts with a base knowledge defined in the `memory` array inside the `main` function. You can change or expand this to give the bot a different starting personality before any user training.
    ```javascript
    memory = [
        { input: "what is your purpose", output: "i am a client-side neural network" },
        { input: "what can you do", output: "i can learn from our conversation" },
        // ... add more initial knowledge pairs here
    ];