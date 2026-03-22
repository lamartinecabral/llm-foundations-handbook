# 📚 LLM Foundations Handbook

Welcome to **LLM Foundations Handbook**. The AI landscape is evolving at breakneck speed, and making sense of the countless models, engines, and frameworks can be overwhelming.

This repository serves as a foundational "mental model" and reference guide for the modern Generative AI ecosystem. It strips away the hype and breaks down exactly how AI systems are built, run, and integrated today.

### 🎯 Who is this for?

- **Software Engineers:** Looking to transition into AI/LLM development and needing to understand the tech stack.
- **Product Managers:** Needing a clear, jargon-free understanding of what is technically possible (and required) to build AI features.
- **Tech Enthusiasts:** Anyone wanting to understand the difference between a parameter, an inference engine, and a vector database.

---

# 0\. The AI Stack

Here is an example of an AI system architecture stack. Each system may have more or fewer layers, but this stack is a common foundational standard.

| Layer       | Example           |
| ----------- | ----------------- |
| Application | VS Code           |
| Agent       | GitHub Copilot    |
| Provider    | AWS Bedrock       |
| Inference   | vLLM              |
| Model       | claude-sonnet-4.6 |

---

# 1\. Model

### 1.1. Parameters (Weights)

Think of parameters as the "brain cells" and "synapses" of an AI. When we say a model is "8 billion parameters" (8B) or "400 billion parameters" (400B), we are referring to the total count of these numbers.

- **What they are:** In a neural network, parameters consist of weights and biases. They are numerical values that determine the strength of the connection between different artificial neurons.
- **What they do:** During training, the model reads vast amounts of text and constantly adjusts these weights. If it guesses the next word correctly, the weights that led to that guess are strengthened. Over billions of adjustments, these weights end up encoding grammar, facts, reasoning pathways, and language structure.
- **Analogy:** If an LLM is a massive audio mixing board, the parameters are the billions of tiny sliders and knobs perfectly tuned to produce intelligent, coherent text.
- **Model Sizes & Families:** Because massive models require expensive hardware, developers release "families" of models. Smaller models (under 10B) are fast and can run locally on laptops or phones. Larger models (100B+) require powerful, multi-GPU servers but offer vastly superior logic, reasoning, and world knowledge. Examples include Meta's Llama 4 family (109B and 400B) and Alibaba's Qwen 2.5 family (scaling from 0.5B up to 72B).

---

### 1.2. Approaches (Dense vs. MoE)

This refers to how parameters are utilized when the model is processing text and generating a response.

| Architecture                 | Description                                                                                                                              | Open-Source Examples              | Pros & Cons                                                                                                                               |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **Dense Models**             | Every single parameter in the neural network is activated and used to process every single word (token).                                 | Llama 3.3 (70B), Qwen 2.5 (72B)   | **Pros:** Simpler architecture, easier and more stable to train.<br>**Cons:** Computationally expensive to run as they scale up.          |
| **Mixture of Experts (MoE)** | The model is divided into smaller sub-networks ("experts"). A router activates only the 1 or 2 experts best suited for a specific token. | Llama 4 Scout (109B), DeepSeek-V4 | **Pros:** Massive efficiency and faster text generation.<br>**Cons:** High VRAM usage. The entire model must still be loaded into memory. |

---

### 1.3. Capabilities

Modern LLMs have expanded drastically beyond just predicting the next word into specific, highly advanced skill sets.

| Capability                   | Description                                                                                                                    | Modern Examples            |
| :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :------------------------- |
| **Reasoning**                | The ability to logically deduce answers and follow "chain-of-thought" processes before outputting an answer.                   | DeepSeek-R1 / V4, GPT-5.4  |
| **Vision (Multimodality)**   | Processing images/video alongside text natively, allowing the model to describe images or solve visual puzzles.                | Llama 4, Claude 4.6 Sonnet |
| **Tools (Function Calling)** | Generating structured commands (like JSON) to trigger external tools (e.g., browsing the web, running code).                   | Llama 4, Hermes 3          |
| **Embedding**                | Turning text into high-dimensional mathematical vectors for semantic search and RAG workflows.                                 | Nomic-embed-text, BGE      |
| **Insertion (FIM)**          | "Fill-in-the-Middle" models trained to look at text before and after a cursor to generate the missing middle (used in coding). | Qwen2.5-Coder, StarCoder2  |

---

### 1.4. Chat Templates

A **Chat Template** is the structural "wrapper" that translates a list of conversational messages (System, User, Assistant) into a single long string of text that the model can actually understand.

- **The Blueprint of Interaction:** Raw LLMs don't naturally know where a user's prompt ends and their own response should begin. Templates use specific **control tokens** (like `<|im_start|>` or `[INST]`) to signal these boundaries.
- **Structure vs. Intelligence:** Advanced features—especially **Tool Calling** and **Object Generation (JSON)**—depend entirely on the precision of the template. If a template is malformed by even a single space or missing newline, the model may fail to trigger a tool or hallucinate its own "end of turn" markers.
- **The "Jinja2" Standard:** Most modern models (Llama, Mistral, Qwen) use **Jinja2** templating. This allows the model to dynamically change its behavior based on whether it needs to call a tool, display a thought process, or provide a standard reply.

[Example of a Chat Template with Tool Calling support](./public/examples/phi4-mini-chat-template.txt)

---

### 1.5. Architecture

The architecture is the structural wiring of the neural network—how it actually processes the data you feed it.

| Architecture Type              | How it Works                                                                                                                            | Pros & Cons                                                                                                                              |
| :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Attention (Transformers)**   | Uses "Self-Attention" to look at all words in a sentence simultaneously and calculate how strongly they relate to one another.          | **Pros:** Incredible reasoning and precise recall.<br>**Cons:** Memory required explodes exponentially as context grows.                 |
| **Mamba (SSMs)**               | Processes text selectively, compressing the history of the conversation into a fixed-size mathematical state as it reads left-to-right. | **Pros:** Handles massive context windows with minimal RAM.<br>**Cons:** Can struggle with recalling specific data buried in the middle. |
| **Hybrid (Mamba + Attention)** | Interleaves Mamba layers with standard Transformer attention layers.                                                                    | **Pros:** Combines Mamba's speed with Transformer's sharp recall.<br>**Cons:** Highly complex to engineer and run on consumer hardware.  |

---

### 1.6. Context Length (Context Window)

Think of context length as the AI's "short-term working memory." It is completely separate from its permanent parameter "brain."

- **Quadratic vs. Linear Scaling:** Expanding this memory is a massive hardware challenge. Standard Transformers scale quadratically (doubling the context quadruples the RAM requirement). Newer architectures like Mamba scale linearly, allowing for massive context windows using a fraction of the RAM.

**Context Limits:** As AI evolves, standard context windows continue to expand:

| Tier                 | Token Limit  | Best Used For                                                          | Examples                             |
| :------------------- | :----------- | :--------------------------------------------------------------------- | :----------------------------------- |
| **Standard**         | 8K to 32K    | Daily chat, summarizing short articles, or debugging short scripts.    | Llama 3 8B, Mistral 7B               |
| **Long Context**     | 128K to 256K | Digesting entire novels, financial reports, or large codebases.        | Qwen 3, Llama 3.3                    |
| **Frontier Context** | 1M to 10M    | Ingesting massive document repositories or retaining long-term memory. | Llama 4 Scout (10M), Claude 4.6 (1M) |

<figure style="text-align: center;">
  <img src="./public/images/context-window.png">
  <figcaption>Diagram showing how multi-turn inputs and outputs consume a 128k token context window, eventually leading to truncated outputs.</figcaption>
</figure>

---

### 1.7. Formats

Once a model is trained, its billions of parameters need to be saved into a file. The format you choose depends entirely on your hardware and software stack.

| Format          | Best For        | Description                                                                              |
| :-------------- | :-------------- | :--------------------------------------------------------------------------------------- |
| **Safetensors** | Cloud / Python  | Hugging Face standard. Loads incredibly fast and cannot harbor malicious code.           |
| **GGUF**        | Local AI / CPUs | llama.cpp standard. Portable file that easily splits the workload between CPUs and GPUs. |
| **ONNX**        | Enterprise Apps | Microsoft/Meta standard for strict interoperability across different software stacks.    |
| **MLX**         | Apple Silicon   | Apple's native format optimized to leverage the unified memory of M-series chips.        |
| **MLC**         | Web / Mobile    | Compressed to run entirely client-side via WebAssembly/WebGPU without a backend server.  |

---

### 1.8. Quantization

Quantization is the process of compressing the model to make it smaller and faster to run, usually for local use.

- **How it works:** Originally, model weights are stored in high-precision formats like 16-bit floating-point (FP16). Quantization rounds these numbers down to lower precisions, like 8-bit, 4-bit, or even 2-bit integers.
- **Benefits:** Drastically reduces the file size and the RAM/VRAM required to run the model, while significantly speeding up text generation.
- **Drawbacks:** You lose a tiny bit of precision, though modern quantization methods make this loss almost unnoticeable for most everyday tasks.

---

### 1.9. Fine-Tuning

Fine-tuning is the process of taking an already trained "base" model and training it further on a smaller, highly targeted dataset to change its behavior or master a specific task.

- **The Concept:** A base model possesses vast general knowledge, but its default instinct is simply to predict the next word. Fine-tuning shapes it into a useful tool like a conversational chatbot or a strict JSON generator.
- **Instruction Tuning:** The most common form of fine-tuning. Models are trained on thousands of structured human conversations to learn how to follow instructions and respect safety guardrails.
- **Full Fine-Tuning:** The computationally expensive method where every single parameter in the model is updated during training.
- **PEFT & LoRA:** Parameter-Efficient Fine-Tuning freezes the base model and trains a tiny, lightweight "adapter" on top of it. LoRA (Low-Rank Adaptation) allows a developer to fine-tune a massive model using a single consumer graphics card in just a few hours.

[LoRA Fine-Tuning Example](./public/examples/lora-fine-tuning.py)

---

### 1.10. Sampling & Generation

When an LLM predicts the next word, it doesn't just pick one; it generates a massive list of probabilities for every token in its vocabulary. "Sampling" is the set of rules that dictates how the model chooses the final winner from that list.

- **What it is:** A collection of mathematical dials that control the randomness, creativity, and predictability of the AI's output.
- **What it does:** Instead of always picking the \#1 most likely next word (which often results in dry, robotic, or repetitive text), sampling allows the model to occasionally pick the 2nd, 10th, or 50th most likely word. This injects variety and a more human-like flow into the response.
- **Analogy:** Imagine a chef deciding what ingredient to add next to a soup. A strict chef (low randomness) always picks the most obvious, safe choice (salt). A creative chef (high randomness) might occasionally throw in something less expected (cinnamon) to create a unique flavor profile.

| Parameter              | How it Works                                                                                                                                                                                                                          | Best Used For                                                                                                                                            |
| :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Temperature**        | The master dial for randomness. A value of 0 makes the model strictly pick the highest-probability token every time. Higher values (e.g., 0.7 to 1.0) flatten the probability curve, giving less likely words a fighting chance.      | **Low (0.0 - 0.3):** Coding, math, factual data extraction.<br>**High (0.7+):** Brainstorming, creative writing, storytelling.                           |
| **Top-K**              | Sorts the predicted tokens by probability and outright discards everything below the "K"th rank. If Top-K is set to 40, the model is only allowed to choose from the top 40 most likely next words.                                   | Trimming the absolute worst guesses. It prevents the model from hallucinating or generating gibberish by cutting off ultra-low probability tokens.       |
| **Top-P (Nucleus)**    | Adds up the probabilities of the top tokens until they hit a combined "P" threshold (e.g., 0.90 or 90%). It then discards all remaining tokens. The pool of choices shrinks or grows dynamically based on how confident the model is. | A smarter, more dynamic alternative to Top-K. Great for maintaining coherent text while still allowing for a natural, controlled variance in vocabulary. |
| **Repetition Penalty** | Artificially lowers the probability of tokens that have already appeared recently in the generated text, making the model look for fresh words.                                                                                       | Preventing the model from getting stuck in an infinite loop where it repeats the exact same phrase over and over.                                        |

---

# 2\. Inference

If an LLM’s weights (the data) are the "brain," the inference engine is the "nervous system" and "muscles." An inference engine is the software responsible for loading the model into memory, processing your prompt, and performing the massive mathematical calculations required to generate words.

### The Inference Landscape

| Category             | Engine             | Description                                                                                 |
| :------------------- | :----------------- | :------------------------------------------------------------------------------------------ |
| **The Standards**    | Transformers       | Hugging Face's "Research Lab" engine. Essential for building, but memory-heavy.             |
|                      | vLLM               | The "Data Center Standard." Fastest for concurrent users using smart memory management.     |
|                      | llama.cpp          | The "Everyman’s Engine." Written in C++ to run GGUF models on everyday CPUs and Macs.       |
| **High-Performance** | TensorRT-LLM       | The absolute highest throughput possible, but locked entirely to NVIDIA chips.              |
|                      | SGLang             | Blazing-fast engine optimized for complex prompt workflows and agents using prefix caching. |
|                      | ExLlamaV2          | The "Local Speed Demon" for single-user generation on consumer NVIDIA GPUs.                 |
| **User-Friendly**    | Ollama / LM Studio | Lightweight desktop applications for browsing, downloading, and chatting with local models. |
|                      | GPT4All            | Privacy-first desktop application focused on reading your local documents (RAG).            |
|                      | Llamafile          | Packages an LLM and its engine into a single executable file (like a portable USB drive).   |

### Which engine should you choose?

| Your Goal                          | Recommended Tool                              |
| :--------------------------------- | :-------------------------------------------- |
| **Learning / Fine-Tuning**         | Transformers                                  |
| **Building a SaaS / Scalable API** | vLLM (General) or SGLang (Agents/JSON)        |
| **Private Desktop Chatbot**        | LM Studio, Ollama, or GPT4All                 |
| **Old Laptop / Mac / No GPU**      | llama.cpp (via Ollama/LM Studio) or Llamafile |
| **Max Speed on Home NVIDIA GPU**   | ExLlamaV2                                     |
| **Mobile App / Web Browser Dev**   | MLC LLM                                       |

---

# 3\. Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is the critical bridge connecting the **Application** layer to the **Model** layer. While standard LLMs are limited by their training cutoff dates and lack access to private information, RAG allows an AI to securely read, analyze, and cite external, live, or proprietary data (like your company's internal PDFs, live databases, or massive code repositories) _without_ needing to retrain or fine-tune the model.

- **The Analogy:** If an LLM is a brilliant student taking an open-book exam, RAG is the hyper-efficient librarian who instantly fetches the exact textbook pages the student needs to read right before answering the question.

### 3.1. The RAG Pipeline (How it Works)

A standard RAG system operates in a multi-step workflow behind the scenes every time a user asks a question:

| Step  | Phase                         | What Happens                                                                                                                                                                                                                                                                 |
| :---- | :---------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Ingestion & Chunking**      | Large documents (PDFs, wikis, code) are broken down into smaller, manageable paragraphs or "chunks." This ensures the data is bite-sized enough for the AI to process efficiently.                                                                                           |
| **2** | **Embedding**                 | A specialized, lightweight AI (an Embedding Model) converts these text chunks into high-dimensional arrays of numbers (vectors). This translates human language into mathematical coordinates.                                                                               |
| **3** | **Storage**                   | These vectors are saved in a **Vector Database**. Texts with similar meanings (e.g., "puppy" and "dog") are stored close together in this mathematical space.                                                                                                                |
| **4** | **Retrieval**                 | When a user types a prompt, their question is also converted into a vector. The database searches for the stored vectors mathematically closest to the question's vector, instantly retrieving the most relevant chunks of text.                                             |
| **5** | **Augmentation & Generation** | The Application layer takes the user's original prompt _plus_ the retrieved text chunks, bundles them together, and sends them to the **Inference** engine. The **Model** reads this augmented prompt and generates an accurate answer based purely on the provided context. |

---

### 3.2. The RAG Tech Stack

Building a RAG system introduces a few specialized tools into the broader AI stack:

| Component            | Purpose                                                                                                                                                                                                 | Popular Examples                                                      |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------- |
| **Orchestrators**    | The "glue" frameworks that wire the Application layer to the Vector DB and the LLM API. They handle the logic of the entire pipeline.                                                                   | **LangChain**, **LlamaIndex**, **Haystack**                           |
| **Vector Databases** | Specialized databases built specifically to store, index, and query vector embeddings at lightning speed.                                                                                               | **ChromaDB** (Local), **Qdrant**, **Pinecone** (Cloud), **Milvus**    |
| **Embedding Models** | Lightweight models designed strictly to translate text into vector coordinates.                                                                                                                         | **Nomic-embed-text** (Local), **OpenAI text-embedding-3**, **BGE-M3** |
| **Rerankers**        | An optional but powerful secondary model that acts as a quality filter. It double-checks the retrieved documents and re-orders them to ensure the LLM only sees the absolute most relevant information. | **Cohere Rerank**, **BGE-Reranker**, **Jina Reranker**                |

---

### 3.3. RAG Approaches

As AI has evolved, the basic RAG pipeline has been upgraded to handle much more complex, messy, and interconnected data. Developers now categorize RAG into a few distinct architectural approaches:

| Approach         | How it Works                                                                                                                                                                                                                                                                                                      | Best Used For                                                                                                             |
| :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Naive RAG**    | The foundational "Chunk → Embed → Retrieve → Generate" pipeline. It takes the user's exact prompt, finds the closest matching text chunks, and feeds them to the LLM.                                                                                                                                             | Simple Q&A bots, querying highly structured and clean internal wikis, or basic customer support.                          |
| **Advanced RAG** | Introduces "Pre-Retrieval" and "Post-Retrieval" optimizations. Before searching, it might use a smaller AI to rewrite or expand the user's query for better search results. After retrieving the chunks, it uses a **Reranker** model to filter out irrelevant noise before sending the final context to the LLM. | Enterprise search, analyzing dense financial PDFs, or any system where accuracy and avoiding hallucinations are critical. |
| **GraphRAG**     | Combines standard vector databases with **Knowledge Graphs**. Instead of just finding paragraphs with similar words, it maps out the mathematical relationships between entities (e.g., mapping that "Person A" works for "Company B" which owns "Product C").                                                    | Investigating complex networks, connecting the dots across hundreds of separate documents, or legal discovery.            |
| **Agentic RAG**  | The most dynamic approach. Instead of a hard-coded pipeline, an AI Agent is given access to search tools and decides _autonomously_ if it needs to search, what queries to run, and if it needs to run follow-up searches based on the first set of results (multi-hop reasoning).                                | Coding assistants debugging a massive repository, complex research tasks, and open-ended analysis.                        |
