# Nexus VectorDB: Setup and Run Instructions

This project is a lightweight, dependency-free Vector Database and Retrieval-Augmented Generation (RAG) engine built entirely from scratch in standard Java and Vanilla HTML/JS/CSS. 

## 1. Prerequisites

1. **Java Development Kit (JDK)**
   - You need Java 11 or higher installed on your system.
   - You can verify your installation by opening a terminal and running:
     `java -version` and `javac -version`

2. **API Keys**
   - The engine uses cloud APIs for generating vector embeddings and text generation. You will need:
     - **Gemini API Key**: For 256-dimensional semantic embeddings (using `gemini-embedding-2`).
     - **Groq API Key**: For ultra-fast LLM generation (using `openai/gpt-oss-120b`).

## 2. Configuration

1. In the root directory of this project, create a file named exactly `.env`.
2. Add your API keys to this file in the following format:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 3. How to Run

Because this project is built from scratch without complex build tools like Maven or Gradle, running it is incredibly simple:

1. **Compile the server**:
   Open your terminal in the project directory and run:
   ```bash
   javac Main.java
   ```

2. **Start the server**:
   After compiling, start the server by running:
   ```bash
   java Main
   ```
   You should see output indicating the server has started (e.g., `=== VectorDB Engine ===`, `http://localhost:8080`).

3. **Open the Frontend**:
   Simply open the `index.html` file in your preferred web browser. The frontend communicates with the local Java backend server running on port 8080.

## 4. Troubleshooting

* **"Cannot reach server" / Features not working**: Ensure `java Main` is running in your terminal and that it hasn't crashed. Ensure port `8080` isn't in use by another application.
* **"Cloud API unavailable" / Connection Reset**: This means either your `.env` file is missing/incorrect, or there is a temporary network hiccup connecting to Groq/Google.
* **Vector Dimension Errors**: If the server crashes on startup mentioning `Index 16 out of bounds`, delete the `vector_db.jsonl` file to clear out corrupt/legacy 16-dimensional cache data.
