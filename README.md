# Basic Chatbot with LangChain and LangGraph

This project demonstrates the creation of a **basic chatbot** using the powerful **LangChain** and **LangGraph** libraries. It is designed with memory-saving capabilities and incorporates **Google's Gemini Model 2.5-Flash** as the language model.

---

## Key Features

- **LangChain Integration**: Utilizes LangChain's `BaseMessage`, `HumanMessage`, and messaging capabilities.
- **LangGraph Library**: Implements a StateGraph workflow for defining chatbot states.
- **Google Gemini Model**: Enables human-like conversational capabilities with `gemini-2.5-flash`.
- **Memory Saver**: Integrates `MemorySaver` to optimize resource utilization for sustainable execution.
- **Rich Stateful Design**: The chatbot workflow is structured with stateful nodes and memory checkpoints.

---

## How It Works

1. **StateGraph Definition**: 
   - A `StateGraph` is defined using LangGraph with nodes and edges to structure the chatbot's workflow.

2. **Chat State Handling**:
   - The chatbot uses a custom `TypedDict` named `ChatState` to maintain the chat session's state.
   - Previous conversation messages are saved using `MemorySaver`.

3. **Integration with Google Gemini Model**:
   - The `GoogleGenerativeAI` library facilitates the language model interactions with `gemini-2.5-flash`.

4. **Chat Loop**:
   - The project runs a continuous user query-response loop until explicitly exited.

---

## Code Overview

- **Chat Node Function**:
  The `chat_node` function processes user messages and retrieves responses from the Gemini Model:
  ```python
  def chat_node(state: ChatState):
      messages = state['messages']
      response = llm.invoke(messages)
      return {'messages': [response]}
  ```

- **Graph Control Flow**:
  The chatbot's control flow follows a linear path: `START -> chat_node -> END`.

- **Memory Checkpoint**:
  Optimized memory management ensures efficient execution:
  ```python
  checkpointer = MemorySaver()
  ```

---

## Installation and Getting Started

### Prerequisites

Ensure you have Python 3.12.4 installed along with the following libraries:

- `langchain-core`
- `langgraph`
- `langchain-google-genai`

You also need to set up credentials to access the Google Gemini Model.

### Steps

1. Clone the repository:

```bash
git clone https://github.com/chaudhary-pawan/LangGraph.git
cd LangGraph
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables for Google Gemini API access:
   ```bash
   export GOOGLE_GENAI_API_KEY='your-api-key-here'
   ```

4. Run the notebook:

Open `9_basic_chatbot.ipynb` in Jupyter Notebook or another compatible notebook viewer and step through the cells to initialize the chatbot.
To start the chatbot, execute the notebook code. You can interact with it in real time by providing inputs to the bot.

---

## Acknowledgements

- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Documentation](https://github.com/langgraph/)
- [Google Generative AI - Gemini](https://github.com/langchain-google-genai)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



# X Post Generator Bot

The **X Post Generator Bot** is an automated solution built using [LangChain](https://github.com/hwchase17/langchain) and [LangGraph](https://langgraph.readthedocs.io/en/latest/). This bot generates, evaluates, and optimizes short, humorous posts (tweets) for X (formerly Twitter) using the **Gemini model** `2.5-flash` as the language model.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow Pipeline](#workflow-pipeline)
- [License](#license)

---

## About

The X Post Generator Bot simplifies the process of creating viral and witty content for social media. From idea generation to critical evaluation, this tool ensures every post meets stringent virality criteria. Using feedback loops, it refines posts to maximize their humor, originality, and audience appeal.

---

## Features

- **Automatic Tweet Generation**
  - Creates short, humorous tweets based on a given topic.

- **Evaluation Engine**
  - Uses structured output to evaluate tweets against criteria like originality, humor, virality, punchiness, and adherence to format.

- **Optimization**
  - Improves tweets based on direct feedback to enhance their humor and engagement potential.

- **Feedback History Tracking**
  - Maintains a history of tweets and feedback to provide a transparent optimization process.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/chaudhary-pawan/LangGraph.git
    ```

2. Navigate to the project directory:
    ```bash
    cd LangGraph
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Install additional libraries:
    ```bash
    pip install langchain_google_genai
    ```

---

## Usage

1. Open the provided Jupyter Notebook `8_X_post_generator.ipynb`.
2. Define the initial state by specifying the topic of the post:
   ```python
   initial_state = {
       "topic": "Relation betweeen Russia and Indian governments",
       "iteration": 1,
       "max_iteration": 5
   }
