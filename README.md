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
