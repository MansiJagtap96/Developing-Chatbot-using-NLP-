# Intents-Based Chatbot using NLP

This is a simple chatbot built using **NLP (Natural Language Processing)** and **Logistic Regression**. It uses **Streamlit** to create a user-friendly interface for chatting.

## Features
- Uses NLP to understand user messages.
- Predicts user intent with **Logistic Regression**.
- Simple **Streamlit interface** for chatting.
- Saves chat history in a **CSV file**.
- Includes a **Jupyter Notebook** for learning and testing.

## Project Structure
```
/Developing-Chatbot-using-NLP/
│
├── chatbot.py               # Main chatbot code
├── intents.json             # Data file for chatbot responses
├── chat_log.csv             # Auto-generated chat history
├── chatbot_notebook.ipynb   # Notebook for practice and testing
└── nltk_data/               # Folder for NLTK data (if required)
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd Developing-Chatbot-using-NLP
   ```

2. **Install the required libraries:**
   ```bash
   pip install streamlit scikit-learn nltk
   ```

3. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   ```

## How to Run
1. Run this command in your terminal:
   ```bash
   streamlit run chatbot.py
   ```
2. Click the provided link (e.g., `http://localhost:8501`) to open the chatbot in your browser.

## Example `intents.json`
```json
[
    {"tag": "greeting", "patterns": ["Hi", "Hello"], "responses": ["Hello!", "Hi there!"]},
    {"tag": "goodbye", "patterns": ["Bye", "Goodbye"], "responses": ["Goodbye!", "Bye!"]}
]
```

## Future Improvements
- Add smarter NLP models like **BERT** or **GPT**.
- Improve the chatbot’s UI with more **Streamlit** features.

## License
This project is licensed under the **MIT License**.

## Developer
This project was developed by **Mansi Jagtap**.

## Contact
If you have any questions or suggestions, feel free to reach out!

