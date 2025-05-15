# 💬 e-commerce chatbot (Gen AI RAG project using LLama3.3 and GROQ)

This intelligent chatbot tailored for an e-commerce platform, enabling seamless user interactions by accurately identifying the intent behind user queries. It leverages real-time access to the platform's database, allowing it to provide precise and up-to-date responses.


This chatbot currently supports 3 intents:

- **faq**: Triggered when users ask questions related to the platform's policies or general information. eg. what is refund policy?
- **sql**: Activated when users request product listings or information based on real-time database queries. eg. Show me all puma shoes below Rs. 5000.
- **smalltalk**: Used to handle colloquial conversations with the user.
* All of these are implemented using semantic-router which will able to classify the intent behind the user question like FAQ's,Product,general conversation..etc
* You have to more number of Utterances or examples while defining the Routes for FAQ's ,product etc to improve performance on classifying the intent behind it.



![Image](https://github.com/user-attachments/assets/266e706d-8879-4231-8dc0-18899f10146f)

## Architecture
![Image](https://github.com/user-attachments/assets/39464f96-84f0-45c9-a605-bb223f3f695b)

### Set-up & Execution

1. Run the following command to install all dependencies. 

    ```bash
    pip install -r app/requirements.txt
    ```

1. Inside app folder, create a .env file with your GROQ credentials as follows:
    ```text
    GROQ_MODEL=<Add the model name, e.g. llama-3.3-70b-versatile>
    GROQ_API_KEY=<Add your groq api key here>
    ```

1. Run the streamlit app by running the following command.

    ```bash
    streamlit run app/main.py
    ```

---
