# RAG Application using LangChain and Amazon Bedrock

## Introduction
For this project, I created a Retrieval Augmented Generation (RAG) application that can use any LLM from the Amazon Bedrock generative AI library. These LLMs are used to respond to queries using the data retrieved from PDF documents.

The Amazon Bedrock library has the following LLMs available:
- Claude by Anthropic
- Llama by Meta
- Amazon Titan
- Jurassic by AI21 Labs
- Command by Cohere
- Mistral
- Stable Diffusion by Stability
### Technologies Used
This application was developed using the following technologies:
- Amazon Bedrock supplies the two text-generation LLMs used in this application:
  - Amazon Titan
  - Llama 2
- Streamlit library to generate the graphical user interface (GUI)
- LangChain library to:
  -  Split the PDF documents into chunks
  -  Generate the embeddings from the chunks
  -  Store the embeddings in a vector store
  -  Create a chain that accepts a query, retrieves relevant data from the PDF documents, and generates a response using the chosen LLM.

### Data
The data for this project consists of the following four books in PDF format:
- From Newton to Einstein - Changing Conceptions on the Universe
- Relativity - The Special and General Theory
- The Einstein Theory of Relativity
- The Meaning of Relativity

### Example output:
<img src="https://github.com/mlsmall/RAG-App-with-Amazon-Bedrock/blob/main/output.png" width="1200" />

## Instructions
### Download the repository
* Go to a terminal and paste `git clone https://github.com/mlsmall/RAG-App-with-Amazon-Bedrock.git`

### AWS Credentials
Before you begin, set up an IAM account in AWS and generate a secret key. You will need to put your credentials in a `.env` file. There is a small cost associated with using the generative AI models in Amazon Bedrock so be sure to check the [prices](https://aws.amazon.com/bedrock/pricing/).
* Go to https://us-east-1.console.aws.amazon.com/iam/home and create a new user.
* Click on the new user and assign the AdministratorAccess policy to it.
* Click on the "Security Credentials" tab, scroll below, and generate a new access key.
* Create a `.env` file in your project directory and add:
`ACCESS_KEY="your_AWS_access_key`
`SECRET_KEY="your_AWS_secret_key"`.
* Create a `.gitignore` file in your project directory and add `.env`.

### Copy your documents (Optional)
* Go to the `data` directory and replace the books with your document files in `.pdf` format.
  
### Running the application
#### Inside a terminal:
* Install the dependencies
`pip install -r requirements.txt`

* Run the Streamlit application
`streamlit run app.py`

#### In your browser:

* Open the GUI in your browser
`http://localhost:8501/`

* Click the `Update Vector Store` button in the sidebar to create the vector store. If you add new documents to the `data` directory in your project folder, you'll have to update the vector store again.
* Type your query in the prompt and click the `Llama2 Output` or the `Titan Output` button.
`What are the main differences between Einstein's and Newton's theories?`
