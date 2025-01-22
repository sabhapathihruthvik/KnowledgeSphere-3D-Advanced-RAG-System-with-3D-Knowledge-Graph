# KnowledgeSphere 3D: Advanced RAG System with 3D Knowledge Graph

## Project Name:
**KnowledgeSphere 3D: Advanced RAG System with 3D Knowledge Graph**

## Description:
KnowledgeSphere 3D is an advanced Retrieval-Augmented Generation (RAG) system that combines Natural Language Processing (NLP), machine learning, and Neo4j-based knowledge graphs to analyze, visualize, and interact with documents. This platform allows users to upload PDF documents, extract relationships and entities, and visualize the data in a 3D knowledge graph. The system supports querying and answering questions based on the content of the uploaded documents while showcasing the relationships in a 3D visual format.

## Features:
- **Upload and Process PDF**: Upload PDF documents to extract text and relationships.
- **Entity & Relationship Extraction**: Extract key entities and relationships from the document using AI.
- **3D Knowledge Graph**: Visualize extracted entities and relationships in a 3D knowledge graph using Plotly.
- **Query Interface**: Ask questions and get answers based on the document using the RAG system.
- **Neo4j Integration**: Store and query entities and relationships in a Neo4j graph database.
- **Interactive UI**: A user-friendly interface powered by Streamlit.

## Usage:
1. Run the Streamlit app by executing the following command:
   ```bash
   streamlit run app.py


## Technical Specifications:
Backend: Neo4j (Graph Database)
Frontend: Streamlit (Web UI)
3D Visualization: Plotly, NetworkX, PyVis
Text Processing: LangChain, Hugging Face Embeddings, RecursiveCharacterTextSplitter
Machine Learning: ChatGroq for relationship extraction and question answering
File Processing: PyPDFLoader for PDF text extraction


## Requirements:
Python 3.7+
Neo4j Graph Database (can be hosted on Neo4j Cloud or locally)
ChatGroq API Key (for AI-powered relationship extraction)
Hugging Face API for embeddings (optional for local model integration)
Setup Instructions:
Install Python: Ensure Python 3.7+ is installed.
Install Neo4j: You can use a Neo4j Cloud instance or install Neo4j locally. Configure the URI, username, and password in the code.
ChatGroq API Key: You need a ChatGroq API key to perform LLM-based relationship extraction. Sign up at ChatGroq to obtain the API key.
Run the Application: Use the streamlit run app.py command to start the app.

## Possible Improvements:
Document Formats: Extend support for other document formats such as Word, Excel, or plain text.
Advanced Graph Layout: Improve the 3D graph layout and interactivity with features like zooming, panning, and tooltips.
Enhanced AI Models: Integrate more advanced models for extracting relationships, including domain-specific models.
Authentication & User Roles: Add user authentication for secure access to the system.
Real-Time Collaboration: Enable multiple users to interact with the platform in real-time.
Performance Optimization: Enhance the performance and scalability of the system, especially when dealing with large documents and graphs.

## Limitations:
The current setup supports only PDF files as input. Adding support for other formats is a future enhancement.
The quality of relationship extraction heavily relies on the performance of the LLM model used.

## Example Output:
Once a PDF is processed, you can ask questions like:

"What are the key relationships in this document?"
"Who are the main entities?"
"How is entity A related to entity B?"
The system will return the answers and display them alongside a knowledge graph showing the relationships between the entities.

## License:
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact:
For any questions or further assistance, you can contact the project owner at your-email@example.com.

## Acknowledgements:
Neo4j - Graph database for storing and querying entities and relationships.
LangChain - For text processing and interacting with LLM models.
ChatGroq - For relationship extraction using an LLM.
Plotly - For 3D graph visualization.
Streamlit - For creating the web interface.


## Run the below command to install the libraries
    ```bash
    pip install streamlit neo4j langchain langchain_groq networkx pyvis plotly huggingface-hub pydantic PyPDF2 pyyaml
