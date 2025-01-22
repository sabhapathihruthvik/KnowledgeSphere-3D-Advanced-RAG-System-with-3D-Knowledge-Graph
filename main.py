import streamlit as st
import os
from typing import List, Dict
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import networkx as nx
from pyvis.network import Network
import tempfile
import plotly.graph_objects as go
# from streamlit_plotly import plotly_chart
from streamlit import plotly_chart

import base64
import re

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="KnowledgeSphere 3D")
# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #1a1a1a, #2d2d2d);
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .graph-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq
GROQ_API_KEY = ""
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

# Neo4j Configuration
NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = ""

class KnowledgeGraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def delete_database(self):
        """Delete all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try:
                session.run("CALL db.index.vector.drop('document_vectors')")
            except Exception as e:
                st.warning("Vector index might not exist or was already deleted.")
            
    def create_vector_store(self, documents: List):
        """Create vector store in Neo4j"""
        vector_store = Neo4jVector.from_documents(
            documents,
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors",
            node_label="Document",
            embedding_node_property="embedding",
            text_node_property="text"
        )
        return vector_store

    def _parse_relationships(self, llm_response: str) -> List[Dict]:
        """Parse LLM relationship extraction response"""
        relationships = []
        # Regular expression pattern to match the relationship format
        pattern = r'\(([^)]+)\)-\[([^\]]+)\]->\(([^)]+)\)'
        
        # Split the response into lines and process each line
        for line in llm_response.split('\n'):
            line = line.strip()
            matches = re.findall(pattern, line)
            
            for match in matches:
                if len(match) == 3:  # Ensure we have all three components
                    entity1, relationship, entity2 = match
                    # Clean up the extracted texts
                    entity1 = entity1.strip()
                    relationship = relationship.strip()
                    entity2 = entity2.strip()
                    
                    if entity1 and relationship and entity2:  # Ensure none are empty
                        relationships.append({
                            'entity1': entity1,
                            'relationship': relationship,
                            'entity2': entity2
                        })
        
        return relationships

    def create_knowledge_graph(self, documents: List):
        """Extract entities and relationships to create knowledge graph"""
        with self.driver.session() as session:
            for doc in documents:
                prompt = f"""
                Extract key entities and their relationships from this text. 
                Format each relationship exactly as: (entity1)-[relationship]->(entity2)
                Return one relationship per line.
                Only include clear, explicit relationships from the text.
                Text: {doc.page_content}
                """
                response = llm.predict(prompt)
                
                relationships = self._parse_relationships(response)
                for rel in relationships:
                    if all(rel.values()):  # Check that no values are empty
                        session.run("""
                        MERGE (e1:Entity {name: $entity1})
                        MERGE (e2:Entity {name: $entity2})
                        MERGE (e1)-[:RELATES {type: $relationship}]->(e2)
                        """, rel)

    def create_3d_graph(self):
        """Generate 3D graph visualization using Plotly"""
        G = nx.DiGraph()
        
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
            RETURN e1.name as source, r.type as relationship, e2.name as target
            """)
            records = list(result)
            
        if not records:  # If no relationships exist
            return None
            
        for record in records:
            G.add_edge(record["source"], record["target"])
            
        pos = nx.spring_layout(G, dim=3)
        
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edges_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_z = []
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        nodes_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hovertext=list(G.nodes()),
            hoverinfo='text',
            marker=dict(
                size=8,
                color='#00ff00',
                line_width=2))

        fig = go.Figure(data=[edges_trace, nodes_trace])
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def query(self, question: str) -> Dict:
        """Query both vector store and knowledge graph"""
        vector_store = Neo4jVector(
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors"
        )
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        rag_answer = qa_chain.run(question)
        
        kg_data = []
        with self.driver.session() as session:
            result = session.run("""
            MATCH path = (e1:Entity)-[r:RELATES]->(e2:Entity)
            WHERE e1.name CONTAINS $question OR e2.name CONTAINS $question
            RETURN e1.name as source, r.type as relationship, e2.name as target
            LIMIT 10
            """, question=question)
            kg_data = [dict(record) for record in result]
        
        return {
            "rag_answer": rag_answer,
            "knowledge_graph": kg_data
        }

def main():
    st.title("🌟 KnowledgeSphere 3D:  Advanced RAG System with 3D Knowledge Graph")
    
    # Initialize system
    rag_system = KnowledgeGraphRAG()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🗑️ Delete Database", help="Clear all data and start fresh"):
            with st.spinner("Deleting database..."):
                rag_system.delete_database()
            st.success("Database cleared successfully!")
        
        st.markdown("---")
        st.markdown("### 📊 Graph Settings")
        st.markdown("Customize your graph visualization here")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Welcome to **KnowledgeSphere 3D**! This platform enables you to analyze, visualize, and interact with a 3D knowledge graph.")

        uploaded_file = st.file_uploader(
            "Drop your PDF document here",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file:
            with st.spinner("🔄 Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(tmp.name)
                    documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)
                
                rag_system.create_vector_store(splits)
                rag_system.create_knowledge_graph(splits)
            st.success("✅ Document processed successfully!")
    
    with col2:
        st.markdown("### 🔍 Query Interface")
        question = st.text_input("Ask a question:", placeholder="Type your question here...")
        
    if question:
        with st.spinner("🤔 Analyzing..."):
            results = rag_system.query(question)
        
        st.markdown("### 📝 Answer")
        st.markdown(f"```\n{results['rag_answer']}\n```")
        
        st.markdown("### 🔗 Knowledge Graph Connections")
        for rel in results["knowledge_graph"]:
            st.markdown(f"🔹 {rel['source']} → *{rel['relationship']}* → {rel['target']}")
    
    # 3D Graph Visualization
    st.markdown("### 🌐 Knowledge Graph Visualization")
    with st.container():
        fig = rag_system.create_3d_graph()
        if fig is not None:
            plotly_chart(fig, use_container_width=True, height=600)
        else:
            st.info("No relationships to visualize yet. Upload a document to create the knowledge graph.")

if __name__ == "__main__":
    main()

