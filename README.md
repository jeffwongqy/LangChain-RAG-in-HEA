<img width="211" height="101" alt="images" src="https://github.com/user-attachments/assets/68a3f812-e1f2-4c38-8ed9-a7587cff0788" />

#### _A*STAR Institute of High Performance Computing (IHPC) ARIA Internship Project on Materials Informatics 2024_


# LangChain RAG in High Entropy Alloys

<img width="1000" height="450" alt="9ce19469-692a-4422-bd27-a4071e291ee7" src="https://github.com/user-attachments/assets/c7f04877-1bad-45c5-8587-d65b2309fbe3" />

## 1. Introduction 
High-Entropy Alloys (HEAs) are advanced materials containing multiple principal elements and have attracted significant research interest due to their unique mechanical, thermal, corrosion, and structural properties. However, extracting relevant information from HEA research papers can be time-consuming because of the large amount of technical information available.

This project develops a High-Entropy Alloy Retrieval-Augmented Generation (RAG) platform using Streamlit, LangChain, Ollama, and ChromaDB. Users can upload a HEA research paper in PDF format and ask questions about its contents. The system retrieves relevant sections from the uploaded paper and uses the Llama 3.2 language model to generate concise answers based only on the retrieved information.

## 2. Project Aim
The aim of this project is to develop a simple AI-assisted research platform for retrieving and answering questions from High-Entropy Alloy research papers using a local RAG architecture.

## 3. Project Objectives
The project objectives are to:
- Develop a Streamlit-based interface for uploading HEA research papers.
- Extract and divide PDF content into smaller text chunks.
- Generate document embeddings using nomic-embed-text.
- Store and retrieve relevant information using Chroma.
- Use Llama 3.2 to generate answers based only on retrieved document content.
- Reduce unsupported or fabricated information by restricting responses to the uploaded paper.
- Provide useful information about HEA composition, processing, phases, microstructure, and properties

## 4. System Workflow
The system follows a simple RAG pipeline:

PDF Upload → PDF Text Extraction → Text Chunking → Embedding Generation → Chroma Vector Store → Similarity Retrieval → Llama 3.2 → Answer

When a user submits a question, the system retrieves the three most relevant text chunks from the uploaded paper. These chunks are then provided to Llama 3.2 as context for generating the answer.

### 4.1 PDF Upload

The user uploads one HEA research paper through the Streamlit interface. The system accepts PDF files and saves the uploaded document for processing.

```python
pdf_file = st.file_uploader("Upload one HEA pdf file:", type = ["pdf"])
```

### 4.2 Text Extraction

PyPDFLoader reads the PDF and extracts its text page by page. The system also displays the number of pages detected.

```python
with open("hea_pdf", "wb") as f:
    f.write(pdf_file.getbuffer())
st.success("PDF uploaded Successfully")

loader = PyPDFLoader("hea_pdf")
documents = loader.load()
st.write("Number of pages:", len(documents))
```

### 4.3 Text Chunking

The extracted text is divided into smaller sections using RecursiveCharacterTextSplitter.

Chunk size: 1,000 characters

Chunk overlap: 150 characters

The overlap helps maintain context between neighbouring chunks.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
chunks = splitter.split_documents(documents)
st.write("Number of chunks:", len(chunks))
```

### 4.4 Embedding Generation

Each text chunk is converted into a numerical vector using the Ollama nomic-embed-text embedding model. These vectors represent the semantic meaning of the text.

```python
embeddings = OllamaEmbeddings(model = "nomic-embed-text")
```

### 4.5 Vector Database

The embeddings and corresponding text chunks are stored in a Chroma vector store. This allows the system to efficiently search for information relevant to a user's question.

```python
vectorstore = Chroma.from_documents(documents = chunks, embedding = embeddings)
```

### 4.6 Question Retrieval

When the user enters a question, the retriever performs a similarity search against the stored document chunks and retrieves the top 3 most relevant chunks.

```python
retriever = vectorstore.as_retriever(search_kwargs = {'k': 3})
```

### 4.7 Prompt Construction

The retrieved chunks are inserted into a predefined LangChain prompt template. The prompt instructs the LLM to answer using only the retrieved context and to preserve numerical values, units, alloy compositions, and experimental conditions.

```python
prompt = ChatPromptTemplate.from_template(
        
    """
    You are an expert research assistant in High-Entropy Alloys (HEAs).
    
    Use ONLY the information provided in the context to answer the question. 
    
    Instructions:
    1. Give a clear and concise answer.
    2. Include specific values, units, alloy compositions, temperature, processing conditions, 
       and properties when available.
    3. Do not change, estimate or invent numerical values.
    4. Distinguish experimental results from general statements.
    5. If information is conflicting, mention the conflict.
    6. Do not use outside knowledge.
    7. If the answer is not in the context, say: "The answer cannot be found in the provided pdf."
    8. Use bullet points when useful. 
    
    Focus on 
    - Alloy composition
    - Processing and manufactuing
    - Heat treatment
    - Crystal structures and phases
    - Microstructure
    - Mechanical Properties
    - Strength, hardness, and ductility
    - Corrosion and thermal properties
    - Experimental results
    - key findings and conclusion
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    
    """)
```

### 4.8 Answer Generation

The retrieved context and user's question are passed to Llama 3.2 through Ollama. The model generates a concise answer based on the information retrieved from the paper.

```python
llm = ChatOllama(model = "llama3.2", temperature = 0)

rag_chain = (
        {
            "context": retriever, 
            "question": lambda x : x
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
```

### 4.9 Display Results

The generated answer is returned to the Streamlit interface and displayed under the Answer section.

```python
question = st.text_input("Ask a question about HEA paper:")
    
if question:
    with st.spinner("Searching the PDF..."):
        answer = rag_chain.invoke(question)
        
    st.subheader("Answer:")
    
    st.write(answer)
```


## 5. Limitations
The current implementation has several limitations:

- Only one PDF can be uploaded at a time.
- The system depends on the quality and completeness of the uploaded paper.
- Answers cannot provide information that is absent from the document.
- Retrieval quality depends on the selected chunk size, overlap, and number of retrieved chunks.
- There is currently no persistent vector database between different uploads.

## 6. Expected Outcome

The expected outcome is a lightweight and user-friendly HEA research assistant that can quickly retrieve relevant information from technical papers and provide concise answers without relying on external knowledge. The project demonstrates how RAG, LangChain, local LLMs, vector databases, and Streamlit can be integrated for materials informatics research.

## References
[1] Wang, J., Kwon, H., Kim, H. S., & Lee, B.-J. (2023). A neural network model for high entropy alloy design. Npj Computational Materials, 9(1). https://doi.org/10.1038/s41524-023-01010-x
