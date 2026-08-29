import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("🔬 High-Entropy Alloy RAG")
st.write("A platform to upload your research paper and ask questions about High Entropy Alloy")

pdf_file = st.file_uploader("Upload one HEA pdf file:", type = ["pdf"])


if pdf_file is not None:
    
    with open("hea_pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    st.success("PDF uploaded Successfully")
    
    loader = PyPDFLoader("hea_pdf")
    documents = loader.load()
    st.write("Number of pages:", len(documents))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
    chunks = splitter.split_documents(documents)
    st.write("Number of chunks:", len(chunks))
    
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    
    vectorstore = Chroma.from_documents(documents = chunks, 
                                        embedding = embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs = {'k': 3})
    
    llm = ChatOllama(model = "llama3.2", temperature = 0)
    
    
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
    
    
    rag_chain = (
        {
            "context": retriever, 
            "question": lambda x : x
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )

    question = st.text_input("Ask a question about HEA paper:")
    
    if question:
        with st.spinner("Searching the PDF..."):
            answer = rag_chain.invoke(question)
            
        st.subheader("Answer:")
        
        st.write(answer)