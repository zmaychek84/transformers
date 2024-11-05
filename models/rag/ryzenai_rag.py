import torch
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from typing import Optional, Tuple, List
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    Pipeline,
)
import os

# load docs
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

# split docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# embeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# db
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from ragatouille import RAGPretrainedModel
from transformers import TextStreamer


torch.random.manual_seed(0)

DOCS_PATH = "./papers"
CHROMA_DB_PATH = "./chroma_database"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
EMBEDDING_MODEL_NAME = "thenlper/gte-small"  # "meta-llama/Llama-2-7b-hf"
KNOWLEDGE_VECTOR_DATABASE = None


def load_docs():
    document_loader = PyPDFDirectoryLoader(DOCS_PATH)
    return document_loader.load()


def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        add_start_index=True,
        separators=MARKDOWN_SEPARATORS,
        is_separator_regex=False,
    )
    docs_processed = []  # chunks
    for doc in documents:
        docs_processed += text_splitter.split_documents([doc])
    return docs_processed


def split_docs_tokenized(
    documents: list[Document],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    chunk_size: int = 800,
):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    docs_processed = []
    for doc in documents:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def visualize_chunks(
    docs_processed: list[Document],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    tok=False,
):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs_processed]
    fig = pd.Series(lengths).hist()
    plt.title(
        "Distribution of document lengths in the knowledge base (in count of tokens)"
    )
    if tok == False:
        plt.savefig("visualize_chunks.png")
    else:
        plt.savefig("visualize_chunks_tokenized.png")
    plt.close()


def generate_knowledge_vector_database(docs_processed: list[Document] = None):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # Set `True` for cosine similarity
    )

    global KNOWLEDGE_VECTOR_DATABASE
    index_name = "faiss_index"
    if not os.path.exists(index_name):
        KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        KNOWLEDGE_VECTOR_DATABASE.save_local(index_name)
    else:
        KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
            index_name, embedding_model, allow_dangerous_deserialization=True
        )
    return embedding_model


def visualize_db(embedding_model, docs_processed):
    user_query = "How to create a pipeline object?"
    query_vector = embedding_model.embed_query(user_query)
    import pacmap
    import numpy as np
    import plotly.express as px

    global KNOWLEDGE_VECTOR_DATABASE
    embedding_projector = pacmap.PaCMAP(
        n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
    )

    embeddings_2d = [
        list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0])
        for idx in range(len(docs_processed))
    ] + [query_vector]

    # Fit the data (the index of transformed data corresponds to the index of the original data)
    documents_projected = embedding_projector.fit_transform(
        np.array(embeddings_2d), init="pca"
    )
    print("here now")
    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": docs_processed[i].metadata["source"].split("\\")[1],
                "extract": docs_processed[i].page_content[:100] + "...",
                "symbol": "circle",
                "size_col": 4,
            }
            for i in range(len(docs_processed))
        ]
        + [
            {
                "x": documents_projected[-1, 0],
                "y": documents_projected[-1, 1],
                "source": "User query",
                "extract": user_query,
                "size_col": 100,
                "symbol": "star",
            }
        ]
    )

    print("here now 2")
    # Visualize the embedding
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={"User query": "black"},
        width=1000,
        height=700,
    )
    print("here now 3")
    fig.update_traces(
        marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    print("here now 4")
    fig.update_layout(
        legend_title_text="<b>Chunk source</b>",
        title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
    )
    print("here now 5")
    # fig.write_image("visualize_db.png")
    fig.write_image("visualize_db.pdf")
    print("here now 6")


def test():
    documents = load_docs()
    # chunks = split_docs(documents)
    # visualize_chunks(chunks)

    chunks_tokenized = split_docs_tokenized(documents, chunk_size=256)
    # print(len(chunks_tokenized))
    print(chunks_tokenized[0].page_content)
    print(chunks_tokenized[0].metadata)
    # visualize_chunks(chunks_tokenized, tok=True)

    # input("Enter a key")
    print(f"KNOWLEDGE_VECTOR_DATABASE: {KNOWLEDGE_VECTOR_DATABASE}")
    embedding_model = generate_knowledge_vector_database(chunks_tokenized)
    print(f"KNOWLEDGE_VECTOR_DATABASE: {KNOWLEDGE_VECTOR_DATABASE}")

    # visualize_db(embedding_model, chunks_tokenized) - plotly installation crashes

    user_query = "How to quantize Llama2 with best accuracy"
    query_vector = embedding_model.embed_query(user_query)
    print(f"\nStarting retrieval for {user_query=}...")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    print(
        "\n==================================Top document=================================="
    )
    print(retrieved_docs[0].page_content)
    print(
        "==================================Metadata=================================="
    )
    print(retrieved_docs[0].metadata)

    READER_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    """

    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    from transformers import TextStreamer

    streamer = TextStreamer(tokenizer)
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=50,
        streamer=streamer,
    )

    answer = READER_LLM("What is 4+4? Answer:")
    print(answer)

    ###############################
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    print(RAG_PROMPT_TEMPLATE)

    retrieved_docs_text = [
        doc.page_content for doc in retrieved_docs
    ]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(
        question="How to create a pipeline object?", context=context
    )

    # Redact an answer
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    print(f"final answer: {answer}")


def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_index: FAISS,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[Document]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )

    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs


if __name__ == "__main__":
    # test()

    embedding_model = generate_knowledge_vector_database()
    READER_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

    # reranker fails on windows 11
    # from colbert.infra.config.config import ColBERTConfig
    # colbert_config = ColBERTConfig()
    # print(colbert_config)
    # RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RERANKER = None
    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    streamer = TextStreamer(tokenizer)
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=50,
        streamer=streamer,
    )

    question = "how to create a pipeline object?"
    answer, relevant_docs = answer_with_rag(
        question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER
    )
    print(f"rag answer: {answer}")

    question = "how to quantize using AWQ?"
    answer, relevant_docs = answer_with_rag(
        question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER
    )
    print(f"rag answer: {answer}")
