"""LangGraph Agent"""
import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a: first int
        b: second int
    """
    return a - b


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.

    Args:
        a: first int
        b: second int
    """
    return a % b


@tool
def wiki_search(query: str) -> dict:
    """Search Wikipedia for a query and return maximum 2 results.

    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


@tool
def web_search(query: str) -> dict:
    """Search Tavily for a query and return maximum 3 results,
    formatted with source URL, title, and content.

    Args:
        query: The search query.
    """

    tavily_tool = TavilySearchResults(max_results=3)

    # 'search_docs' is expected to be a list of dictionaries based on your sample.
    # Each dictionary contains keys like 'url', 'content', 'title'.
    search_docs = tavily_tool.invoke(query)

    final_formatted_docs = []

    if isinstance(search_docs, list):
        for doc_dict in search_docs:  # Iterate through the list of result dictionaries
            if isinstance(doc_dict, dict):
                # Extract data using dictionary keys found in your sample:
                source_url = doc_dict.get(
                    "url",
                    "N/A"
                    )  # From your sample, e.g., 'https://www.biblegateway.com/...'
                page_content = doc_dict.get(
                    "content",
                    ""
                    )  # From your sample, e.g., '8\xa0When the king’s order...'
                title = doc_dict.get(
                    "title",
                    "No Title Provided"
                    )  # From your sample, e.g., 'Esther 1-10 NIV...'

                # Format the output string including source, title, and content
                final_formatted_docs.append(
                    f'<Document source="{source_url}" title="{title}"/>\n{page_content}\n</Document>'
                )
            else:
                # This handles cases where an item in the list returned by Tavily might not be a dictionary.
                print(
                    f"[web_search_DEBUG] Expected a dictionary in search_docs list, but got {type(doc_dict)}: {str(doc_dict)[:100]}"
                    )
    elif isinstance(search_docs, str):
        # This handles cases where the Tavily tool might return a single string (e.g., an error message)
        print(
            f"[web_search_DEBUG] Tavily search returned a string, possibly an error: {search_docs}"
            )
        final_formatted_docs.append(
            f'<Document source="Error" title="Error"/>\n{search_docs}\n</Document>'
        )
    else:
        # This handles any other unexpected types for search_docs
        print(
            f"[web_search_DEBUG] Expected search_docs to be a list or string, but got {type(search_docs)}. Output may be empty."
            )

    joined_formatted_docs = "\n\n---\n\n".join(final_formatted_docs)

    return {"web_results": joined_formatted_docs}


@tool
def arvix_search(query: str) -> dict:
    """Search Arxiv for a query and return maximum 3 result.

    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()

    # print(f"[arvix_search_DEBUG] ArxivLoader found {len(search_docs)} documents.")

    processed_docs_str_list = []
    for i, doc in enumerate(search_docs):
        # print(f"\n--- [arvix_search_DEBUG] Document {i+1} ---")
        # print(f"Metadata: {doc.metadata}")
        # print(f"Page Content (first 200 chars): {doc.page_content[:200]}...")
        # print(f"--- End Debug for Document {i+1} ---\n")

        # Your original logic to format the document (with the fix for 'source')
        title = doc.metadata.get("Title", "N/A")
        published = doc.metadata.get(
            "Published",
            "N/A"
            )  # 'page' might often be empty for ArxivLoader results
        # content_snippet = doc.page_content[:3000]
        content_snippet = doc.page_content

        formatted_doc_str = f'<Document title="{title}" published="{published}"/>\n{content_snippet}\n</Document>'
        processed_docs_str_list.append(formatted_doc_str)

    formatted_search_results = "\n\n---\n\n".join(processed_docs_str_list)

    # print(f"[arvix_search_DEBUG] Returning: {{\"arvix_results\": \"{formatted_search_results[:100]}...\"}}")

    return {"arvix_results": formatted_search_results}


@tool
def similar_question_search(question: str) -> dict:
    """Search the vector database for similar questions and return the first results.

    Args:
        question: the question human provided."""
    matched_docs = vector_store.similarity_search(question, 3)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in matched_docs
        ]
    )
    return {"similar_questions": formatted_search_docs}


# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

# build a retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_SERVICE_KEY"))
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="question_retriever",
    description="A tool to retrieve similar questions from a vector store.",
)

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
    similar_question_search,
]

# Build graph function
def build_graph(provider: str = "google"):
    """Build the graph"""
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
    # elif provider == "groq":
    #     # Groq https://console.groq.com/docs/models
    #     llm = ChatGroq(model="qwen-qwq-32b", temperature=0) # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        # TODO: Add huggingface endpoint
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()

# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph(provider="google")
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
