from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class Rag:
  def __init__(self):
    # 1. Загрузка и обработка документов
    loader = DirectoryLoader('./data', glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()

    # 2. Разделение текста на чанки
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 3. Выбор модели для эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",  # Хорошая модель
        model_kwargs={'device': 'cpu'},       # Используйте 'cuda' для GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Создание векторной базы данных
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Данные сохранятся на диск
    )
    vector_db.persist()

    # 5. Настройка локальной LLM через Ollama
    llm = Ollama(model="deepseek-llm", temperature=0)

    # 6. Создание цепочки QA с RAG
    # Настройте промпт под вашу модель и задачу
    custom_prompt = PromptTemplate(
        template=(
            "Используй следующие фрагменты контекста, чтобы ответить на вопрос. "
            "Если не знаешь ответа, просто скажи, что не знаешь. Не придумывай ничего.\n\n"
            "Контекст: {context}\n\n"
            "Вопрос: {question}\n\n"
            "Полезный ответ:"
        ),
        input_variables=["context", "question"],
    )

    self.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),  # Ищем 3 самых релевантных чанка
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
    )

  def query(self, query):
    result = self.qa_chain({"query": query})
    return result["result"]

rag = Rag()
