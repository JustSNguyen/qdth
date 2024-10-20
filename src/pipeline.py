from typing import List
from langchain_core.documents.base import Document
from langchain_community.document_loaders import UnstructuredRTFLoader, TextLoader
import os 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from sklearn.metrics import accuracy_score
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

#######################################
# Loading the documents 
#######################################

# Test Documents (for making sure the model is working positive cases)

test_langchain_docs = []
test_document_type = []
folder_path = "/Users/kelley/qdth/TrainTestData/Relevant_Positive"
for filename in os.listdir(folder_path): # Finds every file in the folder
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path) # Loads a txt file (RTF Loader is always works )
    docs = loader.load()
    test_langchain_docs.append(docs[0])
    test_document_type.append(filename.split(".")[0])

# Magazine articles 

document_type = []
magazine_langchain_docs = []

folder_path = "/Users/kelley/qdth/FullData/Magazines_Journals2014-Present1-500byRelevence_cleaned"

# Loop through each document and load it into a langchain document 
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        magazine_langchain_docs.append(docs[0])
        document_type.append("Magazine")

folder_path = "/Users/kelley/qdth/FullData/Magazines_Journals2014-Present501-808byRelevence_cleaned"
# Loop through each document and load it into a langchain document 
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        magazine_langchain_docs.append(docs[0])
        document_type.append("Magazine")

# Newpaper

newspaper_langchain_docs = []
folder_path = "/Users/kelley/qdth/FullData/Magazines_Journals2014-Present1-500byRelevence_cleaned"
# Loop through each document and load it into a langchain document  
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        newspaper_langchain_docs.append(docs[0])
        document_type.append("Newspaper")

folder_path = "/Users/kelley/qdth/FullData/Newspapers2014-Present501-1000byRelevence_cleaned"
# Loop through each document and load it into a langchain document  
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        newspaper_langchain_docs.append(docs[0])
        document_type.append("Newspaper")

# News Transcripts

news_transcript_langchain_docs = []
folder_path = "/Users/kelley/qdth/FullData/NewsTranscripts2014-Present1-500byRelevence_cleaned"
# Loop through each document and load it into a langchain document  
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        news_transcript_langchain_docs.append(docs[0])
        document_type.append("News Transcript")

folder_path = "/Users/kelley/qdth/FullData/NewsTranscripts2014-Present501-1000byRelevence_cleaned"
# Loop through each document and load it into a langchain document  
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        news_transcript_langchain_docs.append(docs[0])
        document_type.append("News Transcript")
# Related with example 

# Newswires

newswire_langchain_docs = []
folder_path = "/Users/kelley/qdth/FullData/Newswires_Press_Releases2014-Present1-500byRelevence_cleaned"
# Loop through each document and load it into a langchain document  
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        newswire_langchain_docs.append(docs[0])
        document_type.append("Newswire")

folder_path = "/Users/kelley/qdth/FullData/Newswires_Press_Releases2014-Present501-1000byRelevence_cleaned-1"
# Loop through each document and load it into a langchain document  
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        newswire_langchain_docs.append(docs[0])
        document_type.append("Newswire")

# Web Based

web_langchain_docs = []
folder_path = "/Users/kelley/qdth/FullData/Web_based_2014-Present1-500byRelevence_cleaned"
# Loop through each document and load it into a langchain document  
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    loader = TextLoader(file_path)
    docs = loader.load()
    if len(docs[0].page_content) > 0:
        web_langchain_docs.append(docs[0])
        document_type.append("Web Based")

langchain_docs = magazine_langchain_docs + newspaper_langchain_docs + news_transcript_langchain_docs + newswire_langchain_docs + web_langchain_docs
print(len(langchain_docs))
print(len(document_type))

if len(langchain_docs) != len(document_type):
    print("Error: The length of langchain_docs is not equal to the length of document_type.")
    raise ValueError("Mismatch in lengths of langchain_docs and document_type")

filtered_langchain_docs = langchain_docs[:400]
filtered_document_type = document_type[:400]

# Embeddings for Retrieval Augmented Generation (RAG) Testing

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))

#######################################
# Classifying the documents 
#######################################

# Structured outputs - https://python.langchain.com/v0.1/docs/modules/model_io/chat/structured_output/

# LLMs really like structure and tend to be more accurate when you "program" them, so know we can get back a class with variables that we can use later without parsing the output. 

class ClassificationResponse(BaseModel):
    """Class to parse the output of the LLM"""
    about_pharmaceutical_refusals: str = Field(description="""Answer with 'True' if it is about this topic or 'False' if it is not.If it is about pharmaceutical refusals, but does not meet all three conditions, answer 'Unclear'.""") #  but does not meet all three conditions, answer 'Unclear'.
    additional_information: str = Field(description="Any extra context or details about the classification.")

#######################################
# Rag Chain for Classification
#######################################

def split_into_chunks(doc: Document, chunk_size: int = 200, chunk_overlap: int = 40) -> List[Document]:
    """Take in a document, split it into chunks of a specified size and overlap."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents([doc])
    return chunks

def run_rag_chain(doc: Document, llm: ChatOpenAI, embeddings: OpenAIEmbeddings, chat_prompt:ChatPromptTemplate) -> ClassificationResponse:
    """Take in a document, use a language model, embeddings, and chat prompt and return a retrieval based classification response.
    Accepts:
        doc (Document): The langchain document to classify. Contains the text content to be analyzed.
        llm (ChatOpenAI): The language model to use for classification. Configured for structured output.
        chat_prompt (ChatPromptTemplate): The chat prompt template to use for generating the classification question.
        embeddings (OpenAIEmbeddings): The embeddings to use for the vector store.
    Returns:
        Response: A Structured Response class defined with pydantic, containing:
            - true_or_false (str): "True" if the document relates to pharmacy refusals, "False" otherwise.
            - additional_information (str): Any extra context or details about the classification.

    Description:
    This function takes a document, uses semantic search and keyword search to classify the document, and returns a structured response.
    """
    print(doc.page_content)
    chunks = split_into_chunks(doc)
    vector_store = FAISS.from_documents(chunks, embeddings) # for semantic search (Vector Search)
    vector_store.k = 10
    vector_retriever = vector_store.as_retriever()
    rag_chain = RunnableSequence({"question": RunnablePassthrough(), "source": vector_retriever} | chat_prompt | llm)
    response = rag_chain.invoke(f"")
    return response

#######################################
# Regular LLM for Classification
#######################################

def run_structured_llm(doc: Document, llm: ChatOpenAI, chat_prompt:ChatPromptTemplate) -> ClassificationResponse:
    """Take in a document, use a language model and chat prompt and return a structured response.
    Accepts:
        doc (Document): The langchain document to classify. Contains the text content to be analyzed.
        llm (ChatOpenAI): The language model to use for classification. Configured for structured output.
        chat_prompt (ChatPromptTemplate): The chat prompt template to use for generating the classification question.

    Returns:
        Response: A Structured Response class defined with pydantic, containing:
            - true_or_false (str): "True" if the document relates to pharmacy refusals, "False" otherwise.
            - additional_information (str): Any extra context or details about the classification.

    Description:
    This function takes a document, uses a specified language model and chat prompt to analyze
    the document's content, and determines whether it's related to pharmacy refusals.
    The result is returned in a structured format for easy parsing and further processing.
    """
    chatting_chain = RunnableSequence({"question": RunnablePassthrough(), "source": lambda x: doc.page_content} | chat_prompt | llm)
    response = chatting_chain.invoke("")
    return response

# Run the model for a single document

classify_chat_prompt = ChatPromptTemplate.from_template(template="""
    We are searching for specific examples of pharmaceutical refusals, or the refusal to fulfill a prescription medication at a pharmacy by a pharmacist based on religious or moral objections. Our current corpus contains news articles or legal cases.
    To qualify as a specific example of a pharmaceutical refusal, the news article or legal case must have all three conditions:
    1. Involve a specific person who was refused a prescription at a pharmacy (name not necessary).
    2. Mention the drug or type of drug that was refused (e.g. emergency contraception, birth control, abortion medication, hormones, HIV medication, etc.).
    3. State that the refusal was based on moral or religious grounds. It can also relate to an alternative conscientious objection.
    Based on these conditions, read each of the attached documents and determine if it mentions specific instances of prescriptions being refused on moral or religious grounds.    
    Answer based on the following document:{source} Do not include any other information in your answer.
    {question}""")
openai_model = ChatOpenAI(model = "gpt-4o", api_key = os.getenv("OPENAI_API_KEY"))
classify_structured_llm = openai_model.with_structured_output(ClassificationResponse)
response = run_structured_llm(test_langchain_docs[0], classify_structured_llm, classify_chat_prompt)
print(response)

predicted_pharmaceutical_refusals = []
document_names = []
for i, doc in enumerate(filtered_langchain_docs):
    print(len(doc.page_content))
    response = run_rag_chain(doc, classify_structured_llm, classify_chat_prompt)
    predicted_pharmaceutical_refusals.append(response.about_pharmaceutical_refusals)
    document_names.append(f'{doc.metadata["source"].split("/")[-2]}/{doc.metadata["source"].split("/")[-1]}')

classify_df = pd.DataFrame({"document_name": document_names, "document_type": filtered_document_type, "Pharmaceutical Refusal": predicted_pharmaceutical_refusals})
classify_df

#######################################
# Extracting the information from the documents 
#######################################

# Structured Output Blueprint  

class ExtractionResponse(BaseModel):
    """Class to parse the output of the LLM"""
    date: str = Field(description="The date when the incident occurred. Only list the date if it refers to when a specific pharmacist refused a prescription, not legal case timelines or rulings, or the date the article was published or uploaded. Answer with None if not mentioned.")
    location: str = Field(description="The state, city, or county where a specific pharmacist refused a prescription. Answer with None if not mentioned.")
    pharmacy_name: str = Field(description="The pharmacy that originally refused the medication. Answer with None if not mentioned.")
    drug_or_classification: str = Field(description="The drug, item, or broad drug category that was refused. Answer with None if not mentioned.")
    patient_name: str = Field(description="The name of the patient who was refused medication. Answer with None if not mentioned.")
    patient_demographics: str = Field(description="The demographics of the patient (e.g. Age, Race, Gender, Sexuality, etc.). Answer with None if not mentioned.")
    refusal_reason: str = Field(description="The reason the pharmacist refused to provide the desired medication. Answer with None if not mentioned.")
    patient_outcome: str = Field(description="The outcome for the patient. Did they eventually receive the drug? If yes, indicate if it was the same pharmacy or a different one. Answer with None if not mentioned.")
    pharmacist_outcome: str = Field(description="The outcome for the pharmacist. Was legal action brought against the pharmacist or pharmacy, and if so, what was the result? Answer with None if not mentioned.")
    news_source: str = Field(description="Where the story was reported (name of newspaper, publication, headline, and date published). Answer with None if not mentioned.")
    additional_information: str = Field(description="Any important additional information about the refusal.")

extraction_structured_llm = openai_model.with_structured_output(ExtractionResponse)
extraction_chat_prompt = ChatPromptTemplate.from_template(template="""Answer the following questions based on the following document:{source}. From the document, which clarifies specific instances of pharmaceutical refusals based upon moral or religious grounds, extract the following information. If the information is not available, return 'None'""")

# Run the extraction process for all pharmaceutical refusals documents

extraction_responses = []
date = []
location = []
pharmacy_name = []
drug_or_classification = []
patient_name = []
patient_demographics = []
refusal_reason = []
patient_outcome = []
pharmacist_outcome = []
news_source = []
additional_information = []
for i, doc in enumerate(filtered_langchain_docs):
    bool = predicted_pharmaceutical_refusals[i]
    if bool == "True":
        response = run_structured_llm(doc, extraction_structured_llm, extraction_chat_prompt)
        date.append(response.date)
        location.append(response.location)
        pharmacy_name.append(response.pharmacy_name)
        drug_or_classification.append(response.drug_or_classification)
        patient_name.append(response.patient_name)
        patient_demographics.append(response.patient_demographics)
        refusal_reason.append(response.refusal_reason)
        patient_outcome.append(response.patient_outcome)
        pharmacist_outcome.append(response.pharmacist_outcome)
        news_source.append(response.news_source)
        additional_information.append(response.additional_information)
    else:
        date.append("None")
        location.append("None")
        pharmacy_name.append("None")
        drug_or_classification.append("None")
        patient_name.append("None")
        patient_demographics.append("None")
        refusal_reason.append("None")
        patient_outcome.append("None")
        pharmacist_outcome.append("None")
        news_source.append("None")
        additional_information.append("None")

#######################################
# Saving info to a dataframe and exporting it 
#######################################

extraction_df = pd.DataFrame({
    "date": date,
    "location": location,
    "pharmacy_name": pharmacy_name,
    "drug_or_classification": drug_or_classification,
    "patient_name": patient_name,
    "patient_demographics": patient_demographics,
    "refusal_reason": refusal_reason,
    "patient_outcome": patient_outcome,
    "pharmacist_outcome": pharmacist_outcome,
    "news_source": news_source,
    "additional_information": additional_information
})
# Add new columns from the earlier DataFrame
final_df = pd.concat([classify_df, extraction_df], axis=1)

final_df.to_csv("Classification_and_Extraction_Final_Documents_Full.csv", index=False)
## Filtering for only the pharmaceutical refusals 

final_df[final_df['Pharmaceutical Refusal'] == "True"]

final_df.to_csv("Classification_and_Extraction_Final_Documents_Filtered.csv", index=False)