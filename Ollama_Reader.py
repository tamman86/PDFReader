import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer, BigBirdPegasusForConditionalGeneration, PegasusTokenizer, BigBirdForQuestionAnswering, AutoModelForQuestionAnswering

BASE_CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

Answer the question based on the above context: {question}
"""

def list_databases():
    if not os.path.exists(BASE_CHROMA_PATH):
        return []
    databases = []
    for title in os.listdir(BASE_CHROMA_PATH):
        title_path = os.path.join(BASE_CHROMA_PATH, title)
        if os.path.isdir(title_path):
            for revision in os.listdir(title_path):
                revision_path = os.path.join(title_path, revision)
                if os.path.isdir(revision_path):
                    databases.append(f"{title}/{revision}")
    return databases

def format_sources(sources):
    formatted_sources = [
        f"{os.path.basename(source).split(' - ')[0]} - Page {source.split('(Page ')[-1].rstrip(')')}"
        if "(Page " in source else "Unknown Page" for source in sources
    ]
    return "\n".join(f"{i + 1}. {src}" for i, src in enumerate(formatted_sources))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--database", type=str, choices=list_databases() + ["All"], default="All",
                        help="Select a specific database to query or choose 'All' to query all.")
    parser.add_argument("--model", type=str, choices=["GPT-2", "Zephyr", "Roberta-Base", "Bert-Large-Squad"], default="GPT-2",
                        help="Choose the transformer model to generate responses.")
    args = parser.parse_args()
    query_text = args.query_text
    selected_db = args.database
    selected_model = args.model

    embedding_function = HuggingFaceEmbeddings(model_name="local_models/all-mpnet-base-v2")
    databases = list_databases()
    if not databases:
        print("No databases found")
        return

    if selected_db == "All":
        dbs = [Chroma(persist_directory=os.path.join(BASE_CHROMA_PATH, db), embedding_function=embedding_function) for db in databases]
        print(f"Querying ALL databases: {databases}")
    else:
        db_path = os.path.join(BASE_CHROMA_PATH, selected_db)
        if not os.path.exists(db_path):
            print(f"Database '{selected_db}' not found in {BASE_CHROMA_PATH}.")
            return
        dbs = [Chroma(persist_directory=db_path, embedding_function=embedding_function)]
        print(f"Querying database: {selected_db}")

    results = []
    for db in dbs:
        results.extend(db.similarity_search_with_relevance_scores(query_text, k=3))

    results.sort(key=lambda x: x[1], reverse=True)
    if not results or results[0][1] < 0.3:
        print("Unable to find matching results.")
        return

    context_texts = []
    sources = []
    for doc, score in results:
        chunk_id = doc.metadata.get("id", "Unknown ID")
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        formatted_chunk = f"[{chunk_id}] {doc.page_content}"
        context_texts.append(formatted_chunk)
        sources.append(f"{source} (Page {page})")

    context_text = "\n\n---\n\n".join(context_texts)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    if selected_model == "GPT-2":
        model_path = "local_models/gpt2"

        pipeline_model = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True),
        )

        response_text = pipeline_model(
            prompt,
            max_new_tokens=100,  # ✅ Controls output length separately
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            truncation=True  # ✅ Ensures input is properly truncated if needed
        )[0]["generated_text"]


    elif selected_model == "Zephyr":
        model_path = "local_models/zephyr-7b-beta"

        pipeline_model = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map="auto"),
            tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True),
        )

        response_text = pipeline_model(
            prompt,
            max_new_tokens=100,  # ✅ Controls how many new tokens are generated
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            truncation=True  # ✅ Ensures input is properly truncated if needed
        )[0]['generated_text']

    elif selected_model == "Roberta-Base":
        model_path = "local_models/roberta-base-squad2"

        pipeline_model = pipeline(
            "question-answering",
            model=AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True),
        )

        response_text = pipeline_model(question=query_text, context=context_text)["answer"]

    elif selected_model == "Bert-Large-Squad":
        model_path = "local_models/bert-large-squad"

        pipeline_model = pipeline(
            "question-answering",
            model=AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True),
        )

        response_text = pipeline_model(question=query_text, context=context_text)["answer"]

    formatted_sources = format_sources(sources)

    formatted_response = f"""
Answer:
{response_text}

Sources:
{formatted_sources}
"""
    print(formatted_response)

if __name__ == "__main__":
    main()
