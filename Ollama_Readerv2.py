import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM

'''
For testing extractor-generator combinations
Modify variables "extractor" and "generator" to use different models
For extractor:
    1 = bert_large_squad
    2 = longformer-base-4096
    3 = DistilBERT (Very Fast)
    4 = RoBERTa (Balanced)
    5 = bart-large
    6 = bert-base-uncased
    7 = bert-large-squad
    8 = bert2bert
    9 = bigbird-pegasus-large-arxiv
    10 = bigbird-roberta-base
    11 = roberta-base-squad2
For generator:
    1 = GPT-2
    2 = GPT-J-6B
    3 = Mistral-7B
'''
extractor = 3
generator = 3



BASE_CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following extracted answer:

{extracted_answer}

Provide a well-structured response.
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
    parser.add_argument("--relevance", type=float, default=0.3, help="Relevance threshold for retrieved chunks.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature setting for GPT-2 generation.")
    args = parser.parse_args()
    query_text = args.query_text
    selected_db = args.database
    relevance_threshold = args.relevance
    temperature = args.temperature

    embedding_function = HuggingFaceEmbeddings(model_name="local_models/all-mpnet-base-v2")
    databases = list_databases()
    if not databases:
        return

    if selected_db == "All":
        dbs = [Chroma(persist_directory=os.path.join(BASE_CHROMA_PATH, db), embedding_function=embedding_function) for
               db in databases]
    else:
        db_path = os.path.join(BASE_CHROMA_PATH, selected_db)
        if not os.path.exists(db_path):
            return
        dbs = [Chroma(persist_directory=db_path, embedding_function=embedding_function)]

    results = []
    sources = []
    for db in dbs:
        search_results = db.similarity_search_with_relevance_scores(query_text, k=3)
        results.extend(search_results)
        for doc, _ in search_results:
            sources.append(doc.metadata.get("source", "Unknown source"))
    results.sort(key=lambda x: x[1], reverse=True)

    for doc, score in results:
        print(f"Chunk Score: {score}, Content: {doc.page_content[:300]}")

    if not results or results[0][1] < relevance_threshold:
        print("No relevant results extracted")
        return

    max_context_size = 3500  # Prevent exceeding model limit
    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])[:max_context_size]
    context_text = results[0][0].page_content if results else ""

    #print("Query:", query_text)
    #print("Context (first 500 chars):", context_text[:200])

    #context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    for i in range(1,12):
        extractor = i
        try:
            if extractor == 1:
                extractor_model_path = "local_models/bert-large-squad"
            elif extractor == 2:
                extractor_model_path = "local_models/longformer-base-4096"
            elif extractor == 3:
                extractor_model_path = "distilbert-base-uncased"
            elif extractor == 4:
                extractor_model_path = "roberta-base"
            elif extractor == 5:
                extractor_model_path = "bart-large"
            elif extractor == 6:
                extractor_model_path = "bert-base-uncased"
            elif extractor == 7:
                extractor_model_path = "bert-large-squad"
            elif extractor == 8:
                extractor_model_path = "bert2bert"
            elif extractor == 9:
                extractor_model_path = "bigbird-pegasus-large-arxiv"
            elif extractor == 10:
                extractor_model_path = "bigbird-roberta-base"
            elif extractor == 11:
                extractor_model_path = "roberta-base-squad2"

            extractor_pipeline = pipeline("question-answering",
                                     model=AutoModelForQuestionAnswering.from_pretrained(extractor_model_path,
                                                                                         local_files_only=True),
                                     tokenizer=AutoTokenizer.from_pretrained(extractor_model_path, local_files_only=True))

            extractor_result = extractor_pipeline(question=query_text,
                                                  context=context_text,
                                                  padding = "max_length",
                                                  truncation = True,
                                                  max_length = 4096)
                                                  #stride = 256)
            extracted_answer = extractor_result["answer"]

            print("Extracted answer for: " + extractor_model_path)
            print(extracted_answer)
            print(" ")

            if not extracted_answer.strip():
                return
        except:
            print("Extraction failed for: " + extractor_model_path)





    ################Stop for now to debug extractor
    return

    tokenizer = AutoTokenizer.from_pretrained(generator_model_path, local_files_only=True, padding_side="left")

    if generator == 1:
        generator_model_path = "local_models/gpt2"
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif generator == 2:
        generator_model_path = "local_models/gpt-j-6b"
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif generator == 3:
        generator_model_path = "local_models/mistral-7b"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")

    generator_pipeline = pipeline("text-generation",
                             model=AutoModelForCausalLM.from_pretrained(generator_model_path, local_files_only=True),
                             tokenizer=tokenizer)

    #generator_pipeline.tokenizer.pad_token_id = generator_pipeline.tokenizer.eos_token_id  # Explicitly set pad token

    # Ensure input prompt is within GPT-2 token limits
    #tokenizer = AutoTokenizer.from_pretrained(generator_model_path, local_files_only=True)
    full_prompt = PROMPT_TEMPLATE.format(extracted_answer=extracted_answer)
    encoded_prompt = tokenizer(full_prompt, return_tensors="pt")

    # Truncate input if it exceeds 4096 tokens
    MAX_INPUT_TOKENS = 4096
    if encoded_prompt["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        truncated_prompt = tokenizer.decode(encoded_prompt["input_ids"][0, -MAX_INPUT_TOKENS:],
                                            skip_special_tokens=True)
    else:
        truncated_prompt = full_prompt

    response_text = generator_pipeline(truncated_prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True,
                                  temperature=temperature)[0]["generated_text"]

    print("Answer:")
    print(response_text)
    print("\nSources:")
    print(format_sources(sources))


if __name__ == "__main__":
    main()