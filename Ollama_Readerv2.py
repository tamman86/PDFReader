import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import torch

'''
For testing extractor-generator combinations
Modify variables "extractor" and "generator" to use different models
For extractor:
    1 = bart-large
    2 = bert-base-uncased
    3 = bert-large-squad
    4 = bert2bert
    5 = bigbird-pegasus-large
    6 = bigbird-pegasus-large-arxiv
    7 = bigbird-roberta-base
    8 = distilbert-base-uncased
    9 = longformer-base-4096
    10 = roberta-base
    11 = roberta-base-squad2
    12 = scibert-scivocab-uncased
    
For generator:
    1 = GPT-2
    2 = GPT-J-6B
    3 = Mistral-7B
'''

generator = 1



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

    '''
    for doc, score in results:
        print(f"Chunk Score: {score}, Content: {doc.page_content[:300]}")
    '''

    if not results or results[0][1] < relevance_threshold:
        print("No relevant results extracted")
        return

    top_results = results[:3]
    top_extractions = []

    models = [3, 4, 5, 11]
    for i in models:
        extractor = i
        try:
            if extractor == 1:
                extractor_model_path = "local_models/bart-large"
                task = "question-answering"
                max_len = 1024
            elif extractor == 2:
                extractor_model_path = "local_models/bert-base-uncased"
                task = "question-answering"
                max_len = 512
            elif extractor == 3:
                extractor_model_path = "local_models/bert-large-squad"
                task = "question-answering"
                max_len = 512
            elif extractor == 4:
                extractor_model_path = "local_models/bert2bert"
                task = "text2text-generation"
                max_len = 512
            elif extractor == 5:
                extractor_model_path = "local_models/bigbird-pegasus-large"
                task = "summarization"
                max_len = 1024
            elif extractor == 6:
                extractor_model_path = "local_models/bigbird-pegasus-large-arxiv"
                task = "summarization"
                max_len = 1024
            elif extractor == 7:
                extractor_model_path = "local_models/bigbird-roberta-base"
                task = "question-answering"
                max_len = 4096
            elif extractor == 8:
                extractor_model_path = "local_models/distilbert-base-uncased"
                task = "question-answering"
                max_len = 512
            elif extractor == 9:
                extractor_model_path = "local_models/longformer-base-4096"
                task = "question-answering"
                max_len = 4096
            elif extractor == 10:
                extractor_model_path = "local_models/roberta-base"
                task = "question-answering"
                max_len = 512
            elif extractor == 11:
                extractor_model_path = "local_models/roberta-base-squad2"
                task = "question-answering"
                max_len = 512
            elif extractor == 12:
                extractor_model_path = "local_models/scibert-scivocab-uncased"
                task = "question-answering"
                max_len = 512

            print("Now using: " + extractor_model_path)

            model = AutoModelForQuestionAnswering.from_pretrained(extractor_model_path, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(extractor_model_path, local_files_only=True)
            extractor_pipeline = pipeline(task=task, model=model, tokenizer=tokenizer, max_len=max_len)

            for doc, chunk_score in top_results:
                context_text = doc.page_content

                try:
                    extractor_result = extractor_pipeline(question=query_text,
                                                          context=context_text,
                                                          padding = "max_length",
                                                          truncation = "only_second",
                                                          max_length = max_len,
                                                          stride = max_len / 2,
                                                          max_answer_len=50)

                    if isinstance(extractor_result, list):
                        extractor_result = extractor_result[0]

                    extracted_answer = extractor_result["answer"]
                    extractor_score = extractor_result["score"]
                    final_score = chunk_score * extractor_score

                    print(f"- Chunk score:     {chunk_score:.3f}")
                    print(f"- Extractor score: {extractor_score:.3f}")
                    print(f"- Final score:     {final_score:.3f}")
                    print(f"- Extracted span:  {extracted_answer}")

                    top_extractions.append({
                        "model": extractor_model_path,
                        "chunk": context_text,
                        "span": extracted_answer,
                        "chunk_score": chunk_score,
                        "extractor_score": extractor_score,
                        "final_score": final_score
                    })


                except Exception as e:
                    print(f"Extractor {extractor_model_path} failed on chunk:\n{str(e)}")

        except:
            print("Extraction failed for: " + extractor_model_path)

    top_extractions.sort(key=lambda x: x["final_score"], reverse=True)

    top_3_for_generation = top_extractions[:3]

    spans_for_prompt = "\n".join(
        [f"{i + 1}. {e['span']}" for i, e in enumerate(top_3_for_generation)]
    )

    best_chunk = top_extractions[0]["chunk"]

    generator_prompt = f"""You are an engineering assistant AI. Given a user query and extracted document spans, generate a clear and technically accurate answer suitable for an engineer.

    Context from document:
    {best_chunk}
        
    Q: {query_text}
    The following answers were extracted from relevant documents:
    {spans_for_prompt}

    A:"""

    print("\n🧠 Prompt to Generator:\n", generator_prompt)

    if generator == 1:
        generator_model_path = "local_models/gpt2"
        #tokenizer.pad_token_id = tokenizer.eos_token_id
    elif generator == 2:
        generator_model_path = "local_models/gpt-j-6b"
        #tokenizer.pad_token_id = tokenizer.eos_token_id
    elif generator == 3:
        generator_model_path = "local_models/mistral-7b"
        #if tokenizer.pad_token_id is None:
        #    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")

    tokenizer = AutoTokenizer.from_pretrained(generator_model_path, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(generator_model_path, local_files_only=True)

    input_ids = tokenizer.encode(generator_prompt, return_tensors="pt")

    for searchTechnique in range(2):
        if searchTechnique == 0:
            print("Sampling based text generation:")
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,  # or whatever you choose
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        elif searchTechnique == 1:
            print("Beam search based text generation:")
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generator Output:\n{response_text}")



    '''
    generator_pipeline = pipeline("text-generation",
                             model=model,
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
    '''

if __name__ == "__main__":
    main()