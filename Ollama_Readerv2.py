import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import LlamaCpp
from chromadb.config import Settings
from langchain_community.llms import CTransformers
from sentence_transformers.cross_encoder import CrossEncoder

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
    2 = Deepseek-7b
    3 = Mistral-7b 
    4 = Mistral-7b 
    5 = Mistral-7b 
    6 = GPT-J-6B
    7 = Bertin-GPT-J-6b
'''

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

def full_model(generator_model_path, generator_prompt, temperature):
    tokenizer = AutoTokenizer.from_pretrained(generator_model_path, local_files_only=True, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(generator_model_path, local_files_only=True).to("cuda")

    inputs = tokenizer(generator_prompt, return_tensors="pt", padding = True, truncation = True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")


    print("Sampling based text generation:")
    output_ids = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_new_tokens=128,  # or whatever you choose
        temperature=temperature,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

    prompt_length = input_ids.shape[1]
    generated_text = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
    print(generated_text)
    #return generated_text

def gguf_model(generator_model_path, generator_prompt, temperature):
    """
    Loads and runs a GGUF model using the reliable llama-cpp-python library.
    """
    # Find the actual .gguf file in the model directory
    gguf_file = [f for f in os.listdir(generator_model_path) if f.endswith(".gguf")]
    if not gguf_file:
        raise RuntimeError(f"No .gguf file found in {generator_model_path}")

    gguf_file_path = os.path.join(generator_model_path, gguf_file[0])

    print(f"Loading GGUF model from: {gguf_file_path}")
    llm = LlamaCpp(
        model_path=gguf_file_path,
        # -1 means offload all possible layers to the GPU
        n_gpu_layers=-1,
        # The number of tokens to process in parallel
        n_batch=512,
        # The maximum context size the model can handle
        n_ctx=4096,
        # We pass generation parameters directly when we call the model
        temperature=temperature,
        max_tokens=512,
        # Set to True to display detailed C++-level performance info
        verbose=False,
    )
    print("Generating response with LlamaCpp (GGUF)...")
    response = llm.invoke(generator_prompt)
    return response

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

    embedding_function = HuggingFaceEmbeddings(model_name="local_models/bge-large-en-v1.5")

    databases = list_databases()
    print(f"DEBUG: list_databases() found: {databases}")   ####################################
    if not databases:
        return

    # Define the setting to disable telemetry
    chroma_settings = Settings(anonymized_telemetry=False)

    if selected_db == "All":
        # --- ADD A PRINT STATEMENT INSIDE THIS LIST COMPREHENSION (requires changing to a loop) ---
        print("DEBUG: Loading ALL databases...")
        dbs = []
        for db_name in databases:
            db_path = os.path.join(BASE_CHROMA_PATH, db_name)
            print(f"  -> Attempting to load DB from: '{db_path}'")
            dbs.append(Chroma(persist_directory=db_path, embedding_function=embedding_function))
    else:
        db_path = os.path.join(BASE_CHROMA_PATH, selected_db)
        # --- ADD THIS LINE ---
        print(f"DEBUG: Attempting to load SINGLE DB from: '{db_path}'")
        if not os.path.exists(db_path):
            print(f"DEBUG: ERROR - Path does not exist: '{db_path}'")
            return
        dbs = [Chroma(persist_directory=db_path, embedding_function=embedding_function)]

    '''
    if selected_db == "All":
        dbs = [Chroma(
            persist_directory=os.path.join(BASE_CHROMA_PATH, db),
            embedding_function=embedding_function,
            client_settings=chroma_settings
        ) for db in databases]
    else:
        db_path = os.path.join(BASE_CHROMA_PATH, selected_db)
        if not os.path.exists(db_path):
            return
        dbs = [Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function,
            client_settings=chroma_settings
        )]
    '''

    # --- ADD THIS ENTIRE BLOCK ---
    print("\nDEBUG: Verifying content of loaded databases...")
    if not dbs:
        print("  -> ERROR: The 'dbs' list is empty. No databases were loaded.")
    for i, db in enumerate(dbs):
        try:
            count = db._collection.count()
            print(f"  -> DB #{i + 1}: Contains {count} document chunks.")
            if count == 0:
                print("  -> WARNING: This database appears to be empty!")
        except Exception as e:
            print(f"  -> ERROR checking DB #{i + 1}: {e}")
    print("-" * 30)
    # --- END OF ADDED BLOCK ---

    print("Step 1: Retrieving diverse document chunks with MMR...")
    results_with_scores = []
    for db in dbs:
        # Use the direct similarity search to get a larger pool of candidates for the re-ranker.
        # We'll get 20 chunks to give the re-ranker a good selection to choose from.
        results_with_scores.extend(db.similarity_search_with_relevance_scores(query_text, k=20))

    # Remove duplicates based on page content, keeping the one with the highest score.
    unique_docs_map = {}
    for doc, score in results_with_scores:
        if doc.page_content not in unique_docs_map or score > unique_docs_map[doc.page_content][1]:
            unique_docs_map[doc.page_content] = (doc, score)

    # We only need the documents themselves for the re-ranker, not the initial scores.
    unique_docs = [doc for doc, score in unique_docs_map.values()]

    print("Step 2: Re-ranking retrieved chunks for best relevance...")
    reranker_model = CrossEncoder('local_models/bge-reranker-large', max_length=512)

    pairs = [[query_text, doc.page_content] for doc in unique_docs]

    if not pairs:
        print("No documents were retrieved. Aborting.")
        return

    reranker_scores = reranker_model.predict(pairs)

    reranked_results = list(zip(reranker_scores, unique_docs))
    reranked_results.sort(key=lambda x: x[0], reverse=True)

    # Re-format into the [(doc, score), ...] structure the rest of the script expects
    results = [(doc, score) for score, doc in reranked_results]

    if not results or results[0][1] < relevance_threshold:
        print("No relevant results found after re-ranking, or top score is below threshold.")
        return

    print("Step 3: Extracting specific answers from top 3 re-ranked chunks...")
    top_results = results[:3]
    top_extractions = []

    models = [3, 11]
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
                    print(f"Extractor {extractor_model_path} failed on chunk. Error: {e}")

        except:
            print(f"Extraction failed for: {extractor_model_path}. Error: {e}")

    top_extractions.sort(key=lambda x: x["final_score"], reverse=True)

    top_3_for_generation = top_extractions[:3]

    # If no extractions were made, you can't generate an answer.
    if not top_3_for_generation:
        print("Could not extract any specific answer spans to generate a final response.")
        return

    # List of extracted facts
    context_for_prompt = "\n".join(
        [f"{i + 1}. {extraction['span']}" for i, extraction in enumerate(top_3_for_generation)]
    )

    # A much cleaner and more direct prompt template.
    generator_prompt = f"""You are an expert technical assistant. Your task is to synthesize the following extracted pieces of information into a single, coherent answer. Do not use any information not provided below.

    USER'S QUESTION:
    "{query_text}"

    RELEVANT EXTRACTED INFORMATION:
    {context_for_prompt}

    Based ONLY on the information above, provide a comprehensive and coherent answer:
    """

    print("\n🧠 Prompt to Generator:\n", generator_prompt)

    if generator == 1:
        generator_model_path = "local_models/gpt2"
        response_text = full_model(generator_model_path, generator_prompt, temperature)
    elif generator == 2:
        generator_model_path = "local_models/deepseek-7b-gptq"
        response_text = gguf_model(generator_model_path, generator_prompt, temperature)
    elif generator == 3:
        generator_model_path = "local_models/mistral-7b-gguf"
        response_text = gguf_model(generator_model_path, generator_prompt, temperature)
    elif generator == 4:
        generator_model_path = "local_models/mistral-7b-gpo"
        response_text = gguf_model(generator_model_path, generator_prompt, temperature)
    elif generator == 5:
        generator_model_path = "local_models/mistral-7b-OpenHermes"
        response_text = gguf_model(generator_model_path, generator_prompt, temperature)
    elif generator == 6:
        generator_model_path = "local_models/gpt-j-6b"
        response_text = full_model(generator_model_path, generator_prompt, temperature)
    elif generator == 7:
        generator_model_path = "local_models/bertin-gpt-j-6b"
        response_text = gguf_model(generator_model_path, generator_prompt, temperature)

    print(f"Generator Output:\n{response_text}")



if __name__ == "__main__":
    main()