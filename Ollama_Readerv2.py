import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering
from langchain_community.llms import LlamaCpp
from sentence_transformers.cross_encoder import CrossEncoder
import torch
import time

# Program config settings
CONFIG = {
    "chroma_base_path": "chroma",
    "embedding_model": "local_models/bge-large-en-v1.5",
    "reranker_model": "local_models/bge-reranker-large",
    "extractor_models": {
        #"bert-squad": "local_models/bert-large-squad",
        "roberta-squad2": "local_models/roberta-base-squad2",
    },
    "generator_models": {
        "mistral-q4": {
            "path": "local_models/mistral-7b-instruct-v0.2-gguf",
            "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "type": "gguf"
        },
        "mistral-q3": {
            "path": "local_models/mistral-7b-instruct-v0.2-gguf",
            "filename": "mistral-7b-instruct-v0.2.Q3_K_M.gguf",
            "type": "gguf"
        },
        "zephyr-q4": {
            "path": "local_models/zephyr-7b-beta-gguf",
            "filename": "zephyr-7b-beta.Q4_K_M.gguf",
            "type": "gguf"
        },
        "codellama-q4": {
            "path": "local_models/codellama-7b-instruct-gguf",
            "filename": "codellama-7b-instruct.Q4_K_M.gguf",
            "type": "gguf"
        }
    },
    "query_transformer_model": "mistral-q4",
    "extractor_confidence_threshold": 0.05,
    "retrieval_k": 20,
    "top_k_results": 3
}

# Display all embedded databases
def list_databases():
    base_path = CONFIG["chroma_base_path"]
    if not os.path.exists(base_path):
        return []
    databases = []
    for title in os.listdir(base_path):
        title_path = os.path.join(base_path, title)
        if os.path.isdir(title_path):
            for revision in os.listdir(title_path):
                revision_path = os.path.join(title_path, revision)
                if os.path.isdir(revision_path):
                    databases.append(f"{title}/{revision}")
    return databases


# Class used for loading models and executing queries
class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self._embedding_function = None
        self._reranker_model = None
        self._extractor_pipelines = {}
        self._generator_llms = {}

    @property
    def embedding_function(self):
        # Specify embedding function
        if self._embedding_function is None:
            print(f"Loading embedding model: {self.config['embedding_model']}...")
            self._embedding_function = HuggingFaceEmbeddings(model_name=self.config["embedding_model"])
        return self._embedding_function

    @property
    def reranker_model(self):
        # Specify reranker model
        if self._reranker_model is None:
            print(f"Loading re-ranker model: {self.config['reranker_model']}...")
            self._reranker_model = CrossEncoder(self.config["reranker_model"], max_length=512)
        return self._reranker_model

    def get_extractor_pipeline(self, model_name):
        # Specify extractor model
        if model_name not in self._extractor_pipelines:
            print(f"Loading extractor model: {model_name}...")
            model_path = self.config["extractor_models"][model_name]
            self._extractor_pipelines[model_name] = pipeline("question-answering", model=model_path,
                                                             tokenizer=model_path, local_files_only=True)
        return self._extractor_pipelines[model_name]

    # Clear cached Extractor and Re-Ranker from GPU
    def _clear_cached_specialists(self):
        if self._reranker_model is None and not self._extractor_pipelines:
            return

        print("Clearing specialist models (re-ranker, extractor) from VRAM...")
        self._reranker_model = None
        self._extractor_pipelines.clear()

        torch.cuda.empty_cache()
        print("VRAM cleared.")

    def _clear_cached_generators(self):
        if not self._generator_llms:
            return

        print("Clearing cached generator model from VRAM...")
        keys = list(self._generator_llms.keys())
        for key in keys:
            del self._generator_llms[key]

        torch.cuda.empty_cache()
        print("VRAM cleared.")

    def get_generator_llm(self, model_name):
        # Eject the specialists BEFORE loading the generator.
        self._clear_cached_specialists()

        if model_name not in self._generator_llms:
            # Eject any other model that might be loaded.
            self._clear_cached_generators()

            model_info = self.config["generator_models"][model_name]
            print(f"Loading generator model for the first time: {model_name}...")

            if model_info["type"] == "gguf":
                gguf_path = model_info["path"]
                gguf_filename = model_info["filename"]
                gguf_file_path = os.path.join(gguf_path, gguf_filename)
                if not os.path.exists(gguf_file_path):
                    raise RuntimeError(f"Model file not found: {gguf_file_path}")

                llm = LlamaCpp(
                    model_path=gguf_file_path, n_gpu_layers=-1, n_batch=512,
                    n_ctx=4096, max_tokens=512, verbose=False
                )
                self._generator_llms[model_name] = llm
            else:
                raise NotImplementedError(f"Generator type '{model_info['type']}' not implemented.")

        # Return the model from the cache
        return self._generator_llms[model_name]

    # User option to use LLM to help fine tune their query
    def _transform_query(self, query_text: str, status_callback=None) -> str:
        if status_callback: status_callback("Step 1/5: Transforming query...")

        print("  -> Transforming query with LLM...")
        # Rules for system to follow when enhancing query
        prompt = f"""You are an expert search query creator. Your task is to take a user's question and rephrase it into a more detailed, "ideal" query that is perfectly suited for a semantic search against a technical knowledge base.

RULES:
1. PRESERVE THE CORE INTENT: The transformed query's meaning must be identical to the original query.
2. DO NOT ADD NEW CONCEPTS: Do not introduce any new keywords, topics, or entities that are not explicitly present in the original query.
3. EXPAND ON EXISTING CONCEPTS: Rephrase by adding detail and context based only on the words already in the query.
4. OUTPUT ONLY THE QUERY: Your entire response must be only the new search query, with no preamble or explanation.
5. IF THE QUERY IS ALREADY GOOD, DO NOT CHANGE IT: If the user's query is already specific and detailed, simply return it exactly as it is.

User's Query:
"{query_text}"

Ideal Search Query:
"""
        # Use the configured transformer model with a very low temperature for deterministic results
        transformer_llm = self.get_generator_llm(self.config["query_transformer_model"])
        transformer_llm.temperature = 0.01
        transformed_query = transformer_llm.invoke(prompt).strip()

        print(f"  -> Original Query:    '{query_text}'")
        print(f"  -> Transformed Query: '{transformed_query}'")
        return transformed_query

    # Main engine to find relevant spans based on prompts and generate a Natural Language response
    def answer_question(self, query_text, selected_db="All", relevance_threshold=0.3, temperature=0.7,
                        generator_name="mistral-q4", use_query_transform=False, status_callback=None):

        if status_callback: status_callback("Starting query process...")

        try:
            # Step 1: Retrieval - Pull the most relevant chunks which apply to the query
            print("Step 1: Retrieving document chunks...")
            dbs = self._load_databases(selected_db)
            if not dbs: return "Error: Could not load any valid databases.", []

            # If the user decides to "enhance" their query
            if use_query_transform:
                print("  -> Using hybrid query approach.")
                transformed_query = self._transform_query(query_text, status_callback)
                if status_callback: status_callback("Step 2/5: Retrieving documents (Hybrid)...")

                # Retrieve for both original and transformed queries
                original_results = self._retrieve_docs(dbs, query_text)
                transformed_results = self._retrieve_docs(dbs, transformed_query)

                # Combine and de-duplicate the results
                all_docs = {doc.page_content: doc for doc in original_results + transformed_results}
                retrieved_docs = list(all_docs.values())
            else:
                print("  -> Using direct query approach.")
                if status_callback: status_callback("Step 1/4: Retrieving documents...")
                retrieved_docs = self._retrieve_docs(dbs, query_text)

            if not retrieved_docs: return "No documents were retrieved from the database.", []

            step_num = 3 if use_query_transform else 2
            total_steps = 5 if use_query_transform else 4
            if status_callback: status_callback(f"Step {step_num}/{total_steps}: Re-ranking retrieved chunks...")

            # Step 2: Re-ranking - Ranking the pulled chunks so the generator has a priority list
            print("Step 2: Re-ranking retrieved chunks...")
            reranked_results = self._rerank_docs(query_text, retrieved_docs)
            if not reranked_results or reranked_results[0][
                1] < relevance_threshold: return "No relevant results found after re-ranking.", []
            top_docs_with_scores = reranked_results[:self.config["top_k_results"]]

            step_num += 1
            if status_callback: status_callback(f"Step {step_num}/{total_steps}: Extracting specific answers...")

            # Step 3: Extraction - Pulling the info out of the ranked spans
            print("Step 3: Extracting specific answers...")
            top_extractions = self._extract_answers(query_text, top_docs_with_scores)
            if not top_extractions: return "Could not extract any answer spans.", []

            step_num += 1
            if status_callback: status_callback(f"Step {step_num}/{total_steps}: Generating final answer...")

            # Step 4: Generation - Feed the info into the response generator with instructions
            print("Step 4: Generating final answer...")
            final_answer = self._generate_answer(query_text, top_extractions, generator_name, temperature)

            if status_callback: status_callback("Done.")
            return final_answer, top_extractions[:self.config["top_k_results"]]

        # Clear generators when done
        finally:
            self._clear_cached_generators()

    # User selects which databases the query is to be applied to. No Selection = All Databases
    def _load_databases(self, selected_db):
        dbs = []
        db_names_to_load = list_databases() if selected_db == "All" else selected_db.split(',')
        for db_name in db_names_to_load:
            db_path = os.path.join(self.config["chroma_base_path"], db_name)
            if os.path.exists(db_path):
                print(f"  -> Loading DB from: '{db_path}'")
                dbs.append(Chroma(persist_directory=db_path, embedding_function=self.embedding_function))
        return dbs

    def _retrieve_docs(self, dbs, query_text):
        results_with_scores = []
        for db in dbs:
            results_with_scores.extend(
                db.similarity_search_with_relevance_scores(query_text, k=self.config["retrieval_k"]))
        unique_docs_map = {}
        for doc, score in results_with_scores:
            if doc.page_content not in unique_docs_map or score > unique_docs_map[doc.page_content][1]:
                unique_docs_map[doc.page_content] = (doc, score)
        return [doc for doc, score in unique_docs_map.values()]

    def _rerank_docs(self, query_text, docs):
        pairs = [[query_text, doc.page_content] for doc in docs]
        reranker_scores = self.reranker_model.predict(pairs)
        reranked_results = list(zip(reranker_scores, docs))
        reranked_results.sort(key=lambda x: x[0], reverse=True)
        return [(doc, score) for score, doc in reranked_results]

    # Extract answer spans from the best documents and elaborate using surrounding context
    def _extract_answers(self, query_text, top_docs_with_scores):
        top_extractions = []

        # Context window for response enhancement
        CONTEXT_WINDOW = 100

        for model_name in self.config["extractor_models"].keys():
            print(f"  -> Using extractor: {model_name}")
            extractor = self.get_extractor_pipeline(model_name)
            for doc, chunk_score in top_docs_with_scores:
                try:
                    result = extractor(question=query_text, context=doc.page_content)

                    if result["score"] < self.config["extractor_confidence_threshold"]:
                        continue

                    # Context expansion
                    span = result["answer"]
                    full_chunk = doc.page_content

                    # Find the start and end of the span in the full chunk
                    start_index = full_chunk.find(span)
                    if start_index != -1:
                        end_index = start_index + len(span)

                        # Expand the context window
                        expanded_start = max(0, start_index - CONTEXT_WINDOW)
                        expanded_end = min(len(full_chunk), end_index + CONTEXT_WINDOW)

                        # Find full sentence boundaries for cleaner context
                        # Find the first period before the expanded start
                        start_sentence = full_chunk.rfind(". ", 0, expanded_start) + 2
                        # Find the first period after the expanded end
                        end_sentence = full_chunk.find(".", expanded_end)
                        if end_sentence == -1:
                            end_sentence = len(full_chunk)  # Go to end if no period found

                        expanded_context = full_chunk[start_sentence:end_sentence + 1].strip()
                    else:
                        expanded_context = span  # Fallback to just the span if it can't be found

                    final_score = chunk_score * result["score"]
                    top_extractions.append({
                        "span": span,  # Original span
                        "context": expanded_context,  # Expanded span
                        "final_score": final_score
                    })
                except Exception as e:
                    print(f"  - Extractor '{model_name}' failed on chunk. Error: {e}")

        top_extractions.sort(key=lambda x: x["final_score"], reverse=True)
        return top_extractions

    # Tailored generator prompt formation
    def _generate_answer(self, query_text, top_extractions, generator_name, temperature):
        context_for_prompt = "\n".join(
            [f"{i + 1}. {ex['context']}" for i, ex in enumerate(top_extractions[:self.config["top_k_results"]])]
        )

        generator_prompt = ""
        if generator_name == "mistral-q4" or generator_name == "mistral-q3":
            generator_prompt = f"""You are an expert technical assistant. Your task is to synthesize the following extracted pieces of information into a single, coherent, and precise answer.

**CRITICAL RULES:**
1. You MUST cite the information you use.
2. To cite, you MUST use the format `[n]` where `n` corresponds to the number from the RELEVANT EXTRACTED INFORMATION list.
3. Every piece of information in your answer must be followed by its citation.
4. If multiple pieces of information come from the same source, cite it each time.
5. Base your answer ONLY on the information provided below. Do not use any outside knowledge.

**EXAMPLE:**
An indoor tank vent must terminate to the outside of the building [1]. This is a critical safety measure [2].

USER'S QUESTION:
"{query_text}"

RELEVANT EXTRACTED INFORMATION:
{context_for_prompt}

Based ONLY on the information and rules above, provide a comprehensive and heavily cited answer:
"""

        elif generator_name == "zephyr-q4":
            generator_prompt = f"""You are an expert explainer. Your task is to summarize the key findings from the following extracted information and explain the main concept in clear, easy-to-understand language.

**CRITICAL RULES:**
1. You MUST cite the information you use.
2. To cite, you MUST use the format `[n]` where `n` corresponds to the number from the RELEVANT EXTRACTED INFORMATION list.
3. Every piece of information in your answer must be followed by its citation.
4. If multiple pieces of information come from the same source, cite it each time.
5. Base your answer ONLY on the information provided below. Do not use any outside knowledge.

**EXAMPLE:**
An indoor tank vent must terminate to the outside of the building [1]. This is a critical safety measure [2].

USER'S QUESTION:
"{query_text}"

RELEVANT EXTRACTED INFORMATION:
{context_for_prompt}

Based ONLY on the information provided, summarize the relevant points and provide a clear explanation that directly answers the user's question with citations.
"""

        elif generator_name == "codellama-q4":
            generator_prompt = f"""You are a senior software engineer and code analyst. Your task is to analyze the following extracted information to answer the user's question, which may be about code, logic, or technical procedures.

**CRITICAL RULES:**
1. You MUST cite the information you use.
2. To cite, you MUST use the format `[n]` where `n` corresponds to the number from the RELEVANT EXTRACTED INFORMATION list.
3. Every piece of information in your answer must be followed by its citation.
4. If multiple pieces of information come from the same source, cite it each time.
5. Base your answer ONLY on the information provided below. Do not use any outside knowledge.

**EXAMPLE:**
An indoor tank vent must terminate to the outside of the building [1]. This is a critical safety measure [2].

USER'S QUESTION:
"{query_text}"

RELEVANT EXTRACTED INFORMATION:
{context_for_prompt}

Based ONLY on the information provided, explain the code or logic in detail with citations. If appropriate, provide a clear, commented code example.
"""
        else:
            generator_prompt = context_for_prompt  # Fallback to just context

        print("\n🧠 Prompt to Generator:\n", generator_prompt)
        llm = self.get_generator_llm(generator_name)
        llm.temperature = temperature
        return llm.invoke(generator_prompt)


def main():
    parser = argparse.ArgumentParser(description="Query documents using a RAG pipeline.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    pipeline = RAGPipeline(CONFIG)
    final_answer, sources = pipeline.answer_question(query_text=args.query_text)
    print("\n--- FINAL ANSWER ---")
    print(final_answer)


if __name__ == "__main__":
    main()
