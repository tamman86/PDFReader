import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForQuestionAnswering
from langchain_community.llms import LlamaCpp
from sentence_transformers.cross_encoder import CrossEncoder
import torch
import time
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

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
    "retrieval_k": 60,
    "top_k_results": 5,
    "hybrid_rrf_k": 60          # Reciprocal Rank Fusion Constant "k"
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


def get_database_description(db_name: str) -> str:
    base_path = CONFIG["chroma_base_path"]
    description_path = os.path.join(base_path, db_name, "description.txt")

    if os.path.exists(description_path):
        try:
            with open(description_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return ""               # Return empty if file is unreadable
    return ""                       # Return empty if no file exists

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

            use_GPU = True
            if use_GPU:
                self._embedding_function = HuggingFaceEmbeddings(
                    model_name = self.config["embedding_model"],
                    encode_kwargs = {'normalize_embeddings': True}
                )
            else:
                model_kwargs = {'device': 'cpu'}
                encode_kwargs = {'normalize_embeddings': True}

                self._embedding_function = HuggingFaceEmbeddings(
                    model_name=self.config["embedding_model"],
                    model_kwargs = model_kwargs,
                    encode_kwargs = encode_kwargs
                )
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

    # Clear cached Embedder, Extractor, and Re-Ranker from GPU
    def _clear_cached_embedder(self):
        if self._embedding_function is None:
            return

        print("Clearing embedding model from VRAM...")

        try:
            if hasattr(self._embedding_function, "client"):
                del self._embedding_function.client
            del self._embedding_function

        except AttributeError:
            try:
                del self._embedding_function
            except Exception:
                pass
        except Exception as e:
            pass

        self._embedding_function = None
        torch.cuda.empty_cache()
        print("VRAM cleared.")

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
    def answer_question(self, query_text, selected_db="All", relevance_threshold=0.5, temperature=0.7,
                        generator_name="mistral-q4", use_query_transform=False, status_callback=None):
        high_confidence_threshold = 0.8
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

            ######## Setup the way to allow user to select if embedder uses RAM or VRAM to facilitate use of this
            print("  -> Retrieval complete. Clearing embedder from VRAM...")
            self._clear_cached_embedder()

            step_num = 3 if use_query_transform else 2
            total_steps = 5 if use_query_transform else 4
            if status_callback: status_callback(f"Step {step_num}/{total_steps}: Re-ranking retrieved chunks...")

            # Step 2: Re-ranking - Ranking the pulled chunks so the generator has a priority list
            print("Step 2: Re-ranking retrieved chunks...")
            reranked_results = self._rerank_docs(query_text, retrieved_docs)

            # Safety net document suggester
            try:
                if selected_db != "All":
                    print("  -> Running 'Safety Net' check on unselected DBs...")
                    all_dbs = list_databases()
                    selected_dbs_list = selected_db.split(',')
                    unselected_dbs = [db for db in all_dbs if db not in selected_dbs_list]

                    suggestion_threshold = 0.5
                    suggestions = []

                    for db_name in unselected_dbs:
                        summary = get_database_description(db_name)
                        if not summary: continue

                        # Use the already-loaded reranker model
                        pair = [query_text, summary]
                        raw_score = self.reranker_model.predict([pair])[0]

                        prob_score = torch.sigmoid(torch.tensor(raw_score)).item()

                        if prob_score > suggestion_threshold:
                            print(f"    -> Suggestion Found! DB: '{db_name}' | Score: {prob_score:.4f}")
                            suggestions.append(db_name)

                    if suggestions:
                        yield {"type": "suggestion", "data": suggestions}

            except Exception as e:
                print(f"  -> ERROR in 'Safety Net' check: {e}")

            if not reranked_results:
                return "No relevant results found after re-ranking.", []

            best_score = reranked_results[0][1]
            confidence_level = "none"
            top_docs_with_scores = []

            # For NO confidence results (score < 0.5)
            if best_score < relevance_threshold:
                print(f"  -> Best score ({best_score:.4f}) is below low_confidence_threshold ({relevance_threshold}).")
                return "I could not find any relevant information in the documents.", []

            # For High confidence results (score > 0.8)
            elif best_score > high_confidence_threshold:
                print(f"  -> Best score ({best_score:.4f}) is HIGH confidence.")
                confidence_level = "high"

                high_confidence_docs = [
                    (doc, score) for doc, score in reranked_results
                    if score >= high_confidence_threshold
                ]
                top_docs_with_scores = high_confidence_docs[:self.config["top_k_results"]]

            # For Low confidence results (0.5 < score < 0.8)
            else:
                print(f"  -> Best score ({best_score:.4f}) is LOW confidence.")
                confidence_level = "low"

                low_confidence_docs = [
                    (doc, score) for doc, score in reranked_results
                    if score >= relevance_threshold
                ]

                top_docs_with_scores = low_confidence_docs[:self.config["top_k_results"]]

            print(f"  -> Selected {len(top_docs_with_scores)} chunks for extraction.")

            step_num += 1
            if status_callback: status_callback(f"Step {step_num}/{total_steps}: Extracting specific answers...")

            # Step 3: Extraction - Pulling the info out of the ranked spans
            print("Step 3: Extracting specific answers...")
            top_extractions = self._extract_answers(query_text, top_docs_with_scores)

            if not top_extractions:
                yield {"type": "error", "data": "Could not extract any answer spans."}
                return

            # Yield the sources first, so the GUI has them before the answer starts streaming.
            final_sources = top_extractions[:self.config["top_k_results"]]
            yield {"type": "sources", "data": final_sources}

            step_num += 1
            if status_callback: status_callback(f"Step {step_num}/{total_steps}: Generating final answer...")

            # Step 4: Generation - Stream the answer
            print("Step 4: Generating final answer...")
            # Use 'yield from' to pass all items from the _generate_answer generator
            yield from self._generate_answer(query_text, top_extractions, generator_name, temperature, confidence_level)

            if status_callback: status_callback("Done.")

        except Exception as e:
            yield {"type": "error", "data": f"An error occurred: {e}"}
        finally:
            # Clear generators when done
            self._clear_cached_generators()

    # User selects which databases the query is to be applied to. No Selection = All Databases
    def _load_databases(self, selected_db):
        dbs = []
        db_names_to_load = list_databases() if selected_db == "All" else selected_db.split(',')

        for db_name in db_names_to_load:
            db_path = os.path.join(self.config["chroma_base_path"], db_name)
            if os.path.exists(db_path):
                print(f"  -> Loading DB from: '{db_path}'")

                ### Hybrid Retrieval
                # Part 1: Loading Chroma (Dense)
                chroma_db = Chroma(persist_directory=db_path, embedding_function=self.embedding_function)

                # Part 2: Build BM25 index from Chroma (Sparse)
                print(f"  -> Building in-memory BM25 index for '{db_name}'...")
                try:
                    # Get all documents saved in the Chroma collection
                    all_docs_data = chroma_db.get(include=["documents", "metadatas"])

                    # Reconstructing full Document objects
                    corpus_docs = []
                    for i, content in enumerate(all_docs_data['documents']):
                        metadata = all_docs_data['metadatas'][i]
                        corpus_docs.append(Document(page_content=content, metadata=metadata))

                    if not corpus_docs:
                        print(f"    -> WARNING: No documents found in '{db_name}'. Skipping BM25.")
                        continue

                    # Create the tokenized corpus for BM25
                    tokenized_corpus = [doc.page_content.split(" ") for doc in corpus_docs]
                    bm25_index = BM25Okapi(tokenized_corpus)

                    # Part 3: Store Dense and Sparse results
                    dbs.append((chroma_db, bm25_index, corpus_docs))
                except Exception as e:
                    print(f"    -> ERROR: Failed to build BM25 index for '{db_name}'. Skipping. Error {e}")
            else:
                print(f"  -> WARNING: Database path not found '{db_name}'")
        return dbs

    def _retrieve_docs(self, dbs, query_text):
        RRF_K = self.config.get("hybrid_rrf_k", 60)
        retrieval_k = self.config["retrieval_k"]

        fused_results = {}
        tokenized_query = query_text.split(" ")

        # 'db' is a list of tuples now (the return of _load_databases)
        for db, bm25, corpus_docs in dbs:
            # Part 1: Chroma (Dense)
            dense_results_with_scores = db.similarity_search_with_relevance_scores(query_text, k=retrieval_k)
            dense_results = [doc for doc, score in dense_results_with_scores]

            # Part 2: BM25 (Sparse)
            sparse_scores = bm25.get_scores(tokenized_query)
            top_k_indices = np.argsort(sparse_scores)[::-1][:retrieval_k]
            sparse_results = [corpus_docs[i] for i in top_k_indices if sparse_scores[i] > 0]

            # Part 3: Reciprocal Rank Fusion (RRF)
            # Adding a score to each doc based on its rank in each list. Higher score if in both lists

            # Process dense results
            for i, doc in enumerate(dense_results):
                rank = i + 1
                rrf_score = 1.0 / (RRF_K + rank)

                doc_content = doc.page_content
                if doc_content not in fused_results:
                    fused_results[doc_content] = (doc, rrf_score)
                else:
                    # Adding score if already present
                    fused_results[doc_content] = (fused_results[doc_content][0], fused_results[doc_content][1] + rrf_score)

            # Process sparse results
            for i, doc in enumerate(sparse_results):
                rank = i + 1
                rrf_score = 1.0 / (RRF_K + rank)

                doc_content = doc.page_content
                if doc_content not in fused_results:
                    fused_results[doc_content] = (doc, rrf_score)
                else:
                    fused_results[doc_content] = (fused_results[doc_content][0], fused_results[doc_content][1] + rrf_score)

        # Sorting the fused list by RFF score
        sorted_fused_list = sorted(fused_results.values(), key=lambda x: x[1], reverse=True)

        #Return Document objects
        final_docs = [doc for doc, score in sorted_fused_list]

        print(f"  -> Hybrid retrieval: Found {len(dense_results)} dense, {len(sparse_results)} sparse. Fused to {len(final_docs)} unique docs.")

        return final_docs

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
    def _generate_answer(self, query_text, top_extractions, generator_name, temperature, confidence_level="high"):
        context_for_prompt = "\n".join(
            [f"{i + 1}. {ex['context']}" for i, ex in enumerate(top_extractions[:self.config["top_k_results"]])]
        )

        generator_prompt = ""

        ##### Low Confidence Prompts
        if confidence_level == "low":
            print("  -> Using LOW CONFIDENCE (cautious) generator prompts.")
            context_for_prompt = "\n".join(
                [f"{i + 1}. {ex['context']}" for i, ex in enumerate(top_extractions[:self.config["top_k_results"]])]
            )

            # Cautious prompt for Mistral
            if "mistral" in generator_name or "zephyr" in generator_name:
                generator_prompt = f"""You are a cautious technical assistant. The following information *may* be related to the user's question, but its relevance score is low.

        **CRITICAL RULES:**
        1. Carefully analyze the provided information and the user's question.
        2. If the information *directly* and *unambiguously* answers the question, provide the answer and cite it using `[n]`.
        3. **If the information is only vaguely related, speculative, or does not answer the core question, YOU MUST NOT use it.**
        4. In that case (Rule 3), you must respond with ONLY the following exact text:
        `I could not find a specific answer for your query in the documents.`

        **EXAMPLE (if you answer):**
        An indoor tank vent must terminate to the outside of the building [1].

        USER'S QUESTION:
        "{query_text}"

        POTENTIALLY RELEVANT INFORMATION:
        {context_for_prompt}

        Based ONLY on the information and rules above, provide a cited answer OR the exact refusal text:
        """
            # Cautious prompt for CodeLlama
            elif "codellama" in generator_name:
                generator_prompt = f"""You are a cautious code analyst. The following code snippets or procedures *may* be related to the user's question, but their relevance score is low.

        **CRITICAL RULES:**
        1. Carefully analyze the provided information and the user's question.
        2. If the information *directly* and *unambiguously* answers the question, provide the answer/code and cite it using `[n]`.
        3. **If the information is only vaguely related, speculative, or does not answer the core question, YOU MUST NOT use it.**
        4. In that case (Rule 3), you must respond with ONLY the following exact text:
        `I could not find a specific technical answer for your query in the documents.`

        USER'S QUESTION:
        "{query_text}"

        POTENTIALLY RELEVANT INFORMATION:
        {context_for_prompt}

        Based ONLY on the information and rules above, provide a cited answer OR the exact refusal text:
        """

        ##### High Confidence Prompt
        else:
            print("  -> Using HIGH CONFIDENCE (expert) generator prompts.")
            context_for_prompt = "\n".join(
                [f"{i + 1}. {ex['context']}" for i, ex in enumerate(top_extractions[:self.config["top_k_results"]])]
            )

            if generator_name == "mistral-q4" or generator_name == "mistral-q3":
                generator_prompt = f"""You are an expert technical assistant. Your task is to answer the user's question directly, then provide supporting details.

        **CRITICAL RULES:**
        1. **ANSWER STRUCTURE:** You MUST start with the direct, concise answer to the user's question in the first sentence. After the direct answer, provide the supporting details, reasoning, or context.
        2. **CITATIONS:** You MUST cite the information you use.
        3. **FORMAT:** To cite, you MUST use the format `[n]` where `n` corresponds to the number from the RELEVANT EXTRACTED INFORMATION list.
        4. Every piece of information in your answer must be followed by its citation.
        5. If multiple pieces of information come from the same source, cite it each time.
        6. **SCOPE:** Base your answer ONLY on the information provided below. Do not use any outside knowledge.

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
        1. **ANSWER STRUCTURE:** You MUST start with the direct, concise answer to the user's question in the first sentence. After the direct answer, provide the supporting explanation and context.
        2. **CITATIONS:** You MUST cite the information you use.
        3. **FORMAT:** To cite, you MUST use the format `[n]` where `n` corresponds to the number from the RELEVANT EXTRACTED INFORMATION list.
        4. Every piece of information in your answer must be followed by its citation.
        5. If multiple pieces of information come from the same source, cite it each time.
        6. **SCOPE:** Base your answer ONLY on the information provided below. Do not use any outside knowledge.

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
        1. **ANSWER STRUCTURE:** You MUST start with the direct, concise answer to the user's question in the first sentence. After the direct answer, provide the supporting explanation and context.
        2. **CITATIONS:** You MUST cite the information you use.
        3. **FORMAT:** To cite, you MUST use the format `[n]` where `n` corresponds to the number from the RELEVANT EXTRACTED INFORMATION list.
        4. Every piece of information in your answer must be followed by its citation.
        5. If multiple pieces of information come from the same source, cite it each time.
        6. **SCOPE:** Base your answer ONLY on the information provided below. Do not use any outside knowledge.

        **EXAMPLE:**
        An indoor tank vent must terminate to the outside of the building [1]. This is a critical safety measure [2].

        USER'S QUESTION:
        "{query_text}"

        RELEVANT EXTRACTED INFORMATION:
        {context_for_prompt}

        Based ONLY on the information provided, explain the code or logic in detail with citations. If appropriate, provide a clear, commented code example.
        """
        if not generator_prompt:
            generator_prompt = context_for_prompt  # Fallback to just context

        print("\n🧠 Prompt to Generator:\n", generator_prompt)
        llm = self.get_generator_llm(generator_name)
        llm.temperature = temperature

        # Now a stream is used in place of a soild block of text response
        for token in llm.stream(generator_prompt):
            yield {"type": "token", "data": token}

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
