import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import shutil
import re
import time
from dotenv import load_dotenv

# --- Backend Imports ---
from Ollama_DB import DatabaseBuilder, CONFIG as DB_CONFIG
from Ollama_Readerv2 import RAGPipeline, CONFIG as READER_CONFIG, list_databases
from Download_Model import ModelDownloader, MODELS_TO_DOWNLOAD, LOCAL_MODEL_DIR

# --- Configuration ---
os.environ['HF_HUB_OFFLINE'] = '1'  # Keep HuggingFace offline
load_dotenv()
HuggingFaceToken = os.getenv("HUGGINGFACE_TOKEN")

# Set the Theme
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class ModernToolTip:

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

    def show(self, event):
        x = event.x_root + 20
        y = event.y_root + 10

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Dark mode style for the tooltip
        label = tk.Label(self.tooltip_window, text=self.text, justify=tk.LEFT,
                         background="#2B2B2B", foreground="#FFFFFF",
                         relief=tk.SOLID, borderwidth=1,
                         wraplength=600, font=("Roboto", 12))
        label.pack(ipadx=10, ipady=10)

    def hide(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None


class DocumentDatabaseGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("AI Document Assistant")
        self.geometry("1100x850")

        # Initialize Backend Pipelines
        print("Initializing backend pipelines...")
        self.db_builder = DatabaseBuilder(DB_CONFIG)
        self.rag_pipeline = RAGPipeline(READER_CONFIG)
        self.model_downloader = ModelDownloader(MODELS_TO_DOWNLOAD, LOCAL_MODEL_DIR, HuggingFaceToken)
        print("✅ Pipelines initialized.")

        self.current_sources = []
        self.database_vars = []

        # Layout Configuration (2 Columns)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ============================
        # LEFT SIDEBAR (Setup & Tools)
        # ============================
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)  # Spacing filler

        # App Title in Sidebar
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="DocuMind AI", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # --- Section: Model Setup ---
        self.dl_btn = ctk.CTkButton(self.sidebar_frame, text="Check/Download Models", command=self.download_all_models,
                                    fg_color="#2E86C1")
        self.dl_btn.grid(row=1, column=0, padx=20, pady=10)

        # --- Section: Create Database ---
        self.create_label = ctk.CTkLabel(self.sidebar_frame, text="Create New Database", anchor="w")
        self.create_label.grid(row=2, column=0, padx=20, pady=(20, 5), sticky="w")

        # File Selection
        self.file_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="File Path")
        self.file_entry.grid(row=3, column=0, padx=20, pady=(0, 5), sticky="ew")

        self.browse_btn = ctk.CTkButton(self.sidebar_frame, text="Browse File...", command=self.browse_file, height=25)
        self.browse_btn.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Meta Data
        self.title_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Document Title")
        self.title_entry.grid(row=5, column=0, padx=20, pady=5, sticky="ew")

        self.rev_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Revision (e.g., 1.0)")
        self.rev_entry.grid(row=6, column=0, padx=20, pady=5, sticky="ew")

        # Chunk Size
        self.chunk_label = ctk.CTkLabel(self.sidebar_frame, text="Chunk Size:", anchor="w")
        self.chunk_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")

        self.chunk_size_var = ctk.StringVar(value="300")
        self.chunk_option = ctk.CTkOptionMenu(self.sidebar_frame, variable=self.chunk_size_var,
                                              values=["200", "300", "500", "700", "1024"])
        self.chunk_option.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Create Action
        self.create_btn = ctk.CTkButton(self.sidebar_frame, text="Process & Embed", command=self.create_database,
                                        fg_color="green")
        self.create_btn.grid(row=9, column=0, padx=20, pady=10, sticky="ew")

        # --- Section: Maintenance ---
        # Button to delete selected database(s)
        self.delete_btn = ctk.CTkButton(self.sidebar_frame, text="Delete Selected DBs",
                                        command=self.delete_selected_databases,
                                        fg_color="transparent", border_width=1, text_color="#FF5555",
                                        hover_color="#550000")
        self.delete_btn.grid(row=11, column=0, padx=20, pady=20, sticky="ew")

        # Button to clear all Valkey caches
        self.clear_cache_btn = ctk.CTkButton(self.sidebar_frame, text="Clear AI Response Cache",
                                             command=self.clear_ai_cache,
                                             fg_color="#800000", hover_color="#B22222")
        self.clear_cache_btn.grid(row=12, column=0, padx=20, pady=10, sticky="ew")

        # ============================
        # RIGHT MAIN AREA (Query)
        # ============================
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(3, weight=1)  # Make result box expandable

        # --- 1. Database Selection ---
        self.search_entry = ctk.CTkEntry(self.main_frame, placeholder_text="Search Available Databases...")
        self.search_entry.pack(fill="x", pady=(0, 10))
        self.search_entry.bind("<KeyRelease>", self.on_search_change)

        # Replaced Canvas with CTkScrollableFrame
        self.db_list_frame = ctk.CTkScrollableFrame(self.main_frame, height=150, label_text="Select Databases to Query")
        self.db_list_frame.pack(fill="x", pady=(0, 10))

        # --- 2. Query Controls ---
        self.query_frame = ctk.CTkFrame(self.main_frame)
        self.query_frame.pack(fill="x", pady=(0, 10))

        # Query Input
        self.query_entry = ctk.CTkEntry(self.query_frame, placeholder_text="Type your question here...", height=40)
        self.query_entry.pack(fill="x", padx=10, pady=(10, 5))

        # Options Grid
        self.opts_frame = ctk.CTkFrame(self.query_frame, fg_color="transparent")
        self.opts_frame.pack(fill="x", padx=10, pady=5)

        # Query Transform Option
        self.query_transform_var = ctk.BooleanVar(value=False)
        self.transform_check = ctk.CTkCheckBox(self.opts_frame, text="AI Query Transform",
                                               variable=self.query_transform_var)
        self.transform_check.pack(side="left", padx=10)

        # Bypass Valkey Cache Check Option
        self.use_cache_var = ctk.BooleanVar(value=True)
        self.cache_check = ctk.CTkCheckBox(self.opts_frame, text="Use Cache",
                                           variable=self.use_cache_var)
        self.cache_check.pack(side="left", padx=10)

        # Set Relevance Value
        self.relevance_entry = ctk.CTkEntry(self.opts_frame, width=60)
        self.relevance_entry.insert(0, "0.5")
        ctk.CTkLabel(self.opts_frame, text="Relevance:").pack(side="left", padx=(10, 5))
        self.relevance_entry.pack(side="left")

        # Set Generator Temperature Value
        self.temp_entry = ctk.CTkEntry(self.opts_frame, width=60)
        self.temp_entry.insert(0, "0.2")
        ctk.CTkLabel(self.opts_frame, text="Temp:").pack(side="left", padx=(10, 5))
        self.temp_entry.pack(side="left")

        # Model Selection (Radio Buttons)
        self.model_frame = ctk.CTkFrame(self.main_frame)
        self.model_frame.pack(fill="x", pady=(0, 10))

        self.selected_generator_model = tk.StringVar(value="mistral-q4")
        models = [
            ("Balanced (Mistral)", "mistral-q4"),
            ("Fast (Mistral)", "mistral-q3"),
            ("Summary (Zephyr)", "zephyr-q4"),
            ("Code (CodeLlama)", "codellama-q4")
        ]

        ctk.CTkLabel(self.model_frame, text="Generator:").pack(side="left", padx=10)
        for text, val in models:
            ctk.CTkRadioButton(self.model_frame, text=text, variable=self.selected_generator_model, value=val).pack(
                side="left", padx=10, pady=10)

        # Send Button
        self.send_btn = ctk.CTkButton(self.main_frame, text="Send Query", command=self.run_query, height=40,
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self.send_btn.pack(fill="x", pady=(0, 10))

        # --- 3. Results Area ---
        self.result_label = ctk.CTkLabel(self.main_frame, text="Response:", anchor="w")
        self.result_label.pack(fill="x")

        self.result_text = ctk.CTkTextbox(self.main_frame, wrap="word", font=("Roboto", 12))
        self.result_text.pack(fill="both", expand=True)

        # Configure Tags for the Textbox
        self.result_text._textbox.tag_config("suggestion_tag", foreground="#FFAA00", font=("Roboto", 11, "bold"))
        self.result_text.tag_config("citation_style", foreground="#3B8ED0", underline=True)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready.")
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w", height=25, fg_color="#222222")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

        # Initial Load
        self.load_databases_to_gui()

    # ==========================================
    #  LOGIC METHODS
    # ==========================================

    def download_all_models(self):
        if messagebox.askyesno("Confirm Download", "Download all models? (Several GB)"):
            threading.Thread(target=self.process_downloads, daemon=True).start()

    def process_downloads(self):
        try:
            self.model_downloader.run_downloads(status_callback=self.update_status_text)
            messagebox.showinfo("Success", "Downloads finished!")
        except Exception as e:
            messagebox.showerror("Error", f"{e}")
        finally:
            self.update_status_text("Ready.")

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Documents", "*.pdf;*.docx;*.md;*.txt")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            base_name = os.path.basename(file_path)
            title, _ = os.path.splitext(base_name)
            self.title_entry.delete(0, tk.END)
            self.title_entry.insert(0, title)
            self.rev_entry.delete(0, tk.END)
            self.rev_entry.insert(0, "1.0")

    def create_database(self):
        file_path = self.file_entry.get().strip()
        title = self.title_entry.get().strip()
        revision = self.rev_entry.get().strip()
        chunk_size = int(self.chunk_size_var.get())

        if not all([file_path, title, revision]):
            messagebox.showerror("Error", "File, Title, and Revision are required.")
            return

        self.update_status_text("Processing document...")
        threading.Thread(target=self.process_creation, args=(file_path, title, revision, chunk_size),
                         daemon=True).start()

    def process_creation(self, file_path, title, revision, chunk_size):
        try:
            self.db_builder.process_document(file_path, title, revision, chunk_size)
            messagebox.showinfo("Success", "Database created successfully!")
            self.after(0, self.load_databases_to_gui)
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
        finally:
            self.update_status_text("Ready.")

    def load_databases_to_gui(self, search_term=""):
        # Clear existing switches
        for widget in self.db_list_frame.winfo_children():
            widget.destroy()
        self.database_vars.clear()

        all_databases = list_databases()
        filtered_databases = [db for db in all_databases if search_term.lower() in db.lower()]

        # Create checkboxes in the scrollable frame
        # We use grid inside the scrollable frame for nice alignment
        for i, db_name in enumerate(filtered_databases):
            var = ctk.BooleanVar()
            # Use CTkCheckBox
            cb = ctk.CTkCheckBox(self.db_list_frame, text=db_name, variable=var)
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=10, pady=5)
            self.database_vars.append((var, db_name))

    def on_search_change(self, event=None):
        search_term = self.search_entry.get()
        self.load_databases_to_gui(search_term)

    def delete_selected_databases(self):
        databases_to_delete = [name for var, name in self.database_vars if var.get()]
        if not databases_to_delete:
            messagebox.showwarning("No Selection", "Please select databases to delete.")
            return

        confirm_message = "Permanently delete:\n\n- " + "\n- ".join(databases_to_delete)
        if messagebox.askyesno("Confirm Deletion", confirm_message):
            threading.Thread(target=self._perform_deletion, args=(databases_to_delete,), daemon=True).start()

    def _perform_deletion(self, databases_to_delete):
        deleted_count = 0
        for db_name in databases_to_delete:
            try:
                path_to_delete = os.path.join(READER_CONFIG["chroma_base_path"], db_name)
                if os.path.exists(path_to_delete):
                    shutil.rmtree(path_to_delete)
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting {db_name}: {e}")

        self.after(0, lambda: messagebox.showinfo("Success", f"Deleted {deleted_count} database(s)."))
        self.after(0, self.load_databases_to_gui)

    def run_query(self):
        query_text = self.query_entry.get().strip()
        if not query_text:
            messagebox.showerror("Error", "Query cannot be empty.")
            return
        try:
            relevance = float(self.relevance_entry.get().strip())
            temperature = float(self.temp_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Relevance/Temp must be numbers.")
            return

        use_transform = self.query_transform_var.get()
        use_cache = self.use_cache_var.get()
        selected_dbs = [name for var, name in self.database_vars if var.get()]
        selected_db_str = "All" if not selected_dbs else ",".join(selected_dbs)
        generator_name = self.selected_generator_model.get()

        start_time = time.monotonic()
        self.clear_result_text()
        self.update_status_text("Thinking...")

        threading.Thread(target=self.process_query, args=(query_text, selected_db_str, relevance, temperature,
                                                          use_transform, use_cache, generator_name, start_time),
                         daemon=True).start()

    def process_query(self, query_text, selected_db, relevance, temperature, use_transform, use_cache, generator_name, start_time):
        try:
            stream_generator = self.rag_pipeline.answer_question(
                query_text=query_text,
                selected_db=selected_db,
                relevance_threshold=relevance,
                temperature=temperature,
                use_query_transform=use_transform,
                use_cache=use_cache,
                status_callback=self.update_status_text,
                generator_name=generator_name
            )

            for item in stream_generator:
                item_type = item.get("type")
                item_data = item.get("data")

                if item_type == "sources":
                    self.current_sources = item_data
                elif item_type == "suggestion":
                    self.display_suggestion(item_data)
                elif item_type == "token":
                    self.append_to_result_text(item_data)
                elif item_type == "error":
                    self.update_result_text(f"Error: {item_data}")
                    return

            end_time = time.monotonic()
            elapsed = int(end_time - start_time)
            self.apply_citations(elapsed)

        except Exception as e:
            self.update_result_text(f"Critical Error:\n{e}")
        finally:
            self.update_status_text("Ready.")

    # --- UI Update Helpers ---

    def clear_result_text(self):
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.configure(state=tk.DISABLED)
        self.current_sources = []

    def append_to_result_text(self, token):
        def task():
            self.result_text.configure(state=tk.NORMAL)
            self.result_text.insert(tk.END, token)
            self.result_text.configure(state=tk.DISABLED)
            self.result_text.see(tk.END)

        self.after(0, task)

    def update_result_text(self, text):
        def task():
            self.clear_result_text()
            self.result_text.configure(state=tk.NORMAL)
            self.result_text.insert("1.0", text)
            self.result_text.configure(state=tk.DISABLED)

        self.after(0, task)

    def display_suggestion(self, suggestions_list):
        def task():
            if not suggestions_list: return
            try:
                suggest_text = "\n".join(suggestions_list)
                self.result_text.configure(state=tk.NORMAL)
                self.result_text.insert(
                    "1.0",
                    f"💡 SUGGESTION: Try adding these databases:\n{suggest_text}\n\n",
                    "suggestion_tag"
                )
                self.result_text.configure(state=tk.DISABLED)
            except Exception as e:
                print(f"Suggestion error: {e}")

        self.after(0, task)

    def apply_citations(self, elapsed_seconds):
        def task():
            self.result_text.configure(state=tk.NORMAL)
            full_text = self.result_text.get("1.0", tk.END)
            prefix = f"({elapsed_seconds}s) "
            self.result_text.insert("1.0", prefix)
            full_text = prefix + full_text

            # Re-apply tags for citations
            for match in re.finditer(r'\[(\d+)\]', full_text):
                start, end = match.span()
                citation_index = int(match.group(1)) - 1

                if 0 <= citation_index < len(self.current_sources):
                    source_text = self.current_sources[citation_index]['context']
                    tag_name = f"citation-{start}"

                    # Calculate CTk indices
                    start_index = f"1.0+{start}c"
                    end_index = f"1.0+{end}c"

                    self.result_text.tag_add("citation_style", start_index, end_index)
                    self.result_text.tag_add(tag_name, start_index, end_index)

                    tooltip = ModernToolTip(self.result_text, text=source_text)
                    self.result_text.tag_bind(tag_name, "<Enter>", lambda e, t=tooltip: t.show(e))
                    self.result_text.tag_bind(tag_name, "<Leave>", lambda e, t=tooltip: t.hide(e))

            self.result_text.configure(state=tk.DISABLED)

        self.after(0, task)

    def update_status_text(self, text):
        self.after(0, lambda: self.status_var.set(text))

    def clear_ai_cache(self):
        # Get cache stats
        self.update_status_text("Fetching cache stats...")
        cache_stats = self.rag_pipeline.get_cache_stats()

        # Check which databases the user has selected
        selected_dbs = [name for var, name in self.database_vars if var.get()]

        # If nothing selected, assume deleting ALL
        if not selected_dbs:
            total_items = sum(cache_stats.values())
            if total_items == 0:
                messagebox.showinfo("Cache Empty", "There are no responses in this cache to clear.")
                self.update_status_text("Ready")
                return

            if messagebox.askyesno("Clear All?", "No databases selected. Clear the ENTIRE cache?"):
                self.rag_pipeline.clear_entire_cache()
                messagebox.showinfo("Success", "Entire cache cleared.")
            self.update_status_text("Ready")
            return

        lines = []
        total_to_delete = 0
        for db in selected_dbs:
            count = cache_stats.get(db, 0)
            lines.append(f" • {db} - ({count} entries)")
            total_to_delete += count

        if total_to_delete == 0:
            messagebox.showinfo("Nothing to Clear", "Selected databases have no cached responses.")
            self.update_status_text("Ready.")
            return

        confirm_msg = f"Permanently clear {total_to_delete} cached responses for:\n\n" + "\n".join(lines)

        if messagebox.askyesno("Confirm Targeted Clear", confirm_msg):
            self.update_status_text("Clearing targeted cache...")
            success, deleted = self.rag_pipeline.clear_cache_by_databases(selected_dbs)
            if success:
                messagebox.showinfo("Success", f"Cleared {deleted} cached responses.")

        self.update_status_text("Ready.")

if __name__ == "__main__":
    app = DocumentDatabaseGUI()
    app.mainloop()