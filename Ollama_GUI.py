import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
os.environ['HF_HUB_OFFLINE'] = '1' # Keep HuggingFace from online communication
import shutil
import re
import time

from Ollama_DB import DatabaseBuilder, CONFIG as DB_CONFIG
from Ollama_Readerv2 import RAGPipeline, CONFIG as READER_CONFIG, list_databases
from Download_Model import ModelDownloader, MODELS_TO_DOWNLOAD, LOCAL_MODEL_DIR
from dotenv import load_dotenv
load_dotenv()
HuggingFaceToken = os.getenv("HUGGINGFACE_TOKEN")

# Tooltips for citations
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

    def show(self, event):
        x = event.x_root + 20
        y = event.y_root + 10

        # Create a Toplevel window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # Removes window borders
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Style the label inside the popup
        label = tk.Label(self.tooltip_window, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         wraplength=500) # Wraps long source text
        label.pack(ipadx=1)

    def hide(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

class DocumentDatabaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Database Manager")
        self.root.geometry("800x850")

        # Initialize Backend Pipelines
        print("Initializing backend pipelines... (This may take a moment)")
        self.db_builder = DatabaseBuilder(DB_CONFIG)
        self.rag_pipeline = RAGPipeline(READER_CONFIG)
        self.model_downloader = ModelDownloader(MODELS_TO_DOWNLOAD, LOCAL_MODEL_DIR, HuggingFaceToken)
        print("✅ Pipelines initialized.")

        # Variable for selected generator (Mistral default)
        self.selected_generator_model = tk.StringVar(value="mistral-q4")

        # UI Setup

        # Model Download Button
        setup_frame = tk.LabelFrame(root, text="Initial Setup", padx=10, pady=10)
        setup_frame.pack(padx=10, pady=10, fill="x")

        tk.Button(
            setup_frame,
            text="Download All Required Models",
            command=self.download_all_models,
            bg="#D6EAF8" # Light Blue
        ).pack(pady=5)

        # Database Creation Frame
        self.create_db_frame = tk.LabelFrame(root, text="1. Create or Re-build a Database", padx=10, pady=10)
        self.create_db_frame.pack(padx=10, pady=10, fill="x")

        # Select file to be chunked/embedded
        tk.Label(self.create_db_frame, text="Select File:").grid(row=0, column=0, sticky="w", pady=2)
        self.file_path_entry = tk.Entry(self.create_db_frame, width=50)
        self.file_path_entry.grid(row=0, column=1, padx=5)
        tk.Button(self.create_db_frame, text="Browse...", command=self.browse_file).grid(row=0, column=2)

        # User selected title for processed document
        tk.Label(self.create_db_frame, text="Document Title:").grid(row=1, column=0, sticky="w", pady=2)
        self.title_entry = tk.Entry(self.create_db_frame, width=50)
        self.title_entry.grid(row=1, column=1, padx=5)

        # User specified revision for processed document
        tk.Label(self.create_db_frame, text="Document Revision:").grid(row=2, column=0, sticky="w", pady=2)
        self.revision_entry = tk.Entry(self.create_db_frame, width=50)
        self.revision_entry.grid(row=2, column=1, padx=5)

        # User specified chunk size determination
        tk.Label(self.create_db_frame, text="Chunk Size:").grid(row=3, column=0, sticky="w", pady=2)
        self.chunk_size_var = tk.StringVar(value="300")  # Default value set to 300
        chunk_size_options = ["200", "300", "500", "700", "1024"]   # Chunk size options
        self.chunk_size_dropdown = tk.OptionMenu(self.create_db_frame, self.chunk_size_var, *chunk_size_options)
        self.chunk_size_dropdown.grid(row=3, column=1, sticky="w", padx=5)

        # Create database button
        tk.Button(self.create_db_frame, text="Create Database", command=self.create_database).grid(row=4, column=1, pady=10)

        # 2. Query Frame
        self.query_frame = tk.LabelFrame(root, text="2. Query Databases", padx=10, pady=10)
        self.query_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Creation and maintenance of status bar
        self.status_var = tk.StringVar(value="Ready.")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5)
        self.status_bar.pack(side=tk.BOTTOM, fill="x")

        # Database search bar
        db_management_frame = tk.Frame(self.query_frame)
        db_management_frame.pack(fill="x", pady=(0, 10))
        tk.Label(db_management_frame, text="Search Databases:").pack(side="left", padx=(0, 5))
        self.search_entry = tk.Entry(db_management_frame, width=40)
        self.search_entry.pack(side="left", fill="x", expand=True)
        # The on_search_change function allows for instant search updates
        self.search_entry.bind("<KeyRelease>", self.on_search_change)

        # Database deletion button
        tk.Button(db_management_frame, text="Delete Selected", command=self.delete_selected_databases, fg="red").pack(
            side="right", padx=(10, 0))

        # Create scrollable database selection area
        db_list_container = tk.Frame(self.query_frame)
        db_list_container.pack(fill="x", pady=5)
        db_list_container.config(height=100)

        canvas = tk.Canvas(db_list_container, height=100)
        scrollbar = tk.Scrollbar(db_list_container, orient="vertical", command=canvas.yview)
        self.scrollable_db_frame = tk.Frame(canvas)

        self.scrollable_db_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_db_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Query and response section
        query_controls_frame = tk.Frame(self.query_frame)
        query_controls_frame.pack(fill="both", expand=True, pady=(10,0))

        tk.Label(query_controls_frame, text="Your Question:").pack(anchor="w")
        self.query_entry = tk.Entry(query_controls_frame, width=80)
        self.query_entry.pack(pady=5, fill="x")

        # AI-assisted query transform
        self.query_transform_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            query_controls_frame,
            text="Query Transform Assist? (Slower, but may improve relevance)",
            variable=self.query_transform_var
        ).pack(anchor="w")

        # Adding radio buttons for generator model selection
        model_selection_frame = tk.LabelFrame(query_controls_frame, text="Select Generator Model", padx=10, pady=5)
        model_selection_frame.pack(fill="x", pady=(10, 5))

        tk.Radiobutton(
            model_selection_frame, text="Balanced (Mistral)", variable=self.selected_generator_model,
            value="mistral-q4"
        ).pack(side="left", padx=10)

        tk.Radiobutton(
            model_selection_frame, text="Fast (Mistral)", variable=self.selected_generator_model,
            value="mistral-q3"
        ).pack(side="left", padx=10)

        tk.Radiobutton(
            model_selection_frame, text="Summarization (Zephyr)", variable=self.selected_generator_model,
            value="zephyr-q4"
        ).pack(side="left", padx=10)

        tk.Radiobutton(
            model_selection_frame, text="Coding & Logic (CodeLlama)", variable=self.selected_generator_model,
            value="codellama-q4"
        ).pack(side="left", padx=10)

        # Relevance Threshold and Temperature User input
        settings_frame = tk.Frame(query_controls_frame)
        settings_frame.pack(anchor="w", pady=5)
        tk.Label(settings_frame, text="Relevance Threshold:").pack(side="left")
        self.relevance_entry = tk.Entry(settings_frame, width=10)
        self.relevance_entry.insert(0, "0.3")
        self.relevance_entry.pack(side="left", padx=(5, 20))

        tk.Label(settings_frame, text="Temperature:").pack(side="left")
        self.temperature_entry = tk.Entry(settings_frame, width=10)
        self.temperature_entry.insert(0, "0.2")
        self.temperature_entry.pack(side="left", padx=5)

        # Submit Query button
        tk.Button(query_controls_frame, text="Send Query", command=self.run_query).pack(pady=10)

        # Response window
        tk.Label(query_controls_frame, text="Answer:").pack(anchor="w")
        self.result_text = scrolledtext.ScrolledText(query_controls_frame, height=10, wrap="word")
        self.result_text.pack(pady=5, fill="both", expand=True)

        self.database_vars = []
        self.load_databases_to_gui()

    # Model Downloader
    def download_all_models(self):
        # Ask the user for confirmation before starting the large download.
        if messagebox.askyesno("Confirm Download",
                               "This will check for and download all required AI models (several GB). This may take a long time. Continue?"):
            threading.Thread(target=self.process_downloads, daemon=True).start()

    def process_downloads(self):
        try:
            self.model_downloader.run_downloads(status_callback=self.update_status_text)
            messagebox.showinfo("Success", "Model download process finished successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during download: {e}")
        finally:
            self.update_status_text("Ready.")

    # File browser for selecting files to chunk/embed
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Documents", "*.pdf;*.docx;*.md;*.txt")])
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            base_name = os.path.basename(file_path)
            title, _ = os.path.splitext(base_name)
            self.title_entry.delete(0, tk.END)
            self.title_entry.insert(0, title)
            self.revision_entry.delete(0, tk.END)
            self.revision_entry.insert(0, "1.0")

    # Collect database inputs
    def create_database(self):
        file_path = self.file_path_entry.get().strip()
        title = self.title_entry.get().strip()
        revision = self.revision_entry.get().strip()
        chunk_size = int(self.chunk_size_var.get())

        if not all([file_path, title, revision]):
            messagebox.showerror("Error", "File, Title, and Revision are required.")
            return

        threading.Thread(target=self.process_creation, args=(file_path, title, revision, chunk_size),
                         daemon=True).start()

    # Provide success/fail response for chunking/embedding
    def process_creation(self, file_path, title, revision, chunk_size):
        try:
            self.db_builder.process_document(file_path, title, revision, chunk_size)
            messagebox.showinfo("Success", "Database created successfully!")
            self.root.after(0, self.load_databases_to_gui)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create database: {e}")

    # Load embedded databases to selection list
    def load_databases_to_gui(self, search_term=""):
        # Clear the existing checkboxes from the scrollable frame
        for widget in self.scrollable_db_frame.winfo_children():
            widget.destroy()
        self.database_vars.clear()
        all_databases = list_databases()

        # Filter databases based on the search term
        filtered_databases = [db for db in all_databases if search_term.lower() in db.lower()]

        num_columns = 6

        for i in range(num_columns):
            self.scrollable_db_frame.columnconfigure(i, weight=1)

        # Database placement on grid shaped list
        for i, db_name in enumerate(filtered_databases):
            # Calculate the row and column for the grid
            row = i // num_columns
            column = i % num_columns

            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.scrollable_db_frame, text=db_name, variable=var)
            cb.grid(row=row, column=column, sticky="w", padx=5, pady=2)

            self.database_vars.append((var, db_name))

    # Database search/browser
    def on_search_change(self, event=None):
        search_term = self.search_entry.get()
        self.load_databases_to_gui(search_term)

    # Database embedding deletion selection function
    def delete_selected_databases(self):
        databases_to_delete = [name for var, name in self.database_vars if var.get()]

        if not databases_to_delete:
            messagebox.showwarning("No Selection", "Please check the databases you want to delete.")
            return
        # Confirm choice to delete selected databases
        confirm_message = "Are you sure you want to permanently delete the following databases?\n\n- " + "\n- ".join(
            databases_to_delete)
        if messagebox.askyesno("Confirm Deletion", confirm_message):
            threading.Thread(target=self._perform_deletion, args=(databases_to_delete,), daemon=True).start()

    # Database embedding deletion function
    def _perform_deletion(self, databases_to_delete):
        deleted_count = 0
        for db_name in databases_to_delete:
            try:
                # Construct the full path e.g., "chroma/Title/Revision"
                path_to_delete = os.path.join(READER_CONFIG["chroma_base_path"], db_name)
                if os.path.exists(path_to_delete):
                    shutil.rmtree(path_to_delete)
                    print(f"Deleted: {path_to_delete}")
                    deleted_count += 1
            except Exception as e:
                messagebox.showerror("Deletion Error", f"Failed to delete {db_name}:\n{e}")

        messagebox.showinfo("Success", f"Successfully deleted {deleted_count} database(s).")
        # Refresh the GUI list from the main thread
        self.root.after(0, self.load_databases_to_gui)

    # Submitting the query into the machine
    def run_query(self):
        query_text = self.query_entry.get().strip()
        if not query_text:
            messagebox.showerror("Error", "Query cannot be empty.")
            return
        try:
            relevance = float(self.relevance_entry.get().strip())
            temperature = float(self.temperature_entry.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Relevance and Temperature must be numbers.")
            return

        # Check to see if AI-assisted transform is used
        use_transform = self.query_transform_var.get()

        # Correctly get selected databases from the potentially filtered list
        selected_dbs = [name for var, name in self.database_vars if var.get()]
        selected_db_str = "All" if not selected_dbs else ",".join(selected_dbs)

        # Selected generator
        generator_name = self.selected_generator_model.get()

        start_time = time.monotonic()  # Start the timer
        self.update_result_text("", [])  # Clear the old answer
        self.update_status_text("Starting query...")  # Set the initial status

        threading.Thread(target=self.process_query, args=(query_text, selected_db_str, relevance, temperature,
                                                          use_transform, generator_name, start_time), daemon=True).start()

    # LLM generation step
    def process_query(self, query_text, selected_db, relevance, temperature, use_transform, generator_name, start_time):
        self.update_result_text("Processing query... Please wait.", [])
        try:
            final_answer, top_extractions = self.rag_pipeline.answer_question(
                query_text=query_text,
                selected_db=selected_db,
                relevance_threshold=relevance,
                temperature=temperature,
                use_query_transform=use_transform,
                status_callback=self.update_status_text,  # Pass the GUI function to the backend
                generator_name=generator_name
            )

            end_time = time.monotonic()  # Stop the timer
            elapsed_seconds = int(end_time - start_time)

            self.update_result_text(final_answer, top_extractions, elapsed_seconds)
        except Exception as e:
            self.update_result_text(f"An error occurred during query processing:\n{e}", [])
        finally:
            # This ensures the status is always reset when the process is over.
            self.update_status_text("Ready.")

    # Update GUI text box via background thread
    def update_result_text(self, text, sources, elapsed_seconds=None):
        def task():
            # 1. Clear previous content and any old tags
            self.result_text.config(state=tk.NORMAL)  # Make widget editable
            self.result_text.delete("1.0", tk.END)
            for tag in self.result_text.tag_names():
                if "citation-" in tag:
                    self.result_text.tag_delete(tag)

            # 2. Insert the new answer text with query time
            prefix = f"({elapsed_seconds}s) " if elapsed_seconds is not None else ""
            full_text = f"{prefix}{text}"
            self.result_text.insert("1.0", full_text)

            # 3. Configure the visual style for our citation "links"
            self.result_text.tag_configure("citation_style", foreground="blue", underline=True)

            # 4. Find all citation markers like [1], [2], etc.
            for match in re.finditer(r'\[(\d+)\]', full_text):
                start, end = match.span()
                citation_num_str = match.group(1)
                citation_index = int(citation_num_str) - 1

                # 5. Check if this citation number is valid for the sources we received
                if 0 <= citation_index < len(sources):
                    # The full text of the source chunk
                    source_text = sources[citation_index]['context']
                    tag_name = f"citation-{start}"  # A unique tag for this specific link

                    # Get the start and end position in Tkinter's text index format
                    start_index = f"1.0+{start}c"
                    end_index = f"1.0+{end}c"

                    # 6. Apply the blue, underlined style
                    self.result_text.tag_add("citation_style", start_index, end_index)

                    # 7. Apply the unique tag for event binding
                    self.result_text.tag_add(tag_name, start_index, end_index)

                    # 8. Create the tooltip and bind mouse events
                    tooltip = ToolTip(self.result_text, text=source_text)
                    self.result_text.tag_bind(tag_name, "<Enter>", lambda e, t=tooltip: t.show(e))
                    self.result_text.tag_bind(tag_name, "<Leave>", lambda e, t=tooltip: t.hide(e))

            self.result_text.config(state=tk.DISABLED)  # Make widget read-only again

        # Schedule the GUI update to run on the main thread
        self.root.after(0, task)

    # Update GUI status bar via background thread
    def update_status_text(self, text):
        def task():
            self.status_var.set(text)

        # Schedule the GUI update to run on the main Tkinter thread.
        self.root.after(0, task)

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentDatabaseGUI(root)
    root.mainloop()

