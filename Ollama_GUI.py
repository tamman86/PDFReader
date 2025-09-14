import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import os
import shutil

from Ollama_DB import DatabaseBuilder, CONFIG as DB_CONFIG
from Ollama_Readerv2 import RAGPipeline, CONFIG as READER_CONFIG, list_databases


class DocumentDatabaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Database Manager")
        self.root.geometry("800x800")  # Increased size for new features

        # --- Initialize Backend Pipelines ---
        print("Initializing backend pipelines... (This may take a moment)")
        self.db_builder = DatabaseBuilder(DB_CONFIG)
        self.rag_pipeline = RAGPipeline(READER_CONFIG)
        print("✅ Pipelines initialized.")

        # --- UI SETUP ---
        # 1. Database Creation Frame
        self.create_db_frame = tk.LabelFrame(root, text="1. Create or Re-build a Database", padx=10, pady=10)
        self.create_db_frame.pack(padx=10, pady=10, fill="x")

        tk.Label(self.create_db_frame, text="Select File:").grid(row=0, column=0, sticky="w", pady=2)
        self.file_path_entry = tk.Entry(self.create_db_frame, width=50)
        self.file_path_entry.grid(row=0, column=1, padx=5)
        tk.Button(self.create_db_frame, text="Browse...", command=self.browse_file).grid(row=0, column=2)

        tk.Label(self.create_db_frame, text="Document Title:").grid(row=1, column=0, sticky="w", pady=2)
        self.title_entry = tk.Entry(self.create_db_frame, width=50)
        self.title_entry.grid(row=1, column=1, padx=5)

        tk.Label(self.create_db_frame, text="Document Revision:").grid(row=2, column=0, sticky="w", pady=2)
        self.revision_entry = tk.Entry(self.create_db_frame, width=50)
        self.revision_entry.grid(row=2, column=1, padx=5)

        ### --- NEW FEATURE: Chunk Size Dropdown --- ###
        tk.Label(self.create_db_frame, text="Chunk Size:").grid(row=3, column=0, sticky="w", pady=2)
        self.chunk_size_var = tk.StringVar(value="300")  # Set default value to 300
        chunk_size_options = ["200", "300", "500", "700", "1024"]
        self.chunk_size_dropdown = tk.OptionMenu(self.create_db_frame, self.chunk_size_var, *chunk_size_options)
        self.chunk_size_dropdown.grid(row=3, column=1, sticky="w", padx=5)

        tk.Button(self.create_db_frame, text="Create Database", command=self.create_database).grid(row=4, column=1,
                                                                                                   pady=10)

        # 2. Query Frame
        self.query_frame = tk.LabelFrame(root, text="2. Query Databases", padx=10, pady=10)
        self.query_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ### --- NEW FEATURE: Search and Delete Frame --- ###
        db_management_frame = tk.Frame(self.query_frame)
        db_management_frame.pack(fill="x", pady=(0, 10))

        tk.Label(db_management_frame, text="Search Databases:").pack(side="left", padx=(0, 5))
        self.search_entry = tk.Entry(db_management_frame, width=40)
        self.search_entry.pack(side="left", fill="x", expand=True)
        # The on_search_change function will be called whenever the user types
        self.search_entry.bind("<KeyRelease>", self.on_search_change)

        tk.Button(db_management_frame, text="Delete Selected", command=self.delete_selected_databases, fg="red").pack(
            side="right", padx=(10, 0))

        ### --- NEW FEATURE: Scrollable Checkbox List --- ###
        # Create a container frame for the canvas and scrollbar
        db_list_container = tk.Frame(self.query_frame)
        db_list_container.pack(fill="x", pady=5)
        db_list_container.config(height=150)

        canvas = tk.Canvas(db_list_container, height=150)
        scrollbar = tk.Scrollbar(db_list_container, orient="vertical", command=canvas.yview)
        # This is the frame that will actually hold the checkboxes
        self.scrollable_db_frame = tk.Frame(canvas)

        self.scrollable_db_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_db_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Query and Results Section (Moved down for better layout) ---
        query_controls_frame = tk.Frame(self.query_frame)
        query_controls_frame.pack(fill="both", expand=True, pady=(10,0))

        tk.Label(query_controls_frame, text="Your Question:").pack(anchor="w")
        self.query_entry = tk.Entry(query_controls_frame, width=80)
        self.query_entry.pack(pady=5, fill="x")

        ### --- NEW FEATURE: Query Transform Checkbox --- ###
        self.query_transform_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            query_controls_frame,
            text="Query Transform Assist? (Slower, but may improve relevance)",
            variable=self.query_transform_var
        ).pack(anchor="w")

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

        tk.Button(query_controls_frame, text="Send Query", command=self.run_query).pack(pady=10)

        tk.Label(query_controls_frame, text="Answer:").pack(anchor="w")
        self.result_text = scrolledtext.ScrolledText(query_controls_frame, height=10, wrap="word")
        self.result_text.pack(pady=5, fill="both", expand=True)

        self.database_vars = []
        self.load_databases_to_gui()

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

    def create_database(self):
        file_path = self.file_path_entry.get().strip()
        title = self.title_entry.get().strip()
        revision = self.revision_entry.get().strip()
        ### --- MODIFICATION: Get chunk size from the new dropdown --- ###
        chunk_size = int(self.chunk_size_var.get())

        if not all([file_path, title, revision]):
            messagebox.showerror("Error", "File, Title, and Revision are required.")
            return

        ### --- MODIFICATION: Pass chunk_size to the processing thread --- ###
        threading.Thread(target=self.process_creation, args=(file_path, title, revision, chunk_size),
                         daemon=True).start()

    def process_creation(self, file_path, title, revision, chunk_size):
        try:
            ### --- MODIFICATION: Pass chunk_size to the db_builder --- ###
            self.db_builder.process_document(file_path, title, revision, chunk_size)
            messagebox.showinfo("Success", "Database created successfully!")
            self.root.after(0, self.load_databases_to_gui)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create database: {e}")

    def load_databases_to_gui(self, search_term=""):
        """Refreshes the scrollable list of databases, applying an optional search filter."""
        # Clear the existing checkboxes from the scrollable frame
        for widget in self.scrollable_db_frame.winfo_children():
            widget.destroy()
        self.database_vars.clear()

        all_databases = list_databases()

        # Filter databases based on the search term
        filtered_databases = [db for db in all_databases if search_term.lower() in db.lower()]

        num_columns = 6
        # Configure the columns in the scrollable frame to have equal weight.
        # This makes the layout responsive if the window is resized.
        for i in range(num_columns):
            self.scrollable_db_frame.columnconfigure(i, weight=1)

        # Use enumerate to get both the index (i) and the name of the database
        for i, db_name in enumerate(filtered_databases):
            # Calculate the row and column for the grid
            row = i // num_columns
            column = i % num_columns

            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.scrollable_db_frame, text=db_name, variable=var)
            # Use .grid() instead of .pack() to place the checkbox
            cb.grid(row=row, column=column, sticky="w", padx=5, pady=2)

            self.database_vars.append((var, db_name))

    ### --- NEW FEATURE: Search Handler --- ###
    def on_search_change(self, event=None):
        """Called whenever the user types in the search box."""
        search_term = self.search_entry.get()
        self.load_databases_to_gui(search_term)

    ### --- NEW FEATURE: Delete Handler --- ###
    def delete_selected_databases(self):
        """Finds all checked databases and prompts the user for deletion."""
        databases_to_delete = [name for var, name in self.database_vars if var.get()]

        if not databases_to_delete:
            messagebox.showwarning("No Selection", "Please check the databases you want to delete.")
            return

        confirm_message = "Are you sure you want to permanently delete the following databases?\n\n- " + "\n- ".join(
            databases_to_delete)
        if messagebox.askyesno("Confirm Deletion", confirm_message):
            threading.Thread(target=self._perform_deletion, args=(databases_to_delete,), daemon=True).start()

    def _perform_deletion(self, databases_to_delete):
        """The actual deletion logic, run in a background thread."""
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

        # Correctly get selected databases from the potentially filtered list
        selected_dbs = [name for var, name in self.database_vars if var.get()]
        selected_db_str = "All" if not selected_dbs else ",".join(selected_dbs)

        threading.Thread(target=self.process_query, args=(query_text, selected_db_str, relevance, temperature),
                         daemon=True).start()

    def process_query(self, query_text, selected_db, relevance, temperature):
        self.update_result_text("Processing query... Please wait.")
        try:
            final_answer, sources = self.rag_pipeline.answer_question(
                query_text=query_text, selected_db=selected_db,
                relevance_threshold=relevance, temperature=temperature
            )
            self.update_result_text(final_answer)
        except Exception as e:
            self.update_result_text(f"An error occurred during query processing:\n{e}")

    def update_result_text(self, text):
        """Helper function to safely update the GUI text box from a background thread."""

        def task():
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, text)

        # Schedule the GUI update to run on the main thread.
        self.root.after(0, task)


if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentDatabaseGUI(root)
    root.mainloop()

