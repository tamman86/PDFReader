import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
from Ollama_DB import process_document
from Ollama_Readerv2 import list_databases
import nltk
import traceback

class DocumentDatabaseGUI:
    def __init__(self, root):
        self.root = root

        # GUI Window title
        self.root.title("Document Database Manager")

        # GUI sub-window 1
        self.create_db_frame = tk.LabelFrame(root, text="Create Database", padx=10, pady=10)
        self.create_db_frame.pack(padx=10, pady=5, fill="x")

        # First Line in sub-window 1
        tk.Label(self.create_db_frame, text="Select File:").grid(row=0, column=0, sticky="w")
        self.file_path_entry = tk.Entry(self.create_db_frame, width=40)
        self.file_path_entry.grid(row=0, column=1, padx=5)
        tk.Button(self.create_db_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)

        '''
        title_frame = tk.Frame(self.create_db_frame)
        title_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        tk.Label(title_frame, text="Document Title:").grid(row=0, column=0, sticky="w")
        self.title_entry = tk.Entry(title_frame, width=40)
        self.title_entry.grid(row=0, column=1, padx=5)
        '''

        tk.Label(self.create_db_frame, text="Title:").grid(row=1, column=0, sticky="w")
        self.title_entry = tk.Entry(self.create_db_frame, width=40)
        self.title_entry.grid(row=1, column=1, padx=5)

        '''
        revision_frame = tk.Frame(self.create_db_frame)
        revision_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        tk.Label(revision_frame, text="Document Revision:").grid(row=0, column=0, sticky="w")
        self.revision_entry = tk.Entry(revision_frame, width=40)
        self.revision_entry.grid(row=0, column=1, padx=5)
        '''

        tk.Label(self.create_db_frame, text="Revision:").grid(row=2, column=0, sticky="w")
        self.revision_entry = tk.Entry(self.create_db_frame, width=40)
        self.revision_entry.grid(row=2, column=1, padx=5)

        # Nested Frame for Chunk Size & Chunk Overlap
        chunk_frame = tk.Frame(self.create_db_frame)
        chunk_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=2)

        tk.Label(chunk_frame, text="Chunk Size:").grid(row=0, column=0, sticky="w")
        self.chunk_size_entry = tk.Entry(chunk_frame, width=10)  # Unique name, adjusted width
        self.chunk_size_entry.grid(row=0, column=1, padx=5)

        self.sentence_split_var = tk.BooleanVar(value=False)
        self.sentence_split_checkbox = tk.Checkbutton(
            chunk_frame,
            text = "Sentence Split",
            variable=self.sentence_split_var
        )
        self.sentence_split_checkbox.grid(row=0, column=3, padx=5)

        tk.Button(self.create_db_frame, text="Create Database", command=self.create_database).grid(row=4, column=1, pady=5)

        self.query_frame = tk.LabelFrame(root, text="Query Database", padx=10, pady=10)
        self.query_frame.pack(padx=10, pady=5, fill="x")

        self.database_checkboxes = []
        self.database_vars = []
        self.load_databases()

        tk.Label(self.query_frame, text="Query:").pack(anchor="w")
        self.query_entry = tk.Entry(self.query_frame, width=50)
        self.query_entry.pack(pady=5)

        tk.Label(self.query_frame, text="Relevance Threshold:").pack(anchor="w")
        self.relevance_entry = tk.Entry(self.query_frame, width=10)
        self.relevance_entry.insert(0, "0.3")
        self.relevance_entry.pack(pady=5)

        tk.Label(self.query_frame, text="Temperature:").pack(anchor="w")
        self.temperature_entry = tk.Entry(self.query_frame, width=10)
        self.temperature_entry.insert(0, "0.7")
        self.temperature_entry.pack(pady=5)

        tk.Button(self.query_frame, text="Send Query", command=self.run_query).pack(pady=5)

        self.result_text = tk.Text(self.query_frame, height=10, width=75, wrap="word")
        self.result_text.pack(pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Documents", "*.pdf;*.docx;*.md")])
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

    def create_database(self):
        file_path = self.file_path_entry.get().strip()
        title = self.title_entry.get().strip()
        revision = self.revision_entry.get().strip()
        chunk_size = self.chunk_size_entry.get().strip()
        sentence_split = self.sentence_split_var.get()

        if not file_path or not title or not revision:
            messagebox.showerror("Error", "All fields are required.")
            return

        threading.Thread(target=self.process_creation, args=(file_path, title, revision, chunk_size, sentence_split), daemon=True).start()

    def process_creation(self, file_path, title, revision, chunk_size, sentence_split):
        try:
            process_document(file_path, title, revision, chunk_size, sentence_split)
            messagebox.showinfo("Success", "Database created successfully!")
            self.load_databases()
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to create database: {e}")

    def load_databases(self):
        for cb in self.database_checkboxes:
            cb.destroy()
        self.database_checkboxes.clear()
        self.database_vars.clear()

        databases = list_databases()
        for db in databases:
            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.query_frame, text=db, variable=var)
            cb.pack(anchor="w")
            self.database_checkboxes.append(cb)
            self.database_vars.append(var)

    def run_query(self):
        query_text = self.query_entry.get().strip()
        relevance = self.relevance_entry.get().strip()
        temperature = self.temperature_entry.get().strip()

        if not query_text:
            messagebox.showerror("Error", "Query cannot be empty.")
            return

        selected_dbs = [db for db, var in zip(list_databases(), self.database_vars) if var.get()]
        selected_db = "All" if not selected_dbs else ",".join(selected_dbs)

        threading.Thread(target=self.process_query, args=(query_text, selected_db, relevance, temperature), daemon=True).start()

    def process_query(self, query_text, selected_db, relevance, temperature):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Processing query...\n")
        try:
            os.system(f'python Ollama_Readerv2.py "{query_text}" --database "{selected_db}" --relevance "{relevance}" --temperature "{temperature}"')
        except Exception as e:
            messagebox.showerror("Error", f"Query failed: {e}")

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    root = tk.Tk()
    app = DocumentDatabaseGUI(root)
    root.mainloop()
