# src/chatbot.py
import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
import subprocess
import textwrap

# Optional imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None


class LegalChatbot:
    def __init__(
        self,
        index_path='./faiss_index',
        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
        llm_model_name='google/flan-t5-base',
        device=None
    ):
        load_dotenv()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Using device: {self.device}")

        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.index_path = index_path
        self._load_index()

        # Detect backend
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if self.groq_key:
            print("🔗 Using Groq API (llama-3.3-70b-versatile)")
            self.client = Groq(api_key=self.groq_key)
            self.backend = "groq"

        elif self.openai_key:
            print("🔗 Using OpenAI GPT (gpt-4o-mini)")
            self.client = OpenAI(api_key=self.openai_key)
            self.backend = "openai"

        else:
            print("⚙️ Using local model:", llm_model_name)
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=hf_token)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                llm_model_name,
                token=hf_token,
                device_map="auto",
                torch_dtype="auto",
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            self.backend = "local"

        # Initialize chat memory
        self.chat_history = []
        print(f"✅ Chatbot ready with backend: {self.backend}")

    # ---------------- FAISS INDEX ----------------
    def _load_index(self):
        idx_file = os.path.join(self.index_path, 'index.faiss')
        meta_file = os.path.join(self.index_path, 'meta.pkl')

        if not os.path.exists(idx_file) or not os.path.exists(meta_file):
            print("⚠️ FAISS index not found. Running ingest.py...")
            subprocess.run(["python", "src/ingest.py"], check=True)

        self.index = faiss.read_index(idx_file)
        with open(meta_file, 'rb') as f:
            data = pickle.load(f)
        self.meta, self.texts = data['meta'], data['texts']
        print(f"✅ Loaded FAISS index with {len(self.texts)} entries.")

    # ---------------- RETRIEVAL ----------------
    def _retrieve(self, query, top_k=5):
        q_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        q_emb = normalize(q_emb).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0:
                results.append({
                    'question': self.meta[int(idx)]['question'],
                    'answer': self.meta[int(idx)]['answer'],
                    'meta': self.meta[int(idx)],
                    'score': float(score)
                })
        return results

    # ---------------- GENERATION ----------------
    def _generate(self, prompt):
        """Auto-select backend for response generation."""
        print(f"🧩 Using backend: {self.backend}")

        if self.backend == "groq":
            try:
                completion = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.25,
                )
                print("🧠 Using Groq model: llama-3.3-70b-versatile")
            except Exception as e:
                print("⚠️ Primary model failed, using fallback:", e)
                completion = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.25,
                )
                print("🧠 Using Groq model: llama-3.1-8b-instant")
            return completion.choices[0].message.content.strip()

        elif self.backend == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.25
            )
            return response.choices[0].message.content.strip()

        else:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=300)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ---------------- MAIN CHAT ----------------
    def ask(self, query, top_k=5, show_sources=False, min_score=0.25):
        chunks = self._retrieve(query, top_k=top_k)
        if not chunks:
            return "Sorry, I couldn't find anything relevant in the knowledge base."

        # Include recent chat context (for continuity)
        recent_context = "\n".join([
            f"User: {q}\nAssistant: {a}" for q, a in self.chat_history[-3:]
        ])

        combined = "\n\n".join([f"{c['question']}: {c['answer']}" for c in chunks])
        prompt = (
            "You are **LawGlance**, a polite and knowledgeable Indian legal assistant.\n"
            "Follow these rules:\n"
            "1. Answer only from the given legal text.\n"
            "2. Use clear sentences and line breaks.\n"
            "3. Add bullet points if explaining multiple steps.\n"
            "4. Keep tone factual and helpful.\n\n"
            f"Recent conversation:\n{recent_context}\n\n"
            f"Relevant legal information:\n{combined}\n\n"
            f"User question: {query}\n\nAnswer:"
        )

        # Store query temporarily
        self.chat_history.append((query, ""))

        # Generate response
        answer = self._generate(prompt)

        # Store final answer in memory
        self.chat_history[-1] = (query, answer)

        # Format output neatly
        formatted_answer = textwrap.fill(answer, width=110)
        if show_sources:
            formatted_answer += "\n\n**Sources:**\n" + "\n".join(
                f"- {c['meta'].get('source', 'unknown')} (score={c['score']:.2f})"
                for c in chunks
            )

        return formatted_answer.strip()
