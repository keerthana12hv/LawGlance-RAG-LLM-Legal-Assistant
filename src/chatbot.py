import os
import pickle
import faiss
import torch
import subprocess
import re
import textwrap
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

# Optional provider clients
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
        index_path: str = "./faiss_index",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base",
        device: str | None = None,
    ):
        load_dotenv()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Using device: {self.device}")

        # Embedding model + index
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.index_path = index_path
        self._load_index()

        # Choose backend: Groq -> OpenAI -> Local HF
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if self.groq_key:
            print("🔗 Using Groq API")
            self.client = Groq(api_key=self.groq_key)
            self.backend = "groq"
        elif self.openai_key:
            print("🔗 Using OpenAI API")
            self.client = OpenAI(api_key=self.openai_key)
            self.backend = "openai"
        else:
            print("⚙️ Using local HuggingFace model (fallback)")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True, token=hf_token)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                llm_model_name, use_auth_token=hf_token, device_map="auto"
            ).to(self.device)
            self.backend = "local"

        # lightweight chat memory (last N Q/A pairs)
        self.chat_history: list[tuple[str, str]] = []
        print(f"✅ Chatbot ready with backend: {self.backend}")

    # ---------------- FAISS ----------------
    def _load_index(self):
        idx_file = os.path.join(self.index_path, "index.faiss")
        meta_file = os.path.join(self.index_path, "meta.pkl")

        if not os.path.exists(idx_file) or not os.path.exists(meta_file):
            print("⚠️ FAISS index missing — running ingest.py to create it...")
            subprocess.run(["python", "src/ingest.py"], check=True)

        self.index = faiss.read_index(idx_file)
        with open(meta_file, "rb") as f:
            data = pickle.load(f)
        self.meta, self.texts = data["meta"], data["texts"]
        print(f"✅ Loaded FAISS index with {len(self.texts)} entries.")

    # ---------------- RETRIEVE ----------------
    def _retrieve(self, query: str, top_k: int = 5):
        q_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        q_emb = normalize(q_emb).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0:
                results.append(
                    {
                        "question": self.meta[int(idx)]["question"],
                        "answer": self.meta[int(idx)]["answer"],
                        "meta": self.meta[int(idx)],
                        "score": float(score),
                    }
                )
        return results

    # ---------------- GENERATE ----------------
    def _generate(self, prompt: str, max_tokens: int = 300):
        """Call the configured LLM, handle exceptions and return plain text."""
        print(f"🧩 Generating with backend: {self.backend}")
        try:
            if self.backend == "groq":
                completion = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                answer = completion.choices[0].message.content

            elif self.backend == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content

            else:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            print(f"❌ LLM error: {e}")
            return "⚠️ Sorry — I couldn't generate an answer right now. Please try again."

        # Normalize whitespace & bullets so app can render properly
        answer = re.sub(r"\r\n?", "\n", answer)
        # ensure bullet characters are on their own line
        answer = re.sub(r"\s*•\s*", "\n- ", answer)
        answer = re.sub(r"\s*-\s+", "\n- ", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer.strip()

    # ---------------- ASK ----------------
    def ask(self, query: str, top_k: int = 5, show_sources: bool = False, min_score: float = 0.25):
        q = (query or "").strip()
        if not q:
            return "⚠️ Please type a question."

        q_lower = q.lower()

        # Short small-talk responses
        greetings = ("hello", "hi", "hey", "namaste")
        if any(g in q_lower for g in greetings) and len(q_lower.split()) <= 2:
            return "👋 Hello — I'm LawGlance. Ask a legal question (e.g. 'How to file an FIR?')."

        # Retrieve relevant chunks
        chunks = self._retrieve(q, top_k=top_k)
        if not chunks:
            return "⚖️ I couldn't find relevant documents in the knowledge base. Try rephrasing or use a more specific query."

        # Save small conversation context (last 3)
        recent_ctx = "\n".join([f"User: {a}\nAssistant: {b}" for a, b in self.chat_history[-3:]]) if self.chat_history else ""

        # Determine a friendly heading (UI uses this for the first line)
        heading_map = {
            "fir": "Steps to file an FIR",
            "bail": "Applying for bail - key points",
            "defamation": "Defamation (IPC) - quick facts",
            "article 21": "Article 21 — Right to life & personal liberty",
            "fundamental rights": "Fundamental Rights — overview",
            "cyber": "Registering a cybercrime complaint",
            "vote": "Right age to vote in India",
        }
        heading = "Legal Information"
        for k, v in heading_map.items():
            if k in q_lower:
                heading = v
                break

        combined = "\n\n".join([f"{c['question']}: {c['answer']}" for c in chunks])

        # Strong, explicit instructions to produce neat bullet output (plain text)
        prompt = (
            "You are LawGlance, a concise professional Indian legal assistant. "
            "Answer ONLY using the information below. Output plain text (no JSON), formatted like:\n\n"
            "Summary: <one-line summary>\n\n"
            "Steps or Key Points:\n"
            "- first point\n"
            "- second point\n\n"
            "Rules:\n"
            "• Use short sentences.\n"
            "• Use '-' as the bullet marker at line start.\n"
            "• Do not output Markdown bold (**) or other markup — plain text only.\n"
            "• If the question is a short greeting, reply with one line.\n\n"
            f"Recent chat context (if any):\n{recent_ctx}\n\n"
            f"Relevant texts:\n{combined}\n\n"
            f"User question: {q}\n\nAnswer:"
        )

        answer = self._generate(prompt)

        # store in chat memory
        self.chat_history.append((q, answer))
        # keep memory short
        if len(self.chat_history) > 30:
            self.chat_history = self.chat_history[-30:]

        # if the user requested sources, append a short one-line sources summary
        if show_sources:
            sources_text = "Sources: " + ", ".join(
                f"{c['meta'].get('source','unknown')} (score={c['score']:.2f})" for c in chunks[:top_k]
            )
            answer = f"{answer}\n\n{sources_text}"

        # final cleanup: ensure bullets are on separate lines and first line is a short heading
        answer = re.sub(r"\n\s*-\s*", "\n- ", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return f"{heading}\n\n{answer.strip()}"
