/*
=============================================================================
 App.jsx  -  The chat UI for our RAG app
=============================================================================
 What it does:
   - lets you UPLOAD a PDF (sends it to POST /upload on the backend)
   - lets you ASK a question (sends it to POST /ask)
   - shows the answer and the source chunks it came from

 The backend must be running at http://127.0.0.1:8000
=============================================================================
*/

import { useState } from "react";
import "./App.css";

const BACKEND = "http://127.0.0.1:8000";

function App() {
  // --- state: little pieces of memory the UI keeps track of ---
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]); // chat history
  const [loading, setLoading] = useState(false);

  // --- UPLOAD a PDF to the backend ---
  async function handleUpload() {
    if (!file) {
      setUploadStatus("Please choose a PDF first.");
      return;
    }
    setUploadStatus("Uploading and indexing...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(BACKEND + "/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setUploadStatus(
        `Indexed "${data.filename}" into ${data.chunks_indexed} chunks. Ask away!`
      );
    } catch (err) {
      setUploadStatus("Upload failed. Is the backend running?");
    }
  }

  // --- ASK a question ---
  async function handleAsk() {
    if (!question.trim()) return;

    const myQuestion = question;
    setQuestion("");
    setLoading(true);

    // show the user's question immediately
    setMessages((prev) => [...prev, { role: "user", text: myQuestion }]);

    try {
      const res = await fetch(BACKEND + "/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: myQuestion, top_k: 3 }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, sources: data.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error: is the backend running?" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>📄 AI Document Q&A</h1>
        <p>Upload a PDF, then ask questions about it. Powered by RAG.</p>
      </header>

      {/* Upload section */}
      <section className="upload-box">
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button onClick={handleUpload}>Upload & Index</button>
        {uploadStatus && <p className="status">{uploadStatus}</p>}
      </section>

      {/* Chat section */}
      <section className="chat">
        {messages.length === 0 && (
          <p className="empty">No questions yet. Upload a PDF and ask something!</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="bubble">
              {m.text}
              {m.sources && m.sources.length > 0 && (
                <details className="sources">
                  <summary>Sources ({m.sources.length})</summary>
                  <ul>
                    {m.sources.map((s, j) => (
                      <li key={j}>{s}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </div>
        ))}
        {loading && <div className="message assistant"><div className="bubble">Thinking…</div></div>}
      </section>

      {/* Ask box */}
      <section className="ask-box">
        <input
          type="text"
          placeholder="Ask a question about your document…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
        />
        <button onClick={handleAsk} disabled={loading}>
          Ask
        </button>
      </section>
    </div>
  );
}

export default App;
