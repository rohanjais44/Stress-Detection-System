import { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [page, setPage] = useState("home");
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  const recognitionRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 🎤 Voice Input
  const startVoice = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Voice not supported");
      return;
    }

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.onresult = (event) => {
      setText(event.results[0][0].transcript);
    };
    recognitionRef.current.start();
  };

  // 🔗 API Call
  const predict = async () => {
    if (!text.trim()) return;

    const userMsg = { sender: "user", text };
    setMessages((prev) => [...prev, userMsg]);

    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      const botMsg = {
        sender: "bot",
        text: data.prediction,
        confidence: Math.round(data.confidence * 100),
        suggestions: data.suggestions,
        explanations: data.explanations, // now simple text
      };

      setMessages((prev) => [...prev, botMsg]);
      setText("");
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠ Server Error" },
      ]);
    }

    setLoading(false);
  };

  // 📥 Download Report
  const downloadReport = () => {
    const content = messages
      .map((m) =>
        m.sender === "user"
          ? `User: ${m.text}`
          : `Prediction: ${m.text} (${m.confidence || ""}%)`
      )
      .join("\n");

    const blob = new Blob([content], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "stress_report.txt";
    a.click();
  };

  // ---------------- HOME ----------------

  const renderHome = () => (
    <>
      <h2 className="title">🧠 Stress Analyzer</h2>

      <div className="chat">
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.sender === "user" ? "userMsg" : "botMsg"}
          >
            {m.sender === "user" ? (
              m.text
            ) : (
              <>
                {/* Prediction */}
                <div className="status">
                  {m.text === "Stress"
                    ? "⚠ High Stress"
                    : "✅ Relaxed"}
                </div>

                {/* Confidence */}
                <small>Confidence: {m.confidence}%</small>

                {/* Progress */}
                <div className="progressOuter">
                  <div
                    className="progressInner"
                    style={{
                      width: `${m.confidence}%`,
                      background:
                        m.text === "Stress"
                          ? "#ff4d4f"
                          : "#2ecc71",
                    }}
                  />
                </div>

                {/* 🔍 USER-FRIENDLY EXPLANATION */}
                {m.explanations && (
                  <div style={{ marginTop: "10px" }}>
                    <b>🔍 Why this result?</b>
                    <ul style={{ fontSize: "13px" }}>
                      {m.explanations.map((e, i) => (
                        <li key={i}> {e}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Suggestions */}
                {m.suggestions && (
                  <ul className="suggestions">
                    {m.suggestions.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </div>
        ))}

        {loading && <div className="botMsg">Analyzing...</div>}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <textarea
        className="textarea"
        value={text}
        placeholder="Tell me how you feel..."
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && predict()}
      />

      {/* Buttons */}
      <div className="buttonRow">
        <button className="btn" onClick={predict}>
          Analyze
        </button>

        <button className="voiceBtn" onClick={startVoice}>
          🎤
        </button>

        <button className="downloadBtn" onClick={downloadReport}>
          Save
        </button>
      </div>
    </>
  );

  // ---------------- OTHER PAGES ----------------

  const renderBlogs = () => (
    <div>
      <h2>📚 Blogs</h2>
      <p>• Understanding stress triggers</p>
      <p>• Workplace mental health</p>
      <p>• Digital wellbeing</p>
    </div>
  );

  const renderResearch = () => (
    <div>
      <h2>📄 Research</h2>
      <p>
        Hybrid NLP + ML model with Explainable AI for stress detection.
      </p>
    </div>
  );

  const renderHelp = () => (
    <div>
      <h2>🧘 Stress Help</h2>
      <p>✔ Exercise regularly</p>
      <p>✔ Sleep well</p>
      <p>✔ Practice meditation</p>
      <p>✔ Talk to someone</p>
    </div>
  );

  const renderAbout = () => (
    <div>
      <h2>ℹ About</h2>
      <p>
        This system detects stress using AI and explains results in a
        simple and understandable way.
      </p>
    </div>
  );

  // ---------------- UI ----------------

  return (
    <div className="page">
      <div className="bubbles">
        <span></span><span></span><span></span><span></span><span></span>
      </div>

      <div className="navbar">
        <div className="logo">Stress AI</div>
        <div className="menu">
          <span onClick={() => setPage("home")}>Home</span>
          <span onClick={() => setPage("blogs")}>Blogs</span>
          <span onClick={() => setPage("research")}>Research</span>
          <span onClick={() => setPage("help")}>Help</span>
          <span onClick={() => setPage("about")}>About</span>
        </div>
      </div>

      <div className="card">
        {page === "home" && renderHome()}
        {page === "blogs" && renderBlogs()}
        {page === "research" && renderResearch()}
        {page === "help" && renderHelp()}
        {page === "about" && renderAbout()}
      </div>
    </div>
  );
}

export default App;