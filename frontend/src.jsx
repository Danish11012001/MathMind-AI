
import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import {
    Plus,
    MessageSquare,
    Trash2,
    Send,
    Sparkles,
    Calculator,
    Copy,
    Menu,
} from "lucide-react";

import "katex/dist/katex.min.css";
import "./style.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
    const [chats, setChats] = useState([]);
    const [cid, setCid] = useState(null);
    const [msgs, setMsgs] = useState([]);
    const [text, setText] = useState("");
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState("");
    const [side, setSide] = useState(true);

    const end = useRef(null);
    const input = useRef(null);

    useEffect(() => {
        load();
    }, []);

    useEffect(() => {
        end.current?.scrollIntoView({ behavior: "smooth" });
    }, [msgs, busy]);

    async function load() {
        try {
            const response = await fetch(`${API}/api/conversations`);

            if (!response.ok) {
                throw new Error("Could not load conversations.");
            }

            const data = await response.json();
            setChats(data);

            if (data.length && !cid) {
                await open(data[0].id);
            }
        } catch {
            setErr("Backend unavailable. Start FastAPI on port 8000.");
        }
    }

    async function open(id) {
        try {
            const response = await fetch(`${API}/api/conversations/${id}`);

            if (!response.ok) {
                throw new Error("Could not open conversation.");
            }

            const data = await response.json();

            setCid(id);
            setMsgs(
                data.messages.map((message) => ({
                    role: message.role,
                    content: message.content,
                    metadata: message.metadata,
                }))
            );

            setErr("");
        } catch {
            setErr("Unable to open conversation.");
        }
    }

    function fresh() {
        setCid(null);
        setMsgs([]);
        setText("");
        setErr("");
        input.current?.focus();
    }

    async function send(event) {
        event?.preventDefault();

        const question = text.trim();

        if (!question || busy) return;

        setText("");
        setMsgs((messages) => [
            ...messages,
            { role: "user", content: question },
        ]);
        setBusy(true);
        setErr("");

        try {
            const response = await fetch(`${API}/api/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: question,
                    conversation_id: cid,
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Request failed.");
            }

            setCid(data.conversation_id);

            setMsgs((messages) => [
                ...messages,
                {
                    role: "assistant",
                    content: data.response,
                    metadata: data,
                },
            ]);

            await load();
        } catch (error) {
            setMsgs((messages) => [
                ...messages,
                {
                    role: "assistant",
                    content: `Could not reach the backend. ${error.message}`,
                },
            ]);

            setErr("Check that FastAPI and Ollama are running.");
        } finally {
            setBusy(false);
            input.current?.focus();
        }
    }

    async function del(id, event) {
        event.stopPropagation();

        if (!confirm("Delete this conversation?")) return;

        try {
            const response = await fetch(`${API}/api/conversations/${id}`, {
                method: "DELETE",
            });

            if (!response.ok) {
                throw new Error("Unable to delete conversation.");
            }

            if (id === cid) {
                fresh();
            }

            await load();
        } catch {
            setErr("Unable to delete conversation.");
        }
    }

    // The backend already normalizes math delimiters. This is only a light
    // safety net for \( \) / \[ \] that slip through; it must NOT split on
    // "\\n" (that is a literal backslash-n, which also appears in \neq, \nu...).
    function normalizeMathMarkdown(content) {
        return String(content || "")
            .replace(/\\\(([^\n]*?)\\\)/g, (_, m) => `$${m.trim()}$`)
            .replace(/^([ \t]*)\\\[([\s\S]*?)\\\]/gm, (_, indent, m) =>
                `\n${indent}$$\n${m
                    .trim()
                    .split("\n")
                    .map((l) => indent + l.trim())
                    .join("\n")}\n${indent}$$\n`
            );
    }

    function renderMessage(message) {
        if (message.role === "user") {
            return message.content;
        }

        return (
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                {normalizeMathMarkdown(message.content)}
            </ReactMarkdown>
        );
    }

    return (
        <div className="app">
            {side && (
                <aside>
                    <div className="brand">
                        <Calculator />
                        MathMind
                        <button onClick={() => setSide(false)}>×</button>
                    </div>

                    <button className="new" onClick={fresh}>
                        <Plus />
                        New chat
                    </button>

                    <small>RECENT CHATS</small>

                    {chats.map((chat) => (
                        <div
                            className={`chat ${chat.id === cid ? "active" : ""}`}
                            key={chat.id}
                            onClick={() => open(chat.id)}
                        >
                            <MessageSquare />
                            <span>{chat.title}</span>

                            <button onClick={(event) => del(chat.id, event)}>
                                <Trash2 size={14} />
                            </button>
                        </div>
                    ))}

                    <footer>● Local AI · Ollama</footer>
                </aside>
            )}

            <main>
                {!side && (
                    <button className="menu" onClick={() => setSide(true)}>
                        <Menu />
                    </button>
                )}

                <header>
                    <Sparkles />
                    Mathematical AI
                    <span>Qwen + SymPy</span>
                </header>

                <section className="feed">
                    {!msgs.length ? (
                        <div className="welcome">
                            <div className="mark">
                                <Calculator />
                            </div>

                            <h1>What math can we solve?</h1>

                            <p>
                                Ask naturally. I'll calculate with SymPy and explain the steps.
                            </p>

                            <div className="examples">
                                {[
                                    "Solve x**2 - 5*x + 6",
                                    "Derivative of x**3 + 2*x",
                                    "Integrate x**2 + 3*x",
                                    "Determinant of [[5,2],[1,3]]",
                                ].map((example) => (
                                    <button
                                        key={example}
                                        onClick={() => setText(example)}
                                    >
                                        {example} ↗
                                    </button>
                                ))}
                            </div>
                        </div>
                    ) : (
                        msgs.map((message, index) => (
                            <article className={message.role} key={index}>
                                {message.role === "assistant" && (
                                    <div className="avatar">
                                        <Sparkles />
                                    </div>
                                )}

                                <div className="bubble">
                                    <div className="markdown-content">
                                        {renderMessage(message)}
                                    </div>

                                    {message.role === "assistant" && (
                                        <button
                                            className="copy"
                                            onClick={() =>
                                                navigator.clipboard.writeText(message.content)
                                            }
                                        >
                                            <Copy size={13} />
                                            Copy
                                        </button>
                                    )}
                                </div>
                            </article>
                        ))
                    )}

                    {busy && (
                        <article className="assistant">
                            <div className="avatar">
                                <Sparkles />
                            </div>

                            <div className="bubble">Thinking and calculating…</div>
                        </article>
                    )}

                    <div ref={end} />
                </section>

                <div className="compose">
                    {err && <p className="error">{err}</p>}

                    <form onSubmit={send}>
            <textarea
                ref={input}
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Message MathMind…"
                rows="1"
                onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        send();
                    }
                }}
            />

                        <button disabled={!text.trim() || busy}>
                            <Send />
                        </button>
                    </form>

                    <small>
                        Enter to send · Shift+Enter for newline · Verify important results
                    </small>
                </div>
            </main>
        </div>
    );
}

createRoot(document.getElementById("root")).render(<App />);
