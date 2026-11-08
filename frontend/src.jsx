import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
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
    Paperclip,
    Camera,
    X,
    LogOut,
    Eye,
    EyeOff,
    Loader2,
} from "lucide-react";

import "katex/dist/katex.min.css";
import "./style.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TOKEN_KEY = "mathmind_token";

const ACCEPT = "image/*,.pdf,.txt,.md,.tex,.csv,.docx";
const MAX_FILES = 5;
const MAX_MB = 10;

// ---------------------------------------------------------------------------
// Session token + API helper
// ---------------------------------------------------------------------------

function getToken() {
    try {
        return localStorage.getItem(TOKEN_KEY);
    } catch {
        return window.__mathmindToken || null; // storage blocked (private mode)
    }
}

function setToken(token) {
    try {
        if (token) localStorage.setItem(TOKEN_KEY, token);
        else localStorage.removeItem(TOKEN_KEY);
    } catch {
        window.__mathmindToken = token || null;
    }
}

async function apiFetch(path, options = {}) {
    const token = getToken();
    const response = await fetch(`${API}${path}`, {
        ...options,
        headers: {
            ...(options.headers || {}),
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
    });

    const isAuthCall = path.startsWith("/api/auth/login") || path.startsWith("/api/auth/register");
    if (response.status === 401 && !isAuthCall) {
        setToken(null);
        window.dispatchEvent(new Event("mathmind:signed-out"));
    }

    return response;
}

// FastAPI returns a string for most errors but a list of objects for validation errors.
function errorText(detail, fallback) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((item) => item.msg).join(" ") || fallback;
    return fallback;
}

// ---------------------------------------------------------------------------
// Sign in / create account
// ---------------------------------------------------------------------------

function AuthScreen({ onAuth, notice }) {
    const [mode, setMode] = useState("signin");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [name, setName] = useState("");
    const [show, setShow] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");

    const signup = mode === "signup";

    async function submit(event) {
        event.preventDefault();
        setError("");

        const cleanEmail = email.trim();

        if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(cleanEmail)) {
            setError("Enter a valid email address.");
            return;
        }

        if (signup && password.length < 8) {
            setError("Password must be at least 8 characters.");
            return;
        }

        setBusy(true);

        try {
            const response = await apiFetch(`/api/auth/${signup ? "register" : "login"}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email: cleanEmail,
                    password,
                    ...(signup && name.trim() ? { name: name.trim() } : {}),
                }),
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(errorText(data.detail, "Something went wrong. Please try again."));
            }

            setToken(data.token);
            onAuth(data.user);
        } catch (err) {
            setError(
                err instanceof TypeError
                    ? "Can't reach the server. Please try again in a moment."
                    : err.message
            );
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="auth">
            <form className="auth-card" onSubmit={submit} noValidate>
                <div className="mark">
                    <Calculator />
                </div>

                <h1>{signup ? "Create your account" : "Welcome back"}</h1>
                <p className="sub">
                    {signup
                        ? "Save your chats and pick up on any device."
                        : "Sign in to open your MathMind chats."}
                </p>

                <div className="tabs" role="tablist">
                    <button
                        type="button"
                        className={!signup ? "on" : ""}
                        onClick={() => { setMode("signin"); setError(""); }}
                    >
                        Sign in
                    </button>
                    <button
                        type="button"
                        className={signup ? "on" : ""}
                        onClick={() => { setMode("signup"); setError(""); }}
                    >
                        Create account
                    </button>
                </div>

                {notice && <p className="auth-notice">{notice}</p>}
                {error && <p className="auth-error" role="alert">{error}</p>}

                {signup && (
                    <label className="field">
                        <span>Name (optional)</span>
                        <input
                            value={name}
                            onChange={(event) => setName(event.target.value)}
                            autoComplete="name"
                            maxLength={60}
                        />
                    </label>
                )}

                <label className="field">
                    <span>Email</span>
                    <input
                        type="email"
                        value={email}
                        onChange={(event) => setEmail(event.target.value)}
                        autoComplete="email"
                        required
                    />
                </label>

                <label className="field">
                    <span>Password</span>
                    <div className="input-wrap">
                        <input
                            type={show ? "text" : "password"}
                            value={password}
                            onChange={(event) => setPassword(event.target.value)}
                            autoComplete={signup ? "new-password" : "current-password"}
                            placeholder={signup ? "At least 8 characters" : ""}
                            required
                        />
                        <button
                            type="button"
                            onClick={() => setShow((value) => !value)}
                            aria-label={show ? "Hide password" : "Show password"}
                        >
                            {show ? <EyeOff size={17} /> : <Eye size={17} />}
                        </button>
                    </div>
                </label>

                <button className="primary" disabled={busy}>
                    {busy && <Loader2 size={16} className="spin" />}
                    {signup ? "Create account" : "Sign in"}
                </button>

                <p className="fine">Your chats are private to your account.</p>
            </form>
        </div>
    );
}

// ---------------------------------------------------------------------------
// Math rendering helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

const isMobile = () =>
    typeof window.matchMedia === "function" && window.matchMedia("(max-width: 700px)").matches;

function ChatApp({ user, onSignOut }) {
    const [chats, setChats] = useState([]);
    const [cid, setCid] = useState(null);
    const [msgs, setMsgs] = useState([]);
    const [text, setText] = useState("");
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState("");
    // On phones the sidebar is an overlay, so start with it closed.
    const [side, setSide] = useState(() => !isMobile());
    const [files, setFiles] = useState([]);
    const [hint, setHint] = useState("");
    const [dragging, setDragging] = useState(false);

    const end = useRef(null);
    const input = useRef(null);
    const fileInput = useRef(null);
    const cameraInput = useRef(null);

    const reading = files.some((file) => file.status === "reading");
    const initial = (user.name || user.email || "?").trim().charAt(0);

    useEffect(() => {
        load();
    }, []);

    useEffect(() => {
        end.current?.scrollIntoView({ behavior: "smooth" });
    }, [msgs, busy]);

    // Grow the input with its content (up to the CSS max-height, then scroll).
    useLayoutEffect(() => {
        const element = input.current;
        if (!element) return;
        element.style.height = "auto";
        element.style.height = `${element.scrollHeight}px`;
    }, [text]);

    async function load() {
        try {
            const response = await apiFetch("/api/conversations");

            if (!response.ok) {
                throw new Error("Could not load conversations.");
            }

            const data = await response.json();
            setChats(data);

            if (data.length && !cid) {
                await open(data[0].id);
            }
        } catch {
            setErr("Backend unavailable. Please try again shortly.");
        }
    }

    async function open(id) {
        try {
            const response = await apiFetch(`/api/conversations/${id}`);

            if (!response.ok) {
                throw new Error("Could not open conversation.");
            }

            const data = await response.json();

            setCid(id);
            if (isMobile()) setSide(false);
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
        if (isMobile()) setSide(false);
        setCid(null);
        setMsgs([]);
        setText("");
        setErr("");
        setHint("");
        setFiles([]);
        input.current?.focus();
    }

    async function send(event) {
        event?.preventDefault();

        const question = text.trim();

        if (!question || busy || reading) return;

        setText("");
        setHint("");
        setMsgs((messages) => [
            ...messages,
            { role: "user", content: question },
        ]);
        setBusy(true);
        setErr("");

        try {
            const response = await apiFetch("/api/chat", {
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
                throw new Error(errorText(data.detail, "Request failed."));
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
            const offline = error instanceof TypeError;

            setMsgs((messages) => [
                ...messages,
                {
                    role: "assistant",
                    content: offline
                        ? "Could not reach the server. Please check your connection and try again."
                        : error.message,
                },
            ]);

            setErr(offline ? "Connection problem. Please try again." : "");
        } finally {
            setBusy(false);
            input.current?.focus();
        }
    }

    async function del(id, event) {
        event.stopPropagation();

        if (!confirm("Delete this conversation?")) return;

        try {
            const response = await apiFetch(`/api/conversations/${id}`, {
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

    // -- Attachments: photos, handwriting, PDFs, documents -> editable text ----

    async function addFiles(fileList) {
        const list = Array.from(fileList || []);

        if (!list.length) return;

        if (list.length > MAX_FILES) {
            setErr(`Upload at most ${MAX_FILES} files at a time.`);
            return;
        }

        const tooBig = list.find((file) => file.size > MAX_MB * 1024 * 1024);

        if (tooBig) {
            setErr(`${tooBig.name || "This file"} is larger than ${MAX_MB} MB.`);
            return;
        }

        setErr("");
        setHint("");

        const entries = list.map((file) => ({
            id: crypto.randomUUID(),
            name: file.name || "pasted image",
            status: "reading",
        }));
        const ids = entries.map((entry) => entry.id);

        setFiles((current) => [...current, ...entries]);

        const form = new FormData();
        list.forEach((file, index) => form.append("files", file, file.name || `image-${index}.png`));

        try {
            const response = await apiFetch("/api/extract", { method: "POST", body: form });
            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(errorText(data.detail, "Could not read the file."));
            }

            const items = data.items || [];
            const failed = entries
                .map((entry, index) => ({
                    ...entry,
                    status: "error",
                    error: items[index]?.error || "Could not read this file.",
                }))
                .filter((_, index) => !items[index]?.ok);

            setFiles((current) => [...current.filter((file) => !ids.includes(file.id)), ...failed]);

            if (data.text) {
                setText((previous) => [previous.trim(), data.text].filter(Boolean).join("\n\n"));

                const readNames = entries
                    .filter((_, index) => items[index]?.ok)
                    .map((entry) => entry.name)
                    .join(", ");
                const notes = items.flatMap((item) => item.notes || []);

                setHint(
                    [
                        `Text read from ${readNames}. Check it before sending, especially handwriting and formulas.`,
                        data.truncated ? "It was long, so it was shortened." : "",
                        ...notes,
                    ]
                        .filter(Boolean)
                        .join(" ")
                );

                input.current?.focus();
            }
        } catch (error) {
            const message =
                error instanceof TypeError
                    ? "Could not reach the server."
                    : error.message;

            setFiles((current) =>
                current.map((file) =>
                    ids.includes(file.id) ? { ...file, status: "error", error: message } : file
                )
            );
        }
    }

    function dismissFile(id) {
        setFiles((current) => current.filter((file) => file.id !== id));
    }

    function pickFiles(event) {
        const picked = Array.from(event.target.files || []);
        event.target.value = "";
        addFiles(picked);
    }

    return (
        <div className="app">
            {side && <div className="scrim" onClick={() => setSide(false)} />}

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

                    <div className="chat-list">
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
                    </div>

                    <footer>
                        <div className="who" title={user.email}>
                            <span className="who-avatar">{initial}</span>
                            <span className="who-text">{user.name || user.email}</span>
                        </div>

                        <button className="signout" onClick={onSignOut} title="Sign out" aria-label="Sign out">
                            <LogOut size={16} />
                        </button>
                    </footer>
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
                    <span>AI + SymPy</span>
                </header>

                <section className="feed">
                    {!msgs.length ? (
                        <div className="welcome">
                            <div className="mark">
                                <Calculator />
                            </div>

                            <h1>What math can we solve?</h1>

                            <p>
                                Type a problem, or attach a photo, handwritten page, PDF or document.
                                I'll calculate with SymPy, explain the steps and verify the result.
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
                                        onClick={() => {
                                            setText(example);
                                            input.current?.focus();
                                        }}
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

                <div
                    className={`compose ${dragging ? "dragging" : ""}`}
                    onDragOver={(event) => {
                        event.preventDefault();
                        setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(event) => {
                        event.preventDefault();
                        setDragging(false);
                        addFiles(event.dataTransfer.files);
                    }}
                >
                    {err && <p className="error">{err}</p>}
                    {hint && <p className="hint">{hint}</p>}

                    {files.length > 0 && (
                        <div className="chips">
                            {files.map((file) => (
                                <span
                                    key={file.id}
                                    className={`chip ${file.status}`}
                                    title={file.error || file.name}
                                >
                                    {file.status === "reading" && <Loader2 size={13} className="spin" />}

                                    <span>
                                        {file.status === "reading"
                                            ? `Reading ${file.name}…`
                                            : `${file.name}: ${file.error}`}
                                    </span>

                                    {file.status === "error" && (
                                        <button
                                            type="button"
                                            onClick={() => dismissFile(file.id)}
                                            aria-label="Dismiss"
                                        >
                                            <X size={13} />
                                        </button>
                                    )}
                                </span>
                            ))}
                        </div>
                    )}

                    <form onSubmit={send}>
                        <input
                            ref={fileInput}
                            type="file"
                            accept={ACCEPT}
                            multiple
                            hidden
                            onChange={pickFiles}
                        />

                        <input
                            ref={cameraInput}
                            type="file"
                            accept="image/*"
                            capture="environment"
                            hidden
                            onChange={pickFiles}
                        />

                        <button
                            type="button"
                            className="icon"
                            onClick={() => fileInput.current?.click()}
                            title="Attach a photo, PDF or document"
                            aria-label="Attach a photo, PDF or document"
                        >
                            <Paperclip />
                        </button>

                        <button
                            type="button"
                            className="icon"
                            onClick={() => cameraInput.current?.click()}
                            title="Take a photo of a problem"
                            aria-label="Take a photo of a problem"
                        >
                            <Camera />
                        </button>

                        <textarea
                            ref={input}
                            value={text}
                            onChange={(event) => setText(event.target.value)}
                            placeholder="Message MathMind, or attach a photo or file…"
                            rows="1"
                            onKeyDown={(event) => {
                                if (event.key === "Enter" && !event.shiftKey) {
                                    event.preventDefault();
                                    send();
                                }
                            }}
                            onPaste={(event) => {
                                const pasted = Array.from(event.clipboardData?.files || []);

                                if (pasted.length) {
                                    event.preventDefault();
                                    addFiles(pasted);
                                }
                            }}
                        />

                        <button className="send" disabled={!text.trim() || busy || reading} aria-label="Send">
                            <Send />
                        </button>
                    </form>

                    <small>
                        Enter to send · Shift+Enter for newline · Paste or drop images and files · Verify important results
                    </small>
                </div>
            </main>
        </div>
    );
}

// ---------------------------------------------------------------------------
// App: sign-in gate
// ---------------------------------------------------------------------------

function App() {
    const [user, setUser] = useState(null);
    const [ready, setReady] = useState(false);
    const [notice, setNotice] = useState("");

    useEffect(() => {
        let alive = true;

        const signedOut = () => {
            setUser(null);
            setNotice("Your session ended. Please sign in again.");
        };

        window.addEventListener("mathmind:signed-out", signedOut);

        (async () => {
            if (getToken()) {
                try {
                    const response = await apiFetch("/api/auth/me");

                    if (response.ok && alive) {
                        setUser(await response.json());
                    }
                } catch {
                    if (alive) setNotice("Can't reach the server right now. Please try again shortly.");
                }
            }

            if (alive) setReady(true);
        })();

        return () => {
            alive = false;
            window.removeEventListener("mathmind:signed-out", signedOut);
        };
    }, []);

    async function signOut() {
        try {
            await apiFetch("/api/auth/logout", { method: "POST" });
        } catch {
            // Signing out locally is enough even if the server can't be reached.
        }

        setToken(null);
        setNotice("");
        setUser(null);
    }

    if (!ready) {
        return (
            <div className="splash">
                <Loader2 className="spin" />
            </div>
        );
    }

    if (!user) {
        return (
            <AuthScreen
                notice={notice}
                onAuth={(signedIn) => {
                    setNotice("");
                    setUser(signedIn);
                }}
            />
        );
    }

    return <ChatApp key={user.id} user={user} onSignOut={signOut} />;
}

createRoot(document.getElementById("root")).render(<App />);
