import { useEffect, useRef, useState } from "react";
import "./App.css";

type Message = {
  id: number;
  role: "user" | "ai";
  content: string;
};

function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const hasMessages = messages.length > 0;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    console.log(messages);
  }, [messages]);

  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    const userMessage: Message = {
      id: Date.now(),
      role: "user",
      content: trimmedInput,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: trimmedInput }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from backend.");
      }

      const data = await response.json();
      const aiMessage: Message = {
        id: Date.now() + 1,
        role: "ai",
        content: data?.response || "No response received.",
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "ai" as const,
          content: "Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-inner">
          {!hasMessages && (
            <div className="intro-container intro-enter">
              <div className="intro-title">
                <span className="logo-circle" />
                DOTproduct
              </div>
              <p className="intro-subtitle">Ask us anything...</p>
            </div>
          )}

          {hasMessages &&
            messages.map((message) => (
              <div key={message.id} className={`message ${message.role === "user" ? "user" : "ai"} message-enter`}>
                {message.content}
              </div>
            ))}

          {hasMessages && isLoading && (
            <div className="message ai message-enter">
              <span className="thinking">Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="input-container">
        <div className="input-inner">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            disabled={isLoading}
            className="input-box"
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="send-btn"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatApp;
