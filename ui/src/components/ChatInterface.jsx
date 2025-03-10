import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./ChatInterface.css";

const ChatInterface = ({ filename }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [popupImage, setPopupImage] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setIsLoading(true);

    setMessages((prev) => [...prev, { type: "user", content: userMessage }]);

    try {
      const response = await fetch("http://localhost:8080/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userMessage, filename }),
      });

      const data = await response.json();

      if (response.ok) {
        if (data.analyses && Array.isArray(data.analyses) && data.analyses.length > 0) {
          // Handle the analyses array format
          data.analyses.forEach((analysisItem) => {
            setMessages((prev) => [
              ...prev,
              {
                type: "bot",
                content: analysisItem.analysis,
                plot: analysisItem.plot,
              },
            ]);
          });
        } else if (data.response) {
          // Handle the simple response format
          setMessages((prev) => [
            ...prev,
            {
              type: "bot",
              content: data.response,
              plot: data.plot, // This will be undefined if there's no plot
            },
          ]);
        } else {
          // Fallback for unexpected response format
          setMessages((prev) => [
            ...prev,
            {
              type: "bot",
              content: "Received an unexpected response format from the server.",
            },
          ]);
        }
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            content: "Sorry, I encountered an error. Please try again.",
          },
        ]);
      }
    } catch (error) {
      console.error("Failed to get response:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          content: "Network error. Please check your connection.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageClick = (imageUrl) => {
    setPopupImage(imageUrl);
  };

  const closePopup = () => {
    setPopupImage(null);
  };

  return (
    <div className="chat-interface">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.type}`}>
            <div className="content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  table: ({ node, ...props }) => (
                    <div className="table-container">
                      <table {...props} />
                    </div>
                  ),
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </div>
            {msg.plot && (
              <div className="plot-container">
                {Array.isArray(msg.plot) ? (
                  msg.plot.map((plotUrl, index) => (
                    <img
                      key={index}
                      src={plotUrl}
                      alt={`Data visualization ${index + 1}`}
                      className="plot"
                      loading="lazy"
                      onClick={() => handleImageClick(plotUrl)}
                    />
                  ))
                ) : (
                  <img
                    src={msg.plot}
                    alt="Data visualization"
                    className="plot"
                    loading="lazy"
                    onClick={() => handleImageClick(msg.plot)}
                  />
                )}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message bot">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your data..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Processing..." : "Send"}
        </button>
      </form>

      {/* Image Popup */}
      {popupImage && (
        <div className="image-popup-overlay" onClick={closePopup}>
          <div className="image-popup-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-popup" onClick={closePopup}>×</button>
            <img src={popupImage} alt="Enlarged visualization" />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatInterface;