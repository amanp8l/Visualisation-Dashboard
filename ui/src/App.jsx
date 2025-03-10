import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import ChatInterface from "./components/ChatInterface";
import "./App.css";

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);

  return (
    <div className="app">
      <h1 className="app-header" style={{
        backgroundImage: 'linear-gradient(45deg, #ec4899, #4f46e5)',
        color: 'transparent',
        backgroundClip: 'text',
      }}>Visualisation Dashboard</h1>
      {!uploadedFile ? (
        <FileUpload onFileUpload={setUploadedFile} />
      ) : (
        <ChatInterface filename={uploadedFile} />
      )}
    </div>
  );
}

export default App;
