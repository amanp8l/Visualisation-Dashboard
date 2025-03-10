import React, { useState } from "react";
import "./FileUpload.css";

const FileUpload = ({ onFileUpload }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    setIsDragging(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    await uploadFile(file);
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    await uploadFile(file);
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8080/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setUploadedFile(file.name);
        onFileUpload(data.filename);
      }
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <div className="file-upload">
      {!uploadedFile ? (
        <div
          className={`dropzone ${isDragging ? "dragging" : ""}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={handleFileSelect}
            id="file-input"
          />
          <label htmlFor="file-input">
            Drop CSV/Excel file here or click to upload
          </label>
        </div>
      ) : (
        <div className="uploaded-file">File uploaded: {uploadedFile}</div>
      )}
    </div>
  );
};

export default FileUpload;
