'use client'

import { useState, useEffect } from 'react'
import Image from 'next/image'

type Message = {
  role: 'user' | 'ai'
  content: string
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{type: 'success' | 'error', text: string} | null>(null)
  const [uploadedPdf, setUploadedPdf] = useState<string | null>(null)
  const [showUploadSection, setShowUploadSection] = useState(true)
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isAsking, setIsAsking] = useState(false)

  // Clear success message after 2 seconds
  useEffect(() => {
    if (message?.type === 'success') {
      const timer = setTimeout(() => {
        setMessage(null)
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [message])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setMessage(null)
    }
  }

  const handleDelete = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/delete-all-documents`, {
        method: 'DELETE',
      })

      if (response.ok) {
        setUploadedPdf(null)
        setShowUploadSection(true)
        setMessage({ type: 'success', text: 'PDF deleted successfully!' })
        // Clear chat history when PDF is deleted
        setMessages([])
        setInputMessage('')
      } else {
        throw new Error('Failed to delete PDF')
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Failed to delete PDF'
      })
    }
  }

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);

    try {
      console.log('Uploading file:', file.name);
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/process-pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Upload response:', data);

      setMessage({ 
        type: 'success', 
        text: `PDF processed successfully! Created ${data.chunks} chunks and stored ${data.stored_documents} documents.` 
      });
      setUploadedPdf(file.name);
      setShowUploadSection(false);
      setFile(null);
      
      // Verify data was stored by making a test request
      const testResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/test-vectorstore`);
      const testData = await testResponse.json();
      console.log('Vectorstore test:', testData);
      
    } catch (error) {
      console.error('Error:', error);
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'An error occurred while processing the PDF' 
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = inputMessage.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInputMessage('');
    setIsAsking(true);

    try {
      console.log('Sending question:', userMessage);
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: userMessage }),
      });

      const data = await response.json();
      console.log('Question response:', data);

      if (response.ok) {
        setMessages(prev => [...prev, { role: 'ai', content: data.answer }]);
      } else {
        throw new Error(data.error || 'Failed to get response');
      }
    } catch (error) {
      console.error('Error asking question:', error);
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: error instanceof Error ? error.message : 'Sorry, I encountered an error processing your question.' 
      }]);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header - Minimal height when PDF is loaded */}
      <div className={`text-center ${!showUploadSection ? 'py-2' : 'py-8'}`}>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
          DocTalk
        </h1>
        {showUploadSection && (
          <p className="text-lg text-gray-600 dark:text-gray-300 mt-2">
            Upload your PDF document to get started
          </p>
        )}
      </div>

      {/* Main Content - Flex grow to fill available space */}
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full px-4 pb-4">
        {showUploadSection ? (
          /* Upload Section */
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
            <div className="space-y-8">
              {/* File Input */}
              <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 transition-colors hover:border-gray-400 dark:hover:border-gray-500">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center"
                >
                  <svg
                    className="w-12 h-12 text-gray-400 dark:text-gray-500 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V7a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <span className="text-gray-600 dark:text-gray-400">
                    {file ? file.name : 'Click to select a PDF file'}
                  </span>
                </label>
              </div>

              {/* Upload Button */}
              <div className="flex justify-center">
                <button
                  onClick={() => file && handleUpload(file)}
                  disabled={!file || loading}
                  className={`
                    px-6 py-3 rounded-lg text-white font-medium
                    ${!file || loading
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700'}
                    transition-colors duration-200
                  `}
                >
                  {loading ? 'Processing...' : 'Upload PDF'}
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* Chat Interface */
          <div className="flex-1 flex flex-col bg-white dark:bg-gray-800 rounded-lg shadow-lg">
            {/* PDF Info - Top */}
            {uploadedPdf && (
              <div className="flex items-center justify-between p-3 border-b border-gray-200 dark:border-gray-700">
                <span className="text-sm text-gray-700 dark:text-gray-300">{uploadedPdf}</span>
                <button
                  onClick={handleDelete}
                  className="text-red-600 hover:text-red-700 transition-colors duration-200"
                  title="Delete PDF"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            )}

            {/* Messages Container - Flex grow to fill available space */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message, index) => (
                <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[70%] rounded-lg px-4 py-2 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
                    }`}
                    dangerouslySetInnerHTML={{ __html: message.content }}
                  />
                </div>
              ))}
              {isAsking && (
                <div className="flex justify-start">
                  <div className="bg-gray-200 dark:bg-gray-700 rounded-lg px-4 py-2 text-gray-900 dark:text-gray-100">
                    <span className="animate-pulse">...</span>
                  </div>
                </div>
              )}
            </div>

            {/* Chat Input - Fixed height */}
            <div className="border-t border-gray-200 dark:border-gray-700 p-4">
              <div className="flex space-x-4">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask a question about your PDF..."
                  className="flex-1 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-4 py-2 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isAsking || !inputMessage.trim()}
                  className={`px-6 py-2 rounded-lg text-white font-medium ${
                    isAsking || !inputMessage.trim()
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 hover:bg-blue-700'
                  } transition-colors duration-200`}
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Message Display */}
        {message && (
          <div className={`
            mt-4 text-center p-4 rounded-lg
            ${message.type === 'success' 
              ? 'bg-green-100 text-green-800 dark:bg-green-800/20 dark:text-green-400'
              : 'bg-red-100 text-red-800 dark:bg-red-800/20 dark:text-red-400'}
          `}>
            {message.text}
          </div>
        )}
      </div>
    </div>
  )
}