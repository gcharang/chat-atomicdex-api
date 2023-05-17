import React, { useRef, useEffect } from 'react'

const InputBar = ({ input, setInput, handleKeyDown, handleSubmit }) => {
  const inputRef = useRef(null)

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
      inputRef.current.style.height = inputRef.current.scrollHeight + 'px'
    }
  }, [input])

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex items-center justify-center px-4 py-2 md:px-4 md:py-4">
        <div className="flex items-center w-full max-w-xl md:w-1/2">
          <textarea
            ref={inputRef}
            rows="1"
            placeholder="Does the atomicdex api support lightning?"
            className="flex-1 p-2 overflow-hidden text-gray-100 bg-gray-600 border rounded-lg resize-none focus:outline-none focus:ring focus:border-blue-300"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            type="submit"
            className="px-2 py-1 ml-2 text-white bg-blue-500 rounded-lg focus:outline-none hover:bg-blue-600 md:ml-4 md:px-4 md:py-2"
          >
            Send
          </button>
        </div>
      </form>
      <div className="pb-2 text-xs text-center text-gray-400 md:pb-4">
        This is still a WIP, and answers may not be correct
      </div>
    </div>
  )
}

export default InputBar