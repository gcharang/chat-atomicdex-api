import React from 'react';

const Header = () => {
  return (
    <div className="py-4 text-center text-white bg-gray-700">
      <h1 className="text-2xl font-semibold">Chat With AtomicDEX API github repo</h1>
      <div className="pt-2 text-xs text-center text-gray-400">
        <a
          href="https://github.com/KomodoPlatform/atomicDEX-API"
          target="_blank"
          rel="noopener noreferrer"
        >
          https://github.com/KomodoPlatform/atomicDEX-API
        </a>
      </div>
    </div>
  );
};

export default Header;