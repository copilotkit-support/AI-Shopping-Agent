import React, { useState } from 'react';

interface DialogBoxProps {
  contentList: { title: string; url: string }[];
  onAccept: () => void;
  onReject: () => void;
  onNeedInfo: () => void;
  isDisabled: boolean;
}

const DialogBox: React.FC<DialogBoxProps> = ({ contentList, onAccept, onReject, onNeedInfo, isDisabled }) => {
  const [buttonState, setButtonState] = useState<'accept' | 'reject' | 'showAll' | null>(null);

  const handleAccept = () => {
    setButtonState('accept');
    onAccept();
  };

  const handleReject = () => {
    setButtonState('reject');
    onReject();
  };

  const handleNeedInfo = () => {
    setButtonState('showAll');
    onNeedInfo();
  };

  return (
    <div className="border border-gray-300 rounded-lg shadow-lg w-54 max-h-96 overflow-hidden flex flex-col">
      <div className="overflow-y-auto p-4 flex-grow">
      I found and chose the top 5 items you asked for. Would you like to see them, retry, or show all found items?
      </div>
      <div className="flex justify-end p-4 border-t border-gray-200">
        {buttonState === 'accept' && (
          <span className="bg-green-500 text-white py-1 px-3 rounded-full shadow-md text-sm bg-opacity-80 backdrop-filter backdrop-blur-lg opacity-50 cursor-not-allowed">Accepted</span>
        )}
        {buttonState === 'reject' && (
          <span className="bg-red-500 text-white py-1 px-3 rounded-full shadow-md text-sm bg-opacity-80 backdrop-filter backdrop-blur-lg opacity-50 cursor-not-allowed">Rejected</span>
        )}
        {buttonState === 'showAll' && (
          <span className="bg-yellow-500 text-white py-1 px-3 rounded-full shadow-md text-sm bg-opacity-80 backdrop-filter backdrop-blur-lg opacity-50 cursor-not-allowed">Showing All</span>
        )}
        {!buttonState && (
          <>
            <button disabled={isDisabled} onClick={handleAccept} className="bg-green-500 text-white py-1 px-3 rounded-full shadow-md hover:shadow-lg transition-shadow duration-200 text-sm bg-opacity-80 backdrop-filter backdrop-blur-lg mr-2">Accept</button>
            <button disabled={isDisabled} onClick={handleReject} className="bg-red-500 text-white py-1 px-3 rounded-full shadow-md hover:shadow-lg transition-shadow duration-200 text-sm bg-opacity-80 backdrop-filter backdrop-blur-lg mr-2">Reject</button>
            <button disabled={isDisabled} onClick={handleNeedInfo} className="bg-yellow-500 text-white py-1 px-3 rounded-full shadow-md hover:shadow-lg transition-shadow duration-200 text-sm bg-opacity-80 backdrop-filter backdrop-blur-lg">Show All</button>
          </>
        )}
      </div>
    </div>
  );
};

export default DialogBox;
