// background.js
importScripts('config.js');

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'ANALYZE') {
    chrome.tabs.captureVisibleTab(null, { format: 'png' }, async (dataUrl) => {
      if (chrome.runtime.lastError) {
        sendResponse({ error: chrome.runtime.lastError.message });
        return;
      }

      const screenshotBase64 = dataUrl.split(',')[1];

        try {
        console.log('Wysyłam zapytanie do:', CONFIG.API_URL);
        console.log('URL:', CONFIG.API_URL);
        const response = await fetch(CONFIG.API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
            screenshot: screenshotBase64,
            html: msg.html
            })
        });

        console.log('Status odpowiedzi:', response.status);
        const result = await response.json();
        console.log('Wynik:', result);
        
        chrome.tabs.sendMessage(sender.tab.id, {
            type: 'RESULT',
            verdict: result.verdict,
            confidence: result.confidence
        });
        } catch (err) {
        console.error('Błąd fetch:', err);
        chrome.tabs.sendMessage(sender.tab.id, {
            type: 'RESULT',
            error: 'Błąd połączenia z serwerem'
        });
        }
    });

    return true;
  }
});