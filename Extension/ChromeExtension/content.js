setTimeout(() => {
    const html = document.documentElement.outerHTML;

    showBanner({ loading: true});

    chrome.runtime.sendMessage({ type: 'ANALYZE', html});
}, 1500);

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'RESULT') {
        showBanner(msg);
    }
});

function showBanner({ loading, verdict, confidence, error }) {

    const old = document.getElementById('phishguard-banner');
    if (old) old.remove();

    const banner = document.createElement('div');
    banner.id = 'phishguard-banner';
    banner.style.cssText = `
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 2147483647;
        padding: 10px 16px;
        text-align: center;
        font-size: 15px;
        font-family: sans-serif;
        font-weight: bold;
        color: white;
        cursor: pointer;
        transition: opacity 0.3s;
        ${loading  ? 'background: #555;' : ''}
        ${verdict === 'PHISHING' ? 'background: #cc2200;' : ''}
        ${verdict === 'SAFE'     ? 'background: #1a7a1a;' : ''}
        ${error    ? 'background: #886600;' : ''}
    `;

    if (loading) {
        banner.textContent = '🔍 PhishGuard analizuje stronę...';
    } else if (error) {
        banner.textContent = `⚠️ PhishGuard: ${error}`;
    } else if (verdict === 'PHISHING') {
        banner.textContent = `🚨 UWAGA: Możliwy phishing! Pewność: ${confidence}%`;
    } else {
        banner.textContent = `✅ Strona bezpieczna. Pewność: ${confidence}%`;
    }

    banner.addEventListener('click', () => banner.remove());

    document.body.prepend(banner);

}