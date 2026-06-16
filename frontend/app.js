
function setHTML(element, htmlString) {
  if (!element) return;
  element.textContent = '';
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlString, 'text/html');
  while (doc.body.firstChild) {
    element.appendChild(doc.body.firstChild);
  }
}

// DOM Elements
const chatHistory = document.getElementById('chat-history');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');
const streamToggle = document.getElementById('stream-toggle');

// Stats Elements
const statRequests = document.getElementById('stat-requests');
const statCost = document.getElementById('stat-cost');
const barCheap = document.getElementById('bar-cheap');
const barMedium = document.getElementById('bar-medium');
const barStrong = document.getElementById('bar-strong');

// State
let totalRequests = 0;
let totalCost = 0.0;
let tierCounts = { CHEAP: 0, MEDIUM: 0, STRONG: 0 };
let isGenerating = false;

// Auto-resize textarea
promptInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle enter key
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
    const text = promptInput.value.trim();
    if (!text || isGenerating) return;

    // Reset input
    promptInput.value = '';
    promptInput.style.height = 'auto';
    
    // Add user message to UI
    appendMessage('user', text);
    
    // Setup AI response container
    const msgId = 'msg-' + Date.now();
    const contentDiv = appendMessage('assistant', '', msgId);
    
    isGenerating = true;
    sendBtn.disabled = true;

    try {
        const isStreaming = streamToggle.checked;
        
        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'auto',
                messages: [{ role: 'user', content: text }],
                stream: isStreaming
            })
        });

        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        let fullText = '';
        let tier = 'UNKNOWN';
        let latency = 0;
        let modelUsed = 'unknown';

        if (isStreaming) {
            // Read SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            
            // Extract headers for stats
            tier = response.headers.get('X-RouteLLM-Tier') || 'UNKNOWN';
            latency = response.headers.get('X-RouteLLM-Router-Latency-Ms') || 0;
            modelUsed = response.headers.get('X-RouteLLM-Model') || 'unknown';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.choices && data.choices[0].delta.content) {
                                fullText += data.choices[0].delta.content;
                                setHTML(contentDiv, marked.parse(fullText));
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                            }
                        } catch (e) {
                            console.error('JSON parse error:', e, line);
                        }
                    }
                }
            }
        } else {
            // Non-streaming JSON response
            const data = await response.json();
            fullText = data.choices[0].message.content;
            setHTML(contentDiv, marked.parse(fullText));
            
            // Extract _routellm metadata
            const meta = data._routellm || {};
            tier = meta.predicted_tier || (meta.source === 'cache' ? 'CACHED' : 'UNKNOWN');
            latency = meta.total_latency_ms || 0;
            modelUsed = meta.model_used || 'cache';
            
            if (meta.estimated_cost_usd) {
                totalCost += meta.estimated_cost_usd;
            }
        }

        // Add Routing Meta Badge
        appendRoutingMeta(msgId, tier, latency, modelUsed);
        
        // Update Stats
        updateStats(tier);

    } catch (error) {
        setHTML(contentDiv, `<p style="color: #ef4444;">Error: ${error.message}</p>`);
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        promptInput.focus();
    }
}

function appendMessage(role, text, id = null) {
    const div = document.createElement('div');
    div.className = `message ${role}-message`;
    if (id) div.id = id;
    
    const icon = role === 'user' ? '👤' : '⚡';
    
    setHTML(div, `
        <div class="avatar">${icon}</div>
        <div class="message-wrapper">
            <div class="message-content">${marked.parse(text)}</div>
        </div>
    `);
    
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    // Return the content div for updating later
    return div.querySelector('.message-content');
}

function appendRoutingMeta(msgId, tier, latency, modelUsed) {
    const msgDiv = document.getElementById(msgId);
    if (!msgDiv) return;
    
    const wrapper = msgDiv.querySelector('.message-wrapper');
    const metaDiv = document.createElement('div');
    metaDiv.className = 'routing-meta';
    
    const tierLower = tier.toLowerCase();
    const badgeClass = tierLower === 'cached' ? 'badge cached' : `badge tier-${tierLower}`;
    
    setHTML(metaDiv, `
        <span class="${badgeClass}">Tier: ${tier}</span>
        <span class="meta-detail">Latency: ${latency}ms</span>
        <span class="meta-detail">Model: ${modelUsed}</span>
    `);
    
    wrapper.appendChild(metaDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function updateStats(tier) {
    totalRequests++;
    statRequests.innerText = totalRequests;
    statCost.innerText = '$' + totalCost.toFixed(5);
    
    if (tierCounts[tier] !== undefined) {
        tierCounts[tier]++;
    }
    
    // Update bars
    const sum = tierCounts.CHEAP + tierCounts.MEDIUM + tierCounts.STRONG;
    if (sum > 0) {
        barCheap.style.width = (tierCounts.CHEAP / sum * 100) + '%';
        barMedium.style.width = (tierCounts.MEDIUM / sum * 100) + '%';
        barStrong.style.width = (tierCounts.STRONG / sum * 100) + '%';
    }
}
