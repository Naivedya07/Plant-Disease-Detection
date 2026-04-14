let currentDisease = "";

function previewImage(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('preview');
    preview.src = e.target.result;
    preview.style.display = 'block';
    document.getElementById('predictBtn').disabled = false;
  };
  reader.readAsDataURL(file);
}

async function predict() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files[0]) return;

  document.getElementById('results').innerHTML =
    '<div class="loading"><div class="spinner"></div><p style="margin-top:10px">Analyzing leaf...</p></div>';
  document.getElementById('predictBtn').disabled = true;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const res  = await fetch('/predict', { method:'POST', body: formData });
    const data = await res.json();
    displayResults(data.predictions);
  } catch(e) {
    document.getElementById('results').innerHTML =
      '<div class="placeholder">❌ Error connecting to server. Make sure backend is running.</div>';
  }
  document.getElementById('predictBtn').disabled = false;
}

function displayResults(predictions) {
  if (!predictions || predictions.length === 0) {
    document.getElementById('results').innerHTML = '<div class="placeholder">No results found.</div>';
    return;
  }

  currentDisease = predictions[0].class;
  document.getElementById('statusBar').innerHTML =
    `🔬 Active Disease: <b>${currentDisease.replace(/_/g,' ')}</b> — Ask me anything about it!`;

  let html = '';
  predictions.forEach((p, i) => {
    html += `
      <div class="result-item ${i===0?'top':''}">
        <div class="result-header">
          <span class="result-name">${i===0?'🥇':'#'+(i+1)} ${p.class.replace(/_/g,' ')}</span>
          <span class="confidence-badge">${p.confidence}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width:${p.confidence}%"></div>
        </div>
        ${i===0 ? `
          <div class="disease-info">
            <p style="color:#81c784;font-size:0.85rem;margin-top:8px">
              💬 Ask the chatbot below about cause, symptoms, solution or prevention!
            </p>
          </div>` : ''}
      </div>`;
  });
  document.getElementById('results').innerHTML = html;

  addBotMessage(`🌿 Detected: <b>${currentDisease.replace(/_/g,' ')}</b> with <b>${predictions[0].confidence}%</b> confidence. Ask me about the cause, symptoms, treatment or prevention!`);
}

async function sendMessage() {
  const input   = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  addUserMessage(message);
  input.value = '';

  try {
    const res  = await fetch('/chat', {
      method : 'POST',
      headers: { 'Content-Type':'application/json' },
      body   : JSON.stringify({ message, disease: currentDisease })
    });
    const data = await res.json();
    addBotMessage(data.reply);
  } catch(e) {
    addBotMessage('❌ Could not connect to server.');
  }
}

function addUserMessage(text) {
  const chatBox = document.getElementById('chatBox');
  chatBox.innerHTML += `
    <div class="msg user">
      <div class="avatar">👤</div>
      <div class="bubble">${text}</div>
    </div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addBotMessage(text) {
  const chatBox = document.getElementById('chatBox');
  chatBox.innerHTML += `
    <div class="msg bot">
      <div class="avatar">🌿</div>
      <div class="bubble">${text}</div>
    </div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
}