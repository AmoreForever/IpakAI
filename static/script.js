const themeToggle = document.getElementById('theme-toggle');
const html = document.documentElement;

// Check for saved theme preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    html.classList.toggle('dark', savedTheme === 'dark');
}

// Check system preference if no saved theme
if (!savedTheme) {
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    html.classList.toggle('dark', systemDark);
}

// Theme toggle handler
themeToggle.addEventListener('click', () => {
    // Add animation class
    themeToggle.classList.add('animate-toggle');
    
    // Toggle dark mode
    html.classList.toggle('dark');
    localStorage.setItem('theme', html.classList.contains('dark') ? 'dark' : 'light');
    
    // Remove animation class after animation completes
    setTimeout(() => {
        themeToggle.classList.remove('animate-toggle');
    }, 500);
});

const recordBtn = document.getElementById('record-btn');
const stopBtn = document.getElementById('stop-btn');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const status = document.getElementById('status');
const chatHistory = document.getElementById('chat-history');
const ttsPlayer = document.getElementById('tts-player');
const waveform = document.getElementById('waveform');

let mediaRecorder;
let audioChunks = [];

navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');

            fetch('/process_voice', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        status.textContent = data.error;
                    } else {
                        addToChatHistory('You', data.transcript || 'Voice input');
                        addToChatHistory('Bot', data.response);
                        if (data.audio_url) {
                            ttsPlayer.src = data.audio_url;
                            ttsPlayer.play().catch(error => {
                                console.error('Playback error:', error);
                                status.textContent = 'Error playing audio response';
                            });
                        }
                    }
                    status.textContent = '';
                    audioChunks = [];
                    stopBtn.classList.add('hidden');
                    recordBtn.classList.remove('hidden');
                    waveform.style.display = 'none';
                })
                .catch(error => {
                    console.error('Error:', error);
                    status.textContent = 'Error processing voice input';
                    stopBtn.classList.add('hidden');
                    recordBtn.classList.remove('hidden');
                    waveform.style.display = 'none';
                });
        };
    })
    .catch(error => {
        console.error('Microphone access error:', error);
        status.textContent = 'Mikrofon ruxsatini bering.';
    });

recordBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'inactive') {
        mediaRecorder.start();
        recordBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        status.textContent = 'Recording...';
        waveform.style.display = 'flex';
    }
});

stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        status.textContent = 'Processing...';
    }
});

sendBtn.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (text) {
        addToChatHistory('You', text);
        fetch('/process_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `text=${encodeURIComponent(text)}`
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    status.textContent = data.error;
                } else {
                    addToChatHistory('Bot', data.response);
                    if (data.audio_url) {
                        ttsPlayer.src = data.audio_url;
                        ttsPlayer.play().catch(error => {
                            console.error('Playback error:', error);
                            status.textContent = 'Error playing audio response';
                        });
                    }
                }
                status.textContent = '';
            })
            .catch(error => {
                console.error('Error:', error);
                status.textContent = 'Error processing text input';
            });
        textInput.value = '';
    }
});

function addToChatHistory(role, message) {
    const div = document.createElement('div');
    div.classList.add('flex', 'items-start', 'space-x-2', 'mb-2');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar', role === 'You' ? 'user-avatar' : 'bot-avatar');
    avatar.textContent = role === 'You' ? 'U' : 'B';

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', role === 'You' ? 'user-message' : 'bot-message');
    messageDiv.textContent = message;

    if (role === 'You') {
        div.append(messageDiv, avatar);
    } else {
        div.append(avatar, messageDiv);
    }

    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}