// ------------------------------
// Elements
// ------------------------------
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const talkBtn = document.getElementById("talk-btn");
const chatBox = document.getElementById("chat-box");

let recorder;
let recordedChunks = [];
let isRecording = false;

// ------------------------------
// Helper: append message to chat
// ------------------------------
function appendMessage(sender, message) {
    const div = document.createElement("div");
    div.className = sender === "user" ? "text-right text-blue-600" : "text-left text-gray-800";
    div.textContent = message;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ------------------------------
// Text → Text endpoint
// ------------------------------
sendBtn.addEventListener("click", async () => {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage("user", text);
    userInput.value = "";

    try {
        const response = await fetch("http://localhost:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: text })
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();
        appendMessage("bot", data.answer);
    } catch (err) {
        console.error(err);
        appendMessage("bot", "Error: Could not get response.");
    }
});

// ------------------------------
// Audio → STT → RAG → TTS endpoint
// ------------------------------

// ------------------------------
// Audio → STT → RAG → TTS endpoint
// ------------------------------
talkBtn.addEventListener("click", async () => {
    if (!isRecording) {
        // Start recording
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        recordedChunks = [];

        recorder.ondataavailable = e => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        recorder.onstop = async () => {
            const audioBlob = new Blob(recordedChunks, { type: "audio/wav" });

            try {
                // Send audio to backend
                const response = await fetch("http://localhost:8000/chat_audio", {
                    method: "POST",
                    body: audioBlob
                });

                if (!response.ok) throw new Error("Server error");

                const data = await response.json();

                // Display text
                appendMessage("user", data.user_text);
                appendMessage("bot", data.bot_text);

                // Play audio
                const audio = new Audio(data.audio_url);
                audio.play();

            } catch (err) {
                console.error(err);
                appendMessage("bot", "Error: Could not process audio.");
            }
        };

        recorder.start();
        isRecording = true;
        talkBtn.textContent = "Stop Recording";
    } else {
        // Stop recording
        recorder.stop();
        isRecording = false;
        talkBtn.textContent = "Talk";
    }
});

