document.addEventListener("DOMContentLoaded", () => {
    // 1. Efecto de "mirada" (Cat Eye Tracker)
    const pupil = document.querySelector('.pupil');
    const eye = document.querySelector('.cat-eye');
    
    if (pupil && eye) {
        document.addEventListener('mousemove', (e) => {
            const eyeRect = eye.getBoundingClientRect();
            const eyeCenterX = eyeRect.left + eyeRect.width / 2;
            const eyeCenterY = eyeRect.top + eyeRect.height / 2;
            
            const mouseX = e.clientX;
            const mouseY = e.clientY;
            
            const angle = Math.atan2(mouseY - eyeCenterY, mouseX - eyeCenterX);
            const maxDistance = 15;
            const distance = Math.min(maxDistance, Math.hypot(mouseX - eyeCenterX, mouseY - eyeCenterY) / 10);
            
            const x = Math.cos(angle) * distance;
            const y = Math.sin(angle) * distance;
            
            pupil.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
        });
    }

    // 2. Efecto de "aparición" de texto (Fade-in on scroll)
    const fadeElements = document.querySelectorAll('.fade-in-text');
    
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.15
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    fadeElements.forEach(el => observer.observe(el));

    // 3. Palabras clave (Interactividad y efectos en el fondo)
    const keywords = document.querySelectorAll('.keyword');
    const overlay = document.getElementById('flash-overlay');
    
    keywords.forEach(keyword => {
        keyword.addEventListener('click', function() {
            const effect = this.getAttribute('data-effect');
            
            if (effect === 'wall') {
                document.body.classList.add('effect-wall');
                playMeow();
                setTimeout(() => {
                    document.body.classList.remove('effect-wall');
                }, 500); // Duración de la sacudida de la pared
            } else {
                document.body.classList.add(`effect-${effect}`);
                setTimeout(() => {
                    document.body.classList.remove(`effect-${effect}`);
                    // Forzar un reset rápido del overlay temporalmente
                    overlay.style.opacity = '0'; 
                    setTimeout(() => overlay.style.opacity = '', 500); 
                }, 150); // Flash muy rápido de terror
            }
        });
    });

    // 4. Control de Audio Ambiental (Eliminado a petición del usuario)
    // Se ha reemplazado por la sección "Símbolos Clave"

        // 5. Chatbot Logic
    const chatbotBtn = document.getElementById('chatbot-float-btn');
    const chatbotModal = document.getElementById('chatbot-modal');
    const chatbotCloseBtn = document.getElementById('chatbot-close-btn');
    const chatbotClearBtn = document.getElementById('chatbot-clear-btn');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotSendBtn = document.getElementById('chatbot-send-btn');
    const chatbotMessages = document.getElementById('chatbot-messages');

    if (chatbotBtn && chatbotModal) {
        chatbotBtn.addEventListener('click', () => {
            chatbotModal.classList.remove('chatbot-hidden');
            chatbotInput.focus();
        });

        chatbotCloseBtn.addEventListener('click', () => {
            chatbotModal.classList.add('chatbot-hidden');
        });

        if (chatbotClearBtn) {
            chatbotClearBtn.addEventListener('click', () => {
                const messages = chatbotMessages.querySelectorAll('.chat-message');
                for (let i = 1; i < messages.length; i++) {
                    messages[i].remove();
                }
            });
        }

        const appendMessage = (text, sender) => {
            const msgDiv = document.createElement('div');
            msgDiv.classList.add('chat-message');
            msgDiv.classList.add(sender === 'user' ? 'user-msg' : 'bot-msg');
            msgDiv.innerText = text;
            chatbotMessages.appendChild(msgDiv);
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        };

        const handleSend = async () => {
            const text = chatbotInput.value.trim();
            if (text === '') return;
            
            appendMessage(text, 'user');
            chatbotInput.value = '';

            // Show "typing" indicator or just delay
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: text })
                });
                
                const data = await response.json();
                
                setTimeout(() => {
                    appendMessage(data.answer, 'bot');
                    
                    if (Math.random() > 0.8 && document.getElementById('flash-overlay')) {
                        document.body.classList.add('effect-blood');
                        setTimeout(() => document.body.classList.remove('effect-blood'), 150);
                    }
                }, 400);

            } catch (error) {
                console.error("Error al consultar al Oráculo:", error);
                setTimeout(() => {
                    appendMessage("La conexión con el más allá se ha cortado...", 'bot');
                }, 400);
            }
        };

        chatbotSendBtn.addEventListener('click', handleSend);
        chatbotInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSend();
        });
    }
});
