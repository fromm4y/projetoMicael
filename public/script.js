// ----------------- Chatbot (Correto e Único) -----------------
const botaoChat = document.getElementById("btnAbrirChatbot");
const dfMessenger = document.querySelector("df-messenger");


botaoChat.addEventListener("click", () => {
    dfMessenger.classList.toggle("aberto");
    dfMessenger.setAttribute("opened", dfMessenger.classList.contains("aberto"));
});

// ----------------- Variáveis globais -----------------
let stream = null;
let modelosCarregados = false;

// Elementos DOM
document.addEventListener("DOMContentLoaded", () => {
  const abrirIA = document.getElementById('abrirIA');
  const cameraModal = document.getElementById('cameraModal');
  const fecharModal = document.getElementById('fecharModal');
  const video = document.getElementById('video');
  const tirarFoto = document.getElementById('tirarFoto');
  const fotoCanvas = document.getElementById('fotoCanvas');
  const modalStatus = document.getElementById('modalStatus');

  // ----------------- Carregar modelos -----------------
  async function carregarModelos() {
    if (modelosCarregados) return;
    modalStatus.innerText = 'Carregando modelos...';
    try {
      await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
      await faceapi.nets.faceExpressionNet.loadFromUri('/models');
      modelosCarregados = true;
      modalStatus.innerText = 'Modelos carregados.';
    } catch (err) {
      console.error('Erro carregando modelos:', err);
      modalStatus.innerText = 'Erro ao carregar modelos. Veja console.';
    }
  }

  // ----------------- Abrir modal e ativar câmera -----------------
  abrirIA.addEventListener('click', async () => {
    cameraModal.style.display = 'flex';
    cameraModal.setAttribute('aria-hidden', 'false');
    modalStatus.innerText = 'Carregando...';
    await carregarModelos();

    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640 } });
      video.srcObject = stream;
      await video.play();
      modalStatus.innerText = 'Câmera ativa. Posicione seu rosto e clique em 📸';
    } catch (err) {
      console.error('Erro ao acessar a câmera:', err);
      modalStatus.innerText = 'Não foi possível acessar a câmera.';
    }
  });

  // ----------------- Fechar modal e parar câmera -----------------
  fecharModal.addEventListener('click', () => {
    pararCamera();
    cameraModal.style.display = 'none';
    cameraModal.setAttribute('aria-hidden', 'true');
    modalStatus.innerText = 'Aguardando...';
  });

  // ----------------- Parar câmera -----------------
  function pararCamera() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    video.srcObject = null;
  }

  // ----------------- Tirar foto e enviar -----------------
  tirarFoto.addEventListener('click', async () => {
    if (!video || video.readyState < 2) {
      modalStatus.innerText = 'Vídeo não pronto. Tente novamente.';
      return;
    }

    fotoCanvas.width = video.videoWidth || 640;
    fotoCanvas.height = video.videoHeight || 480;
    const ctx = fotoCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, fotoCanvas.width, fotoCanvas.height);

    modalStatus.innerText = 'Enviando foto para processamento...';

    pararCamera();
    cameraModal.style.display = 'none';
    cameraModal.setAttribute('aria-hidden', 'true');

    fotoCanvas.toBlob(async (blob) => {
      if (!blob) {
        modalStatus.innerText = 'Erro ao capturar a foto.';
        return;
      }

      const formData = new FormData();
      formData.append('foto', blob, 'foto.png');

      try {
        const response = await fetch(`${window.location.origin}/processar-foto`, {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (data.facesEncontradas === 0) {
          modalStatus.innerText = 'Nenhuma face detectada na foto.';
          return;
        }

        const dataUrl = fotoCanvas.toDataURL('image/png');
        sessionStorage.setItem('ultimaFoto', dataUrl);

        const emocao = data.emocao || 'neutral';
        const confianca = data.confianca || 0;
        window.location.href =
          `resultado.html?emocao=${encodeURIComponent(emocao)}&conf=${encodeURIComponent(confianca)}`;

      } catch (err) {
        modalStatus.innerText = 'Erro ao enviar foto para o servidor.';
        console.error('Erro fetch:', err);
      }
    }, 'image/png');
  });

  // ====== Animações ======
  window.addEventListener('load', () => {
    document.querySelectorAll('.hero-card, .hero-visual, .card').forEach((el, i) => {
      el.style.opacity = 0;
      el.style.transform = 'translateY(8px)';
      setTimeout(() => {
        el.style.transition = 'opacity 600ms ease, transform 600ms ease';
        el.style.opacity = 1;
        el.style.transform = 'none';
      }, 120 * i);
    });
  });

  // ====== Scroll suave ======
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        const focusEl = target.querySelector('h2') || target;
        focusEl.setAttribute('tabindex', '-1');
        focusEl.focus({ preventScroll: true });
        setTimeout(() => focusEl.removeAttribute('tabindex'), 1200);
      }
    });
  });
});