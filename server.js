import express from "express";
import multer from "multer";
import cors from "cors";
import path from "path";
import fs from "fs";

// Importa TensorFlow.js puro (JS), sem binários nativos
import "@tensorflow/tfjs";
import * as faceapi from "@vladmandic/face-api";

// Importa o canvas e integra com face-api
import canvas from "canvas";

// Configurações do canvas para face-api
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Inicializa o Express
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configuração do multer para upload de imagens
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Caminho para os modelos do face-api
const MODELS_PATH = path.join(process.cwd(), "models");

// Função para carregar os modelos
async function loadModels() {
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODELS_PATH);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_PATH);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODELS_PATH);
    console.log("Modelos carregados com sucesso!");
  } catch (err) {
    console.error("Erro ao carregar os modelos:", err);
  }
}

// Endpoint de teste para upload e detecção facial
app.post("/detect", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).send("Nenhuma imagem enviada.");

  try {
    // Cria imagem a partir do buffer
    const img = await canvas.loadImage(req.file.buffer);

    // Detecta rostos
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors();

    res.json(detections);
  } catch (err) {
    console.error("Erro ao detectar rostos:", err);
    res.status(500).send("Erro na detecção facial.");
  }
});

// Inicializa o servidor
const PORT = process.env.PORT || 3000;
loadModels().then(() => {
  app.listen(PORT, () => console.log(`Servidor rodando em http://localhost:${PORT}`));
});
