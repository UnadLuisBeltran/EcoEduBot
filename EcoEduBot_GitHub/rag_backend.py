"""EcoEduBot: hybrid IPCC retrieval using ONNX, BM25 and RRF."""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / 'IPCC_2023_fragmentos.json'
MODEL_DIR = BASE_DIR / 'embeddings_multilingue'
TOKENIZER_PATH = MODEL_DIR / 'tokenizer.json'
ONNX_PATH = MODEL_DIR / 'onnx' / 'model_quint8_avx2.onnx'


def cargar_fragmentos():
    """Load IPCC text chunks."""
    with JSON_PATH.open('r', encoding='utf-8') as archivo:
        fragmentos = json.load(archivo)
    if not isinstance(fragmentos, list) or not all('texto' in f for f in fragmentos):
        raise ValueError('Invalid IPCC fragments format.')
    return fragmentos


def tokenizar_bm25(texto):
    """Tokenize text for BM25 ranking."""
    return re.findall(r'\b\w+\b', texto.lower(), flags=re.UNICODE)


class RecuperadorIPCC:
    """Retrieve relevant IPCC chunks using semantic search and BM25."""

    def __init__(self):
        self.fragmentos = cargar_fragmentos()
        if not TOKENIZER_PATH.is_file():
            raise FileNotFoundError(f'Missing tokenizer: {TOKENIZER_PATH}')
        if not ONNX_PATH.is_file():
            raise FileNotFoundError(f'Missing ONNX model: {ONNX_PATH}')
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        self.sesion = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
        self.documentos_tokenizados = [tokenizar_bm25(f['texto']) for f in self.fragmentos]
        self.frecuencias = [Counter(doc) for doc in self.documentos_tokenizados]
        self.longitudes = [len(doc) for doc in self.documentos_tokenizados]
        self.longitud_media = sum(self.longitudes) / max(len(self.longitudes), 1)
        self.df = Counter()
        for doc in self.documentos_tokenizados:
            self.df.update(set(doc))
        self.embeddings = self.generar_embeddings([f['texto'] for f in self.fragmentos])

    def generar_embeddings(self, textos, batch_size=16):
        """Generate normalized embeddings from the quantized ONNX model."""
        if not textos:
            return np.empty((0, 384), dtype=np.float32)
        vectores = []
        entradas_modelo = {entrada.name for entrada in self.sesion.get_inputs()}
        nombres_salidas = [salida.name for salida in self.sesion.get_outputs()]
        for inicio in range(0, len(textos), batch_size):
            lote = textos[inicio:inicio + batch_size]
            codificados = self.tokenizer.encode_batch(lote)
            max_len = max(len(c.ids) for c in codificados)
            ids = np.asarray([c.ids + [0] * (max_len - len(c.ids)) for c in codificados], dtype=np.int64)
            mascara = np.asarray([c.attention_mask + [0] * (max_len - len(c.attention_mask)) for c in codificados], dtype=np.int64)
            datos = {}
            if 'input_ids' in entradas_modelo:
                datos['input_ids'] = ids
            if 'attention_mask' in entradas_modelo:
                datos['attention_mask'] = mascara
            if 'token_type_ids' in entradas_modelo:
                datos['token_type_ids'] = np.zeros_like(ids)
            resultados = self.sesion.run(None, datos)
            if 'sentence_embedding' in nombres_salidas:
                embeddings = resultados[nombres_salidas.index('sentence_embedding')]
            else:
                tokens = resultados[0]
                if tokens.ndim != 3:
                    raise ValueError('Unexpected ONNX output dimensions.')
                pesos = mascara[:, :, None]
                embeddings = (tokens * pesos).sum(axis=1) / np.maximum(pesos.sum(axis=1), 1)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            normas = np.linalg.norm(embeddings, axis=1, keepdims=True)
            vectores.append(embeddings / np.maximum(normas, 1e-12))
        return np.vstack(vectores)

    def buscar_semantico(self, pregunta, top_n=20):
        """Rank chunks by cosine similarity."""
        vector = self.generar_embeddings([pregunta])[0]
        similitudes = self.embeddings @ vector
        return [int(i) for i in np.argsort(-similitudes)[:top_n]]

    def buscar_bm25(self, pregunta, top_n=20):
        """Rank chunks using BM25."""
        consulta = tokenizar_bm25(pregunta)
        n = len(self.fragmentos)
        k1, b = 1.5, 0.75
        puntuaciones = np.zeros(n, dtype=np.float32)
        for termino in set(consulta):
            frecuencia_doc = self.df.get(termino, 0)
            idf = math.log(1 + (n - frecuencia_doc + 0.5) / (frecuencia_doc + 0.5))
            for i, frecuencias in enumerate(self.frecuencias):
                tf = frecuencias.get(termino, 0)
                if tf == 0:
                    continue
                denominador = tf + k1 * (1 - b + b * self.longitudes[i] / max(self.longitud_media, 1))
                puntuaciones[i] += idf * tf * (k1 + 1) / denominador
        return [int(i) for i in np.argsort(-puntuaciones)[:top_n]]

    def recuperar(self, pregunta, candidatos=20, top_k=3):
        """Combine semantic and BM25 rankings with reciprocal rank fusion."""
        ranking_semantico = self.buscar_semantico(pregunta, candidatos)
        ranking_bm25 = self.buscar_bm25(pregunta, candidatos)
        puntuaciones_rrf = Counter()
        constante = 60
        for ranking in (ranking_semantico, ranking_bm25):
            for posicion, indice in enumerate(ranking, start=1):
                puntuaciones_rrf[indice] += 1 / (constante + posicion)
        return [self.fragmentos[indice] for indice, _ in puntuaciones_rrf.most_common(top_k)]


def construir_contexto(fragmentos):
    """Build a source-attributed context from retrieved chunks."""
    bloques = []
    for fragmento in fragmentos:
        pagina = fragmento.get('pagina_pdf', 'unknown')
        bloques.append(f"[IPCC PDF page {pagina}]\n{fragmento['texto']}")
    return '\n\n'.join(bloques)
