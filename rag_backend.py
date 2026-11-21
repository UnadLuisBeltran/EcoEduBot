            ]

            if "sentence_embedding" in nombres_salidas:
                indice = nombres_salidas.index(
                    "sentence_embedding"
                )
                embeddings = resultados[indice]

            else:
                tokens = resultados[0]

                if tokens.ndim != 3:
                    raise ValueError(
                        "Unexpected ONNX output dimensions."
                    )

                pesos = mascara[:, :, None]

                embeddings = (
                    (tokens * pesos).sum(axis=1)
                    / np.maximum(pesos.sum(axis=1), 1)
                )

            embeddings = np.asarray(
                embeddings, dtype=np.float32
            )

            normas = np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )

            embeddings = embeddings / np.maximum(
                normas, 1e-12
            )

            vectores.append(embeddings)

        return np.vstack(vectores)

    def buscar_semantico(self, pregunta, top_n=20):
        """Rank chunks using cosine similarity."""

        vector = self.generar_embeddings(
            [pregunta]
        )[0]

        similitudes = self.embeddings @ vector

        indices = np.argsort(-similitudes)[:top_n]

        return [int(i) for i in indices]

    def buscar_bm25(self, pregunta, top_n=20):
        """Rank chunks using BM25."""

        consulta = tokenizar_bm25(pregunta)

        n = len(self.fragmentos)
        k1 = 1.5
        b = 0.75

        puntuaciones = np.zeros(n, dtype=np.float32)

        for termino in set(consulta):
            frecuencia_doc = self.df.get(termino, 0)

            idf = math.log(
                1 + (
                    n - frecuencia_doc + 0.5
                ) / (frecuencia_doc + 0.5)
            )

            for i, frecuencias in enumerate(self.frecuencias):
                tf = frecuencias.get(termino, 0)

                if tf == 0:
                    continue

                denominador = (
                    tf
                    + k1 * (
                        1 - b
                        + b * self.longitudes[i]
                        / max(self.longitud_media, 1)
                    )
                )

                puntuaciones[i] += (
                    idf * tf * (k1 + 1) / denominador
                )

        indices = np.argsort(-puntuaciones)[:top_n]

        return [int(i) for i in indices]

    def recuperar(self, pregunta, candidatos=20, top_k=3):
        """Combine semantic and BM25 rankings with RRF."""

        ranking_semantico = self.buscar_semantico(
            pregunta, candidatos
        )

        ranking_bm25 = self.buscar_bm25(
            pregunta, candidatos
        )

        puntuaciones_rrf = Counter()
        constante = 60

        for ranking in [ranking_semantico, ranking_bm25]:
            for posicion, indice in enumerate(ranking, start=1):
                puntuaciones_rrf[indice] += (
                    1 / (constante + posicion)
                )

        mejores = puntuaciones_rrf.most_common(top_k)

        return [
            self.fragmentos[indice]
            for indice, _ in mejores
        ]


def construir_contexto(fragmentos):
    """Build a context with source page references."""

    bloques = []

