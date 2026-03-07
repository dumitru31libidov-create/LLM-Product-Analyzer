# LLM Product Analyzer API 🤖

API de comparare produse cu Chain-of-Thought, Auto-Verificare și Semantic Cache, construit cu FastAPI + Instructor + Ollama.

---

## Structura proiectului

```
llm-product-analyzer/
├── app/
│   └── main.py              # Codul principal (Task 1 + Task 2 + Bonus)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Task 1: Chain of Thought via Pydantic

### Ce s-a modificat

Câmpul `rationale` a fost adăugat în modelele `FeatureComparison` și `Verdict`, plasat **înainte** de câmpul `winner`/`câștigător`.

```python
class FeatureComparison(BaseModel):
    feature_name: str
    produs_a_value: str
    produs_b_value: str
    winner_score: int

    # ✅ rationale ÎNAINTE de winner
    rationale: str = Field(
        description="Explică PAS CU PAS de ce un produs câștigă această caracteristică..."
    )
    winner: str = Field(pattern="^(A|B|Egal)$")
    relevant_pentru_user: bool


class Verdict(BaseModel):
    # ✅ rationale ÎNAINTE de câștigător
    rationale: str = Field(
        description="Sinteză detaliată a întregii comparații..."
    )
    câștigător: str = Field(pattern="^(A|B|Egal)$")
    scor_a: int
    scor_b: int
    ...
```

### De ce funcționează

LLM-urile generează text **token cu token**, secvențial. Prin plasarea câmpului `rationale` înaintea deciziei finale, modelul este forțat să:
1. Analizeze diferențele concrete între produse
2. Argumenteze fiecare aspect
3. Abia apoi să declare câștigătorul

Rezultat: decizii mai precise și ancorate în date reale.

---

## Task 2: Chain-of-Thought cu Auto-Verificare

### Pipeline în 2 pași

```
Întrebare
    │
    ▼
┌─────────────┐
│  GENERATOR  │  → GÂNDIRE: [pași] | RĂSPUNS: [concluzie] | CONFIDENCE: [0.0-1.0]
└─────────────┘
    │
    ▼
┌─────────────┐
│ VERIFICATOR │  → verdict: da/nu/nesigur | motiv | feedback_pentru_generator
└─────────────┘
    │
    ├── "da" → ✅ Returnează rezultat
    │
    └── "nu"/"nesigur" → Retry cu feedback (max 3 încercări)
```

### Scorul de încredere (`confidence`)

- **Generator** declară un `confidence` între 0.0 și 1.0
- **Verificator** evaluează dacă scorul e realist față de soliditatea argumentelor (`confidence_evaluat`)
- Ambele sunt returnate în răspuns pentru transparență

### Exemplu răspuns `/analyze`

```json
{
  "raspuns_final": "Python este mai potrivit pentru ML datorită ecosistemului...",
  "gandire_finala": "1. Python are NumPy, Pandas, scikit-learn... 2. Java are mai puțin suport...",
  "confidence_final": 0.87,
  "verificat": true,
  "numar_incercari": 1,
  "istoricul_incercarilor": [...]
}
```

---

## Bonus: Semantic Cache

### Cum funcționează

1. La fiecare request, se calculează **embedding-ul** query-ului cu `sentence-transformers`
2. Se compară cu embedding-urile din cache prin **similaritate cosine**
3. Dacă `similaritate > 0.85` → returnează răspuns din cache (fără apel LLM)

### Invalidare la Concept Drift

Dacă un query nou este **foarte diferit** față de media cache-ului existent (similaritate medie < 0.5), întregul cache semantic este invalidat automat.

```python
# Detectare automată în fiecare request
drift_count = semantic_cache_invalidate_drift(semantic_query)
if drift_count > 0:
    logger.warning(f"Concept drift: invalidate {drift_count} intrări")
```

---

## Instalare și rulare

### Opțiunea 1: Docker Compose (recomandat)

```bash
# 1. Clonează repo-ul
git clone <url-repo>
cd llm-product-analyzer

# 2. Copiază și configurează .env
cp .env.example .env

# 3. Pornește serviciile
docker-compose up -d

# 4. Descarcă modelul Ollama
docker exec ollama ollama pull qwen2.5:7b

# 5. Verifică că funcționează
curl http://localhost:8000/health
```

### Opțiunea 2: Local (fără Docker)

```bash
# 1. Instalează dependențele
pip install -r requirements.txt

# 2. Instalează Playwright
playwright install chromium

# 3. Pornește Ollama separat
ollama pull qwen2.5:7b
ollama serve

# 4. Configurează .env
cp .env.example .env
# Editează OLLAMA_HOST=http://localhost:11434

# 5. Pornește API-ul
uvicorn app.main:app --reload --port 8000
```

---

## Endpoints

| Endpoint | Metodă | Descriere |
|----------|--------|-----------|
| `/compare` | POST | Task 1: Comparare produse cu CoT via Pydantic |
| `/analyze` | POST | Task 2: Analiză CoT cu Auto-Verificare |
| `/health` | GET | Status serviciu și Ollama |
| `/cache/stats` | GET | Statistici semantic cache |
| `/cache` | DELETE | Golește tot cache-ul |
| `/docs` | GET | Swagger UI interactiv |

---

## Exemple de request

### Task 1 - Comparare produse

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "produs_a": {"sursa": "MacBook Air M3 8GB 256GB 1.24kg", "este_url": false},
    "produs_b": {"sursa": "ThinkPad X1 Carbon i7 16GB 512GB 1.12kg", "este_url": false},
    "preferinte": "Programare Python și ML, portabilitate maximă"
  }'
```

### Task 2 - Analiză CoT

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "intrebare": "Ce algoritm de ML este mai potrivit pentru clasificare text: SVM sau Random Forest?",
    "context": "Dataset cu 50000 exemple, 10 clase, features TF-IDF",
    "max_incercari": 3
  }'
```
