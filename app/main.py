"""
Product Comparison Engine cu Instructor + Ollama.

Task 1: Chain of Thought via Pydantic - câmp `rationale` în FeatureComparison și Verdict
Task 2: Chain-of-Thought cu Auto-Verificare - pipeline Generator → Verificator + retry
Bonus: Semantic Cache cu sentence-transformers + invalidare concept drift
"""

import hashlib
import json
import os
import time
import logging
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import instructor
import openai
from diskcache import Cache
from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURARE
# =============================================================================

cache = Cache(directory=os.getenv("CACHE_DIR", "./cache"))

# Client OpenAI configurat pentru Ollama
client = openai.OpenAI(
    base_url=f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/v1",
    api_key="ollama",
)

instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# =============================================================================
# BONUS: SEMANTIC CACHE cu sentence-transformers
# =============================================================================

# Import opțional - dacă nu e instalat, semantic cache e dezactivat
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    SEMANTIC_CACHE_ENABLED = True
    logger.info("✅ Semantic cache activat (sentence-transformers disponibil)")
except ImportError:
    SEMANTIC_CACHE_ENABLED = False
    logger.warning("⚠️  sentence-transformers lipsă - semantic cache dezactivat")


def _get_embedding(text: str) -> Optional[List[float]]:
    """Calculează embedding pentru un text."""
    if not SEMANTIC_CACHE_ENABLED:
        return None
    vec = _embedder.encode(text, convert_to_numpy=True)
    return vec.tolist()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Similaritate cosine între două vectori."""
    import numpy as np
    va, vb = np.array(a), np.array(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


SEMANTIC_SIMILARITY_THRESHOLD = 0.85
SEMANTIC_CACHE_PREFIX = "sem_cache:"
SEMANTIC_INDEX_KEY = "sem_index"


def semantic_cache_lookup(query_text: str) -> Optional[dict]:
    """
    Caută în semantic cache un răspuns similar semantic (cosine > 0.85).
    Returnează rezultatul cached dacă găsește, altfel None.
    """
    if not SEMANTIC_CACHE_ENABLED:
        return None

    query_emb = _get_embedding(query_text)
    if query_emb is None:
        return None

    # Indexul conține lista de {key, embedding, timestamp}
    index: List[dict] = cache.get(SEMANTIC_INDEX_KEY, [])

    best_sim = 0.0
    best_key = None

    for entry in index:
        sim = _cosine_similarity(query_emb, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_key = entry["key"]

    if best_sim >= SEMANTIC_SIMILARITY_THRESHOLD and best_key:
        cached_result = cache.get(best_key)
        if cached_result is not None:
            logger.info(f"✅ Semantic cache HIT (similaritate: {best_sim:.3f})")
            return cached_result

    logger.info(f"❌ Semantic cache MISS (best sim: {best_sim:.3f})")
    return None


def semantic_cache_store(query_text: str, result: dict) -> None:
    """
    Stochează un rezultat în semantic cache cu embedding-ul query-ului.
    """
    if not SEMANTIC_CACHE_ENABLED:
        return

    query_emb = _get_embedding(query_text)
    if query_emb is None:
        return

    cache_key = SEMANTIC_CACHE_PREFIX + hashlib.sha256(query_text.encode()).hexdigest()[:16]

    # Salvăm rezultatul
    cache.set(cache_key, result, expire=3600 * 24)

    # Actualizăm indexul
    index: List[dict] = cache.get(SEMANTIC_INDEX_KEY, [])
    index.append({
        "key": cache_key,
        "embedding": query_emb,
        "query_preview": query_text[:80],
        "timestamp": time.time(),
    })
    cache.set(SEMANTIC_INDEX_KEY, index)
    logger.info(f"✅ Semantic cache STORED (key: {cache_key})")


def semantic_cache_invalidate_drift(new_query: str, drift_threshold: float = 0.5) -> int:
    """
    BONUS: Invalidare la concept drift.
    Dacă query-ul nou e FOARTE diferit față de media cache-ului existent
    (similaritate medie < drift_threshold), șterge tot cache-ul semantic.
    Returnează numărul de intrări șterse.
    """
    if not SEMANTIC_CACHE_ENABLED:
        return 0

    index: List[dict] = cache.get(SEMANTIC_INDEX_KEY, [])
    if len(index) < 3:
        # Nu avem suficiente date pentru a detecta drift
        return 0

    new_emb = _get_embedding(new_query)
    if new_emb is None:
        return 0

    # Calculăm similaritatea față de ultimele 10 intrări
    recent = index[-10:]
    similarities = [_cosine_similarity(new_emb, e["embedding"]) for e in recent]
    avg_sim = sum(similarities) / len(similarities)

    if avg_sim < drift_threshold:
        logger.warning(
            f"⚠️  Concept drift detectat! Similaritate medie: {avg_sim:.3f} < {drift_threshold}. "
            f"Invalidez {len(index)} intrări din semantic cache."
        )
        # Ștergem toate intrările
        for entry in index:
            cache.delete(entry["key"])
        cache.delete(SEMANTIC_INDEX_KEY)
        return len(index)

    return 0


# =============================================================================
# TASK 1: MODELE PYDANTIC cu câmp `rationale` (Chain of Thought)
# =============================================================================

class ProductData(BaseModel):
    """Date extrase despre produs."""
    titlu: str = Field(description="Numele produsului")
    descriere: str = Field(description="Descriere scurtă")
    specificatii: str = Field(description="Specificații tehnice cheie")
    preț: str = Field(default="")
    extras_din: str = Field(description="'scraping' sau 'text'")


class FeatureComparison(BaseModel):
    """
    O linie din tabelul comparativ.

    TASK 1: Câmpul `rationale` este plasat ÎNAINTE de `winner`.
    Deoarece LLM-urile generează token cu token, forțând modelul să scrie
    mai întâi raționamentul, decizia finală (winner) devine mai precisă
    și ancorată în datele reale ale produselor.
    """
    feature_name: str = Field(description="Numele caracteristicii comparate")
    produs_a_value: str = Field(description="Valoarea caracteristicii pentru produsul A")
    produs_b_value: str = Field(description="Valoarea caracteristicii pentru produsul B")
    winner_score: int = Field(ge=1, le=10, description="Scorul diferenței între produse pe scala 1-10")

    # TASK 1: rationale ÎNAINTE de winner — LLM gândește înainte să decidă
    rationale: str = Field(
        description=(
            "Explică PAS CU PAS de ce un produs câștigă această caracteristică. "
            "Compară valorile concrete (A vs B), menționează avantajele și dezavantajele "
            "fiecăruia, apoi justifică decizia finală. Scrie înainte de a da verdictul."
        )
    )
    winner: str = Field(
        pattern="^(A|B|Egal)$",
        description="Câștigătorul acestei caracteristici: A, B sau Egal"
    )
    relevant_pentru_user: bool = Field(
        description="True dacă această caracteristică e relevantă pentru preferințele userului"
    )


class Verdict(BaseModel):
    """
    Verdict final al comparației.

    TASK 1: Câmpul `rationale` este plasat ÎNAINTE de `câștigător`.
    Modelul trebuie să sintetizeze toate argumentele înainte de verdictul final.
    """
    # TASK 1: rationale ÎNAINTE de câștigător
    rationale: str = Field(
        description=(
            "Sinteză detaliată a întregii comparații: puncte forte și slabe ale fiecărui produs, "
            "cum se aliniază cu preferințele userului, ce compromisuri implică fiecare alegere. "
            "Construiește argumentul complet ÎNAINTE de a declara câștigătorul."
        )
    )
    câștigător: str = Field(
        pattern="^(A|B|Egal)$",
        description="Produsul câștigător pe baza raționamentului de mai sus"
    )
    scor_a: int = Field(ge=0, le=100, description="Scorul final pentru produsul A (0-100)")
    scor_b: int = Field(ge=0, le=100, description="Scorul final pentru produsul B (0-100)")
    diferență_semificativă: bool = Field(
        description="True dacă există o diferență clară între produse, False dacă sunt apropiate"
    )
    argument_principal: str = Field(max_length=500, description="Argumentul principal pentru câștigător")
    compromisuri: str = Field(max_length=500, description="Ce sacrifici alegând câștigătorul")


class ComparisonResult(BaseModel):
    """
    Model final pe care Instructor îl forțează din LLM.
    Dacă LLM returnează JSON invalid, Instructor retrimite automat.
    """
    produs_a_titlu: str = Field(description="Titlu produs A")
    produs_b_titlu: str = Field(description="Titlu produs B")
    features: List[FeatureComparison] = Field(description="Tabel comparativ cu raționament per feature")
    verdict: Verdict = Field(description="Verdict final cu raționament complet")
    preferinte_procesate: str = Field(description="Rezumat preferințe user luate în calcul")


# =============================================================================
# TASK 2: MODELE pentru Chain-of-Thought cu Auto-Verificare
# =============================================================================

class GeneratorOutput(BaseModel):
    """
    Output-ul pasului Generator din pipeline-ul CoT cu auto-verificare.
    Forțează modelul să separe gândirea de răspuns și să declare un scor de încredere.
    """
    gandire: str = Field(
        description=(
            "Pași de raționament explicit, scrieți fiecare pas logic înainte de concluzie. "
            "Format: '1. [observație] 2. [analiză] 3. [concluzie intermediară]...'"
        )
    )
    raspuns: str = Field(
        description="Concluzia finală bazată pe pașii de gândire de mai sus"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Scor de încredere între 0.0 și 1.0 cât de sigur ești de răspuns. "
            "0.0 = complet nesigur, 1.0 = absolut sigur. Fii realist și calibrat."
        )
    )


class VerificatorOutput(BaseModel):
    """
    Output-ul pasului Verificator din pipeline-ul CoT cu auto-verificare.
    Evaluează validitatea logicii din GeneratorOutput.
    """
    verdict: str = Field(
        pattern="^(da|nu|nesigur)$",
        description="'da' dacă logica e validă, 'nu' dacă e greșită, 'nesigur' dacă e ambiguă"
    )
    motiv: str = Field(
        description=(
            "Explicație detaliată a verdictului: ce e corect sau greșit în raționament, "
            "ce lipsește, ce contradicții există. Obligatoriu completat indiferent de verdict."
        )
    )
    feedback_pentru_generator: str = Field(
        description=(
            "Instrucțiuni concrete pentru îmbunătățire (folosite la retry). "
            "Dacă verdict='da', scrie 'Raționamentul este valid, nu sunt necesare modificări.'"
        )
    )
    confidence_evaluat: float = Field(
        ge=0.0, le=1.0,
        description="Evaluarea ta a scorului de încredere declarat de Generator. E realist?"
    )


class CoTResult(BaseModel):
    """Rezultatul final al pipeline-ului CoT cu auto-verificare."""
    raspuns_final: str
    gandire_finala: str
    confidence_final: float
    verificat: bool
    numar_incercari: int
    istoricul_incercarilor: List[dict]


# =============================================================================
# API Request/Response models
# =============================================================================

class ProductInput(BaseModel):
    sursa: str = Field(..., min_length=3)
    este_url: bool = Field(default=False)

    class Config:
        json_schema_extra = {
            "example": {
                "sursa": "iPhone 15: A16, 6GB RAM, 48MP camera",
                "este_url": False
            }
        }


class ComparisonRequest(BaseModel):
    produs_a: ProductInput
    produs_b: ProductInput
    preferinte: str = Field(..., min_length=5, max_length=1000)
    buget_maxim: Optional[int] = Field(None, ge=100)


class CoTRequest(BaseModel):
    intrebare: str = Field(..., min_length=5, description="Întrebarea sau problema de analizat")
    context: Optional[str] = Field(None, description="Context adițional opțional")
    max_incercari: int = Field(default=3, ge=1, le=5, description="Numărul maxim de reîncercări")


# =============================================================================
# SCRAPING (neschimbat din original)
# =============================================================================

async def scrape_product(url: str) -> ProductData:
    """Scrapează orice pagină de produs cu Playwright + BeautifulSoup."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            page = await browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            await page.goto(url, wait_until="networkidle", timeout=25000)
            await page.wait_for_timeout(2000)
            html = await page.content()
            title = await page.title()
            await browser.close()

            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup.find_all([
                'script', 'style', 'nav', 'footer', 'header',
                'aside', 'noscript', 'iframe', 'svg', 'canvas',
                'button', 'input', 'form', 'select', 'textarea',
            ]):
                tag.decompose()

            content_parts = []
            h1 = soup.find('h1')
            if h1:
                content_parts.append(f"PRODUCT: {h1.get_text(strip=True)}")

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content_parts.append(f"DESCRIPTION: {meta_desc['content'][:500]}")

            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 30:
                    content_parts.append(text)

            for ul in soup.find_all(['ul', 'ol']):
                items = [li.get_text(strip=True) for li in ul.find_all('li') if len(li.get_text(strip=True)) > 5]
                if items:
                    content_parts.append(" | ".join(items[:15]))

            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr')[:25]:
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(": ".join(cells[:2]))
                if rows:
                    content_parts.append("SPECS: " + " | ".join(rows[:10]))

            full_text = "\n\n".join(content_parts[:40])

            return ProductData(
                titlu=title[:300] if title else url.split('/')[-1][:50],
                descriere=full_text[:6000],
                specificatii="",
                preț="",
                extras_din="scraping"
            )
    except Exception as e:
        raise HTTPException(422, f"Scraping failed: {str(e)}")


def parse_text_input(text: str) -> ProductData:
    """Parsează input text liber."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return ProductData(
        titlu=lines[0][:200] if lines else "Unknown",
        descriere='\n'.join(lines[:20]),
        specificatii="",
        preț="",
        extras_din="text"
    )


# =============================================================================
# TASK 1: COMPARARE cu Chain of Thought via Pydantic
# =============================================================================

async def compară_produse_instructor(
    prod_a: ProductData,
    prod_b: ProductData,
    preferinte: str
) -> ComparisonResult:
    """
    Folosește Instructor pentru a forța output validat Pydantic.
    Task 1: modelele includ câmpul `rationale` ÎNAINTE de winner/câștigător,
    forțând LLM-ul să gândească înainte să decidă.
    """
    system_prompt = """Ești un expert în compararea produselor.
Analizează datele reale ale produselor și compară-le STRICT pe criteriile userului.
IMPORTANT: Pentru fiecare feature și pentru verdictul final, scrie mai întâi raționamentul
detaliat (câmpul 'rationale') ÎNAINTE de a declara câștigătorul. Gândește pas cu pas."""

    user_prompt = f"""Compară aceste produse pentru userul care vrea: "{preferinte}"

PRODUS A: {prod_a.titlu}
Descriere: {prod_a.descriere[:4000]}

PRODUS B: {prod_b.titlu}
Descriere: {prod_b.descriere[:4000]}

Generează tabel comparativ cu feature-urile relevante pentru preferințele userului.
Pentru fiecare feature, explică raționamentul ÎNAINTE de a da verdictul."""

    try:
        result = instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=ComparisonResult,
            max_retries=2,
            temperature=0,
            max_tokens=4000
        )
        return result
    except Exception as e:
        raise HTTPException(503, f"Instructor/LLM error: {str(e)}")


# =============================================================================
# TASK 2: CHAIN-OF-THOUGHT cu AUTO-VERIFICARE
# =============================================================================

def _run_generator(intrebare: str, context: str, feedback: Optional[str] = None) -> GeneratorOutput:
    """
    Pasul 1 din pipeline: Generator.
    Produce răspuns în format GÂNDIRE + RĂSPUNS + scor de încredere.
    Dacă primește feedback de la Verificator, îl include în prompt.
    """
    system_prompt = """Ești un analist expert. Când răspunzi la o întrebare:
1. Scrie pașii de gândire expliciți (câmpul 'gandire') - fiecare pas pe rând
2. Formulează concluzia clară (câmpul 'raspuns')
3. Estimează un scor de încredere realist între 0.0 și 1.0 (câmpul 'confidence')

Fii riguros și calibrat - nu exagera încrederea."""

    feedback_section = ""
    if feedback:
        feedback_section = f"\n\nFEEDBACK DE LA VERIFICATOR (încercare anterioară):\n{feedback}\nCorectează problemele identificate!"

    user_prompt = f"""Întrebare: {intrebare}
{f'Context: {context}' if context else ''}
{feedback_section}

Gândește pas cu pas și oferă un răspuns bine fundamentat."""

    try:
        result = instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=GeneratorOutput,
            max_retries=2,
            temperature=0.3,
            max_tokens=1500
        )
        return result
    except Exception as e:
        raise HTTPException(503, f"Generator LLM error: {str(e)}")


def _run_verificator(intrebare: str, generator_output: GeneratorOutput) -> VerificatorOutput:
    """
    Pasul 2 din pipeline: Verificator.
    Evaluează dacă logica din GeneratorOutput este validă.
    Analizează și scorul de încredere declarat.
    """
    system_prompt = """Ești un verificator critic și imparțial.
Evaluează dacă raționamentul prezentat este logic, complet și corect.
Analizează și dacă scorul de încredere declarat este realist față de calitatea argumentelor.
Fii strict - acceptă doar raționamente solide și bine fundamentate."""

    user_prompt = f"""Evaluează acest raționament pentru întrebarea: "{intrebare}"

GÂNDIRE (pașii de raționament):
{generator_output.gandire}

RĂSPUNS FINAL:
{generator_output.raspuns}

SCOR ÎNCREDERE DECLARAT: {generator_output.confidence:.2f}

Verifică:
1. Pașii de gândire sunt logici și corecți?
2. Concluzia decurge din pași?
3. Există contradicții sau omisiuni majore?
4. Scorul de încredere ({generator_output.confidence:.2f}) este realist față de soliditatea argumentelor?"""

    try:
        result = instructor_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=VerificatorOutput,
            max_retries=2,
            temperature=0,
            max_tokens=1000
        )
        return result
    except Exception as e:
        raise HTTPException(503, f"Verificator LLM error: {str(e)}")


async def cot_cu_auto_verificare(
    intrebare: str,
    context: Optional[str] = None,
    max_incercari: int = 3
) -> CoTResult:
    """
    Pipeline complet Chain-of-Thought cu Auto-Verificare:
    1. Generator produce GÂNDIRE + RĂSPUNS + confidence
    2. Verificator evaluează validitatea logicii
    3. Dacă e respins, se retrimite la Generator cu feedback (max 3 încercări)
    """
    istoricul = []
    feedback_curent = None

    for incercare in range(1, max_incercari + 1):
        logger.info(f"🔄 CoT încercarea {incercare}/{max_incercari}")

        # PASUL 1: Generator
        gen_output = _run_generator(intrebare, context or "", feedback_curent)
        logger.info(f"   Generator confidence: {gen_output.confidence:.2f}")

        # PASUL 2: Verificator
        ver_output = _run_verificator(intrebare, gen_output)
        logger.info(f"   Verificator verdict: {ver_output.verdict}")

        # Salvăm în istoric
        istoricul.append({
            "incercare": incercare,
            "gandire": gen_output.gandire,
            "raspuns": gen_output.raspuns,
            "confidence_generator": gen_output.confidence,
            "verdict_verificator": ver_output.verdict,
            "motiv_verificator": ver_output.motiv,
            "confidence_evaluat": ver_output.confidence_evaluat,
        })

        # Dacă e acceptat, returnăm rezultatul
        if ver_output.verdict == "da":
            logger.info(f"✅ CoT acceptat la încercarea {incercare}")
            return CoTResult(
                raspuns_final=gen_output.raspuns,
                gandire_finala=gen_output.gandire,
                confidence_final=gen_output.confidence,
                verificat=True,
                numar_incercari=incercare,
                istoricul_incercarilor=istoricul
            )

        # Dacă e respins sau nesigur, pregătim feedback pentru retry
        feedback_curent = ver_output.feedback_pentru_generator
        logger.info(f"   ↩️  Retry cu feedback: {feedback_curent[:100]}...")

    # Am epuizat încercările - returnăm cel mai bun răspuns (ultima încercare)
    logger.warning(f"⚠️  CoT nevalidat după {max_incercari} încercări. Returnez ultima variantă.")
    ultima = istoricul[-1]
    return CoTResult(
        raspuns_final=ultima["raspuns"],
        gandire_finala=ultima["gandire"],
        confidence_final=ultima["confidence_generator"],
        verificat=False,
        numar_incercari=max_incercari,
        istoricul_incercarilor=istoricul
    )


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="LLM Product Analyzer API",
    description="""
## LLM Product Analyzer cu Chain-of-Thought și Semantic Cache

### Task 1: Chain of Thought via Pydantic
Câmpul `rationale` adăugat în `FeatureComparison` și `Verdict`, plasat **înainte** de winner/câștigător.
LLM-ul gândește înainte să decidă → decizii mai precise.

### Task 2: CoT cu Auto-Verificare
Pipeline în 2 pași: **Generator** → **Verificator** → retry cu feedback (max 3 încercări).
Include scor de încredere (`confidence`) analizat de Verificator.

### Bonus: Semantic Cache
Răspunsuri similare semantic (cosine > 0.85) sunt returnate din cache.
Invalidare automată la concept drift.
    """,
    version="2.0.0"
)


@app.post("/compare", response_model=ComparisonResult, tags=["Task 1 - CoT via Pydantic"])
async def compare(request: ComparisonRequest):
    """
    **Task 1**: Compară două produse cu Chain-of-Thought via Pydantic.

    Câmpul `rationale` în `FeatureComparison` și `Verdict` forțează LLM-ul
    să explice logica *înainte* de a da verdictul.

    Suportă **Semantic Cache** — întrebări similare returnează răspuns din cache.
    """
    start = time.time()

    # Construim cheia semantică din request
    semantic_query = f"{request.produs_a.sursa} vs {request.produs_b.sursa} | {request.preferinte}"

    # Bonus: verificare concept drift
    drift_count = semantic_cache_invalidate_drift(semantic_query)
    if drift_count > 0:
        logger.info(f"🔄 Concept drift: invalidate {drift_count} intrări din cache")

    # Bonus: semantic cache lookup
    cached = semantic_cache_lookup(semantic_query)
    if cached:
        cached["_din_cache"] = True
        cached["_timp_ms"] = int((time.time() - start) * 1000)
        return cached

    # Extrage date produse
    if request.produs_a.este_url:
        date_a = await scrape_product(request.produs_a.sursa)
    else:
        date_a = parse_text_input(request.produs_a.sursa)

    if request.produs_b.este_url:
        date_b = await scrape_product(request.produs_b.sursa)
    else:
        date_b = parse_text_input(request.produs_b.sursa)

    # Task 1: Comparare cu CoT via Pydantic
    result = await compară_produse_instructor(date_a, date_b, request.preferinte)

    result_dict = result.model_dump()
    result_dict["_timp_ms"] = int((time.time() - start) * 1000)
    result_dict["_din_cache"] = False

    # Bonus: stocăm în semantic cache
    semantic_cache_store(semantic_query, result_dict)

    return result


@app.post("/analyze", response_model=CoTResult, tags=["Task 2 - CoT cu Auto-Verificare"])
async def analyze_cot(request: CoTRequest):
    """
    **Task 2**: Chain-of-Thought cu Auto-Verificare.

    Pipeline:
    1. **Generator** produce GÂNDIRE + RĂSPUNS + scor de încredere
    2. **Verificator** evaluează validitatea logicii (da/nu/nesigur)
    3. Dacă e respins → retry cu feedback (maxim `max_incercari` ori)

    Returnează istoricul complet al încercărilor pentru transparență.
    """
    start = time.time()

    # Bonus: semantic cache lookup
    semantic_query = f"cot:{request.intrebare}|{request.context or ''}"
    drift_count = semantic_cache_invalidate_drift(semantic_query)
    if drift_count > 0:
        logger.info(f"🔄 Concept drift: invalidate {drift_count} intrări din cache")

    cached = semantic_cache_lookup(semantic_query)
    if cached:
        cached["_din_cache"] = True
        return cached

    result = await cot_cu_auto_verificare(
        intrebare=request.intrebare,
        context=request.context,
        max_incercari=request.max_incercari
    )

    result_dict = result.model_dump()
    result_dict["_timp_ms"] = int((time.time() - start) * 1000)

    semantic_cache_store(semantic_query, result_dict)

    return result


@app.get("/health", tags=["Sistem"])
async def health():
    """Verificare stare serviciu și conexiune Ollama."""
    try:
        client.models.list()
        ollama_ok = True
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "degraded",
        "model": MODEL,
        "instructor": "active",
        "semantic_cache": "enabled" if SEMANTIC_CACHE_ENABLED else "disabled (instalează sentence-transformers)",
        "cache_entries": len(cache.get(SEMANTIC_INDEX_KEY, [])),
    }


@app.delete("/cache", tags=["Sistem"])
async def clear_cache():
    """Golește tot cache-ul (semantic + disk)."""
    cache.clear()
    return {"message": "Cache cleared complet (disk + semantic index)"}


@app.get("/cache/stats", tags=["Sistem"])
async def cache_stats():
    """Statistici despre semantic cache."""
    index = cache.get(SEMANTIC_INDEX_KEY, [])
    return {
        "total_intrari": len(index),
        "semantic_cache_enabled": SEMANTIC_CACHE_ENABLED,
        "similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
        "previzualizare": [
            {
                "query": e.get("query_preview", ""),
                "timestamp": e.get("timestamp", 0),
            }
            for e in index[-5:]  # Ultimele 5
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
