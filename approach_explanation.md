# Approach Explanation – Persona-Driven Document Intelligence

## 1. Overview
This solution implements an intelligent document analysis pipeline that extracts and prioritizes relevant sections from PDF documents based on a given persona and job-to-be-done. The goal is to provide a generic framework that adapts to different domains (academic, business, educational, etc.) while staying efficient enough to meet strict runtime and model size constraints.

## 2. Methodology

### a) Text Extraction
The system begins by scanning all PDF files in the input directory and extracting text on a per-page basis using `PyPDF2`. Each page's content is preserved to maintain contextual integrity, which is important for aligning extracted sections with their source locations.

### b) Section Identification
A hybrid rule-based method is applied to segment the document into meaningful sections:
- Regular expressions detect common patterns for headings (capitalized titles, numbered sections, etc.).
- When explicit headings are absent, the fallback mechanism splits text into paragraphs and uses the first few words as a pseudo-title.
- Only sections with substantial content are kept to avoid noise.

### c) Content Refinement
Each section is cleaned, whitespace-normalized, and truncated intelligently using sentence boundaries to maintain coherence. The `refine_content()` function ensures the output remains concise while retaining core meaning, which is critical for generating high-quality summaries.

### d) Semantic Relevance Ranking
The core intelligence comes from embedding-based semantic similarity:
- The model `sentence-transformers/all-mpnet-base-v2` (under 1GB, CPU-compatible) generates vector representations of all sections.
- The persona and job-to-be-done are combined into a “query context” and encoded.
- Cosine similarity is computed between query embeddings and section embeddings, and the top-N sections are ranked by relevance.

### e) Output Generation
The final output is a structured JSON containing:
1. **Metadata:** input documents, persona, job description, and timestamp.
2. **Extracted Sections:** document name, page number, section title, and importance rank.
3. **Subsection Analysis:** document name, refined text, and page number.

This format allows downstream systems or evaluators to interpret and use the results consistently.

## 3. Design Considerations
- **Generality:** Regex patterns and embedding-based scoring make the system domain-agnostic.
- **Performance:** The model is preloaded once; all operations are CPU-optimized to ensure processing 3–5 documents within 60 seconds.
- **Robustness:** Handles missing headings, noisy PDFs, and varying document structures.
- **No Internet Dependency:** The model is packaged inside the Docker container for offline execution.

## 4. Execution Flow
1. Place PDF files in `/app/input`.
2. Provide `PERSONA` and `JOB` as environment variables or defaults.
3. Run the Docker container.
4. The system processes documents and generates `analysis_results.json` in `/app/output`.


### **Build the Docker image**

```powershell
docker build -t mysolution:latest .
```

### **Run the container (PowerShell)**

```powershell
docker run --rm `
  -v ${PWD}\input:/app/input `
  -v ${PWD}\output:/app/output `
  -e PERSONA="PhD Researcher in Computational Biology" `
  -e JOB="Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks" `
  mysolution:latest
```

### **Run the container (Git Bash/Linux)**

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PERSONA="PhD Researcher in Computational Biology" \
  -e JOB="Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks" \
  mysolution:latest
```

### **Output**

* The processed results will be saved as:

  ```
  ./output/analysis_results.json
  ```

---

