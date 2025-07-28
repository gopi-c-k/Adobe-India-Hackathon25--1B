---

## 📌 Approach Explanation – Persona-Driven Document Intelligence

### **1. Overview**

This system is designed to process a set of PDF documents, extract meaningful sections, and rank them based on a given **persona** and **job-to-be-done**. The goal is to deliver **context-aware summaries** and **prioritized content** for different scenarios such as research, business analysis, or trip planning. The entire solution is lightweight, offline-compatible, and optimized to run inside a Docker container.

---

### **2. Methodology**

#### ✅ a) Input Handling

* Accepts a JSON file containing:

  * Challenge info
  * Document list with filenames
  * Persona role
  * Job-to-be-done task
* Reads all the referenced PDFs from the `/app/input` directory.
* Stores outputs in a standardized JSON format for downstream integration.

#### ✅ b) Text Extraction

* Uses `PyPDF2` to extract text **per page** while retaining page numbers.
* Maintains mapping of `filename → page → text` for traceability.

#### ✅ c) Section Identification

* Applies **hybrid rule-based segmentation**:

  * Regex patterns to detect headings (capitalized titles, numbered sections, etc.).
  * Fallback mechanism splits text into paragraphs and generates pseudo-headings when none exist.
* Filters out trivial content to avoid noise.

#### ✅ d) Content Refinement

* Cleans up extracted text (removes redundant whitespace, fixes broken lines).
* Uses `refine_content()` to intelligently truncate content to \~500 characters while preserving sentence boundaries.
* Ensures summaries remain coherent and ready for ranking.

#### ✅ e) Semantic Relevance Ranking

* Loads **`sentence-transformers/all-mpnet-base-v2`**, a compact but high-performing embedding model.
* Encodes:

  * **Sections** (title + content)
  * **Query context** (`persona + job-to-be-done`)
* Uses **cosine similarity** to rank sections by relevance.
* Selects the **Top-N sections** (configurable, default = 5).

#### ✅ f) Output Generation

Produces a JSON file with:

1. **Metadata:** input documents, persona, job description, timestamp.
2. **Extracted Sections:** document, section title, page number, importance rank.
3. **Subsection Analysis:** refined content snippets with document references.

---

### **3. Design Considerations**

* **Domain-Agnostic:** Works with academic, travel, business, or any PDF type due to semantic embeddings.
* **Offline Capable:** Packages model inside Docker; no external API calls needed.
* **Performance:** Preloads the model once; can process multiple PDFs under a minute on CPU.
* **Robustness:** Handles messy PDFs, missing headings, and irregular structures gracefully.

---

### **4. Execution Flow**

1. **Prepare Input:**

   * Place PDFs in `/app/input/`.
   * Create `input.json` with persona and job details.

2. **Build Docker Image:**

   ```bash
   docker build -t persona-doc-intel .
   ```

3. **Run Container:**

   ```bash
   docker run --rm \
     -v $(pwd)/input:/app/input \
     -v $(pwd)/output:/app/output \
     persona-doc-intel
   ```

4. **Output:**

   * Results are saved in:

     ```
     /app/output/analysis_results.json
     ```

---

### **5. Deployment Advantages**

* ✅ Portable & self-contained (via Docker).
* ✅ No internet dependency for model inference.
* ✅ Can be integrated into larger pipelines or evaluation systems.
* ✅ Supports environment variable overrides for persona/job.

---
