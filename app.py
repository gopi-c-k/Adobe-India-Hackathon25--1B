import json, re, os
from datetime import datetime
from typing import List, Dict, Tuple, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

class DocumentAnalyst:
    def __init__(self):
        print("Loading only all-mpnet-base-v2 model...")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print("Model loaded successfully!")

    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        page_texts = {}
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        page_texts[page_num] = text.strip()
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
        return page_texts

    def identify_sections(self, text: str, page_num: int) -> List[Tuple[str, str]]:
        sections = []
        patterns = [
            r'^([A-Z][A-Za-z\s\-\&\(\)]{10,80})\s*$',
            r'^(\d+\.?\s+[A-Za-z][A-Za-z\s\-\&\(\)]{5,60})',
            r'^([A-Z\s]{3,50})\s*$',
            r'^([A-Za-z\s\-\&\(\)]{15,80}:)',
        ]
        lines = text.split('\n')
        current_section, current_content = None, []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            is_heading, heading_text = False, None
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    heading_text = match.group(1).strip()
                    is_heading = True
                    break
            if is_heading and heading_text:
                if current_section and current_content:
                    content = ' '.join(current_content).strip()
                    if len(content) > 50:
                        sections.append((current_section, content))
                current_section = heading_text
                current_content = []
            else:
                if current_section:
                    current_content.append(line)

        if current_section and current_content:
            content = ' '.join(current_content).strip()
            if len(content) > 50:
                sections.append((current_section, content))

        if not sections:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            for para in paragraphs:
                if len(para) > 100:
                    title = ' '.join(para.split()[:8]) + "..."
                    sections.append((title, para))
        return sections

    def refine_content(self, content: str, max_length: int = 500) -> str:
        cleaned = re.sub(r'\s+', ' ', content).strip()
        if len(cleaned) <= max_length:
            return cleaned
        sentences = re.split(r'[.!?]+', cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        refined, current_length = [], 0
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            refined.append(sentence)
            current_length += len(sentence)
        return '. '.join(refined) + '.' if refined else cleaned[:max_length]

    def process_documents(self, pdf_folder: str, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        query_context = f"{persona}: {job_to_be_done}"
        all_sections = []
        for filename in os.listdir(pdf_folder):
            if not filename.lower().endswith('.pdf'):
                continue
            pdf_path = os.path.join(pdf_folder, filename)
            page_texts = self.extract_text_from_pdf(pdf_path)
            for page_num, text in page_texts.items():
                sections = self.identify_sections(text, page_num)
                for section_title, section_content in sections:
                    all_sections.append({
                        'document': filename,
                        'section_title': section_title,
                        'content': section_content,
                        'refined_text': self.refine_content(section_content),
                        'page_number': page_num
                    })

        if all_sections:
            section_texts = [f"{s['section_title']}: {s['content']}" for s in all_sections]
            section_embeddings = self.encode(section_texts)
            query_embedding = self.encode([query_context])
            similarities = cosine_similarity(query_embedding, section_embeddings)[0]
            ranked_indices = np.argsort(similarities)[::-1][:5]

            top_sections, top_subsection_analysis = [], []
            for rank, idx in enumerate(ranked_indices, 1):
                section = all_sections[idx]
                top_sections.append({
                    'document': section['document'],
                    'section_title': section['section_title'],
                    'importance_rank': rank,
                    'page_number': section['page_number']
                })
                top_subsection_analysis.append({
                    'document': section['document'],
                    'refined_text': section['refined_text'],
                    'page_number': section['page_number']
                })
        else:
            top_sections, top_subsection_analysis = [], []

        result = {
            "metadata": {
                "input_documents": os.listdir(pdf_folder),
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": top_sections,
            "subsection_analysis": top_subsection_analysis
        }
        return result

if __name__ == "__main__":
    persona = os.getenv("PERSONA", "Default Persona")
    job_to_be_done = os.getenv("JOB", "Default Job")
    pdf_folder = "./input"
    analyst = DocumentAnalyst()
    result = analyst.process_documents(pdf_folder, persona, job_to_be_done)

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("✅ Analysis complete. Results saved to analysis_results.json")
