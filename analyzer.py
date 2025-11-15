from __future__ import annotations

from typing import List, Dict

# PDF extraction
import fitz  # PyMuPDF

# NLP
import nltk
from nltk.corpus import stopwords


def _ensure_nltk_resources() -> None:
	"""Ensure tokenizers and stopwords are available.

	Handles both legacy and new NLTK resource names (punkt vs punkt_tab).
	Falls back gracefully if downloads fail (tokenize by whitespace).
	"""
	# Stopwords
	try:
		stopwords.words("english")
	except LookupError:
		nltk.download("stopwords", quiet=True)

	# Tokenizers (punkt or punkt_tab depending on NLTK version)
	try:
		nltk.data.find("tokenizers/punkt")
	except LookupError:
		try:
			nltk.data.find("tokenizers/punkt_tab")
		except LookupError:
			# Try download new name first, then legacy
			try:
				nltk.download("punkt_tab", quiet=True)
			except Exception:
				nltk.download("punkt", quiet=True)


def extract_text_from_pdf(pdf_path: str) -> str:
	"""Extract text from a PDF file using PyMuPDF."""
	try:
		doc = fitz.open(pdf_path)
		text_parts: List[str] = []
		for page in doc:
			text_parts.append(page.get_text())
		return "\n".join(text_parts)
	except Exception:
		return ""


skill_keywords: Dict[str, List[str]] = {
	"data science": [
		"python", "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "pytorch",
		"machine learning", "deep learning", "nlp", "statistics", "sql", "modeling",
	],
	"web development": [
		"html", "css", "javascript", "react", "vue", "angular", "node", "express",
		"django", "flask", "typescript",
	],
	"marketing": ["marketing", "seo", "branding", "advertising", "content", "campaign"],
	"sales": ["sales", "crm", "negotiation", "lead generation", "b2b", "b2c"],
	"graphic design": ["photoshop", "illustrator", "figma", "ui", "ux", "adobe"],
	"medicine": ["medical", "clinical", "patient", "diagnosis", "surgery", "nurse"],
	"finance": ["accounting", "finance", "budgeting", "excel", "audit", "tax"],
	"education": ["teacher", "curriculum", "lesson", "training", "student"],
	"engineering": ["cad", "mechanical", "electrical", "civil", "autocad"],
	"law": ["law", "legal", "contract", "compliance", "litigation"],
}


def _tokenize(text: str) -> List[str]:
	_ensure_nltk_resources()
	text = (text or "").lower()
	try:
		from nltk.tokenize import word_tokenize

		tokens = word_tokenize(text)
	except Exception:
		# Fallback to simple whitespace split if punkt is unavailable
		tokens = text.split()

	sw = set()
	try:
		sw = set(stopwords.words("english"))
	except Exception:
		pass

	return [t for t in tokens if t.isalpha() and t not in sw]


def extract_skills(text: str) -> List[str]:
	"""Extract likely skills from resume text using keyword matching."""
	tokens = set(_tokenize(text))
	detected: List[str] = []
	for field, kws in skill_keywords.items():
		for kw in kws:
			# multi-word phrases: all words must appear
			parts = kw.lower().split()
			if len(parts) > 1:
				if all(p in tokens for p in parts):
					detected.append(kw)
			else:
				if kw.lower() in tokens:
					detected.append(kw)
	# de-duplicate while preserving order
	seen = set()
	uniq: List[str] = []
	for k in detected:
		if k not in seen:
			seen.add(k)
			uniq.append(k)
	return uniq


def detect_job_field(text: str) -> str:
	"""Predict the most likely job field based on keyword counts."""
	tokens = set(_tokenize(text))
	best_field = "General"
	best_score = 0
	for field, kws in skill_keywords.items():
		score = 0
		for kw in kws:
			parts = kw.lower().split()
			if len(parts) > 1:
				if all(p in tokens for p in parts):
					score += 1
			else:
				if kw.lower() in tokens:
					score += 1
		if score > best_score:
			best_score = score
			best_field = field
	return best_field

