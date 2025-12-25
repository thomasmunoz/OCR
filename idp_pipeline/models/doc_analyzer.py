"""
ORACLE-DESIGNED DOCUMENT ANALYZER v2.1
======================================
Semantic analysis of documents for:
- Summary generation (extractive + AI-based)
- Tags/keywords extraction with classification
- Word cloud generation with occurrence percentages
"""

import re
import math
import logging
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# STOPWORDS (Multi-language support)
# =============================================================================

STOPWORDS_FR = {
    'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux', 'ce', 'cette',
    'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
    'nos', 'votre', 'vos', 'leur', 'leurs', 'je', 'tu', 'il', 'elle', 'on', 'nous',
    'vous', 'ils', 'elles', 'qui', 'que', 'quoi', 'dont', 'où', 'et', 'ou', 'mais',
    'donc', 'car', 'ni', 'ne', 'pas', 'plus', 'moins', 'très', 'bien', 'mal',
    'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'pouvoir', 'vouloir',
    'est', 'sont', 'a', 'ont', 'été', 'sera', 'seront', 'fait', 'font', 'dit',
    'dans', 'sur', 'sous', 'avec', 'sans', 'pour', 'par', 'en', 'entre', 'vers',
    'chez', 'avant', 'après', 'depuis', 'pendant', 'comme', 'si', 'tout', 'tous',
    'toute', 'toutes', 'autre', 'autres', 'même', 'mêmes', 'aussi', 'ainsi',
    'alors', 'donc', 'puis', 'ensuite', 'enfin', 'cela', 'ceci', 'celui', 'celle',
    'ceux', 'celles', 'quel', 'quelle', 'quels', 'quelles', 'chaque', 'quelque',
    'quelques', 'plusieurs', 'certain', 'certains', 'certaine', 'certaines',
    'peu', 'beaucoup', 'trop', 'assez', 'encore', 'toujours', 'jamais', 'souvent',
    'parfois', 'ici', 'là', 'où', 'quand', 'comment', 'pourquoi', 'oui', 'non',
    'd', 'l', 'n', 's', 'c', 'j', 'm', 't', 'y', 'qu', 'se', 'me', 'te', 'lui',
    'soi', 'peut', 'peuvent', 'doit', 'doivent', 'faut', 'soit', 'aux', 'ces',
    'etc', 'via', 'cas', 'fin', 'mis', 'mise', 'soit', 'été', 'ans', 'an', 'jour',
    'jours', 'fois', 'part', 'suite', 'lieu', 'titre', 'objet', 'article'
}

STOPWORDS_EN = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'mine',
    'yours', 'hers', 'ours', 'theirs', 'what', 'which', 'who', 'whom', 'whose',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'once', 'if', 'because', 'until', 'while', 'although', 'though', 'after',
    'before', 'above', 'below', 'between', 'into', 'through', 'during', 'under',
    'again', 'further', 'any', 'about', 'against', 'being', 'having', 'doing',
    'etc', 'e', 'g', 'ie', 'eg', 'vs', 'per', 'via', 'page', 'date', 'time'
}

ALL_STOPWORDS = STOPWORDS_FR | STOPWORDS_EN


# =============================================================================
# DOCUMENT TYPE CLASSIFICATION
# =============================================================================

class DocumentType(Enum):
    """Document type classification"""
    CONTRACT = "contract"
    FORM = "form"
    INVOICE = "invoice"
    REPORT = "report"
    LETTER = "letter"
    POLICY = "policy"
    LEGAL = "legal"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    HR = "hr"
    MARKETING = "marketing"
    OTHER = "other"


# Document type indicators (keywords that suggest document type)
DOC_TYPE_INDICATORS = {
    DocumentType.CONTRACT: [
        'contrat', 'contract', 'agreement', 'accord', 'convention', 'clause',
        'parties', 'signataire', 'engagement', 'obligation', 'terme', 'durée',
        'résiliation', 'termination', 'avenant', 'amendment'
    ],
    DocumentType.FORM: [
        'formulaire', 'form', 'demande', 'request', 'remplir', 'fill',
        'cocher', 'check', 'signature', 'date', 'nom', 'prénom', 'adresse'
    ],
    DocumentType.INVOICE: [
        'facture', 'invoice', 'montant', 'amount', 'total', 'tva', 'vat',
        'prix', 'price', 'paiement', 'payment', 'référence', 'numéro'
    ],
    DocumentType.REPORT: [
        'rapport', 'report', 'analyse', 'analysis', 'résultats', 'results',
        'conclusion', 'recommandation', 'synthèse', 'summary', 'étude'
    ],
    DocumentType.LETTER: [
        'madame', 'monsieur', 'dear', 'cher', 'chère', 'cordialement',
        'sincerely', 'regards', 'veuillez', 'objet', 'subject'
    ],
    DocumentType.POLICY: [
        'politique', 'policy', 'procédure', 'procedure', 'règlement',
        'regulation', 'directive', 'guideline', 'norme', 'standard'
    ],
    DocumentType.LEGAL: [
        'juridique', 'legal', 'loi', 'law', 'article', 'décret', 'decree',
        'tribunal', 'court', 'avocat', 'lawyer', 'litige', 'dispute'
    ],
    DocumentType.TECHNICAL: [
        'technique', 'technical', 'spécification', 'specification',
        'architecture', 'système', 'system', 'api', 'code', 'logiciel'
    ],
    DocumentType.FINANCIAL: [
        'financier', 'financial', 'budget', 'bilan', 'balance', 'compte',
        'account', 'investissement', 'investment', 'rendement', 'return'
    ],
    DocumentType.HR: [
        'ressources humaines', 'human resources', 'rh', 'hr', 'employé',
        'employee', 'recrutement', 'recruitment', 'salaire', 'salary',
        'congé', 'leave', 'formation', 'training', 'prime', 'bonus'
    ],
    DocumentType.MARKETING: [
        'marketing', 'publicité', 'advertising', 'campagne', 'campaign',
        'client', 'customer', 'marque', 'brand', 'promotion', 'vente'
    ]
}


# =============================================================================
# DOCUMENT ANALYZER
# =============================================================================

@dataclass
class DocumentSummary:
    """Document summary with key information"""
    brief: str  # 1-2 sentence summary
    detailed: str  # More detailed summary (3-5 sentences)
    key_points: List[str]  # Bullet points of key information
    confidence: float = 0.8


@dataclass
class DocumentTags:
    """Document tags/keywords"""
    document_type: str  # Primary document type
    categories: List[str]  # Category tags (contract, hr, financial, etc.)
    keywords: List[str]  # Important extracted keywords
    entities: List[str]  # Named entities (companies, people, etc.)
    confidence: float = 0.8


@dataclass
class WordCloudEntry:
    """Single entry in word cloud"""
    word: str
    count: int
    percentage: float


@dataclass
class DocumentWordCloud:
    """Word cloud with top words and percentages"""
    top_words: List[WordCloudEntry]
    others_percentage: float
    total_words: int
    unique_words: int


@dataclass
class DocumentAnalysis:
    """Complete document analysis result"""
    summary: DocumentSummary
    tags: DocumentTags
    word_cloud: DocumentWordCloud


class DocumentAnalyzer:
    """
    ORACLE-DESIGNED: Comprehensive document analysis engine.

    Provides:
    - Summary generation (extractive)
    - Tags/keywords extraction with classification
    - Word cloud generation with occurrence percentages
    """

    def __init__(self, language: str = "auto"):
        """
        Initialize document analyzer.

        Args:
            language: Primary language ("fr", "en", "auto")
        """
        self.language = language
        self.stopwords = ALL_STOPWORDS

    def analyze(self, text: str, full_text: str = None) -> DocumentAnalysis:
        """
        ORACLE-DESIGNED: Complete document analysis.

        Args:
            text: Document text to analyze
            full_text: Optional full text (if text is a sample)

        Returns:
            DocumentAnalysis with summary, tags, and word cloud
        """
        analysis_text = full_text or text

        # Generate each component
        summary = self.generate_summary(analysis_text)
        tags = self.extract_tags(analysis_text)
        word_cloud = self.generate_word_cloud(analysis_text)

        return DocumentAnalysis(
            summary=summary,
            tags=tags,
            word_cloud=word_cloud
        )

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def generate_summary(self, text: str, max_sentences: int = 3) -> DocumentSummary:
        """
        ORACLE-DESIGNED: Generate document summary using extractive method.

        Extracts the most important sentences based on:
        - Position (first sentences often contain key info)
        - Keyword density (sentences with important terms)
        - Length (not too short, not too long)
        """
        # Clean and split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return DocumentSummary(
                brief="Document vide ou illisible.",
                detailed="Le document ne contient pas de texte exploitable.",
                key_points=[],
                confidence=0.0
            )

        # Score sentences
        scored_sentences = []
        word_freq = self._get_word_frequency(text)

        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue

            score = self._score_sentence(sentence, i, len(sentences), word_freq)
            scored_sentences.append((sentence, score, i))

        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Get top sentences for brief summary (top 1-2)
        brief_sentences = sorted(scored_sentences[:2], key=lambda x: x[2])
        brief = ' '.join(s[0] for s in brief_sentences)

        # Get more sentences for detailed summary (top 3-5)
        detailed_sentences = sorted(scored_sentences[:min(5, len(scored_sentences))], key=lambda x: x[2])
        detailed = ' '.join(s[0] for s in detailed_sentences)

        # Extract key points (top scoring unique concepts)
        key_points = self._extract_key_points(text, scored_sentences)

        return DocumentSummary(
            brief=brief[:500] if len(brief) > 500 else brief,
            detailed=detailed[:1000] if len(detailed) > 1000 else detailed,
            key_points=key_points[:5],
            confidence=0.75 if len(scored_sentences) >= 3 else 0.5
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'--- PAGE BREAK ---', ' ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean and filter
        cleaned = []
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and not s.startswith('[') and not s.isupper():
                cleaned.append(s)

        return cleaned

    def _score_sentence(self, sentence: str, position: int, total: int,
                        word_freq: Dict[str, int]) -> float:
        """Score a sentence for summary inclusion"""
        score = 0.0

        # Position bonus (first sentences are often important)
        if position < 3:
            score += 2.0 - (position * 0.5)
        elif position == total - 1:
            score += 0.5  # Last sentence sometimes contains conclusion

        # Keyword density
        words = self._tokenize(sentence)
        for word in words:
            if word in word_freq and word not in self.stopwords:
                score += word_freq[word] * 0.1

        # Length penalty (prefer medium-length sentences)
        word_count = len(words)
        if 10 <= word_count <= 30:
            score += 1.0
        elif word_count < 10:
            score -= 0.5
        elif word_count > 50:
            score -= 1.0

        # Contains numbers (often factual)
        if re.search(r'\d+', sentence):
            score += 0.5

        # Contains key indicators
        key_indicators = ['important', 'essentiel', 'objectif', 'but', 'conclusion',
                         'résultat', 'result', 'key', 'main', 'principal']
        for indicator in key_indicators:
            if indicator in sentence.lower():
                score += 0.5

        return score

    def _extract_key_points(self, text: str, scored_sentences: List[Tuple]) -> List[str]:
        """Extract key points from document"""
        key_points = []

        # Look for bullet points or numbered items
        bullet_pattern = r'[•\-\*]\s*(.+?)(?=\n|$)'
        numbered_pattern = r'\d+[.)\]]\s*(.+?)(?=\n|$)'

        bullets = re.findall(bullet_pattern, text)
        numbered = re.findall(numbered_pattern, text)

        for item in (bullets + numbered)[:5]:
            if len(item) > 20 and len(item) < 200:
                key_points.append(item.strip())

        # If not enough bullet points, extract from top sentences
        if len(key_points) < 3:
            for sentence, score, _ in scored_sentences[:5]:
                if sentence not in key_points:
                    # Shorten if needed
                    point = sentence[:150] + '...' if len(sentence) > 150 else sentence
                    key_points.append(point)
                    if len(key_points) >= 5:
                        break

        return key_points

    # =========================================================================
    # TAGS/KEYWORDS EXTRACTION
    # =========================================================================

    def extract_tags(self, text: str) -> DocumentTags:
        """
        ORACLE-DESIGNED: Extract document tags and keywords.

        Returns:
        - document_type: Primary classification
        - categories: Relevant category tags
        - keywords: Important extracted terms
        - entities: Named entities found
        """
        text_lower = text.lower()

        # Classify document type
        doc_type = self._classify_document_type(text_lower)

        # Extract categories
        categories = self._extract_categories(text_lower)

        # Extract keywords using TF-IDF-like scoring
        keywords = self._extract_keywords(text)

        # Extract named entities
        entities = self._extract_entities(text)

        return DocumentTags(
            document_type=doc_type.value,
            categories=categories,
            keywords=keywords[:15],
            entities=entities[:10],
            confidence=0.8
        )

    def _classify_document_type(self, text_lower: str) -> DocumentType:
        """Classify document type based on content"""
        scores = {}

        for doc_type, indicators in DOC_TYPE_INDICATORS.items():
            score = 0
            for indicator in indicators:
                count = text_lower.count(indicator)
                score += count
            scores[doc_type] = score

        # Get highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type

        return DocumentType.OTHER

    def _extract_categories(self, text_lower: str) -> List[str]:
        """Extract relevant category tags"""
        categories = []

        for doc_type, indicators in DOC_TYPE_INDICATORS.items():
            match_count = 0
            for indicator in indicators:
                if indicator in text_lower:
                    match_count += 1

            if match_count >= 2:  # At least 2 indicators
                categories.append(doc_type.value)

        # Ensure primary type is first
        return list(set(categories))[:5]

    def _extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        """Extract keywords using TF-IDF-like scoring"""
        words = self._tokenize(text)

        # Filter stopwords and short words
        filtered_words = [
            w for w in words
            if w not in self.stopwords
            and len(w) > 3
            and not w.isdigit()
            and not re.match(r'^[\d\W]+$', w)
        ]

        # Count frequencies
        word_counts = Counter(filtered_words)

        # Calculate TF-IDF-like score
        total_words = len(filtered_words)
        scored_words = []

        for word, count in word_counts.items():
            # TF: term frequency
            tf = count / total_words if total_words > 0 else 0

            # IDF approximation: penalize very common words
            idf = math.log(1 + (total_words / (count + 1)))

            # Boost for capitalized words (likely proper nouns/important terms)
            boost = 1.5 if word[0].isupper() else 1.0

            score = tf * idf * boost
            scored_words.append((word, score, count))

        # Sort by score
        scored_words.sort(key=lambda x: x[1], reverse=True)

        return [w[0] for w in scored_words[:top_n]]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (companies, people, etc.)"""
        entities = []

        # Pattern for capitalized multi-word names
        name_pattern = r'\b([A-Z][a-zéèêëàâùûîïôö]+(?:\s+[A-Z][a-zéèêëàâùûîïôö]+)+)\b'
        names = re.findall(name_pattern, text)

        for name in names:
            # Filter out common phrases that aren't entities
            if name not in entities and len(name) > 3:
                # Check it's not a sentence start
                if not any(name.startswith(w) for w in ['Le ', 'La ', 'Les ', 'Un ', 'Une ', 'The ', 'A ']):
                    entities.append(name)

        # Pattern for company names (contains SA, SAS, SARL, Ltd, Inc, etc.)
        company_pattern = r'\b([A-Z][A-Za-z\s&\-\.]+(?:SA|SAS|SARL|Ltd|Inc|LLC|GmbH|AG))\b'
        companies = re.findall(company_pattern, text)
        entities.extend([c.strip() for c in companies if c.strip() not in entities])

        # Pattern for emails (can indicate entities)
        email_pattern = r'@([a-zA-Z0-9\-]+)\.'
        domains = re.findall(email_pattern, text)
        for domain in domains:
            if domain not in ['gmail', 'yahoo', 'hotmail', 'outlook'] and domain not in entities:
                entities.append(domain.capitalize())

        return list(set(entities))[:10]

    # =========================================================================
    # WORD CLOUD GENERATION
    # =========================================================================

    def generate_word_cloud(self, text: str, top_n: int = 10) -> DocumentWordCloud:
        """
        ORACLE-DESIGNED: Generate word cloud with occurrence percentages.

        Returns:
        - top_words: Top N words with count and percentage
        - others_percentage: Percentage for all other words
        - total_words: Total word count
        - unique_words: Number of unique words
        """
        words = self._tokenize(text)

        # Filter
        filtered_words = [
            w for w in words
            if w not in self.stopwords
            and len(w) > 2
            and not w.isdigit()
            and not re.match(r'^[\d\W]+$', w)
        ]

        total_words = len(filtered_words)
        word_counts = Counter(filtered_words)
        unique_words = len(word_counts)

        if total_words == 0:
            return DocumentWordCloud(
                top_words=[],
                others_percentage=0.0,
                total_words=0,
                unique_words=0
            )

        # Get top N words
        top_entries = []
        top_count = 0

        for word, count in word_counts.most_common(top_n):
            percentage = (count / total_words) * 100
            top_entries.append(WordCloudEntry(
                word=word,
                count=count,
                percentage=round(percentage, 2)
            ))
            top_count += count

        # Calculate others percentage
        others_count = total_words - top_count
        others_percentage = (others_count / total_words) * 100 if total_words > 0 else 0

        return DocumentWordCloud(
            top_words=top_entries,
            others_percentage=round(others_percentage, 2),
            total_words=total_words,
            unique_words=unique_words
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Normalize
        text = text.lower()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # Split into words
        words = text.split()

        return words

    def _get_word_frequency(self, text: str) -> Dict[str, int]:
        """Get word frequency dictionary"""
        words = self._tokenize(text)
        return dict(Counter(words))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def analyze_document(text: str, language: str = "auto") -> Dict[str, Any]:
    """
    ORACLE-DESIGNED: Quick document analysis function.

    Returns dictionary with summary, tags, and word_cloud ready for JSON output.
    """
    analyzer = DocumentAnalyzer(language)
    analysis = analyzer.analyze(text)

    return {
        "summary": {
            "brief": analysis.summary.brief,
            "detailed": analysis.summary.detailed,
            "key_points": analysis.summary.key_points,
            "confidence": analysis.summary.confidence
        },
        "tags": {
            "document_type": analysis.tags.document_type,
            "categories": analysis.tags.categories,
            "keywords": analysis.tags.keywords,
            "entities": analysis.tags.entities,
            "confidence": analysis.tags.confidence
        },
        "word_cloud": {
            "top_words": [
                {"word": w.word, "count": w.count, "percentage": w.percentage}
                for w in analysis.word_cloud.top_words
            ],
            "others_percentage": analysis.word_cloud.others_percentage,
            "total_words": analysis.word_cloud.total_words,
            "unique_words": analysis.word_cloud.unique_words
        }
    }
