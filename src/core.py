import os
import re
import torch
import nltk
import pdfplumber
import PyPDF2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ValueError):
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReader:
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        return text
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {pdf_path}: {e}")
        return text
    
    def extract_text(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        text = self.extract_text_pdfplumber(pdf_path)
        if len(text.strip()) < 100:
            logger.info(f"Falling back to PyPDF2 for {pdf_path}")
            text = self.extract_text_pypdf2(pdf_path)
        if not text.strip():
            logger.warning(f"Could not extract text from {pdf_path}")
        return text.strip()
    
    def read_court_proceedings(self, folder_path: str) -> Dict[str, str]:
        proceedings = {}
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        pdf_files = list(folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        for pdf_file in pdf_files:
            try:
                text = self.extract_text(str(pdf_file))
                proceedings[pdf_file.name] = text
                logger.info(f"Extracted text from {pdf_file.name} ({len(text)} characters)")
            except Exception as e:
                logger.error(f"Error reading {pdf_file.name}: {e}")
        return proceedings
    
    def read_ipc_code(self, folder_path: str) -> Dict[str, str]:
        ipc_documents = {}
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        pdf_files = list(folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        for pdf_file in pdf_files:
            try:
                text = self.extract_text(str(pdf_file))
                ipc_documents[pdf_file.name] = text
                logger.info(f"Extracted text from {pdf_file.name} ({len(text)} characters)")
            except Exception as e:
                logger.error(f"Error reading {pdf_file.name}: {e}")
        return ipc_documents


class CaseAnalyzer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.transformer_running = False
        logger.info(f"Using device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            ).to(self.device)
            self.model.eval()
            self.model_loaded = True
            if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    logger.warning(f"Could not load sentence transformer: {e}")
                    self.sentence_model = None
            else:
                self.sentence_model = None
            try:
                self.ner_pipeline = pipeline(
                    "ner", model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=0 if self.device == "cuda" else -1
                )
            except:
                self.ner_pipeline = None
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def is_transformer_running(self):
        return self.model_loaded and self.model is not None
    
    def extract_case_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {'persons': [], 'organizations': [], 'locations': [], 'dates': []}
        if self.ner_pipeline:
            try:
                self.transformer_running = True
                ner_results = self.ner_pipeline(text[:5000])
                self.transformer_running = False
                for entity in ner_results:
                    entity_type = entity['entity_group']
                    entity_text = entity['word']
                    if entity_type == 'PER':
                        entities['persons'].append(entity_text)
                    elif entity_type == 'ORG':
                        entities['organizations'].append(entity_text)
                    elif entity_type == 'LOC':
                        entities['locations'].append(entity_text)
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
                self.transformer_running = False
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        entities['dates'] = re.findall(date_pattern, text)
        return entities
    
    def extract_charges(self, text: str) -> List[Dict[str, Any]]:
        charges = []
        section_pattern = r'(?:Section|Sec\.?|IPC)\s*(\d+[A-Z]?)\s*(?:of|IPC)?\s*([^\.]{50,200})'
        matches = re.finditer(section_pattern, text, re.IGNORECASE)
        for match in matches:
            section = match.group(1)
            context = match.group(2).strip()
            charges.append({'section': section, 'description': context, 'position': match.start()})
        return charges
    
    def extract_outcome(self, text: str) -> Dict[str, Any]:
        outcome = {'verdict': None, 'punishment': None, 'punishment_details': {}}
        conviction_keywords = ['convicted', 'guilty', 'found guilty', 'sentenced', 'punishment', 'imprisonment', 'fine']
        acquittal_keywords = ['acquitted', 'not guilty', 'discharged', 'dismissed', 'absolved', 'exonerated']
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in conviction_keywords):
            outcome['verdict'] = 'convicted'
        elif any(keyword in text_lower for keyword in acquittal_keywords):
            outcome['verdict'] = 'acquitted'
        else:
            outcome['verdict'] = 'pending'
        if outcome['verdict'] == 'convicted':
            imprisonment_patterns = [
                r'(\d+)\s*(?:years?|yrs?)\s*(?:imprisonment|rigorous|simple)?',
                r'imprisonment\s*(?:of|for)?\s*(\d+)\s*(?:years?|yrs?)',
            ]
            for pattern in imprisonment_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    outcome['punishment_details']['years'] = int(match.group(1))
                    break
            fine_pattern = r'fine\s*(?:of|Rs\.?|₹)?\s*(\d+(?:,\d+)*(?:\.\d+)?)'
            fine_match = re.search(fine_pattern, text, re.IGNORECASE)
            if fine_match:
                fine_amount = fine_match.group(1).replace(',', '')
                outcome['punishment_details']['fine'] = float(fine_amount)
        return outcome
    
    def extract_demographics(self, text: str) -> Dict[str, Any]:
        demographics = {
            'accused_gender': None, 'victim_gender': None, 'accused_caste': None,
            'victim_caste': None, 'accused_religion': None, 'victim_religion': None,
            'accused_age': None, 'victim_age': None
        }
        accused_patterns = [
            r'accused.*?(?:\.|victim|$)', r'defendant.*?(?:\.|victim|$)',
            r'appellant.*?(?:\.|victim|$)', r'petitioner.*?(?:\.|victim|$)',
        ]
        for pattern in accused_patterns:
            accused_section = re.search(pattern, text[:5000], re.IGNORECASE | re.DOTALL)
            if accused_section:
                accused_text = accused_section.group()
                if re.search(r'\b(male|man|men|he|him|his|m\.?|gentleman|boy)\b', accused_text, re.IGNORECASE):
                    demographics['accused_gender'] = 'male'
                    break
                elif re.search(r'\b(female|woman|women|she|her|f\.?|lady|girl)\b', accused_text, re.IGNORECASE):
                    demographics['accused_gender'] = 'female'
                    break
        victim_patterns = [
            r'victim.*?(?:\.|accused|$)', r'complainant.*?(?:\.|accused|$)',
            r'injured.*?(?:\.|accused|$)',
        ]
        for pattern in victim_patterns:
            victim_section = re.search(pattern, text[:5000], re.IGNORECASE | re.DOTALL)
            if victim_section:
                victim_text = victim_section.group()
                if re.search(r'\b(male|man|men|he|him|his|m\.?|gentleman|boy)\b', victim_text, re.IGNORECASE):
                    demographics['victim_gender'] = 'male'
                    break
                elif re.search(r'\b(female|woman|women|she|her|f\.?|lady|girl)\b', victim_text, re.IGNORECASE):
                    demographics['victim_gender'] = 'female'
                    break
        age_pattern = r'(?:age|aged)\s*(?:of|:)?\s*(\d+)\s*(?:years?|yrs?)'
        age_matches = re.findall(age_pattern, text[:3000], re.IGNORECASE)
        if age_matches:
            for i, match in enumerate(age_matches[:2]):
                if i == 0:
                    demographics['accused_age'] = int(match)
                elif i == 1:
                    demographics['victim_age'] = int(match)
        caste_keywords = ['caste', 'community', 'sc', 'st', 'obc', 'general']
        religion_keywords = ['hindu', 'muslim', 'christian', 'sikh', 'buddhist', 'jain']
        for keyword in caste_keywords:
            if re.search(rf'\b{keyword}\b', text[:2000], re.IGNORECASE):
                context = re.search(rf'{keyword}.*?(?:\.|,|$)', text[:2000], re.IGNORECASE)
                if context:
                    if 'accused' in text[:context.start()].lower():
                        demographics['accused_caste'] = keyword
                    elif 'victim' in text[:context.start()].lower():
                        demographics['victim_caste'] = keyword
        return demographics
    
    def analyze_case(self, case_text: str) -> Dict[str, Any]:
        logger.info("Analyzing case...")
        analysis = {
            'entities': self.extract_case_entities(case_text),
            'charges': self.extract_charges(case_text),
            'outcome': self.extract_outcome(case_text),
            'demographics': self.extract_demographics(case_text),
            'ipc_sections': []
        }
        section_pattern = r'(?:Section|Sec\.?|IPC)\s*(\d+[A-Z]?)'
        analysis['ipc_sections'] = list(set(re.findall(section_pattern, case_text, re.IGNORECASE)))
        logger.info(f"Found {len(analysis['charges'])} charges and {len(analysis['ipc_sections'])} IPC sections")
        return analysis


class IPCMatcher:
    def __init__(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Could not load sentence model: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
    
    def parse_ipc_document(self, ipc_text: str) -> Dict[str, Dict[str, Any]]:
        ipc_sections = {}
        section_pattern = r'Section\s*(\d+[A-Z]?)\s*(?:of\s*the\s*Indian\s*Penal\s*Code)?[:\-]?\s*([^S]{100,2000}?)(?=Section\s*\d+|$)'
        matches = re.finditer(section_pattern, ipc_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            section_num = match.group(1).strip()
            section_text = match.group(2).strip()
            punishment = self._extract_punishment_from_section(section_text)
            description = self._extract_description(section_text)
            ipc_sections[section_num] = {
                'section': section_num, 'description': description,
                'full_text': section_text, 'punishment': punishment
            }
        logger.info(f"Parsed {len(ipc_sections)} IPC sections")
        return ipc_sections
    
    def _extract_punishment_from_section(self, section_text: str) -> Dict[str, Any]:
        punishment = {
            'imprisonment': {'type': None, 'years': None, 'months': None, 'minimum': None, 'maximum': None},
            'fine': {'amount': None, 'minimum': None, 'maximum': None},
            'death_penalty': False
        }
        text_lower = section_text.lower()
        if re.search(r'\bdeath\s*(?:penalty|sentence)?\b', text_lower):
            punishment['death_penalty'] = True
        if re.search(r'\b(?:imprisonment\s*for\s*life|life\s*imprisonment)\b', text_lower):
            punishment['imprisonment']['type'] = 'life'
            punishment['imprisonment']['years'] = 'life'
        max_pattern = r'(?:imprisonment|rigorous|simple).*?(?:extend\s*to|up\s*to|of)\s*(\d+)\s*(?:years?|yrs?)'
        max_match = re.search(max_pattern, text_lower)
        if max_match:
            punishment['imprisonment']['maximum'] = int(max_match.group(1))
            punishment['imprisonment']['years'] = int(max_match.group(1))
        min_pattern = r'(?:not\s*less\s*than|minimum|at\s*least)\s*(\d+)\s*(?:years?|yrs?|months?)'
        min_match = re.search(min_pattern, text_lower)
        if min_match:
            punishment['imprisonment']['minimum'] = int(min_match.group(1))
        fine_patterns = [
            r'fine\s*(?:which\s*may\s*extend\s*to|up\s*to|of)?\s*(?:Rs\.?|₹|rupees?)?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rupees?|rs\.?|₹)\s*(?:fine)?'
        ]
        for pattern in fine_patterns:
            fine_match = re.search(pattern, text_lower)
            if fine_match:
                fine_amount = float(fine_match.group(1).replace(',', ''))
                punishment['fine']['amount'] = fine_amount
                punishment['fine']['maximum'] = fine_amount
                break
        return punishment
    
    def _extract_description(self, section_text: str) -> str:
        description = section_text
        punishment_phrases = [
            r'punishment.*?imprisonment.*?', r'imprisonment.*?fine.*?',
            r'fine.*?imprisonment.*?', r'shall\s*be\s*punished.*?',
        ]
        for phrase in punishment_phrases:
            description = re.sub(phrase, '', description, flags=re.IGNORECASE | re.DOTALL)
        sentences = re.split(r'[\.!?]\s+', description)
        if sentences:
            description = sentences[0]
        return description.strip()[:200]
    
    def find_matching_sections(self, case_sections: List[str], ipc_database: Dict[str, Dict]) -> Dict[str, Dict]:
        matched_sections = {}
        for section in case_sections:
            if section in ipc_database:
                matched_sections[section] = ipc_database[section]
            else:
                for ipc_section in ipc_database.keys():
                    if section in ipc_section or ipc_section in section:
                        matched_sections[section] = ipc_database[ipc_section]
                        break
        return matched_sections
    
    def get_expected_punishment(self, ipc_section: Dict[str, Any]) -> Dict[str, Any]:
        if 'punishment' not in ipc_section:
            return {}
        punishment = ipc_section['punishment']
        expected = {
            'imprisonment_years': None, 'imprisonment_type': None,
            'fine_amount': None, 'death_penalty': False
        }
        if punishment.get('death_penalty'):
            expected['death_penalty'] = True
        if punishment.get('imprisonment', {}).get('years'):
            years = punishment['imprisonment']['years']
            if years == 'life':
                expected['imprisonment_years'] = 'life'
            else:
                expected['imprisonment_years'] = years
        if punishment.get('imprisonment', {}).get('type'):
            expected['imprisonment_type'] = punishment['imprisonment']['type']
        if punishment.get('fine', {}).get('amount'):
            expected['fine_amount'] = punishment['fine']['amount']
        return expected


class PunishmentComparator:
    def __init__(self):
        pass
    
    def normalize_punishment(self, punishment: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            'imprisonment_years': None, 'imprisonment_months': None, 'imprisonment_days': None,
            'fine_amount': None, 'death_penalty': False, 'life_imprisonment': False
        }
        if punishment.get('imprisonment_years') == 'life' or punishment.get('life_imprisonment'):
            normalized['life_imprisonment'] = True
            normalized['imprisonment_years'] = 'life'
        if punishment.get('imprisonment_years') and punishment.get('imprisonment_years') != 'life':
            normalized['imprisonment_years'] = float(punishment.get('imprisonment_years', 0))
        if punishment.get('imprisonment_months'):
            months = float(punishment.get('imprisonment_months', 0))
            normalized['imprisonment_months'] = months
            if not normalized['imprisonment_years']:
                normalized['imprisonment_years'] = months / 12
        if punishment.get('imprisonment_days'):
            days = float(punishment.get('imprisonment_days', 0))
            normalized['imprisonment_days'] = days
        if punishment.get('fine_amount') or punishment.get('fine'):
            fine = punishment.get('fine_amount') or punishment.get('fine')
            if isinstance(fine, (int, float)):
                normalized['fine_amount'] = float(fine)
        if punishment.get('death_penalty'):
            normalized['death_penalty'] = True
        return normalized
    
    def compare_punishments(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        expected_norm = self.normalize_punishment(expected)
        actual_norm = self.normalize_punishment(actual)
        comparison = {
            'imprisonment_match': False, 'imprisonment_discrepancy': None,
            'fine_match': False, 'fine_discrepancy': None, 'overall_match': False,
            'severity': None, 'discrepancy_percentage': 0.0
        }
        if expected_norm.get('life_imprisonment'):
            if actual_norm.get('life_imprisonment'):
                comparison['imprisonment_match'] = True
            elif actual_norm.get('imprisonment_years'):
                comparison['imprisonment_match'] = False
                comparison['severity'] = 'lenient'
                comparison['imprisonment_discrepancy'] = 'life_expected_got_years'
        elif expected_norm.get('imprisonment_years') and expected_norm.get('imprisonment_years') != 'life':
            expected_years = expected_norm['imprisonment_years']
            actual_years = actual_norm.get('imprisonment_years', 0)
            if actual_years == 'life':
                comparison['imprisonment_match'] = False
                comparison['severity'] = 'harsh'
                comparison['imprisonment_discrepancy'] = 'years_expected_got_life'
            elif actual_years:
                if expected_years > 0:
                    discrepancy = abs(actual_years - expected_years) / expected_years * 100
                    comparison['imprisonment_discrepancy'] = {
                        'expected': expected_years, 'actual': actual_years,
                        'difference': actual_years - expected_years, 'percentage': discrepancy
                    }
                    if discrepancy < 10:
                        comparison['imprisonment_match'] = True
                    elif actual_years < expected_years:
                        comparison['severity'] = 'lenient'
                    else:
                        comparison['severity'] = 'harsh'
        expected_fine = expected_norm.get('fine_amount', 0)
        actual_fine = actual_norm.get('fine_amount', 0)
        if expected_fine > 0 and actual_fine > 0:
            if abs(actual_fine - expected_fine) / expected_fine < 0.1:
                comparison['fine_match'] = True
            else:
                comparison['fine_discrepancy'] = {
                    'expected': expected_fine, 'actual': actual_fine,
                    'difference': actual_fine - expected_fine,
                    'percentage': abs(actual_fine - expected_fine) / expected_fine * 100
                }
                if actual_fine < expected_fine:
                    comparison['severity'] = 'lenient'
                else:
                    comparison['severity'] = 'harsh'
        elif expected_fine > 0 and actual_fine == 0:
            comparison['fine_discrepancy'] = {
                'expected': expected_fine, 'actual': 0,
                'difference': -expected_fine, 'percentage': 100
            }
            comparison['severity'] = 'lenient'
        if comparison['imprisonment_match'] and (comparison['fine_match'] or expected_fine == 0):
            comparison['overall_match'] = True
            comparison['severity'] = 'appropriate'
        discrepancies = []
        if comparison.get('imprisonment_discrepancy') and isinstance(comparison['imprisonment_discrepancy'], dict):
            discrepancies.append(comparison['imprisonment_discrepancy'].get('percentage', 0))
        if comparison.get('fine_discrepancy'):
            discrepancies.append(comparison['fine_discrepancy'].get('percentage', 0))
        if discrepancies:
            comparison['discrepancy_percentage'] = sum(discrepancies) / len(discrepancies)
        return comparison


class BiasDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    logger.warning(f"Could not load sentence transformer: {e}")
                    self.sentence_model = None
            else:
                self.sentence_model = None
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if self.device == "cuda" else -1
                )
            except:
                self.sentiment_pipeline = None
        except Exception as e:
            logger.warning(f"Error initializing bias detector: {e}")
            self.sentence_model = None
            self.sentiment_pipeline = None
    
    def detect_gender_bias(self, text: str, demographics: Dict[str, Any], 
                          punishment_comparison: Dict[str, Any]) -> Dict[str, Any]:
        bias_score = 0.0
        indicators = []
        gender_biased_patterns = {
            'negative_female': [
                r'\b(?:she|her|woman|girl|female|lady).*?(?:provocative|promiscuous|character|reputation|dress|clothing|appearance)',
                r'(?:questionable|doubtful|suspicious|bad|poor).*?(?:character|conduct|behavior|reputation).*?(?:woman|girl|female|lady)',
                r'(?:woman|girl|female|lady).*?(?:consented|willing|agreed|invited|provoked)',
                r'(?:victim|complainant).*?(?:woman|girl|female).*?(?:character|reputation|past)',
                r'(?:mitigating|extenuating).*?(?:circumstances).*?(?:woman|girl|female)',
            ],
            'negative_male': [
                r'\b(?:he|him|man|boy|male|gentleman).*?(?:aggressive|violent|predator|dangerous|habitual)',
                r'(?:criminal|offender|accused|defendant).*?(?:man|male|boy|gentleman)',
                r'(?:previous|past|criminal).*?(?:record|history).*?(?:man|male)',
            ],
            'stereotypical': [
                r'(?:women|females|ladies).*?(?:emotional|hysterical|unreliable|weak|vulnerable)',
                r'(?:men|males|gentlemen).*?(?:strong|dominant|aggressive|powerful)',
                r'(?:gender|sex).*?(?:role|stereotype|expectation)',
            ],
            'legal_gender_bias': [
                r'(?:considering|taking.*?into.*?account).*?(?:gender|sex).*?(?:of|the)',
                r'(?:mitigating|extenuating).*?(?:gender|sex|female|male)',
                r'(?:lenient|reduced).*?(?:punishment|sentence).*?(?:because|due.*?to).*?(?:gender|sex)',
            ]
        }
        text_lower = text.lower()
        for bias_type, patterns in gender_biased_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    indicators.append(f"{bias_type}_language")
                    bias_score += 0.1
        if punishment_comparison.get('severity') in ['lenient', 'harsh']:
            discrepancy = punishment_comparison.get('discrepancy_percentage', 0)
            if punishment_comparison.get('severity') == 'lenient' and discrepancy > 10:
                accused_gender = demographics.get('accused_gender')
                victim_gender = demographics.get('victim_gender')
                if accused_gender == 'male' and victim_gender == 'female':
                    indicators.append('lenient_punishment_male_accused_female_victim')
                    bias_score += min(discrepancy / 100, 0.5)
                elif accused_gender == 'male' or victim_gender == 'female':
                    indicators.append('lenient_punishment_gender_context')
                    bias_score += min(discrepancy / 150, 0.3)
            elif punishment_comparison.get('severity') == 'harsh' and discrepancy > 10:
                accused_gender = demographics.get('accused_gender')
                victim_gender = demographics.get('victim_gender')
                if accused_gender == 'female' and victim_gender == 'male':
                    indicators.append('harsh_punishment_female_accused_male_victim')
                    bias_score += min(discrepancy / 100, 0.5)
                elif accused_gender == 'female' or victim_gender == 'male':
                    indicators.append('harsh_punishment_gender_context')
                    bias_score += min(discrepancy / 150, 0.3)
        gender_reasoning_patterns = [
            r'(?:because|due\s*to|considering).*?(?:she|her|woman|girl).*?(?:gender|sex)',
            r'(?:mitigating|extenuating).*?(?:circumstances).*?(?:gender|sex)',
        ]
        for pattern in gender_reasoning_patterns:
            if re.search(pattern, text_lower):
                indicators.append('gender_based_reasoning')
                bias_score += 0.15
        bias_score = min(bias_score, 1.0)
        detected = bias_score > 0.05
        return {
            'bias_type': 'gender', 'bias_score': bias_score,
            'bias_percentage': bias_score * 100, 'indicators': list(set(indicators)),
            'detected': detected
        }
    
    def detect_caste_bias(self, text: str, demographics: Dict[str, Any],
                         punishment_comparison: Dict[str, Any]) -> Dict[str, Any]:
        bias_score = 0.0
        indicators = []
        caste_keywords = ['caste', 'sc', 'st', 'obc', 'scheduled', 'tribe', 'community', 'dalit', 'untouchable', 'backward']
        text_lower = text.lower()
        caste_mentioned = any(keyword in text_lower for keyword in caste_keywords)
        if caste_mentioned:
            caste_reasoning_patterns = [
                r'(?:because|due\s*to|considering).*?(?:caste|community|background)',
                r'(?:mitigating|extenuating).*?(?:caste|community)',
                r'(?:caste|community).*?(?:factor|consideration|reason)',
            ]
            for pattern in caste_reasoning_patterns:
                if re.search(pattern, text_lower):
                    indicators.append('caste_based_reasoning')
                    bias_score += 0.2
            if demographics.get('accused_caste') and demographics.get('victim_caste'):
                if demographics.get('accused_caste') != demographics.get('victim_caste'):
                    if punishment_comparison.get('severity') in ['lenient', 'harsh']:
                        discrepancy = punishment_comparison.get('discrepancy_percentage', 0)
                        if discrepancy > 10:
                            indicators.append(f'{punishment_comparison.get("severity")}_punishment_caste_difference')
                            bias_score += min(discrepancy / 100, 0.4)
            elif caste_mentioned and punishment_comparison.get('severity') in ['lenient', 'harsh']:
                discrepancy = punishment_comparison.get('discrepancy_percentage', 0)
                if discrepancy > 15:
                    indicators.append('caste_mentioned_with_punishment_discrepancy')
                    bias_score += 0.1
        bias_score = min(bias_score, 1.0)
        detected = bias_score > 0.05
        return {
            'bias_type': 'caste', 'bias_score': bias_score,
            'bias_percentage': bias_score * 100, 'indicators': list(set(indicators)),
            'detected': detected
        }
    
    def detect_religious_bias(self, text: str, demographics: Dict[str, Any],
                             punishment_comparison: Dict[str, Any]) -> Dict[str, Any]:
        bias_score = 0.0
        indicators = []
        religion_keywords = ['hindu', 'muslim', 'christian', 'sikh', 'buddhist', 'jain', 'religion', 'religious', 'faith', 'community']
        text_lower = text.lower()
        religion_mentioned = any(keyword in text_lower for keyword in religion_keywords)
        if religion_mentioned:
            religion_reasoning_patterns = [
                r'(?:because|due\s*to|considering).*?(?:religion|religious|faith)',
                r'(?:mitigating|extenuating).*?(?:religion|religious)',
                r'(?:religion|religious).*?(?:factor|consideration|reason)',
            ]
            for pattern in religion_reasoning_patterns:
                if re.search(pattern, text_lower):
                    indicators.append('religion_based_reasoning')
                    bias_score += 0.2
        bias_score = min(bias_score, 1.0)
        detected = bias_score > 0.05
        return {
            'bias_type': 'religious', 'bias_score': bias_score,
            'bias_percentage': bias_score * 100, 'indicators': list(set(indicators)),
            'detected': detected
        }
    
    def detect_socioeconomic_bias(self, text: str, demographics: Dict[str, Any],
                                  punishment_comparison: Dict[str, Any]) -> Dict[str, Any]:
        bias_score = 0.0
        indicators = []
        socioeconomic_keywords = ['poor', 'poverty', 'rich', 'wealthy', 'affluent', 'economic', 'financial', 'income', 'status', 'class']
        text_lower = text.lower()
        socioeconomic_mentioned = any(keyword in text_lower for keyword in socioeconomic_keywords)
        if socioeconomic_mentioned:
            socioeconomic_patterns = [
                r'(?:because|due\s*to|considering).*?(?:poor|poverty|economic|financial)',
                r'(?:mitigating|extenuating).*?(?:economic|financial|status)',
            ]
            for pattern in socioeconomic_patterns:
                if re.search(pattern, text_lower):
                    indicators.append('socioeconomic_based_reasoning')
                    bias_score += 0.15
        bias_score = min(bias_score, 1.0)
        detected = bias_score > 0.05
        return {
            'bias_type': 'socioeconomic', 'bias_score': bias_score,
            'bias_percentage': bias_score * 100, 'indicators': list(set(indicators)),
            'detected': detected
        }
    
    def detect_all_biases(self, text: str, demographics: Dict[str, Any],
                         punishment_comparison: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Detecting biases...")
        biases = {
            'gender_bias': self.detect_gender_bias(text, demographics, punishment_comparison),
            'caste_bias': self.detect_caste_bias(text, demographics, punishment_comparison),
            'religious_bias': self.detect_religious_bias(text, demographics, punishment_comparison),
            'socioeconomic_bias': self.detect_socioeconomic_bias(text, demographics, punishment_comparison),
        }
        bias_scores = [bias['bias_score'] for bias in biases.values() if isinstance(bias, dict) and 'bias_score' in bias]
        overall_bias_score = max(bias_scores) if bias_scores else 0.0
        if bias_scores:
            weighted_score = sum(bias_scores) / len(bias_scores) * 0.7 + overall_bias_score * 0.3
            overall_bias_score = max(overall_bias_score, weighted_score)
        biases['overall'] = {
            'bias_score': overall_bias_score, 'bias_percentage': overall_bias_score * 100,
            'bias_types_detected': [bias_type for bias_type, bias_data in biases.items() if bias_data['detected']],
            'total_bias_types': len([b for b in biases.values() if b['detected']])
        }
        return biases

