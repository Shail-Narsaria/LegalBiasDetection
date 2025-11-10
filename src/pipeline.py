import os
from typing import Dict, List, Optional, Any
import pandas as pd
from tqdm import tqdm
import logging

from .core import PDFReader, CaseAnalyzer, IPCMatcher, PunishmentComparator, BiasDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalBiasPipeline:
    def __init__(self, court_proceedings_folder: str, ipc_code_folder: str):
        self.court_proceedings_folder = court_proceedings_folder
        self.ipc_code_folder = ipc_code_folder
        logger.info("Initializing pipeline components...")
        self.pdf_reader = PDFReader()
        self.case_analyzer = CaseAnalyzer()
        self.ipc_matcher = IPCMatcher()
        self.punishment_comparator = PunishmentComparator()
        self.bias_detector = BiasDetector()
        self.ipc_database = {}
        self.results = []
    
    def load_ipc_code(self):
        logger.info("Loading IPC penal code...")
        ipc_documents = self.pdf_reader.read_ipc_code(self.ipc_code_folder)
        for filename, ipc_text in ipc_documents.items():
            logger.info(f"Parsing IPC document: {filename}")
            sections = self.ipc_matcher.parse_ipc_document(ipc_text)
            self.ipc_database.update(sections)
        logger.info(f"Loaded {len(self.ipc_database)} IPC sections")
    
    def process_case(self, case_filename: str, case_text: str) -> Dict[str, Any]:
        logger.info(f"Processing case: {case_filename}")
        result = {
            'case_filename': case_filename,
            'case_text_length': len(case_text),
            'analysis': {},
            'bias_detection': {},
            'summary': {}
        }
        try:
            case_analysis = self.case_analyzer.analyze_case(case_text)
            result['analysis'] = case_analysis
            case_sections = case_analysis.get('ipc_sections', [])
            matched_sections = self.ipc_matcher.find_matching_sections(
                case_sections, self.ipc_database
            )
            result['matched_ipc_sections'] = matched_sections
            expected_punishments = {}
            for section_num, section_data in matched_sections.items():
                expected_punishments[section_num] = self.ipc_matcher.get_expected_punishment(
                    section_data
                )
            result['expected_punishments'] = expected_punishments
            actual_outcome = case_analysis.get('outcome', {})
            actual_punishment = {
                'imprisonment_years': None,
                'fine_amount': None,
                'death_penalty': False
            }
            if actual_outcome.get('punishment_details'):
                details = actual_outcome['punishment_details']
                if details.get('years'):
                    actual_punishment['imprisonment_years'] = details['years']
                if details.get('fine'):
                    actual_punishment['fine_amount'] = details['fine']
            punishment_comparisons = {}
            if matched_sections and expected_punishments:
                first_section = list(matched_sections.keys())[0]
                expected = expected_punishments[first_section]
                comparison = self.punishment_comparator.compare_punishments(
                    expected, actual_punishment
                )
                punishment_comparisons[first_section] = comparison
                result['punishment_comparison'] = comparison
            demographics = case_analysis.get('demographics', {})
            punishment_comp = punishment_comparisons.get(
                list(punishment_comparisons.keys())[0] if punishment_comparisons else None,
                {}
            )
            bias_results = self.bias_detector.detect_all_biases(
                case_text, demographics, punishment_comp
            )
            result['bias_detection'] = bias_results
            result['summary'] = self._generate_summary(
                case_analysis, punishment_comparisons, bias_results
            )
        except Exception as e:
            logger.error(f"Error processing case {case_filename}: {e}")
            result['error'] = str(e)
        return result
    
    def _generate_summary(self, case_analysis: Dict, punishment_comparisons: Dict,
                         bias_results: Dict) -> Dict[str, Any]:
        summary = {
            'verdict': case_analysis.get('outcome', {}).get('verdict', 'unknown'),
            'ipc_sections': case_analysis.get('ipc_sections', []),
            'punishment_match': False,
            'biases_detected': [],
            'overall_bias_score': 0.0
        }
        if punishment_comparisons:
            first_comp = list(punishment_comparisons.values())[0]
            summary['punishment_match'] = first_comp.get('overall_match', False)
            summary['punishment_severity'] = first_comp.get('severity', 'unknown')
        for bias_type, bias_data in bias_results.items():
            if bias_type != 'overall' and bias_data.get('detected'):
                summary['biases_detected'].append(bias_type.replace('_bias', ''))
        if bias_results.get('overall'):
            summary['overall_bias_score'] = bias_results['overall']['bias_percentage']
        return summary
    
    def run(self, output_file: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        logger.info("Starting legal bias detection pipeline...")
        if not self.ipc_database:
            self.load_ipc_code()
        logger.info("Loading court proceedings...")
        court_proceedings = self.pdf_reader.read_court_proceedings(
            self.court_proceedings_folder
        )
        if limit:
            court_proceedings = dict(list(court_proceedings.items())[:limit])
            logger.info(f"Limited to {limit} cases for processing")
        logger.info(f"Processing {len(court_proceedings)} cases...")
        for case_filename, case_text in tqdm(court_proceedings.items(), desc="Processing cases"):
            result = self.process_case(case_filename, case_text)
            self.results.append(result)
        df_results = self._results_to_dataframe()
        if output_file:
            df_results.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        logger.info(f"Pipeline completed. Processed {len(self.results)} cases.")
        return df_results
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            row = {
                'case_filename': result.get('case_filename', ''),
                'verdict': result.get('summary', {}).get('verdict', ''),
                'ipc_sections': ', '.join(result.get('summary', {}).get('ipc_sections', [])),
                'punishment_match': result.get('summary', {}).get('punishment_match', False),
                'punishment_severity': result.get('summary', {}).get('punishment_severity', ''),
                'gender_bias_score': result.get('bias_detection', {}).get('gender_bias', {}).get('bias_percentage', 0),
                'caste_bias_score': result.get('bias_detection', {}).get('caste_bias', {}).get('bias_percentage', 0),
                'religious_bias_score': result.get('bias_detection', {}).get('religious_bias', {}).get('bias_percentage', 0),
                'socioeconomic_bias_score': result.get('bias_detection', {}).get('socioeconomic_bias', {}).get('bias_percentage', 0),
                'overall_bias_score': result.get('summary', {}).get('overall_bias_score', 0),
                'biases_detected': ', '.join(result.get('summary', {}).get('biases_detected', [])),
            }
            rows.append(row)
        return pd.DataFrame(rows)
