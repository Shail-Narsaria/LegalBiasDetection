import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import LegalBiasPipeline
from src.core import PDFReader
import logging

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Legal Bias Detection System",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #ffffff;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        padding: 2rem 0 1rem 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666666;
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    .bias-card-high {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #e53e3e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .bias-card-moderate {
        background: linear-gradient(135deg, #fffaf0 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #f6ad55;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .bias-card-low {
        background: linear-gradient(135deg, #f0fff4 0%, #e6ffed 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #48bb78;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .info-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0.25rem 0;
    }
    
    .indicator-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .indicator-title {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .indicator-text {
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2c5282;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #718096;
        padding: 0.75rem 1.5rem;
        border-radius: 6px 6px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #edf2f7;
        color: #2d3748;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: #718096;
    }
    
    .download-section {
        background-color: #f7fafc;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'case_result' not in st.session_state:
        st.session_state.case_result = None
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'current_case_name' not in st.session_state:
        st.session_state.current_case_name = None

def generate_chart_image(fig, filename):
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format='png', width=800, height=400)
    img_bytes.seek(0)
    return img_bytes

def generate_pdf_report(result, case_name):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    story = []
    
    story.append(Paragraph("Legal Bias Detection Report", title_style))
    story.append(Spacer(1, 12))
    
    info_data = [
        ['Case File:', case_name],
        ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Verdict:', result.get('summary', {}).get('verdict', 'Unknown').title()],
        ['IPC Sections:', ', '.join(result.get('summary', {}).get('ipc_sections', [])) or 'Not found']
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2d3748')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Bias Analysis Summary", heading_style))
    
    bias_results = result.get('bias_detection', {})
    overall_bias = bias_results.get('overall', {})
    overall_score = overall_bias.get('bias_percentage', 0)
    
    bias_summary_data = [
        ['Metric', 'Value'],
        ['Overall Bias Score', f"{overall_score:.2f}%"],
        ['Gender Bias', f"{bias_results.get('gender_bias', {}).get('bias_percentage', 0):.2f}%"],
        ['Caste Bias', f"{bias_results.get('caste_bias', {}).get('bias_percentage', 0):.2f}%"],
        ['Religious Bias', f"{bias_results.get('religious_bias', {}).get('bias_percentage', 0):.2f}%"],
        ['Socioeconomic Bias', f"{bias_results.get('socioeconomic_bias', {}).get('bias_percentage', 0):.2f}%"]
    ]
    
    bias_table = Table(bias_summary_data, colWidths=[3*inch, 3*inch])
    bias_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    
    story.append(bias_table)
    story.append(Spacer(1, 20))
    
    if overall_score > 0:
        story.append(Paragraph("Bias Indicators Detected", heading_style))
        
        for bias_type, bias_data in bias_results.items():
            if bias_type != 'overall' and isinstance(bias_data, dict):
                indicators = bias_data.get('indicators', [])
                if indicators:
                    bias_name = bias_type.replace('_', ' ').title()
                    story.append(Paragraph(f"<b>{bias_name}:</b>", body_style))
                    
                    for indicator in indicators:
                        indicator_text = indicator.replace('_', ' ').title()
                        story.append(Paragraph(f"• {indicator_text}", body_style))
                    
                    story.append(Spacer(1, 12))
    
    story.append(PageBreak())
    story.append(Paragraph("Punishment Analysis", heading_style))
    
    punishment_comp = result.get('punishment_comparison', {})
    severity = result.get('summary', {}).get('punishment_severity', 'Unknown')
    
    punishment_text = f"Punishment Severity: {severity.title()}<br/>"
    punishment_text += f"Matches IPC Standards: {'Yes' if result.get('summary', {}).get('punishment_match', False) else 'No'}"
    
    story.append(Paragraph(punishment_text, body_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Demographic Information", heading_style))
    demographics = result.get('analysis', {}).get('demographics', {})
    
    demo_data = [
        ['Category', 'Accused', 'Victim'],
        ['Gender', demographics.get('accused_gender', 'Not specified') or 'Not specified', 
         demographics.get('victim_gender', 'Not specified') or 'Not specified'],
        ['Age', str(demographics.get('accused_age', 'Not specified')) if demographics.get('accused_age') else 'Not specified',
         str(demographics.get('victim_age', 'Not specified')) if demographics.get('victim_age') else 'Not specified'],
        ['Caste', demographics.get('accused_caste', 'Not specified') or 'Not specified',
         demographics.get('victim_caste', 'Not specified') or 'Not specified']
    ]
    
    demo_table = Table(demo_data, colWidths=[2*inch, 2*inch, 2*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    
    story.append(demo_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def display_case_analysis(result, case_name):
    st.markdown(f'<div class="section-header">Analysis Results: {case_name}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("*Case Information*")
        st.write(f"*File:* {case_name}")
        st.write(f"*Verdict:* {result.get('summary', {}).get('verdict', 'Unknown').title()}")
        
        ipc_sections = result.get('summary', {}).get('ipc_sections', [])
        if ipc_sections:
            st.write(f"*IPC Sections:* {', '.join(ipc_sections)}")
        else:
            st.write("*IPC Sections:* Not found")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("*Punishment Analysis*")
        punishment_match = result.get('summary', {}).get('punishment_match', False)
        severity = result.get('summary', {}).get('punishment_severity', '')
        
        if severity:
            severity_label = {'lenient': 'Lenient', 'harsh': 'Harsh', 'appropriate': 'Appropriate'}.get(severity.lower(), severity.title())
            st.write(f"*Severity:* {severity_label}")
        
        if punishment_match:
            st.success("Punishment matches IPC standards")
        else:
            st.warning("Punishment discrepancy detected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    bias_results = result.get('bias_detection', {})
    overall_bias = bias_results.get('overall', {})
    overall_score = overall_bias.get('bias_percentage', 0)
    
    st.markdown('<div class="section-header">Bias Detection Results</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if overall_score >= 20:
            st.markdown('<div class="bias-card-high">', unsafe_allow_html=True)
            st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="High Risk", delta_color="inverse")
            st.markdown('</div>', unsafe_allow_html=True)
        elif overall_score >= 10:
            st.markdown('<div class="bias-card-moderate">', unsafe_allow_html=True)
            st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="Moderate Risk", delta_color="off")
            st.markdown('</div>', unsafe_allow_html=True)
        elif overall_score > 0:
            st.markdown('<div class="bias-card-low">', unsafe_allow_html=True)
            st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="Low Risk", delta_color="normal")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("No Bias Detected")
        
        bias_types_detected = overall_bias.get('bias_types_detected', [])
        if bias_types_detected:
            st.markdown("*Detected Bias Types:*")
            for bt in bias_types_detected:
                st.write(f"• {bt.replace('_', ' ').title()}")
    
    with col2:
        if overall_score > 0:
            bias_data = {
                'Bias Type': ['Gender', 'Caste', 'Religious', 'Socioeconomic'],
                'Score': [
                    bias_results.get('gender_bias', {}).get('bias_percentage', 0),
                    bias_results.get('caste_bias', {}).get('bias_percentage', 0),
                    bias_results.get('religious_bias', {}).get('bias_percentage', 0),
                    bias_results.get('socioeconomic_bias', {}).get('bias_percentage', 0)
                ]
            }
            df_bias = pd.DataFrame(bias_data)
            fig = px.bar(df_bias, x='Bias Type', y='Score', 
                        color='Score', color_continuous_scale='Reds',
                        title='Bias Scores by Type')
            fig.update_layout(height=300, showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
    
    if overall_score > 0:
        st.markdown('<div class="section-header">Detailed Bias Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        gender_bias = bias_results.get('gender_bias', {})
        caste_bias = bias_results.get('caste_bias', {})
        religious_bias = bias_results.get('religious_bias', {})
        socioeconomic_bias = bias_results.get('socioeconomic_bias', {})
        
        with col1:
            gender_score = gender_bias.get('bias_percentage', 0)
            st.metric("Gender Bias", f"{gender_score:.2f}%")
            if gender_bias.get('detected'):
                st.caption("⚠ Detected")
        
        with col2:
            caste_score = caste_bias.get('bias_percentage', 0)
            st.metric("Caste Bias", f"{caste_score:.2f}%")
            if caste_bias.get('detected'):
                st.caption("⚠ Detected")
        
        with col3:
            religious_score = religious_bias.get('bias_percentage', 0)
            st.metric("Religious Bias", f"{religious_score:.2f}%")
            if religious_bias.get('detected'):
                st.caption("⚠ Detected")
        
        with col4:
            socioeconomic_score = socioeconomic_bias.get('bias_percentage', 0)
            st.metric("Socioeconomic Bias", f"{socioeconomic_score:.2f}%")
            if socioeconomic_bias.get('detected'):
                st.caption("⚠ Detected")
        
        st.markdown('<div class="section-header">Bias Indicators & Analysis</div>', unsafe_allow_html=True)
        
        indicator_explanations = {
            'lenient_punishment_male_accused_female_victim': 
                'Male accused received lenient punishment compared to IPC standards when victim was female. This may indicate gender bias favoring male offenders.',
            'harsh_punishment_female_accused_male_victim': 
                'Female accused received harsher punishment than expected when victim was male. This may indicate gender bias against female offenders.',
            'lenient_punishment_gender_context': 
                'Punishment was more lenient than expected in a case with gender context, suggesting potential gender-based bias.',
            'harsh_punishment_gender_context': 
                'Punishment was harsher than expected in a case with gender context, suggesting potential gender-based bias.',
            'gender_based_reasoning': 
                'The judgment explicitly mentions gender as a factor in the decision, which may indicate bias.',
            'negative_female_language': 
                'Language used in the judgment contains negative stereotypes about women.',
            'negative_male_language': 
                'Language used contains negative stereotypes about men.',
            'stereotypical_language': 
                'The judgment uses gender stereotypes.',
            'legal_gender_bias': 
                'Legal reasoning explicitly considers gender in a way that may indicate bias.',
            'caste_based_reasoning': 
                'The judgment mentions caste as a factor in the decision, which may indicate bias.',
            'lenient_punishment_caste_difference': 
                'Punishment was lenient when accused and victim were from different castes.',
            'harsh_punishment_caste_difference': 
                'Punishment was harsh when accused and victim were from different castes.',
            'caste_mentioned_with_punishment_discrepancy': 
                'Caste was mentioned in the case and there was a significant punishment discrepancy.',
            'religion_based_reasoning': 
                'The judgment mentions religion as a factor in the decision, which may indicate bias.',
            'socioeconomic_based_reasoning': 
                'The judgment considers economic status in a way that may indicate bias.',
        }
        
        bias_categories = {}
        for bias_type, bias_data in bias_results.items():
            if bias_type != 'overall' and isinstance(bias_data, dict):
                indicators = bias_data.get('indicators', [])
                if indicators:
                    bias_categories[bias_type] = {
                        'name': bias_type.replace('_', ' ').title(),
                        'indicators': indicators,
                        'score': bias_data.get('bias_percentage', 0)
                    }
        
        if bias_categories:
            for bias_type, category_data in bias_categories.items():
                bias_name = category_data['name']
                bias_score = category_data['score']
                indicators = category_data['indicators']
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); 
                            padding: 1.5rem; border-radius: 8px; margin: 1rem 0; 
                            border-left: 4px solid #2b6cb0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                        <h3 style='margin: 0; color: #2d3748; font-size: 1.25rem; font-weight: 600;'>{bias_name}</h3>
                        <span style='background-color: #2b6cb0; color: white; padding: 0.25rem 0.75rem; 
                                     border-radius: 12px; font-size: 0.875rem; font-weight: 600;'>{bias_score:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                for idx, indicator in enumerate(indicators, 1):
                    explanation = indicator_explanations.get(indicator, 
                        f'This indicator suggests potential {bias_name.lower()} in the judgment.')
                    
                    st.markdown(f"""
                    <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; 
                                padding: 1.25rem; margin: 0.75rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                        <div style='display: flex; align-items: start;'>
                            <div style='background-color: #edf2f7; color: #2b6cb0; width: 28px; height: 28px; 
                                       border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                       font-weight: 600; font-size: 0.875rem; margin-right: 1rem; flex-shrink: 0;'>{idx}</div>
                            <div style='flex: 1;'>
                                <div style='font-weight: 600; color: #2d3748; margin-bottom: 0.5rem; font-size: 1rem;'>
                                    {indicator.replace('_', ' ').title()}
                                </div>
                                <div style='color: #4a5568; font-size: 0.95rem; line-height: 1.6;'>
                                    {explanation}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with st.expander("View Complete Analysis Data"):
            st.json(result)
    else:
        st.info("The analysis did not find evidence of bias in this case.")
    
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("*Export Analysis Report*")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pdf_report = generate_pdf_report(result, case_name)
        st.download_button(
            label="Download PDF Report",
            data=pdf_report,
            file_name=f"bias_analysis_{case_name.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    
    with col2:
        json_data = json.dumps(result, indent=2)
        st.download_button(
            label="Download JSON Data",
            data=json_data,
            file_name=f"bias_analysis_{case_name.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col3:
        summary_text = f"""Legal Bias Analysis Report
Case: {case_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CASE INFORMATION
Verdict: {result.get('summary', {}).get('verdict', 'Unknown').title()}
IPC Sections: {', '.join(result.get('summary', {}).get('ipc_sections', [])) or 'Not found'}

BIAS ANALYSIS
Overall Bias Score: {overall_score:.2f}%
Gender Bias: {bias_results.get('gender_bias', {}).get('bias_percentage', 0):.2f}%
Caste Bias: {bias_results.get('caste_bias', {}).get('bias_percentage', 0):.2f}%
Religious Bias: {bias_results.get('religious_bias', {}).get('bias_percentage', 0):.2f}%
Socioeconomic Bias: {bias_results.get('socioeconomic_bias', {}).get('bias_percentage', 0):.2f}%

PUNISHMENT ANALYSIS
Severity: {result.get('summary', {}).get('punishment_severity', 'Unknown').title()}
Matches IPC Standards: {'Yes' if result.get('summary', {}).get('punishment_match', False) else 'No'}
"""
        st.download_button(
            label="Download Text Summary",
            data=summary_text,
            file_name=f"bias_summary_{case_name.replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

initialize_session_state()

st.markdown('<h1 class="main-header">Legal Bias Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced analysis of court proceedings for potential judicial bias</p>', unsafe_allow_html=True)

if st.session_state.pipeline is None:
    with st.spinner("Initializing system components..."):
        try:
            st.session_state.pipeline = LegalBiasPipeline(
                court_proceedings_folder='court_proceedings',
                ipc_code_folder='ipc_penal_code'
            )
            st.session_state.pipeline.load_ipc_code()
            st.success("System initialized successfully")
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()

tab1, tab2, tab3 = st.tabs(["Case Analysis", "Batch Processing", "Comparison & Insights"])

with tab1:
    st.markdown('<div class="section-header">Individual Case Analysis</div>', unsafe_allow_html=True)
    
    analysis_mode = st.radio(
        "Select input method:",
        ["Upload PDF File", "Select from Database"],
        horizontal=True
    )
    
    uploaded_file = None
    selected_file = None
    
    if analysis_mode == "Upload PDF File":
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a court proceeding PDF file for analysis"
        )
    else:
        case_files = list(Path('court_proceedings').glob('*.pdf'))
        
        if not case_files:
            st.warning("No PDF files found in the court_proceedings folder")
        else:
            case_file_names = [f.name for f in case_files]
            selected_file = st.selectbox(
                "Select a case file:",
                case_file_names,
                help="Choose a case file from the database"
            )
    
    if st.button("Analyze Case", type="primary"):
        case_to_analyze = None
        case_name = None
        
        if analysis_mode == "Upload PDF File" and uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                case_to_analyze = tmp_file.name
            case_name = uploaded_file.name
        elif analysis_mode == "Select from Database" and selected_file is not None:
            case_to_analyze = str(Path('court_proceedings') / selected_file)
            case_name = selected_file
        
        if case_to_analyze:
            try:
                with st.spinner("Analyzing case... This may take a few moments."):
                    pdf_reader = PDFReader()
                    case_text = pdf_reader.extract_text(case_to_analyze)
                    
                    if len(case_text.strip()) < 100:
                        st.warning("Could not extract sufficient text from PDF. The file might be scanned or corrupted.")
                    else:
                        result = st.session_state.pipeline.process_case(case_name, case_text)
                        st.session_state.case_result = result
                        st.session_state.current_case_name = case_name
                        st.success("Analysis completed successfully")
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error analyzing case: {str(e)}")
                st.exception(e)
            finally:
                if analysis_mode == "Upload PDF File" and os.path.exists(case_to_analyze):
                    os.unlink(case_to_analyze)
    
    if st.session_state.case_result is not None:
        display_case_analysis(st.session_state.case_result, st.session_state.current_case_name)

with tab2:
    st.markdown('<div class="section-header">Batch Processing</div>', unsafe_allow_html=True)
    st.write("Process all cases in the court proceedings database and generate comprehensive statistics.")
    
    if st.button("Start Batch Analysis", type="primary"):
        with st.spinner("Processing all cases... This may take several minutes."):
            try:
                results_df = st.session_state.pipeline.run(
                    output_file='bias_detection_results.csv',
                    limit=None
                )
                st.session_state.batch_results = results_df
                st.success(f"Analysis complete! Processed {len(results_df)} cases.")
                st.rerun()
            
            except Exception as e:
                st.error(f"Error during batch analysis: {str(e)}")
                st.exception(e)
    
    if st.session_state.batch_results is not None:
        results_df = st.session_state.batch_results
        
        st.markdown('<div class="section-header">Batch Analysis Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cases", len(results_df))
        
        with col2:
            cases_with_bias = len(results_df[results_df['overall_bias_score'] >= 5.0])
            st.metric("Cases with Bias (≥5%)", cases_with_bias)
        
        with col3:
            cases_significant = len(results_df[results_df['overall_bias_score'] >= 20.0])
            st.metric("Significant Bias (≥20%)", cases_significant)
        
        with col4:
            avg_bias = results_df['overall_bias_score'].mean()
            st.metric("Average Bias Score", f"{avg_bias:.2f}%")
        
        st.markdown("---")
        
        st.markdown('<div class="section-header">Statistical Distribution</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Bias Type Distribution*")
            gender_count = len(results_df[results_df['gender_bias_score'] >= 5.0])
            caste_count = len(results_df[results_df['caste_bias_score'] >= 5.0])
            religious_count = len(results_df[results_df['religious_bias_score'] >= 5.0])
            socioeconomic_count = len(results_df[results_df['socioeconomic_bias_score'] >= 5.0])
            
            bias_data = pd.DataFrame({
                'Bias Type': ['Gender', 'Caste', 'Religious', 'Socioeconomic'],
                'Cases Detected': [gender_count, caste_count, religious_count, socioeconomic_count]
            })
            fig = px.bar(bias_data, x='Bias Type', y='Cases Detected', 
                        color='Cases Detected', color_continuous_scale='Blues',
                        title='Number of Cases with Bias by Type')
            fig.update_layout(height=400, showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("*Bias Score Distribution*")
            if len(results_df[results_df['overall_bias_score'] > 0]) > 0:
                fig = px.histogram(results_df[results_df['overall_bias_score'] > 0], 
                                 x='overall_bias_score',
                                 nbins=20,
                                 title='Distribution of Bias Scores',
                                 labels={'overall_bias_score': 'Bias Score (%)', 'count': 'Number of Cases'})
                fig.update_layout(height=400, plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No bias detected in any cases")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Average Bias Score by Verdict*")
            if 'verdict' in results_df.columns:
                verdict_bias = results_df.groupby('verdict')['overall_bias_score'].mean().reset_index()
                fig = px.bar(verdict_bias, x='verdict', y='overall_bias_score',
                            color='overall_bias_score', color_continuous_scale='Reds',
                            title='Average Bias Score by Verdict Type',
                            labels={'overall_bias_score': 'Average Bias Score (%)', 'verdict': 'Verdict'})
                fig.update_layout(height=400, showlegend=False, plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("*Cases with Highest Bias Scores*")
            top_cases = results_df.nlargest(10, 'overall_bias_score')[['case_filename', 'overall_bias_score']]
            fig = px.bar(top_cases, x='overall_bias_score', y='case_filename',
                        orientation='h', color='overall_bias_score',
                        color_continuous_scale='Reds',
                        title='Top 10 Cases by Bias Score',
                        labels={'overall_bias_score': 'Bias Score (%)', 'case_filename': 'Case File'})
            fig.update_layout(height=400, showlegend=False, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="section-header">Detailed Results Table</div>', unsafe_allow_html=True)
        
        display_df = results_df[['case_filename', 'verdict', 'overall_bias_score', 
                                'gender_bias_score', 'caste_bias_score', 
                                'religious_bias_score', 'socioeconomic_bias_score',
                                'biases_detected']].copy()
        
        display_df.columns = ['Case File', 'Verdict', 'Overall Bias %', 
                             'Gender Bias %', 'Caste Bias %', 
                             'Religious Bias %', 'Socioeconomic Bias %',
                             'Detected Biases']
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.markdown("*Export Batch Results*")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f'batch_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col2:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                label="Download Excel Report",
                data=excel_buffer,
                file_name=f'batch_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        with col3:
            summary_stats = f"""Batch Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
Total Cases Analyzed: {len(results_df)}
Cases with Bias (≥5%): {cases_with_bias}
Cases with Significant Bias (≥20%): {cases_significant}
Average Bias Score: {avg_bias:.2f}%

BIAS TYPE BREAKDOWN
Gender Bias Cases: {gender_count}
Caste Bias Cases: {caste_count}
Religious Bias Cases: {religious_count}
Socioeconomic Bias Cases: {socioeconomic_count}

VERDICT ANALYSIS
{verdict_bias.to_string(index=False) if 'verdict' in results_df.columns else 'N/A'}
"""
            st.download_button(
                label="Download Summary Report",
                data=summary_stats,
                file_name=f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain'
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-header">Comparative Analysis & Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.batch_results is not None and st.session_state.case_result is not None:
        results_df = st.session_state.batch_results
        case_result = st.session_state.case_result
        case_name = st.session_state.current_case_name
        
        st.markdown("*Compare Individual Case Against Database*")
        
        case_bias_score = case_result.get('summary', {}).get('overall_bias_score', 0)
        avg_bias_score = results_df['overall_bias_score'].mean()
        median_bias_score = results_df['overall_bias_score'].median()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Case Bias Score", f"{case_bias_score:.2f}%")
        
        with col2:
            diff_from_avg = case_bias_score - avg_bias_score
            st.metric("Database Average", f"{avg_bias_score:.2f}%", 
                     delta=f"{diff_from_avg:+.2f}% from case",
                     delta_color="inverse" if diff_from_avg > 0 else "normal")
        
        with col3:
            percentile = (results_df['overall_bias_score'] < case_bias_score).sum() / len(results_df) * 100
            st.metric("Percentile Ranking", f"{percentile:.1f}th")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Bias Score Comparison*")
            
            comparison_data = pd.DataFrame({
                'Metric': ['Overall', 'Gender', 'Caste', 'Religious', 'Socioeconomic'],
                'Current Case': [
                    case_bias_score,
                    case_result.get('bias_detection', {}).get('gender_bias', {}).get('bias_percentage', 0),
                    case_result.get('bias_detection', {}).get('caste_bias', {}).get('bias_percentage', 0),
                    case_result.get('bias_detection', {}).get('religious_bias', {}).get('bias_percentage', 0),
                    case_result.get('bias_detection', {}).get('socioeconomic_bias', {}).get('bias_percentage', 0)
                ],
                'Database Average': [
                    avg_bias_score,
                    results_df['gender_bias_score'].mean(),
                    results_df['caste_bias_score'].mean(),
                    results_df['religious_bias_score'].mean(),
                    results_df['socioeconomic_bias_score'].mean()
                ]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Current Case', x=comparison_data['Metric'], 
                                y=comparison_data['Current Case'], marker_color='#e53e3e'))
            fig.add_trace(go.Bar(name='Database Average', x=comparison_data['Metric'], 
                                y=comparison_data['Database Average'], marker_color='#4299e1'))
            
            fig.update_layout(
                barmode='group',
                title='Case vs Database Comparison',
                xaxis_title='Bias Type',
                yaxis_title='Bias Score (%)',
                height=400,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("*Position in Distribution*")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=results_df['overall_bias_score'], 
                                      name='All Cases',
                                      marker_color='#cbd5e0',
                                      opacity=0.7))
            fig.add_vline(x=case_bias_score, line_dash="dash", line_color="#e53e3e",
                         annotation_text=f"Current Case: {case_bias_score:.2f}%",
                         annotation_position="top right")
            
            fig.update_layout(
                title='Case Position in Overall Distribution',
                xaxis_title='Bias Score (%)',
                yaxis_title='Number of Cases',
                height=400,
                plot_bgcolor='white',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        
        insights = []
        
        if case_bias_score > avg_bias_score * 1.5:
            insights.append("The analyzed case shows significantly higher bias indicators compared to the database average.")
        elif case_bias_score < avg_bias_score * 0.5:
            insights.append("The analyzed case shows lower bias indicators compared to most cases in the database.")
        else:
            insights.append("The analyzed case has bias indicators consistent with the database average.")
        
        case_verdict = case_result.get('summary', {}).get('verdict', '').lower()
        if case_verdict in results_df['verdict'].values:
            verdict_avg = results_df[results_df['verdict'] == case_verdict]['overall_bias_score'].mean()
            if case_bias_score > verdict_avg * 1.2:
                insights.append(f"This case shows higher bias than average for '{case_verdict}' verdicts.")
            elif case_bias_score < verdict_avg * 0.8:
                insights.append(f"This case shows lower bias than average for '{case_verdict}' verdicts.")
        
        case_biases = case_result.get('bias_detection', {})
        if case_biases.get('gender_bias', {}).get('detected'):
            gender_cases = len(results_df[results_df['gender_bias_score'] >= 5.0])
            gender_pct = (gender_cases / len(results_df)) * 100
            insights.append(f"Gender bias was detected. This occurs in {gender_pct:.1f}% of analyzed cases.")
        
        if case_biases.get('caste_bias', {}).get('detected'):
            caste_cases = len(results_df[results_df['caste_bias_score'] >= 5.0])
            caste_pct = (caste_cases / len(results_df)) * 100
            insights.append(f"Caste bias was detected. This occurs in {caste_pct:.1f}% of analyzed cases.")
        
        for insight in insights:
            st.info(insight)
        
        st.markdown("---")
        
        st.markdown("*Similar Cases in Database*")
        
        bias_tolerance = 5.0
        similar_cases = results_df[
            (results_df['overall_bias_score'] >= case_bias_score - bias_tolerance) &
            (results_df['overall_bias_score'] <= case_bias_score + bias_tolerance)
        ].copy()
        
        if len(similar_cases) > 0:
            st.write(f"Found {len(similar_cases)} cases with similar bias scores (±{bias_tolerance}%)")
            
            display_similar = similar_cases[['case_filename', 'verdict', 'overall_bias_score', 
                                            'biases_detected']].head(10)
            display_similar.columns = ['Case File', 'Verdict', 'Bias Score %', 'Detected Biases']
            st.dataframe(display_similar, use_container_width=True)
        else:
            st.info("No cases found with similar bias scores in the database.")
    
    elif st.session_state.batch_results is not None:
        st.info("Run an individual case analysis in the 'Case Analysis' tab to compare it against the batch results.")
    
    elif st.session_state.case_result is not None:
        st.info("Run a batch analysis in the 'Batch Processing' tab to compare individual cases against the database.")
    
    else:
        st.info("Please run both an individual case analysis and batch processing to see comparative insights.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("*To get started:*")
            st.write("1. Go to 'Case Analysis' tab to analyze a single case")
            st.write("2. Go to 'Batch Processing' tab to analyze all cases")
            st.write("3. Return here to see detailed comparisons")
        
        with col2:
            if st.session_state.batch_results is not None:
                results_df = st.session_state.batch_results
                
                st.markdown("*Database Statistics Available*")
                st.write(f"Total Cases: {len(results_df)}")
                st.write(f"Average Bias: {results_df['overall_bias_score'].mean():.2f}%")
                st.write(f"Cases with Bias: {len(results_df[results_df['overall_bias_score'] >= 5.0])}")
