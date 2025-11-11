import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import LegalBiasPipeline
from src.core import PDFReader
import logging

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Legal Bias Detection System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .bias-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .bias-moderate {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
    }
    .bias-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚖️ Legal Bias Detection System</h1>', unsafe_allow_html=True)
st.markdown("### Analyze court proceedings for potential bias in judgments")

if 'pipeline' not in st.session_state:
    with st.spinner("🔄 Initializing system (this may take a minute on first run)..."):
        try:
            st.session_state.pipeline = LegalBiasPipeline(
                court_proceedings_folder='court_proceedings',
                ipc_code_folder='ipc_penal_code'
            )
            st.session_state.pipeline.load_ipc_code()
            st.success("✅ System initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            st.stop()

tab1, tab2, tab3 = st.tabs(["📄 Upload Case", "📁 Individual File Analysis", "📊 Batch Analysis"])

with tab1:
    st.header("📄 Upload and Analyze Single Case")
    st.markdown("Upload a court proceeding PDF to analyze for potential bias")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a court proceeding PDF file"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("🔍 Analyzing case..."):
                pdf_reader = PDFReader()
                case_text = pdf_reader.extract_text(tmp_path)
                
                if len(case_text.strip()) < 100:
                    st.warning("⚠️ Could not extract sufficient text from PDF. The file might be scanned or corrupted.")
                else:
                    result = st.session_state.pipeline.process_case(uploaded_file.name, case_text)
                    
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Case Information")
                        st.write(f"**File:** {result.get('case_filename', 'Unknown')}")
                        st.write(f"**Verdict:** {result.get('summary', {}).get('verdict', 'Unknown').title()}")
                        
                        ipc_sections = result.get('summary', {}).get('ipc_sections', [])
                        if ipc_sections:
                            st.write(f"**IPC Sections:** {', '.join(ipc_sections)}")
                        else:
                            st.write("**IPC Sections:** Not found")
                    
                    with col2:
                        st.subheader("⚖️ Punishment Analysis")
                        punishment_match = result.get('summary', {}).get('punishment_match', False)
                        severity = result.get('summary', {}).get('punishment_severity', '')
                        if severity:
                            severity_color = {'lenient': '🟡', 'harsh': '🔴', 'appropriate': '🟢'}.get(severity.lower(), '⚪')
                            st.write(f"**Severity:** {severity_color} {severity.title()}")
                        if punishment_match:
                            st.success("✅ Punishment matches IPC standards")
                        else:
                            st.warning("⚠️ Punishment discrepancy detected")
                    
                    st.divider()
                    
                    bias_results = result.get('bias_detection', {})
                    overall_bias = bias_results.get('overall', {})
                    overall_score = overall_bias.get('bias_percentage', 0)
                    
                    st.subheader("🎯 Bias Analysis Results")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if overall_score >= 20:
                            st.markdown(f'<div class="bias-high">', unsafe_allow_html=True)
                            st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="High Risk", delta_color="inverse")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif overall_score >= 10:
                            st.markdown(f'<div class="bias-moderate">', unsafe_allow_html=True)
                            st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="Moderate Risk", delta_color="off")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif overall_score > 0:
                            st.markdown(f'<div class="bias-low">', unsafe_allow_html=True)
                            st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="Low Risk", delta_color="normal")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.success("✅ **No Bias Detected**")
                        
                        bias_types_detected = overall_bias.get('bias_types_detected', [])
                        if bias_types_detected:
                            st.write("**Bias Types:**")
                            for bt in bias_types_detected:
                                st.write(f"  • {bt.replace('_', ' ').title()}")
                    
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
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if overall_score > 0:
                        st.subheader("📊 Detailed Bias Breakdown")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        gender_bias = bias_results.get('gender_bias', {})
                        caste_bias = bias_results.get('caste_bias', {})
                        religious_bias = bias_results.get('religious_bias', {})
                        socioeconomic_bias = bias_results.get('socioeconomic_bias', {})
                        
                        with col1:
                            gender_score = gender_bias.get('bias_percentage', 0)
                            st.metric("Gender Bias", f"{gender_score:.2f}%")
                            if gender_bias.get('detected'):
                                st.caption("⚠️ Detected")
                        
                        with col2:
                            caste_score = caste_bias.get('bias_percentage', 0)
                            st.metric("Caste Bias", f"{caste_score:.2f}%")
                            if caste_bias.get('detected'):
                                st.caption("⚠️ Detected")
                        
                        with col3:
                            religious_score = religious_bias.get('bias_percentage', 0)
                            st.metric("Religious Bias", f"{religious_score:.2f}%")
                            if religious_bias.get('detected'):
                                st.caption("⚠️ Detected")
                        
                        with col4:
                            socioeconomic_score = socioeconomic_bias.get('bias_percentage', 0)
                            st.metric("Socioeconomic Bias", f"{socioeconomic_score:.2f}%")
                            if socioeconomic_bias.get('detected'):
                                st.caption("⚠️ Detected")
                        
                        st.subheader("🔍 Why Bias Was Detected")
                        
                        for bias_type, bias_data in bias_results.items():
                            if bias_type != 'overall' and isinstance(bias_data, dict):
                                indicators = bias_data.get('indicators', [])
                                if indicators:
                                    bias_name = bias_type.replace('_', ' ').title()
                                    st.write(f"**{bias_name} Indicators:**")
                                    
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
                                            'Language used in the judgment contains negative stereotypes about women (e.g., questioning character, reputation).',
                                        'negative_male_language': 
                                            'Language used contains negative stereotypes about men (e.g., aggressive, dangerous).',
                                        'stereotypical_language': 
                                            'The judgment uses gender stereotypes (e.g., women are emotional, men are dominant).',
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
                                    
                                    for indicator in indicators:
                                        explanation = indicator_explanations.get(indicator, 
                                            f'This indicator suggests potential {bias_name.lower()} in the judgment.')
                                        st.markdown(f"  • **{indicator.replace('_', ' ').title()}:** {explanation}")
                                    
                                    st.write("")
                        
                        with st.expander("📝 View Full Analysis Details"):
                            st.json(result)
                    else:
                        st.info("ℹ️ The analysis did not find evidence of gender, caste, religious, or socioeconomic bias in this case.")
        
        except Exception as e:
            st.error(f"❌ Error analyzing case: {str(e)}")
            st.exception(e)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

with tab2:
    st.header("📁 Individual File Analysis")
    st.markdown("Select and analyze a specific case from the court proceedings folder")
    
    case_files = list(Path('court_proceedings').glob('*.pdf'))
    
    if not case_files:
        st.warning("⚠️ No PDF files found in the court_proceedings folder")
    else:
        case_file_names = [f.name for f in case_files]
        selected_file = st.selectbox(
            "Select a case file to analyze:",
            case_file_names,
            help="Choose a case file from the dropdown"
        )
        
        if st.button("🔍 Analyze Selected Case", type="primary"):
            selected_path = Path('court_proceedings') / selected_file
            
            try:
                with st.spinner("🔍 Analyzing case..."):
                    pdf_reader = PDFReader()
                    case_text = pdf_reader.extract_text(str(selected_path))
                    
                    if len(case_text.strip()) < 100:
                        st.warning("⚠️ Could not extract sufficient text from PDF.")
                    else:
                        result = st.session_state.pipeline.process_case(selected_file, case_text)
                        
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📋 Case Information")
                            st.write(f"**File:** {result.get('case_filename', 'Unknown')}")
                            st.write(f"**Verdict:** {result.get('summary', {}).get('verdict', 'Unknown').title()}")
                            
                            ipc_sections = result.get('summary', {}).get('ipc_sections', [])
                            if ipc_sections:
                                st.write(f"**IPC Sections:** {', '.join(ipc_sections)}")
                            else:
                                st.write("**IPC Sections:** Not found")
                        
                        with col2:
                            st.subheader("⚖️ Punishment Analysis")
                            punishment_match = result.get('summary', {}).get('punishment_match', False)
                            severity = result.get('summary', {}).get('punishment_severity', '')
                            if severity:
                                severity_color = {'lenient': '🟡', 'harsh': '🔴', 'appropriate': '🟢'}.get(severity.lower(), '⚪')
                                st.write(f"**Severity:** {severity_color} {severity.title()}")
                            if punishment_match:
                                st.success("✅ Punishment matches IPC standards")
                            else:
                                st.warning("⚠️ Punishment discrepancy detected")
                        
                        st.divider()
                        
                        bias_results = result.get('bias_detection', {})
                        overall_bias = bias_results.get('overall', {})
                        overall_score = overall_bias.get('bias_percentage', 0)
                        
                        st.subheader("🎯 Bias Analysis Results")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if overall_score >= 20:
                                st.markdown(f'<div class="bias-high">', unsafe_allow_html=True)
                                st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="High Risk", delta_color="inverse")
                                st.markdown('</div>', unsafe_allow_html=True)
                            elif overall_score >= 10:
                                st.markdown(f'<div class="bias-moderate">', unsafe_allow_html=True)
                                st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="Moderate Risk", delta_color="off")
                                st.markdown('</div>', unsafe_allow_html=True)
                            elif overall_score > 0:
                                st.markdown(f'<div class="bias-low">', unsafe_allow_html=True)
                                st.metric("Overall Bias Score", f"{overall_score:.2f}%", delta="Low Risk", delta_color="normal")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.success("✅ **No Bias Detected**")
                            
                            bias_types_detected = overall_bias.get('bias_types_detected', [])
                            if bias_types_detected:
                                st.write("**Bias Types:**")
                                for bt in bias_types_detected:
                                    st.write(f"  • {bt.replace('_', ' ').title()}")
                        
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
                                fig.update_layout(height=300, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        if overall_score > 0:
                            st.subheader("📊 Detailed Bias Breakdown")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            gender_bias = bias_results.get('gender_bias', {})
                            caste_bias = bias_results.get('caste_bias', {})
                            religious_bias = bias_results.get('religious_bias', {})
                            socioeconomic_bias = bias_results.get('socioeconomic_bias', {})
                            
                            with col1:
                                gender_score = gender_bias.get('bias_percentage', 0)
                                st.metric("Gender Bias", f"{gender_score:.2f}%")
                                if gender_bias.get('detected'):
                                    st.caption("⚠️ Detected")
                            
                            with col2:
                                caste_score = caste_bias.get('bias_percentage', 0)
                                st.metric("Caste Bias", f"{caste_score:.2f}%")
                                if caste_bias.get('detected'):
                                    st.caption("⚠️ Detected")
                            
                            with col3:
                                religious_score = religious_bias.get('bias_percentage', 0)
                                st.metric("Religious Bias", f"{religious_score:.2f}%")
                                if religious_bias.get('detected'):
                                    st.caption("⚠️ Detected")
                            
                            with col4:
                                socioeconomic_score = socioeconomic_bias.get('bias_percentage', 0)
                                st.metric("Socioeconomic Bias", f"{socioeconomic_score:.2f}%")
                                if socioeconomic_bias.get('detected'):
                                    st.caption("⚠️ Detected")
                            
                            st.subheader("🔍 Why Bias Was Detected")
                            
                            for bias_type, bias_data in bias_results.items():
                                if bias_type != 'overall' and isinstance(bias_data, dict):
                                    indicators = bias_data.get('indicators', [])
                                    if indicators:
                                        bias_name = bias_type.replace('_', ' ').title()
                                        st.write(f"**{bias_name} Indicators:**")
                                        
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
                                                'Language used in the judgment contains negative stereotypes about women (e.g., questioning character, reputation).',
                                            'negative_male_language': 
                                                'Language used contains negative stereotypes about men (e.g., aggressive, dangerous).',
                                            'stereotypical_language': 
                                                'The judgment uses gender stereotypes (e.g., women are emotional, men are dominant).',
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
                                        
                                        for indicator in indicators:
                                            explanation = indicator_explanations.get(indicator, 
                                                f'This indicator suggests potential {bias_name.lower()} in the judgment.')
                                            st.markdown(f"  • **{indicator.replace('_', ' ').title()}:** {explanation}")
                                        
                                        st.write("")
                            
                            with st.expander("📝 View Full Analysis Details"):
                                st.json(result)
                        else:
                            st.info("ℹ️ The analysis did not find evidence of bias in this case.")

            except Exception as e:
                st.error(f"❌ Error analyzing case: {str(e)}")
                st.exception(e)

with tab3:
    st.header("📊 Batch Analysis")
    st.markdown("Analyze all cases in the court_proceedings folder")
    
    if st.button("🚀 Run Batch Analysis", type="primary"):
        with st.spinner("Processing all cases... This may take several minutes."):
            try:
                results_df = st.session_state.pipeline.run(output_file='bias_detection_results.csv')
                
                st.success(f"✅ Analysis complete! Processed {len(results_df)} cases.")
                
                st.subheader("📊 Summary Statistics")
                
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
                
                st.subheader("📈 Bias Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Bias Type Distribution**")
                    gender_count = len(results_df[results_df['gender_bias_score'] >= 5.0])
                    caste_count = len(results_df[results_df['caste_bias_score'] >= 5.0])
                    religious_count = len(results_df[results_df['religious_bias_score'] >= 5.0])
                    socioeconomic_count = len(results_df[results_df['socioeconomic_bias_score'] >= 5.0])
                    
                    bias_data = pd.DataFrame({
                        'Bias Type': ['Gender', 'Caste', 'Religious', 'Socioeconomic'],
                        'Cases Detected': [gender_count, caste_count, religious_count, socioeconomic_count]
                    })
                    fig = px.bar(bias_data, x='Bias Type', y='Cases Detected', 
                                color='Cases Detected', color_continuous_scale='Oranges',
                                title='Number of Cases with Bias by Type')
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Bias Score Distribution**")
                    if len(results_df[results_df['overall_bias_score'] > 0]) > 0:
                        fig = px.histogram(results_df[results_df['overall_bias_score'] > 0], 
                                         x='overall_bias_score',
                                         nbins=20,
                                         title='Distribution of Bias Scores',
                                         labels={'overall_bias_score': 'Bias Score (%)', 'count': 'Number of Cases'})
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No bias detected in any cases")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Bias Score by Verdict**")
                    if 'verdict' in results_df.columns:
                        verdict_bias = results_df.groupby('verdict')['overall_bias_score'].mean().reset_index()
                        fig = px.bar(verdict_bias, x='verdict', y='overall_bias_score',
                                    color='overall_bias_score', color_continuous_scale='Reds',
                                    title='Average Bias Score by Verdict Type',
                                    labels={'overall_bias_score': 'Average Bias Score (%)', 'verdict': 'Verdict'})
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Top 10 Cases with Highest Bias**")
                    top_cases = results_df.nlargest(10, 'overall_bias_score')[['case_filename', 'overall_bias_score']]
                    fig = px.bar(top_cases, x='overall_bias_score', y='case_filename',
                                orientation='h', color='overall_bias_score',
                                color_continuous_scale='Reds',
                                title='Cases with Highest Bias Scores',
                                labels={'overall_bias_score': 'Bias Score (%)', 'case_filename': 'Case File'})
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 Detailed Results Table")
                st.dataframe(
                    results_df[['case_filename', 'verdict', 'overall_bias_score', 
                               'gender_bias_score', 'caste_bias_score', 
                               'biases_detected']].head(20),
                    use_container_width=True
                )
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name='bias_detection_results.csv',
                    mime='text/csv'
                )
            
            except Exception as e:
                st.error(f"❌ Error during batch analysis: {str(e)}")
                st.exception(e)
