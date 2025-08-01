import streamlit as st
import os
import json
from datetime import datetime
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_system import RAGSystem


def main():
    st.set_page_config(
        page_title="RAG 지식재산권 QA 시스템",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* 전체 배경 및 기본 스타일 */
    .main {
        background-color: #f5f5f5;
    }
    
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem;
    }
    
    /* 헤더 스타일 - 더 명확하게 */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #2c3e50;
        color: white;
    }
    
    /* 입력 필드 스타일 */
    .stTextArea > div > div > textarea {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* 버튼 스타일 - 더 명확하게 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* 답변 박스 - 훨씬 명확하게 */
    .answer-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        color: #2c3e50 !important;
        font-size: 1.1rem;
        line-height: 1.7;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    
    /* 문서 박스 - 개선 */
    .document-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        color: #2c3e50 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* 확장 가능한 섹션 */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    
    /* 메트릭 스타일 */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    /* 데이터프레임 스타일 */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* 전체 텍스트 색상 통일 */
    .stMarkdown, .stText, p, div, span, label, h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    /* 섹션 제목 스타일 */
    h3 {
        color: #34495e !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #e9ecef !important;
    }
    
    /* 성능 지표 섹션 */
    .performance-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    /* 푸터 스타일 */
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧠 RAG 지식재산권 QA 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI Hub 지식재산권법 데이터 기반 지능형 질의응답 시스템</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 시스템 설정")
        
        # RAG 시스템 초기화 (training_db 사용)
        if 'rag_system' not in st.session_state:
            with st.spinner("RAG 시스템 초기화 중..."):
                st.session_state.rag_system = RAGSystem(
                    persist_dir="data/embeddings/training_db",
                    use_llm=True
                )  
        
        # 시스템 정보 표시
        info = st.session_state.rag_system.get_system_info()
        
        st.markdown("### 📊 시스템 정보")
        st.metric("총 문서 수", f"{info['total_documents']:,}")
        st.metric("검색 시스템", info['search_system'])
        st.metric("LLM 시스템", info['llm_system'])
        
        st.markdown("### ⚙️ 검색 설정")
        n_results = st.slider("검색 결과 수", 1, 10, 3)
        
        # 필터 옵션
        st.markdown("### 🔍 필터 옵션")
        filter_doc_type = st.selectbox(
            "문서 타입 필터",
            ["전체", "특허권", "상표권", "저작권", "디자인권"],
            help="특정 지식재산권 분야로 검색 범위를 제한합니다"
        )
        
        # 예시 질문들
        st.markdown("### 💡 예시 질문")
        example_questions = [
            "특허권의 보호기간은 얼마나 되나요?",
            "상표권 침해의 구체적인 행위는 무엇인가요?",
            "저작권의 발생 시점은 언제인가요?",
            "디자인권의 등록 요건은 무엇인가요?",
            "특허 출원 절차는 어떻게 되나요?",
            "상표 등록의 효과는 무엇인가요?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"예시 {i+1}", key=f"example_{i}"):
                st.session_state.query = question
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ❓ 질문하기")
        
        # 질문 입력
        query = st.text_area(
            "지식재산권에 대해 궁금한 점을 질문해주세요:",
            value=st.session_state.get('query', ''),
            height=100,
            placeholder="예: 특허권의 보호기간은 얼마나 되나요?"
        )
        
        # 검색 버튼
        col1_1, col1_2, col1_3 = st.columns([1, 1, 2])
        
        with col1_1:
            search_button = st.button("🔍 검색", type="primary", use_container_width=True)
        
        with col1_2:
            clear_button = st.button("🗑️ 초기화", use_container_width=True)
        
        if clear_button:
            st.session_state.query = ""
            st.session_state.result = None
            st.rerun()
        
        # 필터 설정
        filter_dict = None
        if filter_doc_type != "전체":
            # 지식재산권 분야별 필터링 (메타데이터의 instruction 필드 활용)
            filter_dict = {"instruction": {"$contains": filter_doc_type}}
        
        # 검색 실행
        if search_button and query.strip():
            with st.spinner("검색 중..."):
                result = st.session_state.rag_system.answer_question(
                    query, 
                    n_results=n_results,
                    filter_dict=filter_dict,
                    save_result=True
                )
                st.session_state.result = result
                st.session_state.query = query
    
    with col2:
        st.markdown("### 📈 성능 지표")
        
        if 'result' in st.session_state and st.session_state.result:
            result = st.session_state.result
            
            st.metric("검색 시간", f"{result['search_time']:.2f}초")
            st.metric("생성 시간", f"{result['generation_time']:.2f}초")
            st.metric("총 시간", f"{result['total_time']:.2f}초")
            st.metric("검색된 문서", len(result['retrieved_docs']))
    
    # 결과 표시
    if 'result' in st.session_state and st.session_state.result:
        result = st.session_state.result
        
        st.markdown("---")
        st.markdown("### 🤖 답변")
        
        # 답변 박스 - 개선된 표시
        st.markdown("**🤖 생성된 답변:**")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
        
        # 결과 다운로드
        st.markdown("### 💾 결과 다운로드")
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 JSON 다운로드",
            data=result_json,
            file_name=f"rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🧠 RAG 지식재산권 QA 시스템 | AI Hub 지식재산권법 데이터 기반 | Made with Streamlit</p>
        <p>📊 지식재산권법 데이터 | ⚡ 빠른 검색 | 🤖 지능형 답변</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    import pandas as pd
    main() 