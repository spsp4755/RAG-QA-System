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
        page_title="RAG 법률 문서 QA 시스템",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* 전체 텍스트 색상 개선 */
    .stMarkdown, .stText, .stTextInput, .stTextArea {
        color: #333333 !important;
    }
    
    /* 헤더 스타일 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* 답변 박스 - 가독성 개선 */
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #333333 !important;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 문서 박스 */
    .document-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        color: #333333 !important;
    }
    
    /* 사이드바 개선 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* 입력 필드 개선 */
    .stTextInput > div > div > input {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    .stTextArea > div > div > textarea {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    /* 확장 가능한 섹션 */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        font-weight: 600;
    }
    
    /* 데이터프레임 개선 */
    .stDataFrame {
        color: #333333 !important;
    }
    
    /* 성능 지표 개선 */
    .stMetric {
        color: #333333 !important;
    }
    
    /* 전체 배경색 개선 */
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
    }
    
    /* 텍스트 가독성 전역 개선 */
    p, div, span, label {
        color: #333333 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ RAG 법률 문서 QA 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI Hub 법률 문서 기반 지능형 질의응답 시스템</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 시스템 설정")
        
        # RAG 시스템 초기화
        if 'rag_system' not in st.session_state:
            with st.spinner("RAG 시스템 초기화 중..."):
                st.session_state.rag_system = RAGSystem(use_llm=False)
        
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
            ["전체", "1006", "1032", "1037", "1039"],
            help="특정 문서 타입으로 검색 범위를 제한합니다"
        )
        
        # 예시 질문들
        st.markdown("### 💡 예시 질문")
        example_questions = [
            "계약서의 기본 조항은 무엇인가요?",
            "임대차 계약에서 임차인의 의무는?",
            "계약 해지 조건에 대해 알려주세요",
            "법적 책임과 의무는 어떻게 되나요?",
            "합의를 해지하려면 어떻게 해야하나요?",
            "주주들의 권리와 의무는?"
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
            "법률 문서에 대해 궁금한 점을 질문해주세요:",
            value=st.session_state.get('query', ''),
            height=100,
            placeholder="예: 계약서의 기본 조항은 무엇인가요?"
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
            filter_dict = {"doc_type": filter_doc_type}
        
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
        
        # 참고 문서
        st.markdown("### 📄 참고 문서")
        
        for i, doc in enumerate(result['retrieved_docs'], 1):
            with st.expander(f"📄 문서 {i} (유사도: {1-doc['distance']:.2%})"):
                st.markdown(f"**📝 문서 내용:**")
                st.markdown(f'<div class="document-box">{doc["text"]}</div>', unsafe_allow_html=True)
                
                st.markdown("**🏷️ 메타데이터:**")
                metadata_df = st.dataframe(
                    pd.DataFrame([doc['metadata']]).T,
                    use_container_width=True,
                    hide_index=False
                )
        
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
        <p>🔧 RAG QA 시스템 | AI Hub 법률 문서 기반 | Made with Streamlit</p>
        <p>📊 총 551,750개 문서 | ⚡ 빠른 검색 | 🤖 지능형 답변</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    import pandas as pd
    main() 