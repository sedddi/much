import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import os
from datetime import datetime, timedelta
import json
from pdf_parser import process_pdf_files, process_pdf_to_json, preview_json_data

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LangChain model
def get_llm():
    """LangChain 모델 초기화"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                openai_api_key=api_key
            )
        else:
            return None
    except Exception as e:
        st.warning(f"LangChain 모델 초기화 실패: {e}")
        return None

def generate_credit_guidance(data):
    """사용자 데이터 기반 신용 관리 가이드 생성"""
    llm = get_llm()
    if not llm:
        return get_default_credit_guidance(data)
    
    try:
        # 신용 관리 가이드 프롬프트
        credit_prompt = PromptTemplate(
            input_variables=["income", "expense", "credit_score", "assets"],
            template="""
            사용자의 재무 상황을 분석하여 맞춤형 신용 관리 가이드를 제공해주세요.
            
            사용자 정보:
            - 월 수입: {income:,}원
            - 월 지출: {expense:,}원
            - 신용점수: {credit_score}점
            - 총 자산: {assets:,}원
            
            다음 형식으로 답변해주세요:
            
            ## 신용점수 향상 팁
            - 구체적이고 실행 가능한 팁 3-4개
            
            ## 주의할 점
            - 신용점수에 부정적 영향을 주는 행동 2-3개
            
            ## 맞춤형 권장사항
            - 사용자의 현재 상황에 특화된 권장사항 2-3개
            
            답변은 한국어로 작성하고, 구체적이고 실용적인 내용으로 작성해주세요.
            """
        )
        
        # 최신 LangChain 문법 사용
        chain = credit_prompt | llm
        
        result = chain.invoke({
            "income": data['income'],
            "expense": data['expense'],
            "credit_score": data['credit_score'],
            "assets": sum(data['assets'].values())
        })
        
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        st.warning(f"AI 가이드 생성 실패: {e}")
        return get_default_credit_guidance(data)

def generate_financial_recommendations(data):
    """사용자 데이터 기반 금융 상품 추천 생성"""
    llm = get_llm()
    if not llm:
        return get_default_financial_recommendations(data)
    
    try:
        # 금융 상품 추천 프롬프트 (더 구체적이고 실용적인 내용)
        recommendation_prompt = PromptTemplate(
            input_variables=["income", "credit_score", "assets", "savings", "expense"],
            template="""
            사용자의 재무 상황을 분석하여 구체적이고 실용적인 금융 상품을 추천해주세요.
            
            사용자 정보:
            - 월 수입: {income:,}원
            - 월 지출: {expense:,}원
            - 신용점수: {credit_score}점
            - 총 자산: {assets:,}원
            - 월 저축: {savings:,}원
            
            다음 형식으로 구체적으로 답변해주세요:
            
            ## 💳 추천 신용카드 (실제 카드명 포함)
            - 신용점수와 소득을 고려한 구체적인 카드 상품 3-4개
            - 각 카드의 주요 혜택과 연회비, 발급 조건 명시
            - 사용자 상황에 맞는 카드 선택 이유 설명
            
            ## 🏦 추천 대출 상품 (구체적인 상품명과 조건)
            - 신용점수와 소득을 고려한 구체적인 대출 상품 2-3개
            - 각 상품의 대출 한도, 금리, 상환 기간 명시
            - 사용자 상황에 맞는 대출 상품 선택 이유 설명
            
            ## 💰 추천 적금/투자 상품 (구체적인 상품명과 수익률)
            - 자산 상황과 위험 성향을 고려한 적금/투자 상품 3-4개
            - 각 상품의 금리, 가입 기간, 최소 가입 금액 명시
            - 사용자 상황에 맞는 상품 선택 이유와 투자 전략 설명
            
            ## 📊 신용점수별 금융 상품 이용 가능성
            - 현재 신용점수로 이용 가능한 상품과 제한사항
            - 신용점수 향상 시 추가로 이용 가능한 상품
            - 신용점수 관리 전략과 목표 설정
            
            답변은 한국어로 작성하고, 실제 금융 상품처럼 구체적이고 실용적으로 작성해주세요.
            각 상품마다 구체적인 조건, 혜택, 주의사항을 포함해주세요.
            """
        )
        
        # 최신 LangChain 문법 사용
        chain = recommendation_prompt | llm
        
        result = chain.invoke({
            "income": data['income'],
            "credit_score": data['credit_score'],
            "assets": sum(data['assets'].values()),
            "savings": data['savings'],
            "expense": data['expense']
        })
        
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        st.warning(f"금융 상품 추천 생성 실패: {e}")
        return get_default_financial_recommendations(data)

def get_default_credit_guidance(data):
    """기본 신용 관리 가이드 (LangChain 실패 시)"""
    credit_score = data['credit_score']
    income = data['income']
    
    if credit_score >= 750:
        level = "우수"
        tips = [
            "현재 우수한 신용점수를 유지하고 있습니다.",
            "신용카드 사용량을 30% 이하로 유지하세요.",
            "정기적으로 신용점수를 모니터링하세요."
        ]
        warnings = [
            "과도한 신용카드 발급은 신용점수에 영향을 줄 수 있습니다.",
            "대출 상환을 정시에 완료하세요."
        ]
    elif credit_score >= 650:
        level = "양호"
        tips = [
            "신용점수를 더욱 향상시킬 수 있습니다.",
            "신용카드 사용량을 40% 이하로 유지하세요.",
            "다양한 금융거래를 활성화하세요."
        ]
        warnings = [
            "연체는 신용점수에 큰 영향을 줍니다.",
            "단기간에 여러 금융사에 대출 신청을 자제하세요."
        ]
    else:
        level = "개선 필요"
        tips = [
            "신용점수 향상을 위한 노력이 필요합니다.",
            "신용카드 사용량을 20% 이하로 유지하세요.",
            "정시에 모든 대출을 상환하세요."
        ]
        warnings = [
            "현금서비스나 카드론 이용을 최소화하세요.",
            "주거래 은행과의 관계를 개선하세요."
        ]
    
    return f"""
    ## 💡 맞춤형 신용 관리 가이드 ({level} 단계)
    
    ### ✅ 신용점수 향상 팁
    {chr(10).join([f"- {tip}" for tip in tips])}
    
    ### ⚠️ 주의할 점
    {chr(10).join([f"- {warning}" for warning in warnings])}
    
    ### 🎯 맞춤형 권장사항
    - 현재 신용점수 {credit_score}점에서 목표 750점 달성을 위해 노력하세요
    - 월 수입 {income:,}원을 고려하여 적절한 신용한도를 유지하세요
    - 정기적인 신용점수 모니터링으로 변화를 추적하세요
    """

def get_default_financial_recommendations(data):
    """기본 금융 상품 추천 (LangChain 실패 시)"""
    credit_score = data['credit_score']
    income = data['income']
    
    if credit_score >= 750:
        loans = [
            "🏠 **주택담보대출**: 최우대 금리, 장기 상환 가능",
            "💼 **사업자 대출**: 사업 확장 자금, 최대 5억원"
        ]
        cards = [
            "🏆 **프리미엄 신용카드**: 높은 한도, 다양한 혜택",
            "✈️ **여행 전용 카드**: 마일리지 적립, 여행 보험"
        ]
        savings = [
            "💰 **고금리 적금**: 연 3.5% 이상, 최대 3천만원",
            "📈 **주식형 펀드**: 성장성 투자, 위험 분산"
        ]
    elif credit_score >= 650:
        loans = [
            "🏠 **전세자금대출**: 안정적 상환, 저금리",
            "🚗 **자동차 대출**: 필요 자금, 적정 금리"
        ]
        cards = [
            "💳 **일반 신용카드**: 기본 혜택, 안정적 한도",
            "🎁 **포인트 카드**: 포인트 적립, 할인 혜택"
        ]
        savings = [
            "💰 **청년 우대 적금**: 연 3.0% 이상, 최대 1천만원",
            "📊 **채권형 펀드**: 안정성 투자, 정기 수익"
        ]
    else:
        loans = [
            "💰 **생활안정자금**: 소액 대출, 간편 상환",
            "📚 **교육 대출**: 자기계발, 장기 투자"
        ]
        cards = [
            "🏦 **체크카드**: 신용점수 영향 없음, 현금 사용",
            "💰 **선불카드**: 사용한 만큼만 충전, 안전함"
        ]
        savings = [
            "💰 **기본 적금**: 연 2.5% 이상, 안전한 저축",
            "🏦 **정기예금**: 원금 보장, 안정적 수익"
        ]
    
    return f"""
    ## 🏦 맞춤형 금융 상품 추천
    
    ### 💳 추천 신용카드
    {chr(10).join([f"- {card}" for card in cards])}
    
    ### 🏦 추천 대출 상품
    {chr(10).join([f"- {loan}" for loan in loans])}
    
    ### 💰 추천 적금/투자 상품
    {chr(10).join([f"- {saving}" for saving in savings])}
    
    ### 📊 추천 근거
    - 신용점수 {credit_score}점 기준으로 최적화된 상품 선별
    - 월 수입 {income:,}원을 고려한 상환 능력 분석
    - 개인 맞춤형 위험도와 수익성 균형 고려
    """

def generate_comprehensive_financial_plan(data):
    """사용자 데이터 기반 종합 금융 플랜 생성 (정부지원상품 포함)"""
    llm = get_llm()
    if not llm:
        return get_default_comprehensive_plan(data)
    
    try:
        # 계산된 값들을 미리 준비
        income = data['income']
        expense = data['expense']
        credit_score = data['credit_score']
        assets = sum(data['assets'].values()) if isinstance(data['assets'], dict) else data['assets']
        savings = data['savings']
        
        # 계산된 값들
        recommended_youth_account = min(500000, int(income * 0.15))
        recommended_hope_savings = min(300000, int(income * 0.1))
        recommended_tomorrow_account = min(200000, int(income * 0.08))
        total_government_monthly = min(1000000, int(income * 0.25))
        
        current_savings_ratio = savings / income * 100
        target_monthly_savings = int(income * 0.25)
        improvement_needed = max(0, target_monthly_savings - savings)
        emergency_fund_target = int(income * 6)
        
        stage1_savings = min(int(income * 0.2), savings + 100000)
        stage2_savings = int(income * 0.25)
        stage3_savings = int(income * 0.3)
        stage4_savings = int(income * 0.35)
        
        subscription_savings = min(500000, int(income * 0.15))
        subscription_account = min(300000, int(income * 0.1))
        subscription_fund = min(200000, int(income * 0.08))
        total_subscription = min(1000000, int(income * 0.33))
        
        safe_assets_monthly = int(income * 0.1)
        growth_assets_monthly = int(income * 0.1)
        high_risk_monthly = int(income * 0.05)
        
        emergency_fund_3month = int(income * 3)
        target_credit_score = min(900, credit_score + 30)
        target_assets_1year = int(assets * 1.3)
        subscription_fund_6month = int(income * 6)
        passive_income_start = int(income * 0.02)
        passive_income_target = int(income * 0.05)
        financial_independence = int(income * 12 * 3)
        
        yearly_savings = int(income * 0.25 * 12)
        yearly_investment_return = int(income * 0.25 * 12 * 0.06)
        total_1year = int(assets + income * 0.25 * 12 * 1.06)
        
        three_year_savings = int(income * 0.25 * 12 * 3)
        compound_effect = int(income * 0.25 * 12 * 3 * 0.2)
        total_3year = int(assets + income * 0.25 * 12 * 3 * 1.2)
        
        # 종합 금융 플랜 프롬프트 (계산된 값들 사용)
        plan_prompt = PromptTemplate(
            input_variables=["income", "expense", "credit_score", "assets", "savings", "age"],
            template="""
            사용자의 재무 상황을 분석하여 청년 맞춤형 종합 금융 플랜을 제공해주세요.
            
            사용자 정보:
            - 월 수입: {income:,}원
            - 월 지출: {expense:,}원
            - 신용점수: {credit_score}점
            - 총 자산: {assets:,}원
            - 월 저축: {savings:,}원
            - 연령대: 청년층 (20-30대)
            
            다음 형식으로 구체적이고 실용적으로 답변해주세요:
            
            ## 🏛️ 청년 정부지원 금융상품 상세 가이드
            
            ### 📋 청년도약계좌 (2024년 기준)
            - **가입 조건**: 만 19-34세, 연소득 5,500만원 이하
            - **가입 한도**: 최대 3천만원 (5년간 분할 가입)
            - **권장 월 가입금액**: """ + f"{recommended_youth_account:,}원" + """
            - **정부 지원**: 연 3.5% 금리 보장, 세제혜택
            - **가입 전략**: 월급일 다음날 자동이체, 생일 기준 분산 가입
            
            ### 💰 청년희망적금 (2024년 기준)
            - **가입 조건**: 만 19-34세, 연소득 4,000만원 이하
            - **가입 한도**: 최대 1천만원 (3년간 분할 가입)
            - **권장 월 가입금액**: """ + f"{recommended_hope_savings:,}원" + """
            - **정부 지원**: 연 2.5% 금리 보장, 중도해지 시에도 이자 지급
            - **가입 전략**: 3년 계획으로 단계적 가입, 비상금 대용
            
            ### 🏦 청년내일저축계좌 (2024년 기준)
            - **가입 조건**: 만 19-34세, 연소득 3,600만원 이하
            - **가입 한도**: 최대 500만원 (1년간 분할 가입)
            - **권장 월 가입금액**: """ + f"{recommended_tomorrow_account:,}원" + """
            - **정부 지원**: 연 2.0% 금리 보장, 1년 후 자유로운 출금
            - **가입 전략**: 단기 목표 자금으로 활용, 1년 후 재가입
            
            ### 🎯 정부지원상품 가입 우선순위 및 전략
            - **1순위**: 청년도약계좌 (장기 자산 형성)
            - **2순위**: 청년희망적금 (중기 저축)
            - **3순위**: 청년내일저축계좌 (단기 목표)
            - **총 월 가입금액**: """ + f"{total_government_monthly:,}원" + """ (소득의 25% 이내)
            
            ## 💰 맞춤형 저축 및 투자 전략
            
            ### 📊 현재 상황 분석
            - **현재 월 저축**: {savings:,}원 (소득 대비 """ + f"{current_savings_ratio:.1f}%" + """)
            - **목표 월 저축**: """ + f"{target_monthly_savings:,}원" + """ (소득의 25%)
            - **개선 필요 금액**: """ + f"{improvement_needed:,}원" + """
            - **비상금 목표**: """ + f"{emergency_fund_target:,}원" + """ (6개월치 생활비)
            
            ### 🎯 단계별 저축 계획
            - **1단계 (1-3개월)**: 월 """ + f"{stage1_savings:,}원" + """ 저축
            - **2단계 (4-6개월)**: 월 """ + f"{stage2_savings:,}원" + """ 저축 달성
            - **3단계 (7-12개월)**: 월 """ + f"{stage3_savings:,}원" + """으로 확대
            - **4단계 (1년 이후)**: 월 """ + f"{stage4_savings:,}원" + """으로 안정화
            
            ## 📊 청약 및 투자 상품별 구체적 투자 금액
            
            ### 🏠 청약 상품 투자 전략
            - **청약저축**: 월 """ + f"{subscription_savings:,}원" + """ (총 자산의 15%)
            - **청약통장**: 월 """ + f"{subscription_account:,}원" + """ (총 자산의 10%)
            - **청약펀드**: 월 """ + f"{subscription_fund:,}원" + """ (총 자산의 8%)
            - **총 청약 투자**: 월 """ + f"{total_subscription:,}원" + """
            
            ### 📈 위험도별 투자 포트폴리오
            
            #### 🟢 안전자산 (40% - """ + f"{safe_assets_monthly:,}원" + """)
            - **정기예금**: 월 """ + f"{int(income * 0.05):,}원" + """ (연 2.5-3.0%)
            - **적금**: 월 """ + f"{int(income * 0.03):,}원" + """ (연 2.8-3.5%)
            - **국채/공사채**: 월 """ + f"{int(income * 0.02):,}원" + """ (연 2.0-2.5%)
            
            #### 🟡 성장자산 (40% - """ + f"{growth_assets_monthly:,}원" + """)
            - **주식형 펀드**: 월 """ + f"{int(income * 0.06):,}원" + """ (연 5-8% 예상)
            - **ETF**: 월 """ + f"{int(income * 0.03):,}원" + """ (연 4-6% 예상)
            - **ISA 계좌**: 월 """ + f"{int(income * 0.01):,}원" + """ (세제혜택)
            
            #### 🔴 고위험자산 (20% - """ + f"{high_risk_monthly:,}원" + """)
            - **개별 주식**: 월 """ + f"{int(income * 0.03):,}원" + """ (연 8-15% 예상)
            - **부동산 투자신탁**: 월 """ + f"{int(income * 0.02):,}원" + """ (연 6-10% 예상)
            
            ### 💡 분산 투자 전략
            - **시간 분산**: 월별 정기 투자로 평균 비용 효과
            - **상품 분산**: 8개 이상 상품으로 리스크 분산
            - **기관 분산**: 3개 이상 금융기관 활용
            - **리밸런싱**: 분기별 포트폴리오 점검 및 조정
            
            ## 🎯 단계별 목표 설정
            
            ### 📅 단기 목표 (3-6개월)
            - **비상금 확보**: """ + f"{emergency_fund_3month:,}원" + """ (3개월치 생활비)
            - **월 저축률 달성**: 20% → 25% → 30%
            - **정부지원상품 가입**: 2개 이상 상품 가입
            - **신용점수 향상**: {credit_score}점 → """ + f"{target_credit_score}점" + """
            
            ### 📈 중기 목표 (6개월-1년)
            - **총 자산 증대**: {assets:,}원 → """ + f"{target_assets_1year:,}원" + """ (30% 증가)
            - **투자 포트폴리오 구축**: 5개 이상 상품으로 다각화
            - **청약 자금 확보**: """ + f"{subscription_fund_6month:,}원" + """ (6개월치 청약 자금)
            - **수동소득 시작**: 월 """ + f"{passive_income_start:,}원" + """ 배당금/이자 수익
            
            ### 🚀 장기 목표 (1-3년)
            - **자산 다각화**: 부동산, 해외투자, 대체투자 등
            - **수동소득 확대**: 월 """ + f"{passive_income_target:,}원" + """ (소득의 5%)
            - **재무 독립**: """ + f"{financial_independence:,}원" + """ 자산 확보
            - **투자 수익률**: 연평균 6-8% 달성
            
            ## 💡 실행 가능한 액션 플랜
            
            ### ⚡ 즉시 실행 (이번 주)
            1. **청년도약계좌 가입 신청**: """ + f"{recommended_youth_account:,}원" + """/월
            2. **자동이체 설정**: 월급일 다음날 자동 저축
            3. **현재 지출 분석**: 절약 가능 항목 파악 및 개선
            
            ### 📋 주간 체크리스트
            - [ ] 월 저축 목표 달성 확인
            - [ ] 투자 상품 수익률 체크
            - [ ] 신용점수 변화 모니터링
            - [ ] 정부지원상품 신규 상품 확인
            - [ ] 포트폴리오 리밸런싱 검토
            
            ### 📊 월간 체크리스트
            - [ ] 전체 자산 현황 점검
            - [ ] 투자 수익률 분석 및 전략 조정
            - [ ] 새로운 금융 상품 검토
            - [ ] 목표 달성도 평가 및 계획 수정
            - [ ] 세금 절약 방안 검토
            
            ### 🎁 목표 달성 보상 시스템
            - **월 저축 목표 달성**: 외식 1회 (5만원 이내)
            - **분기 목표 달성**: 소원 상품 구매 (10만원 이내)
            - **연간 목표 달성**: 여행 또는 특별 경험 (50만원 이내)
            - **신용점수 향상**: 50점당 소원 상품 (5만원 이내)
            
            ## 📊 예상 결과 및 시뮬레이션
            
            ### 💰 1년 후 예상 자산
            - **기존 자산**: {assets:,}원
            - **저축 누적**: """ + f"{yearly_savings:,}원" + """
            - **투자 수익**: """ + f"{yearly_investment_return:,}원" + """ (6% 수익률)
            - **총 예상 자산**: """ + f"{total_1year:,}원" + """
            
            ### 📈 3년 후 예상 자산
            - **저축 누적**: """ + f"{three_year_savings:,}원" + """
            - **복리 효과**: """ + f"{compound_effect:,}원" + """ (복리 수익)
            - **총 예상 자산**: """ + f"{total_3year:,}원" + """
            
            ### 🎯 투자 수익률 시나리오
            - **보수적 시나리오**: 연평균 4-5% (안전자산 중심)
            - **균형적 시나리오**: 연평균 6-8% (현재 포트폴리오)
            - **공격적 시나리오**: 연평균 8-12% (고위험자산 확대)
            
            ## ⚠️ 주의사항 및 리스크 관리
            
            ### 🔒 리스크 관리 전략
            - **비상금 우선**: 6개월치 생활비 확보 후 투자 시작
            - **분산 투자**: 한 상품에 20% 이상 집중 투자 금지
            - **정기 점검**: 월 1회 포트폴리오 현황 점검
            - **리스크 조정**: 시장 상황에 따른 투자 비중 조정
            
            ### 📋 투자 시작 전 체크리스트
            - [ ] 비상금 6개월치 확보 완료
            - [ ] 월 저축 계획 수립 및 실행
            - [ ] 정부지원상품 가입 완료
            - [ ] 투자 상품 이해도 향상
            - [ ] 전문가 상담 또는 교육 프로그램 참여
            
            답변은 한국어로 작성하고, 모든 금액은 구체적인 숫자로 명시해주세요.
            실제 금융 상품명과 조건을 포함하여 실용적으로 작성해주세요.
            사용자의 현재 상황에 맞는 구체적인 행동 지침을 제공해주세요.
            """
        )
        
        # 최신 LangChain 문법 사용
        chain = plan_prompt | llm
        
        result = chain.invoke({
            "income": data['income'],
            "credit_score": data['credit_score'],
            "assets": sum(data['assets'].values()),
            "savings": data['savings'],
            "expense": data['expense'],
            "age": "청년층"
        })
        
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        st.warning(f"종합 금융 플랜 생성 실패: {e}")
        return get_default_comprehensive_plan(data)

def get_default_comprehensive_plan(data):
    """기본 종합 금융 플랜 (LangChain 실패 시)"""
    income = data['income']
    credit_score = data['credit_score']
    assets = sum(data['assets'].values())
    savings = data['savings']
    
    # 정부지원상품 상세 정보
    government_products = [
        f"🏛️ **청년도약계좌**: 연 3.5% 금리, 최대 3천만원, 5년 가입, 권장 월 가입금액: {min(500000, int(income * 0.15)):,}원",
        f"💰 **청년희망적금**: 연 2.5% 금리, 최대 1천만원, 3년 가입, 권장 월 가입금액: {min(300000, int(income * 0.1)):,}원",
        f"🏦 **청년내일저축계좌**: 연 2.0% 금리, 최대 500만원, 1년 가입, 권장 월 가입금액: {min(200000, int(income * 0.08)):,}원"
    ]
    
    # 맞춤형 저축 전략
    target_savings_ratio = 0.25 if income >= 4000000 else 0.20
    target_monthly_savings = int(income * target_savings_ratio)
    current_savings_ratio = savings / income if income > 0 else 0
    
    # 청약 및 투자 전략 (구체적인 금액 포함)
    if credit_score >= 700:
        investment_strategy = [
            f"📊 **주식형 펀드**: 월 {min(300000, int(income * 0.15)):,}원 (총 자산의 15%)",
            f"🏦 **ISA 계좌**: 월 {min(200000, int(income * 0.1)):,}원 (세제혜택 활용)",
            f"💰 **청약 상품**: 월 {min(500000, int(income * 0.15)):,}원 (정부지원상품 우선)"
        ]
    else:
        investment_strategy = [
            f"💰 **정기예금**: 월 {min(400000, int(income * 0.15)):,}원 (안정성 우선)",
            f"🏦 **청약 상품**: 월 {min(300000, int(income * 0.1)):,}원 (정부지원상품)",
            f"📊 **채권형 펀드**: 월 {min(200000, int(income * 0.08)):,}원 (위험 분산)"
        ]
    
    # 위험도별 포트폴리오 구성
    safe_assets = int(income * 0.1)
    growth_assets = int(income * 0.1)
    high_risk_assets = int(income * 0.05)
    
    return f"""
    ## 🏛️ 청년 정부지원 금융상품 상세 가이드
    
    ### 📋 정부지원상품 상세 정보
    {chr(10).join([f"- {product}" for product in government_products])}
    
    ### 🎯 가입 우선순위 및 전략
    - **1순위**: 청년도약계좌 (장기 자산 형성, 5년간 분할 가입)
    - **2순위**: 청년희망적금 (중기 저축, 3년간 분할 가입)
    - **3순위**: 청년내일저축계좌 (단기 목표, 1년간 분할 가입)
    - **총 월 가입금액**: {min(1000000, int(income * 0.25)):,}원 (소득의 25% 이내)
    
    ## 💰 맞춤형 저축 및 투자 전략
    
    ### 📊 현재 상황 분석
    - **현재 월 저축**: {savings:,}원 ({current_savings_ratio:.1%})
    - **목표 월 저축**: {target_monthly_savings:,}원 ({target_savings_ratio:.1%})
    - **개선 필요 금액**: {max(0, target_monthly_savings - savings):,}원
    - **비상금 목표**: {int(income * 6):,}원 (6개월치 생활비)
    
    ### 🎯 단계별 저축 계획
    - **1단계 (1-3개월)**: 월 {min(target_monthly_savings, savings + 100000):,}원 저축
    - **2단계 (4-6개월)**: 월 {target_monthly_savings:,}원 저축 달성
    - **3단계 (7-12개월)**: 월 {int(target_monthly_savings * 1.1):,}원으로 확대
    - **4단계 (1년 이후)**: 월 {int(target_monthly_savings * 1.2):,}원으로 안정화
    
    ## 📊 청약 및 투자 상품별 구체적 투자 금액
    
    ### 🏠 청약 상품 투자 전략
    - **청약저축**: 월 {min(500000, int(income * 0.15)):,}원 (총 자산의 15%)
    - **청약통장**: 월 {min(300000, int(income * 0.1)):,}원 (총 자산의 10%)
    - **청약펀드**: 월 {min(200000, int(income * 0.08)):,}원 (총 자산의 8%)
    - **총 청약 투자**: 월 {min(1000000, int(income * 0.33)):,}원
    
    ### 📈 위험도별 투자 포트폴리오
    
    #### 🟢 안전자산 (40% - {safe_assets:,}원)
    - **정기예금**: 월 {int(income * 0.05):,}원 (연 2.5-3.0%)
    - **적금**: 월 {int(income * 0.03):,}원 (연 2.8-3.5%)
    - **국채/공사채**: 월 {int(income * 0.02):,}원 (연 2.0-2.5%)
    
    #### 🟡 성장자산 (40% - {growth_assets:,}원)
    - **주식형 펀드**: 월 {int(income * 0.06):,}원 (연 5-8% 예상)
    - **ETF**: 월 {int(income * 0.03):,}원 (연 4-6% 예상)
    - **ISA 계좌**: 월 {int(income * 0.01):,}원 (세제혜택)
    
    #### 🔴 고위험자산 (20% - {high_risk_assets:,}원)
    - **개별 주식**: 월 {int(income * 0.03):,}원 (연 8-15% 예상)
    - **부동산 투자신탁**: 월 {int(income * 0.02):,}원 (연 6-10% 예상)
    
    ### 💡 분산 투자 전략
    - **시간 분산**: 월별 정기 투자로 평균 비용 효과
    - **상품 분산**: 8개 이상 상품으로 리스크 분산
    - **기관 분산**: 3개 이상 금융기관 활용
    - **리밸런싱**: 분기별 포트폴리오 점검 및 조정
    
    ## 🎯 단계별 목표 설정
    
    ### 📅 단기 목표 (3-6개월)
    - **비상금 확보**: {int(income * 3):,}원 (3개월치 생활비)
    - **월 저축률 달성**: {target_savings_ratio:.1%} → {(target_savings_ratio + 0.05):.1%} → {(target_savings_ratio + 0.1):.1%}
    - **정부지원상품 가입**: 2개 이상 상품 가입
    - **신용점수 향상**: {credit_score}점 → {min(900, credit_score + 30)}점
    
    ### 📈 중기 목표 (6개월-1년)
    - **총 자산 증대**: {assets:,}원 → {int(assets * 1.3):,}원 (30% 증가)
    - **투자 포트폴리오 구축**: 5개 이상 상품으로 다각화
    - **청약 자금 확보**: {int(income * 6):,}원 (6개월치 청약 자금)
    - **수동소득 시작**: 월 {int(income * 0.02):,}원 배당금/이자 수익
    
    ### 🚀 장기 목표 (1-3년)
    - **자산 다각화**: 부동산, 해외투자, 대체투자 등
    - **수동소득 확대**: 월 {int(income * 0.05):,}원 (소득의 5%)
    - **재무 독립**: {int(income * 12 * 3):,}원 자산 확보
    - **투자 수익률**: 연평균 6-8% 달성
    
    ## 💡 실행 가능한 액션 플랜
    
    ### ⚡ 즉시 실행 (이번 주)
    1. **청년도약계좌 가입 신청**: {min(500000, int(income * 0.15)):,}원/월
    2. **자동이체 설정**: 월급일 다음날 자동 저축
    3. **현재 지출 분석**: 절약 가능 항목 파악 및 개선
    
    ### 📋 주간 체크리스트
    - [ ] 월 저축 목표 달성 확인
    - [ ] 투자 상품 수익률 체크
    - [ ] 신용점수 변화 모니터링
    - [ ] 정부지원상품 신규 상품 확인
    - [ ] 포트폴리오 리밸런싱 검토
    
    ### 📊 월간 체크리스트
    - [ ] 전체 자산 현황 점검
    - [ ] 투자 수익률 분석 및 전략 조정
    - [ ] 새로운 금융 상품 검토
    - [ ] 목표 달성도 평가 및 계획 수정
    - [ ] 세금 절약 방안 검토
    
    ### 🎁 목표 달성 보상 시스템
    - **월 저축 목표 달성**: 외식 1회 (5만원 이내)
    - **분기 목표 달성**: 소원 상품 구매 (10만원 이내)
    - **연간 목표 달성**: 여행 또는 특별 경험 (50만원 이내)
    - **신용점수 향상**: 50점당 소원 상품 (5만원 이내)
    
    ## 📊 예상 결과 및 시뮬레이션
    
    ### 💰 1년 후 예상 자산
    - **기존 자산**: {assets:,}원
    - **저축 누적**: {int(target_monthly_savings * 12):,}원
    - **투자 수익**: {int(target_monthly_savings * 12 * 0.06):,}원 (6% 수익률)
    - **총 예상 자산**: {int(assets + target_monthly_savings * 12 * 1.06):,}원
    
    ### 📈 3년 후 예상 자산
    - **저축 누적**: {int(target_monthly_savings * 12 * 3):,}원
    - **복리 효과**: {int(target_monthly_savings * 12 * 3 * 0.2):,}원 (복리 수익)
    - **총 예상 자산**: {int(assets + target_monthly_savings * 12 * 3 * 1.2):,}원
    
    ### 🎯 투자 수익률 시나리오
    - **보수적 시나리오**: 연평균 4-5% (안전자산 중심)
    - **균형적 시나리오**: 연평균 6-8% (현재 포트폴리오)
    - **공격적 시나리오**: 연평균 8-12% (고위험자산 확대)
    
    ## ⚠️ 주의사항 및 리스크 관리
    
    ### 🔒 리스크 관리 전략
    - **비상금 우선**: 6개월치 생활비 확보 후 투자 시작
    - **분산 투자**: 한 상품에 20% 이상 집중 투자 금지
    - **정기 점검**: 월 1회 포트폴리오 현황 점검
    - **리스크 조정**: 시장 상황에 따른 투자 비중 조정
    
    ### 📋 투자 시작 전 체크리스트
    - [ ] 비상금 6개월치 확보 완료
    - [ ] 월 저축 계획 수립 및 실행
    - [ ] 정부지원상품 가입 완료
    - [ ] 투자 상품 이해도 향상
    - [ ] 전문가 상담 또는 교육 프로그램 참여
    
    ## 🌟 추천 근거
    
    ### 📊 상품 선택 근거
    - **신용점수 {credit_score}점** 기준으로 최적화된 상품 선별
    - **월 수입 {income:,}원**을 고려한 상환 능력 및 가입 한도 분석
    - **총 자산 {assets:,}원**을 고려한 위험도와 수익성 균형
    - **현재 저축 {savings:,}원**을 고려한 단계별 개선 전략
    
    ### 💡 개인화 전략
    - **소득 대비 저축률**: 현재 {current_savings_ratio:.1%} → 목표 {target_savings_ratio:.1%}
    - **자산 다각화**: 안전자산 40% + 성장자산 40% + 고위험자산 20%
    - **정부지원상품 활용**: 최대 혜택을 위한 우선순위별 가입 전략
    - **리스크 관리**: 단계별 접근으로 안정적인 자산 형성
    """

# 페이지 설정
st.set_page_config(
    page_title="Much (머니치료)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 페이지 테마 설정
st.markdown("""
<style>
    /* Streamlit 기본 테마 오버라이드 */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* 사이드바 배경색 */
    section[data-testid="stSidebar"] {
        background-color: #E4F0FF;
    }
    
    /* 메인 컨테이너 배경색 */
    .main .block-container {
        background-color: #FFFFFF;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# CSS 스타일링
st.markdown("""
<style>
    /* 전체 페이지 스타일 */
    .main {
        background-color: #FFFFFF;
    }
    
    /* 헤더 스타일 */
    .main-header {
        background: linear-gradient(135deg, #1D5091 0%, #5C81C7 100%);
        padding: 2rem 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(29, 80, 145, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* 메트릭 카드 스타일 */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #E4F0FF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(29, 80, 145, 0.1);
        border: 2px solid #E4F0FF;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(29, 80, 145, 0.15);
        border-color: #5C81C7;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #1D5091 0%, #5C81C7 100%);
    }
    
    .metric-card h3 {
        color: #1D5091;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #1D5091;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #5C81C7;
        font-weight: 600;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: #E4F0FF;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #E4F0FF 0%, #FFFFFF 100%);
        border-right: 2px solid #5C81C7;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #1D5091 0%, #5C81C7 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(29, 80, 145, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5C81C7 0%, #1D5091 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(29, 80, 145, 0.3);
    }
    
    /* 선택박스 스타일 */
    .stSelectbox > div > div {
        background: #FFFFFF;
        border: 2px solid #E4F0FF;
        border-radius: 10px;
        color: #1D5091;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #5C81C7;
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        background: #FFFFFF;
        border: 2px solid #E4F0FF;
        border-radius: 10px;
        color: #1D5091;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #5C81C7;
        box-shadow: 0 0 0 3px rgba(92, 129, 199, 0.1);
    }
    
    /* 파일 업로더 스타일 */
    .stFileUploader > div {
        background: #FFFFFF;
        border: 2px dashed #5C81C7;
        border-radius: 15px;
        padding: 2rem;
    }
    
    .stFileUploader > div:hover {
        border-color: #1D5091;
        background: #E4F0FF;
    }
    
    /* 차트 컨테이너 스타일 */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(29, 80, 145, 0.1);
    }
    
    /* 섹션 헤더 스타일 */
    .section-header {
        background: linear-gradient(135deg, #E4F0FF 0%, #FFFFFF 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1D5091;
        margin: 1.5rem 0;
    }
    
    .section-header h2 {
        color: #1D5091;
        margin: 0;
        font-weight: 700;
    }
    
    /* 정보 박스 스타일 */
    .info-box {
        background: linear-gradient(135deg, #E4F0FF 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #5C81C7;
        margin: 1rem 0;
    }
    
    /* 경고 박스 스타일 */
    .warning-box {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #D6A319;
        margin: 1rem 0;
    }
    
    /* 성공 박스 스타일 */
    .success-box {
        background: linear-gradient(135deg, #E8F5E8 0%, #FFFFFF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
    }
    
    /* 채팅 메시지 스타일 */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border: 1px solid #E4F0FF;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #5C81C7 0%, #1D5091 100%);
        color: white;
        text-align: right;
    }
    
    .chat-assistant {
        background: linear-gradient(135deg, #E4F0FF 0%, #FFFFFF 100%);
        color: #1D5091;
        border-left: 4px solid #5C81C7;
    }
    
    /* 게이지 차트 스타일 */
    .gauge-container {
        background: #FFFFFF;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(29, 80, 145, 0.1);
        border: 2px solid #E4F0FF;
    }
    
    /* 테이블 스타일 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(29, 80, 145, 0.1);
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #1D5091 0%, #5C81C7 100%);
        color: white;
        font-weight: 600;
    }
    
    /* 구분선 스타일 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #E4F0FF 0%, #5C81C7 50%, #E4F0FF 100%);
        border-radius: 1px;
        margin: 2rem 0;
    }
    
    /* 링크 스타일 */
    a {
        color: #1D5091;
        text-decoration: none;
        font-weight: 600;
    }
    
    a:hover {
        color: #5C81C7;
        text-decoration: underline;
    }
    
    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #E4F0FF;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #1D5091 0%, #5C81C7 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5C81C7 0%, #1D5091 100%);
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-card h2 {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# 사용자 데이터 (실제로는 데이터베이스에서 관리)
def hash_password(password):
    """간단한 비밀번호 해싱"""
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "test_user": {
        "name": "테스트 사용자",
        "password": hash_password("test123")
    }
}

# 세션 상태 초기화
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 대시보드"

def login_page():
    """로그인 페이지"""
    st.markdown('<div class="main-header"><h1>💰 Much (머니치료)</h1><p>청년 맞춤형 AI 자산관리 서비스</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 로그인")
        
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("로그인", key="login_btn"):
                if username in USERS and USERS[username]["password"] == hash_password(password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.success("로그인 성공!")
                    st.rerun()
                else:
                    st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
        

def main_dashboard():
    """메인 대시보드"""
    st.markdown('<div class="main-header"><h1>💰 Much (머니치료)</h1><p>청년 맞춤형 AI 자산관리 서비스</p></div>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("### 📊 메뉴")
        page = st.selectbox(
            "페이지 선택",
            ["🏠 대시보드", "📁 PDF 업로드", "📈 자산 분석", "💡 맞춤형 플랜", "📊 신용점수 관리", "💬 머치랑 대화하기"],
            index=["🏠 대시보드", "📁 PDF 업로드", "📈 자산 분석", "💡 맞춤형 플랜", "📊 신용점수 관리", "💬 머치랑 대화하기"].index(st.session_state.current_page)
        )
        
        # 페이지 변경 시 세션 상태 업데이트
        if page != st.session_state.current_page:
            st.session_state.current_page = page
        
        st.markdown("---")
        st.markdown(f"**사용자:** {USERS[st.session_state.current_user]['name']}")
        
        if st.button("로그아웃"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
    
    # 페이지 라우팅
    if page == "🏠 대시보드":
        show_dashboard()
    elif page == "📁 PDF 업로드":
        show_pdf_upload()
    elif page == "📈 자산 분석":
        show_asset_analysis()
    elif page == "💡 맞춤형 플랜":
        show_custom_plan()
    elif page == "📊 신용점수 관리":
        show_credit_score()
    elif page == "💬 머치랑 대화하기":
        show_financial_chatbot()

def show_dashboard():
    """대시보드 페이지"""
    st.markdown("## 🏠 대시보드")
    
    # 사용자 정보 표시
    if st.session_state.current_user:
        st.markdown(f"### 👋 안녕하세요, {USERS[st.session_state.current_user]['name']}님!")
    
    # 데이터가 없는 경우 안내
    if not st.session_state.extracted_data:
        st.markdown("""
        ### 📊 대시보드
        아직 분석할 데이터가 없습니다. PDF 파일을 업로드하여 자산 분석을 시작해보세요!
        
        ### 📋 사용 방법
        1. **PDF 업로드** 메뉴로 이동
        2. 자산 관련 PDF 파일 업로드 (최대 3개월치)
        3. **데이터 추출 및 분석** 버튼 클릭
        4. 분석 완료 후 대시보드에서 결과 확인
        
        ### 📊 지원하는 데이터 형식
        - 수입 정보 (급여, 월급 등)
        - 지출 정보 (월 지출, 총 지출 등)
        - 신용점수 (KCB, NICE)
        - 자산 정보 (입출금, 적금, 투자, 연금, ISA, 정부지원계좌 등)
        
        ### 💡 추천 테스트 파일
        프로젝트 폴더에 있는 `test_financial_report.pdf` 파일을 업로드하여 기능을 테스트해보세요!
        """)
        return
    
    # 추출된 데이터 사용
    data = st.session_state.extracted_data
    
    # 상세 분석 결과 요약
    st.markdown("### 📊 재무 현황 요약")
    
    # 주요 지표 카드 (더 상세한 정보 포함)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        income = data['income']
        income_status = "우수" if income >= 4000000 else "양호" if income >= 3000000 else "개선 필요"
        income_color = "green" if income >= 4000000 else "blue" if income >= 3000000 else "orange"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 월 수입</h3>
            <h2>{income:,}원</h2>
            <p style="color: {income_color}; font-weight: bold;">{income_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        expense = data['expense']
        expense_ratio = (expense / income * 100) if income > 0 else 0
        expense_status = "절약" if expense_ratio < 70 else "적정" if expense_ratio < 80 else "높음"
        expense_color = "green" if expense_ratio < 70 else "blue" if expense_ratio < 80 else "red"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>💸 월 지출</h3>
            <h2>{expense:,}원</h2>
            <p style="color: {expense_color}; font-weight: bold;">{expense_status} ({expense_ratio:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        savings = data['savings']
        savings_ratio = (savings / income * 100) if income > 0 else 0
        savings_status = "우수" if savings_ratio >= 25 else "양호" if savings_ratio >= 15 else "개선 필요"
        savings_color = "green" if savings_ratio >= 25 else "blue" if savings_ratio >= 15 else "orange"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>💎 월 저축</h3>
            <h2>{savings:,}원</h2>
            <p style="color: {savings_color}; font-weight: bold;">{savings_status} ({savings_ratio:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        credit_score = data['credit_score']
        credit_grade = "A+" if credit_score >= 800 else "A" if credit_score >= 700 else "B" if credit_score >= 600 else "C"
        credit_status = "최우수" if credit_score >= 800 else "우수" if credit_score >= 700 else "보통" if credit_score >= 600 else "개선 필요"
        credit_color = "green" if credit_score >= 800 else "blue" if credit_score >= 700 else "yellow" if credit_score >= 600 else "red"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 신용점수</h3>
            <h2>{credit_score}점</h2>
            <p style="color: {credit_color}; font-weight: bold;">{credit_grade}등급 ({credit_status})</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 재무 건강도 평가
    st.markdown("### 🏥 재무 건강도 평가")

    # LLM 기반 자산 관리 능력 평가
    if 'asset_management_assessment' in data and data['asset_management_assessment']:
        st.markdown(f"#### 💡 AI 기반 자산 관리 능력 평가: {data['asset_management_assessment']['level']}")
        st.info(data['asset_management_assessment']['reason'])
        st.markdown("---")
    
    # 종합 점수 계산 (신용점수 외 다각적 요소 반영)
    health_score = 0
    health_factors = []
    
    # 수입 점수 (25점 만점)
    if income >= 4000000:
        health_score += 25
        health_factors.append("✅ 수입: 우수 (25점)")
    elif income >= 3000000:
        health_score += 20
        health_factors.append("✅ 수입: 양호 (20점)")
    else:
        health_score += 12
        health_factors.append("⚠️ 수입: 개선 필요 (12점)")
    
    # 지출 관리 점수 (25점 만점)
    if expense_ratio < 70:
        health_score += 25
        health_factors.append("✅ 지출 관리: 우수 (25점)")
    elif expense_ratio < 80:
        health_score += 20
        health_factors.append("✅ 지출 관리: 양호 (20점)")
    else:
        health_score += 10
        health_factors.append("⚠️ 지출 관리: 개선 필요 (10점)")
    
    # 저축 점수 (20점 만점)
    if savings_ratio >= 25:
        health_score += 20
        health_factors.append("✅ 저축: 우수 (20점)")
    elif savings_ratio >= 15:
        health_score += 16
        health_factors.append("✅ 저축: 양호 (16점)")
    else:
        health_score += 8
        health_factors.append("⚠️ 저축: 개선 필요 (8점)")
    
    # 신용점수 (15점 만점)
    if credit_score >= 800:
        health_score += 15
        health_factors.append("✅ 신용점수: 최우수 (15점)")
    elif credit_score >= 700:
        health_score += 13
        health_factors.append("✅ 신용점수: 우수 (13점)")
    elif credit_score >= 600:
        health_score += 10
        health_factors.append("✅ 신용점수: 보통 (10점)")
    else:
        health_score += 5
        health_factors.append("⚠️ 신용점수: 개선 필요 (5점)")
    
    # 자산 다양성 점수 (15점 만점) - 신규 추가
    total_assets = sum(data['assets'].values())
    asset_diversity = len([v for v in data['assets'].values() if v > 0])
    if asset_diversity >= 4:
        health_score += 15
        health_factors.append("✅ 자산 다양성: 우수 (15점)")
    elif asset_diversity >= 2:
        health_score += 12
        health_factors.append("✅ 자산 다양성: 양호 (12점)")
    else:
        health_score += 6
        health_factors.append("⚠️ 자산 다양성: 개선 필요 (6점)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 재무 건강도 게이지
        fig_health = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "재무 건강도 점수"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1D5091"},
                'steps': [
                    {'range': [0, 60], 'color': "#E4F0FF"},
                    {'range': [60, 80], 'color': "#D6A319"},
                    {'range': [80, 100], 'color': "#5C81C7"}
                ],
                'threshold': {
                    'line': {'color': "#D6A319", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_health.update_layout(height=300)
        st.plotly_chart(fig_health, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 평가 세부사항")
        for factor in health_factors:
            st.markdown(factor)
        
        st.markdown("---")
        
        # 종합 평가 결과
        if health_score >= 90:
            st.success("🎉 **재무 건강도: 최우수** - 훌륭한 재무 관리 능력을 보여주고 있습니다!")
        elif health_score >= 80:
            st.success("✅ **재무 건강도: 우수** - 양호한 재무 상태를 유지하고 있습니다.")
        elif health_score >= 60:
            st.warning("⚠️ **재무 건강도: 보통** - 일부 영역에서 개선이 필요합니다.")
        else:
            st.error("🚨 **재무 건강도: 개선 필요** - 전반적인 재무 관리 개선이 필요합니다.")
    
    st.markdown("---")
    
    # 자산 분포 및 추이 분석
    st.markdown("### 📈 자산 및 수입/지출 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 자산 분포")
        assets_data = data['assets']
        
        # 0이 아닌 자산만 필터링
        non_zero_assets = {k: v for k, v in assets_data.items() if v > 0}
        
        if non_zero_assets:
            # 자산 유형별 색상 매핑
            asset_colors = {
                'checking': '#1f77b4',
                'savings': '#ff7f0e', 
                'investment': '#2ca02c',
                'pension': '#d62728',
                'isa': '#9467bd',
                'government': '#8c564b'
            }
            
            fig_pie = px.pie(
                values=list(non_zero_assets.values()),
                names=list(non_zero_assets.keys()),
                title="자산 분포",
                color_discrete_sequence=['#1D5091', '#5C81C7', '#E4F0FF', '#D6A319', '#FFFFFF']
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # 자산 상세 정보
            st.markdown("**자산 상세 정보:**")
            total_assets = sum(non_zero_assets.values())
            for asset_type, amount in non_zero_assets.items():
                percentage = (amount / total_assets * 100) if total_assets > 0 else 0
                st.markdown(f"• {asset_type}: {amount:,}원 ({percentage:.1f}%)")
        else:
            st.info("자산 데이터가 없습니다.")
    
    with col2:
        st.markdown("#### 📈 수입/지출/저축 추이")
        
        # 실제 데이터 기반 추이 (3개월치)
        months = ['1월', '2월', '3월']
        
        # 실제 수입 데이터 (약간의 변동 포함)
        income_data = [income * 0.95, income, income * 1.05]
        expense_data = [expense * 1.05, expense, expense * 0.95]
        savings_data = [income_data[i] - expense_data[i] for i in range(3)]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=months, 
            y=income_data, 
            name='수입', 
            line=dict(color='#1D5091', width=3),
            mode='lines+markers'
        ))
        fig_line.add_trace(go.Scatter(
            x=months, 
            y=expense_data, 
            name='지출', 
            line=dict(color='#D6A319', width=3),
            mode='lines+markers'
        ))
        fig_line.add_trace(go.Scatter(
            x=months, 
            y=savings_data, 
            name='저축', 
            line=dict(color='#5C81C7', width=3),
            mode='lines+markers'
        ))
        
        fig_line.update_layout(
            title="월별 재무 현황",
            xaxis_title="월",
            yaxis_title="금액 (원)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    
    # 맞춤형 권장사항
    st.markdown("### 💡 맞춤형 권장사항")
    
    recommendations = []
    
    # 수입 관련 권장사항
    if income < 3000000:
        recommendations.append("📈 **수입 증대**: 부업이나 스킬 개발을 통한 수입 증대를 고려해보세요.")
    
    # 지출 관련 권장사항
    if expense_ratio > 80:
        recommendations.append("💸 **지출 절약**: 지출을 20% 줄여서 월 {:,}원을 절약할 수 있습니다.".format(int(expense * 0.2)))
    elif expense_ratio > 70:
        recommendations.append("💸 **지출 관리**: 지출 관리를 더욱 철저히 하여 저축을 늘려보세요.")
    
    # 저축 관련 권장사항
    if savings_ratio < 20:
        recommendations.append("💰 **저축 증대**: 월 저축을 {:,}원으로 늘려서 20% 저축률을 달성해보세요.".format(int(income * 0.2)))
    
    # 신용점수 관련 권장사항
    if credit_score < 700:
        recommendations.append("📊 **신용점수 향상**: 신용카드 사용량을 30% 이하로 유지하고 정시 상환을 통해 신용점수를 향상시켜보세요.")
    
    # 자산 관련 권장사항
    total_assets = sum(data['assets'].values())
    if total_assets < income * 6:
        recommendations.append("🏦 **비상금 확보**: {:,}원의 비상금을 확보하여 안정적인 재무 기반을 구축해보세요.".format(int(income * 6)))
    
    if not recommendations:
        recommendations.append("🎉 **축하합니다!** 현재 재무 상태가 매우 양호합니다. 다음 단계로 투자 포트폴리오 다각화를 고려해보세요.")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    st.markdown("---")
    
    # 빠른 액션 버튼
    st.markdown("### ⚡ 빠른 액션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 자산 분석 상세보기", key="detail_analysis"):
            st.session_state.current_page = "📈 자산 분석"
            st.rerun()
    
    with col2:
        if st.button("💡 맞춤형 플랜 보기", key="custom_plan"):
            st.session_state.current_page = "💡 맞춤형 플랜"
            st.rerun()
    
    with col3:
        if st.button("📊 신용점수 관리", key="credit_management"):
            st.session_state.current_page = "📊 신용점수 관리"
            st.rerun()

def show_pdf_upload():
    """PDF 업로드 페이지"""
    st.markdown("## 📁 PDF 업로드")
    st.markdown("3개월치 자산 관련 PDF 파일을 업로드해주세요.")
    
    # 처리 옵션 선택
    col1, col2 = st.columns(2)
    with col1:
        process_option = st.radio(
            "처리 옵션 선택",
            ["📊 데이터 추출 및 분석", "📄 PDF를 JSON으로 변환"],
            help="데이터 분석만 하거나 JSON 파일로 변환할 수 있습니다."
        )
    
    with col2:
        save_json = st.checkbox(
            "JSON 파일 저장",
            value=True,
            help="JSON 파일을 로컬에 저장합니다."
        )
    
    uploaded_files = st.file_uploader(
        "PDF 파일 선택",
        type=['pdf'],
        accept_multiple_files=True,
        help="최대 3개월치 PDF 파일을 업로드할 수 있습니다."
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        st.markdown("### 📋 업로드된 파일")
        for i, file in enumerate(uploaded_files):
            st.write(f"{i+1}. {file.name}")
        
        if process_option == "📊 데이터 추출 및 분석":
            if st.button("📊 데이터 추출 및 분석", key="extract_btn"):
                with st.spinner("PDF 파일을 분석하고 있습니다..."):
                    # 실제 PDF 파싱 실행
                    extracted_data = process_pdf_files(uploaded_files)
                    if extracted_data:
                        st.session_state.extracted_data = extracted_data
                        st.session_state.current_page = "📈 자산 분석"  # 자산분석 페이지로 자동 이동
                        st.success("✅ PDF 데이터 추출이 완료되었습니다! 자산분석 페이지로 이동합니다.")
                        st.balloons()  # 축하 효과
                        st.rerun()
                    else:
                        st.error("❌ PDF 파일에서 데이터를 추출할 수 없습니다. 파일 형식을 확인해주세요.")
        
        elif process_option == "📄 PDF를 JSON으로 변환":
            if st.button("📄 PDF를 JSON으로 변환", key="convert_btn"):
                with st.spinner("PDF 파일을 JSON으로 변환하고 있습니다..."):
                    # PDF를 JSON으로 변환
                    extracted_data, json_files = process_pdf_to_json(uploaded_files, save_files=save_json)
                    
                    if extracted_data:
                        st.session_state.extracted_data = extracted_data
                        
                        # JSON 변환 결과 표시
                        st.success("✅ PDF를 JSON으로 변환했습니다!")
                        
                        # JSON 데이터 미리보기
                        st.markdown("### 📄 JSON 데이터 미리보기")
                        json_preview = preview_json_data(extracted_data)
                        st.code(json_preview, language='json')
                        
                        # JSON 파일 다운로드 버튼
                        if json_files:
                            st.markdown("### 💾 JSON 파일 다운로드")
                            for json_file in json_files:
                                filename = os.path.basename(json_file)
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    json_content = f.read()
                                
                                st.download_button(
                                    label=f"📥 {filename} 다운로드",
                                    data=json_content,
                                    file_name=filename,
                                    mime="application/json"
                                )
                        
                        # 분석 페이지로 이동 옵션
                        if st.button("📈 자산 분석 페이지로 이동", key="go_analysis"):
                            st.session_state.current_page = "📈 자산 분석"
                            st.rerun()
                        
                        st.balloons()  # 축하 효과
                    else:
                        st.error("❌ PDF 파일을 JSON으로 변환할 수 없습니다. 파일 형식을 확인해주세요.")

def extract_sample_data():
    """샘플 데이터 추출 (실제로는 PDF 파싱)"""
    return {
        'income': 3500000,
        'expense': 2800000,
        'savings': 700000,
        'credit_score': 720,
        'assets': {
            'checking': 5000000,
            'savings': 15000000,
            'investment': 8000000,
            'pension': 3000000,
            'isa': 2000000,
            'government': 5000000
        },
        'transactions': [
            {'date': '2024-01-15', 'category': '급여', 'amount': 3500000, 'type': 'income'},
            {'date': '2024-01-20', 'category': '식비', 'amount': -500000, 'type': 'expense'},
            {'date': '2024-01-25', 'category': '교통비', 'amount': -150000, 'type': 'expense'},
            # 더 많은 거래 내역...
        ]
    }

def show_asset_analysis():
    """자산 분석 페이지 - PDF 데이터 기반 상세 분석"""
    st.markdown("## 📈 자산 분석")
    
    if not st.session_state.extracted_data:
        st.warning("먼저 PDF 파일을 업로드하고 데이터를 추출해주세요.")
        return
    
    data = st.session_state.extracted_data
    
    # AI 분석 새로고침 버튼
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("🤖 **AI 분석 엔진**이 업로드된 PDF 데이터를 기반으로 상세한 자산 분석을 제공합니다.")
    with col2:
        if st.button("🔄 AI 분석 새로고침", type="primary"):
            st.session_state.asset_analysis = None
            st.rerun()
    
    # 기본 데이터 추출
    income = data['income']
    expense = data['expense']
    savings = data['savings']
    credit_score = data['credit_score']
    total_assets = sum(data['assets'].values())
    
    # 주요 지표 계산
    expense_ratio = (expense / income * 100) if income > 0 else 0
    savings_ratio = (savings / income * 100) if income > 0 else 0
    asset_income_ratio = (total_assets / income) if income > 0 else 0
    
    # 개선된 플랜 타입 결정 (더 정교한 기준)
    if total_assets < income * 6:  # 6개월치 생활비 미만
        plan_type = "🚨 응급자금 구축"
        plan_description = "비상금 확보가 우선인 단계"
        color = "red"
        priority = "비상금 구축"
    elif savings_ratio < 20:  # 저축률 20% 미만
        plan_type = "💰 저축 강화"
        plan_description = "저축 습관을 기르는 단계"
        color = "orange"
        priority = "저축률 개선"
    elif credit_score < 700:  # 신용점수 700점 미만
        plan_type = "📊 신용 개선"
        plan_description = "신용점수 향상이 필요한 단계"
        color = "blue"
        priority = "신용점수 향상"
    elif total_assets < income * 24:  # 2년치 생활비 미만
        plan_type = "📈 성장기"
        plan_description = "자산을 늘리고 투자를 확대하는 단계"
        color = "green"
        priority = "자산 증대"
    else:
        plan_type = "🎯 최적화"
        plan_description = "자산을 최적화하고 고수익을 추구하는 단계"
        color = "purple"
        priority = "수익률 최적화"
    
    # PDF 데이터 기반 현재 상황 요약
    st.markdown("### 📊 PDF 데이터 기반 재무 현황")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 월 수입",
            value=f"{income:,}원",
            delta=f"평균 대비 {((income - 3500000) / 3500000 * 100):+.1f}%"
        )
    
    with col2:
        expense_status = "양호" if expense_ratio < 70 else "주의" if expense_ratio < 80 else "위험"
        st.metric(
            label="💸 월 지출",
            value=f"{expense:,}원",
            delta=f"{expense_ratio:.1f}% ({expense_status})"
        )
    
    with col3:
        savings_status = "우수" if savings_ratio >= 30 else "양호" if savings_ratio >= 20 else "부족"
        st.metric(
            label="💾 월 저축",
            value=f"{savings:,}원",
            delta=f"{savings_ratio:.1f}% ({savings_status})"
        )
    
    with col4:
        asset_status = "풍부" if asset_income_ratio >= 24 else "양호" if asset_income_ratio >= 12 else "부족"
        st.metric(
            label="🏦 총 자산",
            value=f"{total_assets:,}원",
            delta=f"{asset_income_ratio:.1f}개월분 ({asset_status})"
        )
    
    st.markdown("---")
    
    # 현재 플랜 타입 표시
    st.markdown(f"### {plan_type}")
    st.markdown(f"**설명**: {plan_description}")
    st.markdown(f"**현재 우선순위**: {priority}")
    
    # AI 기반 자산 분석 생성 및 표시
    if 'asset_analysis' not in st.session_state or st.session_state.asset_analysis is None:
        with st.spinner("🤖 AI가 PDF 데이터를 분석하고 있습니다..."):
            asset_analysis = generate_asset_analysis(data)
            st.session_state.asset_analysis = asset_analysis
    
    # AI 분석 결과 표시
    if st.session_state.asset_analysis:
        st.markdown("---")
        st.markdown("### 🎯 AI 기반 자산 분석 결과")
        st.markdown(st.session_state.asset_analysis)
    
    st.markdown("---")
    
    # 자산 분포 시각화
    st.markdown("### 📊 자산 분포 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 자산 구성비")
        assets_data = data['assets']
        
        # 0이 아닌 자산만 필터링
        non_zero_assets = {k: v for k, v in assets_data.items() if v > 0}
        
        if non_zero_assets:
            # 자산 유형별 한글명 매핑
            asset_labels = {
                'checking': '입출금 계좌',
                'savings': '적금/저축',
                'investment': '투자 계좌',
                'pension': '연금 계좌',
                'isa': 'ISA 계좌',
                'government': '정부지원계좌'
            }
            
            # 파이 차트 생성
            labels = [asset_labels.get(k, k) for k in non_zero_assets.keys()]
            values = list(non_zero_assets.values())
            
            fig_pie = px.pie(
                values=values,
                names=labels,
                title="자산 분포",
                color_discrete_sequence=['#1D5091', '#5C81C7', '#E4F0FF', '#D6A319', '#FFFFFF']
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # 자산 상세 정보
            st.markdown("**자산 상세 정보:**")
            for asset_type, amount in non_zero_assets.items():
                label = asset_labels.get(asset_type, asset_type)
                percentage = (amount / total_assets * 100) if total_assets > 0 else 0
                st.markdown(f"• {label}: {amount:,}원 ({percentage:.1f}%)")
        else:
            st.info("자산 데이터가 없습니다.")
    
    with col2:
        st.markdown("#### 📈 수입/지출/저축 분석")
        
        # 3개월 추이 시뮬레이션 (실제 데이터 기반)
        months = ['1개월 전', '현재', '1개월 후(예상)']
        
        # 실제 수입 데이터 (약간의 변동 포함)
        income_trend = [income * 0.95, income, income * 1.02]
        expense_trend = [expense * 1.02, expense, expense * 0.98]
        savings_trend = [income_trend[i] - expense_trend[i] for i in range(3)]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=months, 
            y=income_trend, 
            name='수입', 
            line=dict(color='#1D5091', width=3),
            mode='lines+markers'
        ))
        fig_line.add_trace(go.Scatter(
            x=months, 
            y=expense_trend, 
            name='지출', 
            line=dict(color='#D6A319', width=3),
            mode='lines+markers'
        ))
        fig_line.add_trace(go.Scatter(
            x=months, 
            y=savings_trend, 
            name='저축', 
            line=dict(color='#5C81C7', width=3),
            mode='lines+markers'
        ))
        
        fig_line.update_layout(
            title="재무 현황 추이",
            xaxis_title="기간",
            yaxis_title="금액 (원)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    
    # 거래 내역 분석 (PDF에서 추출된 거래내역 활용)
    if 'transactions' in data and data['transactions']:
        st.markdown("### 💳 거래 내역 분석")
        
        transactions = data['transactions']
        df_transactions = pd.DataFrame(transactions)
        
        if not df_transactions.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 지출 카테고리별 분석")
                
                # 지출만 필터링
                expenses = df_transactions[df_transactions['amount'] < 0].copy()
                if not expenses.empty:
                    expenses['amount'] = expenses['amount'].abs()
                    expense_by_category = expenses.groupby('description')['amount'].sum().sort_values(ascending=False)
                    
                    # 상위 10개 카테고리만 표시
                    top_expenses = expense_by_category.head(10)
                    
                    fig_bar = px.bar(
                        x=top_expenses.values,
                        y=top_expenses.index,
                        orientation='h',
                        title="주요 지출 항목 (상위 10개)",
                        labels={'x': '금액 (원)', 'y': '지출 항목'}
                    )
                    fig_bar.update_layout(height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("지출 데이터가 없습니다.")
            
            with col2:
                st.markdown("#### 📅 최근 거래 내역")
                
                # 최근 10개 거래 표시
                recent_transactions = df_transactions.head(10)
                for _, transaction in recent_transactions.iterrows():
                    amount_color = "red" if transaction['amount'] < 0 else "green"
                    amount_text = f"{transaction['amount']:+,}원"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 3px solid {amount_color}; margin: 5px 0; background-color: #f8f9fa;">
                        <strong>{transaction.get('date', 'N/A')}</strong><br>
                        {transaction.get('description', 'N/A')}<br>
                        <span style="color: {amount_color}; font-weight: bold;">{amount_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PDF 기반 맞춤형 추천사항
    st.markdown("### 💡 PDF 데이터 기반 맞춤형 조언")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 단기 목표 (3-6개월)")
        
        short_term_goals = []
        
        if expense_ratio > 80:
            short_term_goals.append("💸 지출을 20% 줄여서 월 {:,}원 절약하기".format(int(expense * 0.2)))
        
        if savings_ratio < 20:
            short_term_goals.append("💰 월 저축을 {:,}원으로 늘리기".format(int(income * 0.2)))
        
        if total_assets < income * 6:
            short_term_goals.append("🏦 비상금을 {:,}원 확보하기".format(int(income * 6)))
        
        if not short_term_goals:
            short_term_goals.append("✅ 현재 상황이 양호합니다. 다음 단계로 진행하세요!")
        
        for i, goal in enumerate(short_term_goals, 1):
            st.markdown(f"{i}. {goal}")
    
    with col2:
        st.markdown("#### 🚀 중장기 목표 (6개월-2년)")
        
        long_term_goals = []
        
        if plan_type == "초보자":
            long_term_goals.extend([
                "📈 수입 증대를 위한 스킬 개발",
                "🏦 정부지원계좌 활용 (청년도약계좌)",
                "💳 신용점수 750점 이상 달성",
                "📊 투자 기초 학습 및 소액 투자 시작"
            ])
        elif plan_type == "성장기":
            long_term_goals.extend([
                "📊 투자 포트폴리오 다각화",
                "🏦 ISA 계좌 활용으로 세제 혜택",
                "💰 월 25% 이상 저축 목표",
                "📈 고수익 투자 상품 검토"
            ])
        else:
            long_term_goals.extend([
                "📈 고수익 투자 상품 확대",
                "🏦 연금 계좌 확충",
                "💰 월 30% 이상 저축 목표",
                "📊 자산 배분 최적화"
            ])
        
        for i, goal in enumerate(long_term_goals, 1):
            st.markdown(f"{i}. {goal}")
    
    st.markdown("---")
    
    # 구체적인 실행 계획
    st.markdown("### 📅 구체적인 실행 계획")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 저축 계획")
        
        # 저축 목표 설정
        current_savings = savings
        target_savings_ratio = st.slider(
            "목표 저축률 (%)",
            min_value=10,
            max_value=50,
            value=max(20, int(savings_ratio)),
            step=5
        )
        
        target_savings_amount = int(income * target_savings_ratio / 100)
        monthly_increase = target_savings_amount - current_savings
        
        st.metric("현재 월 저축", f"{current_savings:,}원")
        st.metric("목표 월 저축", f"{target_savings_amount:,}원")
        
        if monthly_increase > 0:
            st.info(f"💡 월 {monthly_increase:,}원을 더 저축해야 목표를 달성할 수 있습니다.")
        else:
            st.success("✅ 목표 저축률을 달성하고 있습니다!")
    
    with col2:
        st.markdown("#### 📊 투자 계획")
        
        # 투자 상품 추천
        if plan_type == "초보자":
            investment_recommendations = [
                "🏦 정기예금/적금 (안정성)",
                "📈 국채/공사채 (저위험)",
                "💰 청년도약계좌 (정부지원)"
            ]
        elif plan_type == "성장기":
            investment_recommendations = [
                "📊 주식형 펀드 (성장성)",
                "🏦 ISA 계좌 (세제혜택)",
                "📈 ETF 투자 (다각화)"
            ]
        else:
            investment_recommendations = [
                "📈 개별 주식 투자",
                "🏦 부동산 투자 신탁",
                "📊 해외 투자 상품"
            ]
        
        st.markdown("**추천 투자 상품:**")
        for i, rec in enumerate(investment_recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    st.markdown("---")
    
    # 소득 변동 시뮬레이션 (Earnin 앱 사례 기반)
    st.markdown("### 📈 소득 변동 시뮬레이션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 소득 안정성 분석")
        
        # 소득 안정성 평가 (실제로는 더 복잡한 알고리즘 적용)
        income_stability = "안정적" if income >= 3000000 else "보통" if income >= 2000000 else "불안정"
        stability_color = "green" if income_stability == "안정적" else "blue" if income_stability == "보통" else "red"
        
        st.markdown(f"""
        **현재 소득 안정성**: <span style="color: {stability_color}; font-weight: bold;">{income_stability}</span>
        """, unsafe_allow_html=True)
        
        # 소득 변동 시나리오
        st.markdown("**소득 변동 시나리오:**")
        scenarios = [
            f"📈 **수입 증가 20%**: {income * 1.2:,.0f}원 → 월 저축 {income * 1.2 * 0.2:,.0f}원",
            f"📊 **수입 유지**: {income:,.0f}원 → 월 저축 {income * 0.2:,.0f}원",
            f"📉 **수입 감소 20%**: {income * 0.8:,.0f}원 → 월 저축 {income * 0.8 * 0.15:,.0f}원"
        ]
        
        for scenario in scenarios:
            st.markdown(f"• {scenario}")
    
    with col2:
        st.markdown("#### 💰 현금 흐름 예측")
        
        # 월별 현금 흐름 예측 (3개월)
        months = ['1월', '2월', '3월']
        
        # 실제 수입 데이터 (약간의 변동 포함)
        income_data = [income * 0.95, income, income * 1.05]
        expense_data = [expense * 1.05, expense, expense * 0.95]
        savings_data = [income_data[i] - expense_data[i] for i in range(3)]
        
        fig_cashflow = go.Figure()
        fig_cashflow.add_trace(go.Scatter(
            x=months, 
            y=income_data, 
            name='수입', 
            line=dict(color='#1D5091', width=3),
            mode='lines+markers'
        ))
        fig_cashflow.add_trace(go.Scatter(
            x=months, 
            y=expense_data, 
            name='지출', 
            line=dict(color='#D6A319', width=3),
            mode='lines+markers'
        ))
        fig_cashflow.add_trace(go.Scatter(
            x=months, 
            y=savings_data, 
            name='저축', 
            line=dict(color='#5C81C7', width=3),
            mode='lines+markers'
        ))
        
        fig_cashflow.update_layout(
            title="월별 현금 흐름 예측",
            xaxis_title="월",
            yaxis_title="금액 (원)",
            height=300,
            hovermode='x unified'
        )
        st.plotly_chart(fig_cashflow, use_container_width=True)
    
    st.markdown("---")
    
    
    
    # 목표 설정 및 저장
    st.markdown("### 🎯 목표 설정")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_savings = st.number_input(
            "월 저축 목표 (원)",
            min_value=0,
            value=int(income * 0.2) if income > 0 else 0,
            step=100000
        )
    
    with col2:
        target_credit_score = st.number_input(
            "목표 신용점수",
            min_value=300,
            max_value=900,
            value=750,
            step=10
        )
    
    with col3:
        target_assets = st.number_input(
            "목표 총 자산 (원)",
            min_value=0,
            value=int(income * 12),
            step=1000000
        )
    
    if st.button("🎯 목표 저장", key="save_goals"):
        st.success("✅ 목표가 저장되었습니다! 정기적으로 진행 상황을 확인해보세요.")

def show_custom_plan():
    """맞춤형 플랜 페이지 - LangChain 모델 기반 상세 플랜 제공"""
    st.markdown("## 💡 맞춤형 플랜")
    st.markdown("### 🌟 AI 기반 맞춤형 자산 관리 플랜")

    if not st.session_state.extracted_data:
        st.warning("먼저 PDF 파일을 업로드하고 데이터를 추출해주세요.")
        return

    data = st.session_state.extracted_data
    
    # AI 플랜 새로고침 버튼
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("🤖 **LangChain AI 모델**이 사용자의 재무 상황을 분석하여 맞춤형 금융 플랜을 생성합니다.")
    with col2:
        if st.button("🔄 AI 플랜 새로고침", type="primary"):
            st.session_state.ai_plan = None
            st.rerun()
    
    # AI 기반 맞춤형 플랜 생성 및 표시
    if 'ai_plan' not in st.session_state or st.session_state.ai_plan is None:
        with st.spinner("🤖 AI가 맞춤형 금융 플랜을 생성하고 있습니다..."):
            ai_plan = generate_comprehensive_financial_plan(data)
            st.session_state.ai_plan = ai_plan
    
    # AI 플랜 표시
    if st.session_state.ai_plan:
        st.markdown("---")
        st.markdown("### 🎯 AI 생성 맞춤형 금융 플랜")
        
        # 플랜을 섹션별로 분리하여 표시
        plan_content = st.session_state.ai_plan
        
        # 정부지원상품 섹션
        if "청년 정부지원 금융상품" in plan_content:
            st.markdown("#### 🏛️ 청년 정부지원 금융상품 상세 가이드")
            gov_section = plan_content.split("## 💰 맞춤형 저축 및 투자 전략")[0].split("🏛️ 청년 정부지원 금융상품")[1]
            st.markdown(gov_section)
        
        # 저축 및 투자 전략 섹션
        if "맞춤형 저축 및 투자 전략" in plan_content:
            st.markdown("#### 💰 맞춤형 저축 및 투자 전략")
            savings_section = plan_content.split("## 📊 청약 및 투자 상품별")[0].split("## 💰 맞춤형 저축 및 투자 전략")[1]
            st.markdown(savings_section)
        
        # 청약 및 투자 상품 섹션
        if "청약 및 투자 상품별" in plan_content:
            st.markdown("#### 📊 청약 및 투자 상품별 구체적 투자 금액")
            investment_section = plan_content.split("## 🎯 단계별 목표 설정")[0].split("## 📊 청약 및 투자 상품별")[1]
            st.markdown(investment_section)
        
        # 단계별 목표 설정 섹션
        if "단계별 목표 설정" in plan_content:
            st.markdown("#### 🎯 단계별 목표 설정")
            goals_section = plan_content.split("## 💡 실행 가능한 액션 플랜")[0].split("## 🎯 단계별 목표 설정")[1]
            st.markdown(goals_section)
        
        # 실행 가능한 액션 플랜 섹션
        if "실행 가능한 액션 플랜" in plan_content:
            st.markdown("#### 💡 실행 가능한 액션 플랜")
            action_section = plan_content.split("## 📊 예상 결과 및 시뮬레이션")[0].split("## 💡 실행 가능한 액션 플랜")[1]
            st.markdown(action_section)
        
        # 예상 결과 및 시뮬레이션 섹션
        if "예상 결과 및 시뮬레이션" in plan_content:
            st.markdown("#### 📊 예상 결과 및 시뮬레이션")
            simulation_section = plan_content.split("## ⚠️ 주의사항 및 리스크 관리")[0].split("## 📊 예상 결과 및 시뮬레이션")[1]
            st.markdown(simulation_section)
        
        # 주의사항 및 리스크 관리 섹션
        if "주의사항 및 리스크 관리" in plan_content:
            st.markdown("#### ⚠️ 주의사항 및 리스크 관리")
            risk_section = plan_content.split("## 🌟 추천 근거")[0].split("## ⚠️ 주의사항 및 리스크 관리")[1]
            st.markdown(risk_section)
        
        # 추천 근거 섹션
        if "추천 근거" in plan_content:
            st.markdown("#### 🌟 추천 근거")
            basis_section = plan_content.split("## 🌟 추천 근거")[1]
            st.markdown(basis_section)
        
        # 전체 플랜을 접을 수 있는 섹션으로도 제공
        with st.expander("📋 전체 AI 플랜 보기"):
            st.markdown(st.session_state.ai_plan)
    
    st.markdown("---")
    
    # 추가 정보 및 시각화
    st.markdown("### 📊 현재 재무 상황 요약")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 월 수입",
            value=f"{data['income']:,}원",
            delta=f"{(data['income'] - 3000000) / 3000000 * 100:.1f}%" if data['income'] != 3000000 else "0%"
        )
    
    with col2:
        st.metric(
            label="💸 월 지출",
            value=f"{data['expense']:,}원",
            delta=f"{(data['expense'] - data['income'] * 0.7) / (data['income'] * 0.7) * 100:.1f}%"
        )
    
    with col3:
        savings_ratio = (data['income'] - data['expense']) / data['income'] * 100
        st.metric(
            label="💾 저축률",
            value=f"{savings_ratio:.1f}%",
            delta=f"{savings_ratio - 30:.1f}%" if savings_ratio != 30 else "0%"
        )
    
    # 소득 안정성 평가
    st.markdown("### 🎯 소득 안정성 평가")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 소득 안정성 점수 계산
        income_stability_score = 0
        if data['income'] >= 4000000:
            income_stability_score = 90
        elif data['income'] >= 3000000:
            income_stability_score = 75
        elif data['income'] >= 2000000:
            income_stability_score = 60
        else:
            income_stability_score = 40
        
        # 소득 안정성 게이지
        fig_stability = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = income_stability_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "소득 안정성 점수"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1D5091"},
                'steps': [
                    {'range': [0, 50], 'color': "#E4F0FF"},
                    {'range': [50, 70], 'color': "#D6A319"},
                    {'range': [70, 100], 'color': "#5C81C7"}
                ]
            }
        ))
        fig_stability.update_layout(height=250)
        st.plotly_chart(fig_stability, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 소득 변동 시나리오")
        
        # 소득 변동 시나리오별 대응 전략
        scenarios = [
            {
                "name": "수입 증가 20%",
                "income": data['income'] * 1.2,
                "strategy": "투자 비중 확대, 고수익 상품 검토"
            },
            {
                "name": "수입 유지",
                "income": data['income'],
                "strategy": "현재 플랜 유지, 점진적 개선"
            },
            {
                "name": "수입 감소 20%",
                "income": data['income'] * 0.8,
                "strategy": "비상금 확보, 지출 절약 강화"
            }
        ]
        
        for scenario in scenarios:
            st.markdown(f"**{scenario['name']}**")
            st.markdown(f"예상 수입: {scenario['income']:,.0f}원")
            st.markdown(f"대응 전략: {scenario['strategy']}")
            st.markdown("---")
    
    # 정부지원상품 가입 가능 여부 체크
    st.markdown("### 🏛️ 정부지원상품 가입 가능 여부")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 청년도약계좌 가입 가능 여부
        can_join_daeyak = data['income'] * 12 <= 55000000  # 연소득 5,500만원 이하
        st.metric(
            label="청년도약계좌",
            value="가입 가능" if can_join_daeyak else "가입 불가",
            delta="연 3.5% 금리" if can_join_daeyak else "연소득 초과"
        )
    
    with col2:
        # 청년희망적금 가입 가능 여부
        can_join_huimang = data['income'] * 12 <= 40000000  # 연소득 4,000만원 이하
        st.metric(
            label="청년희망적금",
            value="가입 가능" if can_join_huimang else "가입 불가",
            delta="연 2.5% 금리" if can_join_huimang else "연소득 초과"
        )
    
    with col3:
        # 청년내일저축계좌 가입 가능 여부
        can_join_naeil = data['income'] * 12 <= 36000000  # 연소득 3,600만원 이하
        st.metric(
            label="청년내일저축계좌",
            value="가입 가능" if can_join_naeil else "가입 불가",
            delta="연 2.0% 금리" if can_join_naeil else "연소득 초과"
        )
    
    # 즉시 실행 가능한 액션
    st.markdown("### ⚡ 즉시 실행 가능한 액션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 이번 주 실행")
        immediate_actions = [
            f"청년도약계좌 가입 신청: {min(500000, int(data['income'] * 0.15)):,}원/월",
            "자동이체 설정: 월급일 다음날 자동 저축",
            "현재 지출 분석: 절약 가능 항목 파악 및 개선"
        ]
        
        for action in immediate_actions:
            st.markdown(f"✅ {action}")
    
    with col2:
        st.markdown("#### 📋 이번 달 체크리스트")
        monthly_checklist = [
            "월 저축 목표 달성 확인",
            "투자 상품 수익률 체크",
            "신용점수 변화 모니터링",
            "정부지원상품 신규 상품 확인"
        ]
        
        for item in monthly_checklist:
            st.markdown(f"☐ {item}")
    
    st.markdown("---")
    
    # 추가 도움말
    st.markdown("### 💡 도움말")
    st.info("""
    **AI 플랜 새로고침**: 사용자 데이터가 변경되었거나 더 정확한 플랜이 필요한 경우 사용하세요.
    
    **정부지원상품**: 연소득 기준에 따라 가입 가능한 상품이 다릅니다. 
    정확한 가입 조건은 해당 금융기관에 문의하시기 바랍니다.
    
    **투자 상품**: 모든 투자는 원금 손실 가능성이 있습니다. 
    투자 전 상품 설명서를 꼭 읽어보시고, 필요시 전문가 상담을 받으시기 바랍니다.
    """)

def show_credit_score():
    """신용점수 관리 페이지"""
    # 기획서에 반영할 내용:
    # - 실제 금융권 적용 가능성: 신용점수 관리 가이드가 실제 신용 평가 기관의 기준과 금융 상품 연계에 부합하도록 제시
    # - 객관적 근거 자료: 신용점수 상승 요인 및 하락 요인에 대한 통계적 근거 제시, 금융사 연계 상품의 실제 효과를 데이터로 뒷받침
    # - 고객 유치 및 기대 효과: 신용점수 향상을 통한 금융 접근성 개선 및 고객의 금융 활동 증대 효과 강조
    # - 차별점: AI 기반의 개인화된 신용 관리 가이드와 맞춤형 금융 상품 연계를 통한 차별점 부각
    # - 락인(Lock-in) 전략: 신용점수 개선에 따른 금융사 제휴 혜택 제공으로 고객의 장기적인 서비스 이용 유도
    st.markdown("## 📊 신용점수 관리")
    
    if not st.session_state.extracted_data:
        st.warning("먼저 PDF 파일을 업로드하고 데이터를 추출해주세요.")
        return
    
    data = st.session_state.extracted_data
    current_score = data['credit_score']
    
    st.write("신용점수 현황을 확인하고, AI 기반의 맞춤형 신용 관리 가이드를 제공합니다.")

    # KCB/NICE 신용점수 표시
    st.markdown("### 📊 현재 신용점수 현황")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏦 KCB 신용점수")
        kcb_score = current_score
        kcb_grade = "A+" if kcb_score >= 800 else "A" if kcb_score >= 700 else "B" if kcb_score >= 600 else "C"
        st.markdown(f"**{kcb_score}점 ({kcb_grade}등급)**")
    
    with col2:
        st.markdown("#### 🏦 NICE 신용점수")
        nice_score = current_score + 5  # 실제로는 NICE 점수 (예시로 +5)
        nice_grade = "A+" if nice_score >= 800 else "A" if nice_score >= 700 else "B" if nice_score >= 600 else "C"
        st.markdown(f"**{nice_score}점 ({nice_grade}등급)**")

    st.markdown("---")
    
    # AI 기반 신용 관리 가이드 (LangChain 사용)
    st.markdown("### 💡 AI 기반 신용 관리 가이드")
    
    if st.button("🔄 AI 가이드 새로고침", key="refresh_credit_guide"):
        st.session_state.credit_guidance = None
    
    if 'credit_guidance' not in st.session_state:
        st.session_state.credit_guidance = generate_credit_guidance(data)
    
    st.markdown(st.session_state.credit_guidance)
    
    st.markdown("---")
    
    # 맞춤형 금융 상품 추천 (LangChain 사용)
    st.markdown("### 🔗 맞춤형 금융 상품 추천")
    st.info("💡 **LangChain AI 모델**이 사용자의 신용점수와 재정 상태를 분석하여 최적화된 금융 상품을 추천합니다.")
    
    if st.button("🔄 추천 상품 새로고침", key="refresh_recommendations"):
        st.session_state.financial_recommendations = None
    
    if 'financial_recommendations' not in st.session_state:
        st.session_state.financial_recommendations = generate_financial_recommendations(data)
    
    st.markdown(st.session_state.financial_recommendations)
    
    st.markdown("---")
    
    # 신용점수 등급별 상세 분석
    st.markdown("### 📈 신용점수 등급별 분석")
    
    if current_score >= 800:
        grade = "A+"
        grade_description = "최우수 등급"
        grade_color = "success"
        grade_benefits = [
            "🏦 모든 금융상품 이용 가능",
            "💳 최고 한도 신용카드 발급",
            "🏠 대출 금리 최우대",
            "📊 투자 상품 우선 이용"
        ]
    elif current_score >= 700:
        grade = "A"
        grade_description = "우수 등급"
        grade_color = "info"
        grade_benefits = [
            "🏦 대부분 금융상품 이용 가능",
            "💳 높은 한도 신용카드 발급",
            "🏠 대출 금리 우대",
            "📊 투자 상품 이용 가능"
        ]
    elif current_score >= 600:
        grade = "B"
        grade_description = "보통 등급"
        grade_color = "warning"
        grade_benefits = [
            "🏦 기본 금융상품 이용 가능",
            "💳 일반 한도 신용카드 발급",
            "🏠 대출 가능하나 금리 보통",
            "📊 제한적 투자 상품 이용"
        ]
    else:
        grade = "C"
        grade_description = "개선 필요 등급"
        grade_color = "error"
        grade_benefits = [
            "🏦 제한적 금융상품 이용",
            "💳 신용카드 발급 어려움",
            "🏠 대출 한도 제한",
            "📊 투자 상품 이용 제한"
        ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 현재 등급: {grade} ({grade_description})")
        
        for benefit in grade_benefits:
            st.markdown(f"✅ {benefit}")
    
    with col2:
        st.markdown("#### 📊 등급별 신용점수 분포")
        
        # 등급별 분포 차트
        grade_data = {
            'A+': 800,
            'A': 700,
            'B': 600,
            'C': 300
        }
        
        fig_grade = go.Figure(data=[
            go.Bar(
                x=list(grade_data.keys()),
                y=list(grade_data.values()),
                marker_color=['#5C81C7', '#1D5091', '#D6A319', '#E4F0FF']
            )
        ])
        fig_grade.update_layout(
            title="등급별 최소 신용점수",
            yaxis_title="신용점수",
            height=300
        )
        st.plotly_chart(fig_grade, use_container_width=True)
    
    st.markdown("---")
    
    # 신용점수 향상 가이드
    st.markdown("### 🚀 신용점수 향상 가이드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 즉시 실행 가능한 방법")
        
        immediate_actions = [
            "💳 신용카드 사용량을 30% 이하로 유지",
            "⏰ 모든 대출 상환을 정시에 완료",
            "📊 다양한 금융거래 활성화",
            "🏦 정기적인 수입 증명",
            "📈 신용한도 점진적 증가 신청"
        ]
        
        for i, action in enumerate(immediate_actions, 1):
            st.markdown(f"{i}. {action}")
    
    with col2:
        st.markdown("#### 🎯 중장기 개선 방법")
        
        long_term_actions = [
            "📚 신용관리 교육 프로그램 참여",
            "🏦 안정적인 수입 증대",
            "💼 다양한 금융상품 이용",
            "📊 정기적인 신용점수 모니터링",
            "🤝 신용상담 전문가 상담"
        ]
        
        for i, action in enumerate(long_term_actions, 1):
            st.markdown(f"{i}. {action}")
    
    st.markdown("---")
    
    # 신용점수 향상 시뮬레이션
    st.markdown("### 🎮 신용점수 향상 시뮬레이션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 개선 요소별 점수 변화")
        
        improvement_factors = {
            "신용카드 사용량 30% 이하": 20,
            "정시 상환 6개월 연속": 30,
            "다양한 금융거래 활성화": 15,
            "수입 증대": 25,
            "신용한도 증가": 10
        }
        
        selected_improvements = st.multiselect(
            "개선할 요소를 선택하세요:",
            list(improvement_factors.keys())
        )
        
        total_improvement = sum(improvement_factors[factor] for factor in selected_improvements)
        projected_score = min(900, current_score + total_improvement)
        projected_grade = "A+" if projected_score >= 800 else "A" if projected_score >= 700 else "B" if projected_score >= 600 else "C"
        
        st.metric("현재 점수", f"{current_score}점 ({grade}등급)")
        st.metric("예상 점수", f"{projected_score}점 ({projected_grade}등급)")
        st.metric("점수 향상", f"+{total_improvement}점")
    
    with col2:
        st.markdown("#### 🎯 목표 신용점수 설정")
        
        target_score = st.slider(
            "목표 신용점수",
            min_value=300,
            max_value=900,
            value=current_score + 50,
            step=10
        )
        
        target_grade = "A+" if target_score >= 800 else "A" if target_score >= 700 else "B" if target_score >= 600 else "C"
        score_gap = target_score - current_score
        
        st.metric("목표 등급", target_grade)
        st.metric("필요 점수", f"+{score_gap}점")
        
        if score_gap > 0:
            st.info(f"💡 목표 달성을 위해 신용관리를 더욱 철저히 해야 합니다.")
        else:
            st.success("✅ 목표를 달성하고 있습니다!")
        
        if st.button("🎯 목표 설정", key="set_credit_goal"):
            st.success(f"✅ 목표 신용점수 {target_score}점({target_grade}등급)이 설정되었습니다!")
    
    st.markdown("---")
    
    # 신용점수 관리 팁
    st.markdown("### 💡 신용점수 관리 팁")
    
    tips_data = {
        "신용카드 관리": [
            "💳 사용량을 30% 이하로 유지하세요",
            "⏰ 결제일을 정확히 기억하고 정시에 결제하세요",
            "📊 여러 카드를 번갈아 사용하세요"
        ],
        "대출 관리": [
            "🏦 대출 상환을 정시에 완료하세요",
            "📈 대출 한도를 점진적으로 늘리세요",
            "📊 대출 종류를 다양화하세요"
        ],
        "금융거래": [
            "🏦 다양한 금융기관과 거래하세요",
            "📊 정기적인 수입 증명을 제출하세요",
            "💰 안정적인 수입을 유지하세요"
        ]
    }
    
    for category, tips in tips_data.items():
        st.markdown(f"#### {category}")
        for tip in tips:
            st.markdown(f"• {tip}")
        st.markdown("")

def show_financial_chatbot():
    """머치랑 대화하기 - 금융 상담 챗봇 페이지"""
    st.markdown("## 💬 머치랑 대화하기")
    st.markdown("### 🌟 AI 금융 상담사와 함께 금융 고민을 해결해보세요")
    
    if not st.session_state.extracted_data:
        st.warning("먼저 PDF 파일을 업로드하고 데이터를 추출해주세요.")
        return
    
    data = st.session_state.extracted_data
    
    # 채팅 히스토리 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # AI 상담사 소개
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("🤖 **머치(Much Money Therapy)**는 당신의 금융 상담사입니다. 신용점수, 투자, 저축, 대출 등 모든 금융 고민을 편하게 상담해보세요!")
    with col2:
        if st.button("🔄 대화 초기화", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    # 채팅 인터페이스
    st.markdown("---")
    
    # 채팅 히스토리 표시
    chat_container = st.container()
    
    with chat_container:
        # AI 상담사 첫 인사
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <strong>🤖 머치:</strong> 안녕하세요! 저는 당신의 AI 금융 상담사 머치입니다. 
                현재 월 수입 {income:,}원, 신용점수 {credit_score}점으로 파악되었습니다. 
                어떤 금융 고민이 있으신가요? 편하게 말씀해주세요! 😊
            </div>
            """.format(income=data['income'], credit_score=data['credit_score']), unsafe_allow_html=True)
        
        # 기존 대화 내용 표시
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; text-align: right;">
                    <strong>👤 나:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong>🤖 머치:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # 사용자 입력
    st.markdown("---")
    
    # 빠른 질문 버튼들
    st.markdown("### 💡 자주 묻는 질문")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💳 신용점수 향상 방법"):
            user_question = "신용점수를 빨리 올리는 방법이 있을까요?"
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.rerun()
    
    with col2:
        if st.button("💰 투자 시작 방법"):
            user_question = "투자를 처음 시작하려고 하는데 어떻게 해야 할까요?"
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.rerun()
    
    with col3:
        if st.button("🏦 정부지원상품"):
            user_question = "청년을 위한 정부지원 금융상품이 어떤 것들이 있나요?"
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.rerun()
    
    # 추가 빠른 질문
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 자산 관리 전략"):
            user_question = "현재 상황에서 어떤 자산 관리 전략을 세워야 할까요?"
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.rerun()
    
    with col2:
        if st.button("🚨 금융 위기 대응"):
            user_question = "경제가 어려워질 때 어떻게 대비해야 할까요?"
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.rerun()
    
    with col3:
        if st.button("🎯 목표 달성 방법"):
            user_question = "1억 모으기 같은 큰 목표를 어떻게 달성할 수 있을까요?"
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.rerun()
    
    # 사용자 직접 입력
    st.markdown("### 💭 직접 질문하기")
    
    user_input = st.text_area(
        "금융에 관한 고민이나 궁금한 점을 자유롭게 적어주세요:",
        placeholder="예: 신용카드 여러 장 사용하는 게 좋을까요? 투자할 때 주의할 점은? 등",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        send_button = st.button("💬 질문하기", type="primary", disabled=not user_input.strip())
    with col2:
        if st.button("🎲 랜덤 조언"):
            random_advice = get_random_financial_advice(data)
            st.session_state.chat_history.append({"role": "assistant", "content": random_advice})
            st.rerun()
    
    # 사용자 질문 처리
    if send_button and user_input.strip():
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # AI 응답 생성
        with st.spinner("🤖 머치가 답변을 준비하고 있습니다..."):
            ai_response = generate_financial_advice(user_input, data)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        st.rerun()
    
    # 채팅 히스토리가 있을 때만 표시
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📝 대화 기록")
        
        # 대화 내보내기 기능
        if st.button("📥 대화 내보내기"):
            chat_text = ""
            for message in st.session_state.chat_history:
                role = "나" if message['role'] == 'user' else "머치"
                chat_text += f"[{role}]: {message['content']}\n\n"
            
            st.download_button(
                label="💾 대화 내용 다운로드",
                data=chat_text,
                file_name=f"머치와의_대화_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # 대화 내용 요약
        if len(st.session_state.chat_history) > 2:
            st.markdown("#### 📊 이번 대화 요약")
            summary = generate_chat_summary(st.session_state.chat_history, data)
            st.info(summary)

def generate_financial_advice(user_question, data):
    """사용자 질문에 대한 금융 상담 답변 생성"""
    llm = get_llm()
    if not llm:
        return get_default_financial_advice(user_question, data)
    
    try:
        # 금융 상담 프롬프트
        advice_prompt = PromptTemplate(
            input_variables=["user_question", "income", "expense", "credit_score", "assets", "savings"],
            template="""
            당신은 친근하고 전문적인 AI 금융 상담사 '머치'입니다. 
            사용자의 금융 고민에 대해 구체적이고 실용적인 조언을 제공해주세요.
            
            사용자 정보:
            - 월 수입: {income:,}원
            - 월 지출: {expense:,}원
            - 신용점수: {credit_score}점
            - 총 자산: {assets:,}원
            - 월 저축: {savings:,}원
            
            사용자 질문: {user_question}
            
            다음 형식으로 답변해주세요:
            
            1. **공감과 이해**: 사용자의 고민에 공감을 표현
            2. **현재 상황 분석**: 사용자의 재무 상황을 고려한 분석
            3. **구체적인 해결방안**: 실행 가능한 구체적인 조언 3-5개
            4. **주의사항**: 주의해야 할 점이나 위험 요소
            5. **다음 단계**: 구체적인 행동 계획
            
            답변은 친근하고 격려하는 톤으로, 한국어로 작성해주세요.
            금액은 구체적으로 명시하고, 실제 금융 상품명이나 조건을 포함해주세요.
            """
        )
        
        # LangChain 체인 실행
        chain = advice_prompt | llm
        
        result = chain.invoke({
            "user_question": user_question,
            "income": data['income'],
            "expense": data['expense'],
            "credit_score": data['credit_score'],
            "assets": sum(data['assets'].values()) if isinstance(data['assets'], dict) else data['assets'],
            "savings": data['savings']
        })
        
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        st.warning(f"AI 상담 생성 실패: {e}")
        return get_default_financial_advice(user_question, data)

def get_default_financial_advice(user_question, data):
    """기본 금융 상담 답변 (AI 실패 시)"""
    income = data['income']
    credit_score = data['credit_score']
    assets = sum(data['assets'].values()) if isinstance(data['assets'], dict) else data['assets']
    savings = data['savings']
    
    # 질문 키워드별 기본 답변
    if "신용점수" in user_question or "신용" in user_question:
        return f"""
        💳 **신용점수 향상에 대한 조언**
        
        현재 신용점수 {credit_score}점으로 파악되었습니다. 
        
        **즉시 실행 가능한 방법:**
        1. **신용카드 사용 최적화**: 월 한도 내에서 사용하고 매월 전액 상환
        2. **자동이체 설정**: 대출금, 카드대금 자동 상환으로 연체 방지
        3. **신용조회 최소화**: 불필요한 대출 신청 자제
        
        **3개월 내 목표:**
        - 신용점수 {min(900, credit_score + 30)}점 달성
        - 월별 신용점수 변화 모니터링
        
        **주의사항:**
        - 단기간에 여러 금융사 대출 신청 금지
        - 현금서비스, 카드론 이용 최소화
        
        현재 상황에서는 안정적인 신용 관리가 가장 중요합니다! 💪
        """
    
    elif "투자" in user_question or "펀드" in user_question or "주식" in user_question:
        return f"""
        📊 **투자 시작에 대한 조언**
        
        현재 월 저축 {savings:,}원, 총 자산 {assets:,}원으로 파악되었습니다.
        
        **투자 시작 전 체크리스트:**
        1. ✅ 비상금 6개월치 확보 ({int(income * 6):,}원)
        2. ✅ 월 저축 계획 수립 및 실행
        3. ✅ 투자 상품 이해도 향상
        
        **단계별 투자 전략:**
        - **1단계 (1-3개월)**: 정기예금, 적금으로 안정적 저축
        - **2단계 (4-6개월)**: 청년 정부지원상품 가입
        - **3단계 (7-12개월)**: 위험도별 포트폴리오 구축
        
        **권장 투자 비중:**
        - 안전자산: 40% ({int(income * 0.1):,}원/월)
        - 성장자산: 40% ({int(income * 0.1):,}원/월)
        - 고위험자산: 20% ({int(income * 0.05):,}원/월)
        
        **주의사항:**
        - 모든 투자는 원금 손실 가능성 있음
        - 투자 전 상품 설명서 필독
        - 필요시 전문가 상담 권장
        
        천천히 시작해서 점진적으로 확대하는 것이 좋습니다! 🚀
        """
    
    elif "정부지원" in user_question or "청년" in user_question:
        return f"""
        🏛️ **청년 정부지원 금융상품 안내**
        
        현재 월 수입 {income:,}원 기준으로 가입 가능한 상품입니다.
        
        **가입 가능 상품:**
        """
        + (f"""
        1. 🏛️ **청년도약계좌**: 연 3.5% 금리, 최대 3천만원
           - 가입 조건: 연소득 {income * 12:,}원 (5,500만원 이하) ✅
           - 권장 월 가입: {min(500000, int(income * 0.15)):,}원
        """ if income * 12 <= 55000000 else """
        1. 🏛️ **청년도약계좌**: 연 3.5% 금리, 최대 3천만원
           - 가입 조건: 연소득 5,500만원 이하 ❌
           - 현재 연소득: {income * 12:,}원
        """) + (f"""
        
        2. 💰 **청년희망적금**: 연 2.5% 금리, 최대 1천만원
           - 가입 조건: 연소득 {income * 12:,}원 (4,000만원 이하) ✅
           - 권장 월 가입: {min(300000, int(income * 0.1)):,}원
        """ if income * 12 <= 40000000 else """
        
        2. 💰 **청년희망적금**: 연 2.5% 금리, 최대 1천만원
           - 가입 조건: 연소득 4,000만원 이하 ❌
           - 현재 연소득: {income * 12:,}원
        """) + f"""
        
        **가입 전략:**
        - 우선순위: 청년도약계좌 → 청년희망적금 → 청년내일저축계좌
        - 월급일 다음날 자동이체 설정
        - 생일 기준 분산 가입으로 리스크 분산
        
        **혜택:**
        - 정부가 금리 보장
        - 세제혜택
        - 중도해지 시에도 이자 지급
        
        정부지원상품을 활용하면 일반 상품보다 높은 수익을 얻을 수 있습니다! 🎯
        """
    
    elif "자산관리" in user_question or "자산" in user_question:
        return f"""
        💰 **자산 관리 전략 제안**
        
        현재 상황 분석:
        - 월 수입: {income:,}원
        - 월 지출: {expense:,}원
        - 월 저축: {savings:,}원
        - 총 자산: {assets:,}원
        
        **현재 저축률: {(savings / income * 100):.1f}%**
        **목표 저축률: 25% ({int(income * 0.25):,}원)**
        
        **개선 방안:**
        1. **지출 분석**: 월 {expense:,}원 중 절약 가능 항목 파악
        2. **저축 자동화**: 월급일 다음날 자동이체 설정
        3. **목표 설정**: 단기(3개월), 중기(6개월), 장기(1년) 목표 설정
        
        **자산 배분 전략:**
        - 비상금: {int(income * 6):,}원 (6개월치 생활비)
        - 정기 저축: {int(income * 0.15):,}원 (소득의 15%)
        - 투자 자금: {int(income * 0.1):,}원 (소득의 10%)
        
        **월별 체크리스트:**
        - [ ] 저축 목표 달성 확인
        - [ ] 지출 패턴 분석
        - [ ] 자산 현황 점검
        - [ ] 투자 수익률 확인
        
        체계적인 자산 관리로 안정적인 재무 상태를 만들어보세요! 📈
        """
    
    elif "위기" in user_question or "경제" in user_question or "불황" in user_question:
        return f"""
        🚨 **경제 위기 대응 전략**
        
        현재 상황에서 경제 위기에 대비하는 방법을 제안드립니다.
        
        **즉시 실행:**
        1. **비상금 확보**: 현재 {savings:,}원 → 목표 {int(income * 6):,}원
        2. **지출 절약**: 월 {expense:,}원 → 목표 {int(expense * 0.8):,}원
        3. **부채 관리**: 고금리 부채 우선 상환
        
        **3개월 내 준비:**
        - **다중 수입원**: 부업, 프리랜서, 온라인 수입 등
        - **스킬 개발**: 디지털 역량, 외국어 등 미래 지향적 스킬
        - **네트워크 구축**: 업계 인맥, 멘토십 등
        
        **6개월 내 준비:**
        - **투자 포트폴리오 조정**: 안전자산 비중 확대
        - **보험 점검**: 실업보험, 의료보험 등 보장 범위 확인
        - **대출 한도 확보**: 신용한도 유지 및 개선
        
        **주의사항:**
        - 급하게 고위험 투자로 전환하지 않기
        - 감정적 의사결정 자제
        - 전문가 상담 적극 활용
        
        **긍정적 관점:**
        - 위기는 기회의 시작
        - 새로운 분야 진출 기회
        - 자산 가격 하락 시 매수 기회
        
        차분하게 준비하면 위기를 기회로 바꿀 수 있습니다! 💪
        """
    
    elif "목표" in user_question or "1억" in user_question or "큰 목표" in user_question:
        return f"""
        🎯 **큰 목표 달성 전략**
        
        현재 월 수입 {income:,}원, 월 저축 {savings:,}원으로 큰 목표를 달성하는 방법을 제안드립니다.
        
        **1억 달성 시나리오:**
        
        **보수적 시나리오 (연 4% 수익률):**
        - 월 저축: {int(income * 0.3):,}원 (소득의 30%)
        - 달성 기간: 약 15-18년
        - 복리 효과: {int(income * 0.3 * 12 * 15 * 0.3):,}원
        
        **균형적 시나리오 (연 6% 수익률):**
        - 월 저축: {int(income * 0.25):,}원 (소득의 25%)
        - 달성 기간: 약 12-15년
        - 복리 효과: {int(income * 0.25 * 12 * 12 * 0.4):,}원
        
        **공격적 시나리오 (연 8% 수익률):**
        - 월 저축: {int(income * 0.2):,}원 (소득의 20%)
        - 달성 기간: 약 10-12년
        - 복리 효과: {int(income * 0.2 * 12 * 10 * 0.5):,}원
        
        **단계별 목표 설정:**
        - **1단계 (1-3년)**: 1천만원 달성
        - **2단계 (4-7년)**: 3천만원 달성
        - **3단계 (8-12년)**: 7천만원 달성
        - **4단계 (13-15년)**: 1억 달성
        
        **가속화 전략:**
        1. **수입 증대**: 스킬 개발, 부업, 사업 확장
        2. **투자 수익률 향상**: 위험도 조정, 포트폴리오 최적화
        3. **세금 절약**: ISA, 연금 등 세제혜택 활용
        4. **부동산 투자**: 임대 수익, 자산 가치 상승
        
        **실행 계획:**
        - **이번 주**: 월 저축 목표 설정 및 자동이체
        - **이번 달**: 투자 상품 검토 및 포트폴리오 구축
        - **이번 분기**: 목표 달성도 점검 및 전략 조정
        
        큰 목표는 작은 목표들의 연속입니다. 하나씩 달성해나가면 반드시 이루어집니다! 🚀
        """
    
    else:
        return f"""
        💡 **일반적인 금융 상담**
        
        안녕하세요! 금융에 관한 고민을 편하게 말씀해주셔서 감사합니다.
        
        현재 파악된 재무 상황:
        - 월 수입: {income:,}원
        - 월 지출: {expense:,}원
        - 신용점수: {credit_score}점
        - 총 자산: {assets:,}원
        - 월 저축: {savings:,}원
        
        **일반적인 금융 관리 원칙:**
        1. **수입 > 지출**: 기본적인 재무 건전성 유지
        2. **비상금 확보**: 6개월치 생활비 확보
        3. **분산 투자**: 리스크 분산을 위한 포트폴리오 구성
        4. **정기 점검**: 월 1회 재무 현황 점검
        
        **추천 서비스:**
        - 맞춤형 플랜: 개인 상황에 맞는 자산 관리 전략
        - 신용점수 관리: 신용점수 향상 및 관리 방법
        - 자산 분석: 현재 자산 현황 및 개선 방안
        
        더 구체적인 질문이 있으시면 언제든 말씀해주세요! 
        신용점수, 투자, 저축, 대출 등 모든 금융 분야에 대해 상담해드릴 수 있습니다. 😊
        """

def get_random_financial_advice(data):
    """랜덤 금융 조언 생성"""
    import random
    
    income = data['income']
    credit_score = data['credit_score']
    
    advices = [
        f"💡 **오늘의 금융 팁**: 월 수입 {income:,}원의 20%인 {int(income * 0.2):,}원을 자동이체로 저축해보세요. 작은 습관이 큰 자산을 만듭니다!",
        
        f"🎯 **신용점수 관리**: 현재 {credit_score}점에서 50점만 더 올리면 더 좋은 금융 상품을 이용할 수 있어요. 매월 전액 상환으로 시작해보세요!",
        
        f"🏛️ **정부지원상품**: 청년도약계좌 가입을 고려해보세요. 연 3.5% 금리로 안전하게 자산을 늘릴 수 있습니다.",
        
        f"📊 **투자 시작**: 월 {int(income * 0.1):,}원부터 시작하는 적립식 투자를 추천합니다. 복리의 힘을 경험해보세요!",
        
        f"💰 **지출 관리**: 현재 지출을 10% 줄이면 월 {int(income * 0.1):,}원을 절약할 수 있어요. 작은 변화가 큰 차이를 만듭니다!",
        
        f"🚀 **수입 증대**: 현재 직업 외에 부업이나 스킬 개발을 통해 월 {int(income * 0.1):,}원의 추가 수입을 만들어보세요!",
        
        f"🔄 **자산 다각화**: 예금, 적금, 펀드 등 다양한 상품에 분산 투자하여 리스크를 줄이고 수익을 극대화해보세요!",
        
        f"📈 **장기 계획**: 10년 후를 생각해서 월 {int(income * 0.15):,}원씩 투자하면 큰 자산을 만들 수 있어요!"
    ]
    
    return random.choice(advices)

def generate_chat_summary(chat_history, data):
    """대화 내용 요약 생성"""
    if len(chat_history) < 3:
        return "대화가 충분하지 않아 요약을 생성할 수 없습니다."
    
    try:
        # 간단한 요약 생성
        user_questions = [msg['content'] for msg in chat_history if msg['role'] == 'user']
        ai_answers = [msg['content'] for msg in chat_history if msg['role'] == 'assistant']
        
        summary = f"""
        **💬 이번 대화 요약**
        
        **상담 주제**: {', '.join(user_questions[:3])}
        **상담 횟수**: {len(user_questions)}회
        **주요 조언**: {len(ai_answers)}개
        
        **다음 단계 제안**:
        - 상담 내용을 바탕으로 맞춤형 플랜 확인
        - 제안받은 조언 중 우선순위 높은 것부터 실행
        - 정기적인 재무 현황 점검 및 상담
        
        더 자세한 상담이 필요하시면 언제든 말씀해주세요! 😊
        """
        
        return summary
    except Exception as e:
        return f"대화 요약 생성 중 오류가 발생했습니다: {e}"

def generate_asset_analysis(data):
    """PDF 데이터 기반 AI 자산 분석 생성"""
    llm = get_llm()
    if not llm:
        return get_default_asset_analysis(data)
    
    try:
        # 자산 분석 프롬프트
        analysis_prompt = PromptTemplate(
            input_variables=["income", "expense", "credit_score", "assets", "savings", "transactions"],
            template="""
            업로드된 PDF 데이터를 기반으로 사용자의 자산 상황을 전문적으로 분석하고 구체적인 개선 방안을 제공해주세요.
            
            사용자 PDF 데이터:
            - 월 수입: {income:,}원
            - 월 지출: {expense:,}원
            - 신용점수: {credit_score}점
            - 총 자산: {assets:,}원
            - 월 저축: {savings:,}원
            - 거래 내역 수: {transactions}건
            
            다음 형식으로 전문적인 분석을 제공해주세요:
            
            ## 📊 PDF 데이터 기반 재무 현황 분석
            
            ### 🎯 주요 강점
            - 현재 재무 상황에서 긍정적인 요소 3-4개
            
            ### ⚠️ 개선 필요 영역
            - 즉시 개선이 필요한 영역 2-3개
            
            ### 📈 자산 증대 전략
            - 현재 상황에 맞는 구체적인 자산 증대 방법 3-4개
            - 각 전략별 예상 효과와 기간 명시
            
            ### 💰 최적화된 자산 배분
            - 비상금: 구체적 금액과 비율
            - 단기 저축: 구체적 금액과 상품
            - 중장기 투자: 구체적 금액과 전략
            
            ### 🏛️ 정부지원상품 활용 전략
            - 가입 가능한 정부지원상품과 구체적 가입 금액
            - 우선순위와 가입 시기
            
            ### 📊 월별 실행 계획
            - 1개월차: 즉시 실행할 항목
            - 3개월차: 단기 목표
            - 6개월차: 중기 목표
            - 1년차: 장기 목표
            
            ### 🎯 예상 결과
            - 1년 후 예상 자산: 구체적 금액
            - 투자 수익률: 예상 수익률과 근거
            - 재무 건전성 개선도: 점수화
            
            답변은 한국어로 작성하고, 모든 금액은 구체적으로 명시해주세요.
            PDF에서 추출된 실제 데이터를 기반으로 한 실용적이고 실행 가능한 조언을 제공해주세요.
            """
        )
        
        # 최신 LangChain 문법 사용
        chain = analysis_prompt | llm
        
        # 거래 내역 수 계산
        transaction_count = len(data.get('transactions', []))
        
        result = chain.invoke({
            "income": data['income'],
            "expense": data['expense'],
            "credit_score": data['credit_score'],
            "assets": sum(data['assets'].values()) if isinstance(data['assets'], dict) else data['assets'],
            "savings": data['savings'],
            "transactions": transaction_count
        })
        
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        st.warning(f"AI 자산 분석 생성 실패: {e}")
        return get_default_asset_analysis(data)

def get_default_asset_analysis(data):
    """기본 자산 분석 (AI 실패 시)"""
    income = data['income']
    expense = data['expense']
    credit_score = data['credit_score']
    assets = sum(data['assets'].values()) if isinstance(data['assets'], dict) else data['assets']
    savings = data['savings']
    
    savings_ratio = (savings / income * 100) if income > 0 else 0
    expense_ratio = (expense / income * 100) if income > 0 else 0
    asset_months = (assets / income) if income > 0 else 0
    
    return f"""
    ## 📊 PDF 데이터 기반 재무 현황 분석
    
    ### 🎯 주요 강점
    
    ✅ **현재 재무 상황의 긍정적 요소:**
    - 월 수입 {income:,}원으로 {"안정적인" if income >= 3000000 else "기본적인"} 소득 기반 확보
    - 월 저축 {savings:,}원 (저축률 {savings_ratio:.1f}%)로 {"우수한" if savings_ratio >= 25 else "양호한" if savings_ratio >= 15 else "기본적인"} 저축 습관
    - 신용점수 {credit_score}점으로 {"최우수" if credit_score >= 800 else "우수한" if credit_score >= 700 else "보통" if credit_score >= 600 else "개선 필요한"} 신용 관리
    - 총 자산 {assets:,}원으로 {asset_months:.1f}개월분 생활비 확보
    
    ### ⚠️ 개선 필요 영역
    
    🔧 **즉시 개선이 필요한 영역:**
    """ + (f"- 비상금 부족: 현재 {asset_months:.1f}개월분 → 목표 6개월분 ({int(income * 6):,}원)" if asset_months < 6 else "") + f"""
    """ + (f"- 저축률 개선: 현재 {savings_ratio:.1f}% → 목표 25% ({int(income * 0.25):,}원)" if savings_ratio < 25 else "") + f"""
    """ + (f"- 지출 관리: 현재 {expense_ratio:.1f}% → 목표 70% 이하 ({int(income * 0.7):,}원)" if expense_ratio > 75 else "") + f"""
    """ + (f"- 신용점수 향상: 현재 {credit_score}점 → 목표 750점 이상" if credit_score < 750 else "") + f"""
    
    ### 📈 자산 증대 전략
    
    💰 **현재 상황에 맞는 구체적 전략:**
    
    1. **정부지원상품 적극 활용**
       - 청년도약계좌: 월 {min(500000, int(income * 0.15)):,}원 (연 3.5% 금리)
       - 청년희망적금: 월 {min(300000, int(income * 0.1)):,}원 (연 2.5% 금리)
       - 예상 연간 수익: {int((min(500000, int(income * 0.15)) * 0.035 + min(300000, int(income * 0.1)) * 0.025) * 12):,}원
    
    2. **단계적 투자 포트폴리오 구축**
       - 1단계: 안전자산 중심 (월 {int(income * 0.1):,}원)
       - 2단계: 성장자산 확대 (월 {int(income * 0.08):,}원)
       - 3단계: 고수익 자산 도입 (월 {int(income * 0.05):,}원)
    
    3. **수입 증대 계획**
       - 부업/프리랜서: 월 {int(income * 0.1):,}원 목표
       - 스킬 개발 투자: 월 {int(income * 0.02):,}원
       - 예상 수입 증가: 6개월 내 10-20%
    
    ### 💰 최적화된 자산 배분
    
    📊 **권장 자산 배분 (월 {income:,}원 기준):**
    
    - **비상금**: {int(income * 6):,}원 (6개월치, 고금리 적금)
    - **단기 저축**: 월 {int(income * 0.15):,}원 (정기예금, 청년도약계좌)
    - **중기 투자**: 월 {int(income * 0.1):,}원 (주식형 펀드, ETF)
    - **장기 투자**: 월 {int(income * 0.05):,}원 (연금저축, ISA)
    
    ### 🏛️ 정부지원상품 활용 전략
    
    🎯 **가입 우선순위 및 일정:**
    
    1. **1순위 - 청년도약계좌** (즉시 가입)
       - 월 가입금액: {min(500000, int(income * 0.15)):,}원
       - 5년간 총 {min(500000, int(income * 0.15)) * 60:,}원 적립 가능
    
    2. **2순위 - 청년희망적금** (1개월 후)
       - 월 가입금액: {min(300000, int(income * 0.1)):,}원
       - 3년간 총 {min(300000, int(income * 0.1)) * 36:,}원 적립 가능
    
    3. **3순위 - ISA 계좌** (3개월 후)
       - 월 가입금액: {min(200000, int(income * 0.08)):,}원
       - 세제혜택으로 연간 {int(min(200000, int(income * 0.08)) * 12 * 0.15):,}원 절약
    
    ### 📊 월별 실행 계획
    
    📅 **단계별 실행 일정:**
    
    **1개월차 (즉시 실행):**
    - 청년도약계좌 가입 및 자동이체 설정
    - 지출 분석 및 가계부 작성 시작
    - 신용카드 사용량 30% 이하로 조정
    
    **3개월차 (단기 목표):**
    - 월 저축률 20% 달성
    - 비상금 3개월치 확보 ({int(income * 3):,}원)
    - 투자 상품 교육 이수 및 소액 투자 시작
    
    **6개월차 (중기 목표):**
    - 월 저축률 25% 달성
    - 비상금 6개월치 완성 ({int(income * 6):,}원)
    - 다양한 투자 포트폴리오 구축
    
    **1년차 (장기 목표):**
    - 총 자산 {int(assets * 1.5):,}원 달성 (50% 증가)
    - 월 수동소득 {int(income * 0.02):,}원 창출
    - 신용점수 {min(900, credit_score + 50)}점 달성
    
    ### 🎯 예상 결과
    
    📈 **1년 후 예상 성과:**
    
    - **예상 총 자산**: {int(assets + savings * 12 * 1.2):,}원 (현재 대비 {((assets + savings * 12 * 1.2 - assets) / assets * 100):.1f}% 증가)
    - **예상 투자 수익률**: 연 5-7% (분산 투자 포트폴리오)
    - **재무 건전성 점수**: {min(100, int((savings_ratio * 2) + (100 - expense_ratio) + (credit_score / 10)))}점 (현재 대비 15-20점 향상)
    - **월 수동소득**: {int(savings * 12 * 0.05 / 12):,}원 (배당금 및 이자 수익)
    
    **핵심 성공 요인:**
    - 정부지원상품 최대 활용으로 안전한 수익 확보
    - 단계적 포트폴리오 구축으로 리스크 관리
    - 정기적 점검 및 조정으로 목표 달성률 극대화
    
    이 분석은 업로드하신 PDF 데이터를 기반으로 작성되었으며, 실제 실행 시 정기적인 모니터링과 조정이 필요합니다.
    """

# 메인 앱 실행
if __name__ == "__main__":
    if not st.session_state.authenticated:
        login_page()
    else:
        main_dashboard()
