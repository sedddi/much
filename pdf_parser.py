import PyPDF2
import re
import pandas as pd
from datetime import datetime
# import streamlit as st  # Streamlit 의존성 제거
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
# from langchain_community.llms import OpenAI  # 필요시 주석 해제
# from langchain.chains import LLMChain  # 필요시 주석 해제
import json
import os

class PDFParser:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def extract_text_from_pdf(self, pdf_file):
        """PDF 파일에서 텍스트 추출"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            # st.error(f"PDF 파일 읽기 오류: {str(e)}")  # Streamlit 의존성 제거
            print(f"PDF 파일 읽기 오류: {str(e)}")  # 일반 print로 대체
            return None
    
    def parse_financial_data(self, text):
        """텍스트에서 재무 데이터 추출"""
        data = {
            'income': 0,
            'expense': 0,
            'savings': 0,
            'credit_score': 0,
            'assets': {
                'checking': 0,
                'savings': 0,
                'investment': 0,
                'pension': 0,
                'isa': 0,
                'government': 0
            },
            'transactions': []
        }
        
        # 수입 패턴 매칭
        income_patterns = [
            r'급여[:\s]*([0-9,]+)',
            r'수입[:\s]*([0-9,]+)',
            r'월급[:\s]*([0-9,]+)',
            r'월\s*수입[:\s]*([0-9,]+)',
            r'월\s*소득[:\s]*([0-9,]+)',
            r'연봉[:\s]*([0-9,]+)',
            r'연\s*소득[:\s]*([0-9,]+)'
        ]
        
        for pattern in income_patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    data['income'] = int(amount_str)
                    # 연봉인 경우 월급으로 변환
                    if '연봉' in pattern or '연\s*소득' in pattern:
                        data['income'] = int(amount_str) // 12
                    break
                except ValueError:
                    continue
        
        # 지출 패턴 매칭
        expense_patterns = [
            r'지출[:\s]*([0-9,]+)',
            r'월\s*지출[:\s]*([0-9,]+)',
            r'총\s*지출[:\s]*([0-9,]+)',
            r'월\s*생활비[:\s]*([0-9,]+)',
            r'고정\s*지출[:\s]*([0-9,]+)'
        ]
        
        for pattern in expense_patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    data['expense'] = int(amount_str)
                    break
                except ValueError:
                    continue
        
        # 저축 계산
        if data['income'] > 0 and data['expense'] > 0:
            data['savings'] = data['income'] - data['expense']
        
        # 신용점수 패턴 매칭
        credit_patterns = [
            r'신용점수[:\s]*([0-9]+)',
            r'신용\s*점수[:\s]*([0-9]+)',
            r'KCB[:\s]*([0-9]+)',
            r'NICE[:\s]*([0-9]+)',
            r'신용등급[:\s]*([A-Z][+-]?)',
            r'신용\s*등급[:\s]*([A-Z][+-]?)'
        ]
        
        for pattern in credit_patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                if value.isdigit():
                    data['credit_score'] = int(value)
                    break
                elif re.match(r'[A-Z][+-]?', value):
                    # 등급을 점수로 변환
                    grade_to_score = {
                        'A+': 850, 'A': 750, 'A-': 700,
                        'B+': 650, 'B': 600, 'B-': 550,
                        'C+': 500, 'C': 450, 'C-': 400,
                        'D+': 350, 'D': 300, 'D-': 250
                    }
                    data['credit_score'] = grade_to_score.get(value, 600)
                    break
        
        # 자산 정보 추출
        asset_patterns = {
            'checking': [
                r'입출금[:\s]*([0-9,]+)', 
                r'통장[:\s]*([0-9,]+)', 
                r'현금[:\s]*([0-9,]+)',
                r'계좌[:\s]*([0-9,]+)'
            ],
            'savings': [
                r'적금[:\s]*([0-9,]+)', 
                r'저축[:\s]*([0-9,]+)', 
                r'예금[:\s]*([0-9,]+)',
                r'정기예금[:\s]*([0-9,]+)'
            ],
            'investment': [
                r'투자[:\s]*([0-9,]+)', 
                r'증권[:\s]*([0-9,]+)', 
                r'주식[:\s]*([0-9,]+)',
                r'펀드[:\s]*([0-9,]+)',
                r'ETF[:\s]*([0-9,]+)'
            ],
            'pension': [
                r'연금[:\s]*([0-9,]+)',
                r'퇴직연금[:\s]*([0-9,]+)',
                r'개인연금[:\s]*([0-9,]+)'
            ],
            'isa': [
                r'ISA[:\s]*([0-9,]+)', 
                r'개인형퇴직연금[:\s]*([0-9,]+)',
                r'IRP[:\s]*([0-9,]+)'
            ],
            'government': [
                r'청년도약계좌[:\s]*([0-9,]+)', 
                r'희망두배통장[:\s]*([0-9,]+)', 
                r'정부지원[:\s]*([0-9,]+)',
                r'청년희망적금[:\s]*([0-9,]+)',
                r'청년내일저축계좌[:\s]*([0-9,]+)'
            ]
        }
        
        for asset_type, patterns in asset_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        data['assets'][asset_type] = int(amount_str)
                        break
                    except ValueError:
                        continue
        
        return data
    
    def extract_transactions(self, text):
        """거래 내역 추출"""
        transactions = []
        
        # 날짜 패턴
        date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
        # 금액 패턴
        amount_pattern = r'([+-]?[0-9,]+)'
        # 카테고리 패턴
        category_patterns = [
            r'(급여|월급|수입|소득)',
            r'(식비|음식|식사|외식)',
            r'(교통비|교통|지하철|버스|택시)',
            r'(주거비|월세|전세|관리비|집세)',
            r'(통신비|전화비|인터넷|휴대폰)',
            r'(의료비|병원|약|치료)',
            r'(교육비|학원|강의|도서)',
            r'(문화생활|영화|공연|취미)',
            r'(쇼핑|의류|화장품|생활용품)',
            r'(저축|적금|투자|펀드)',
            r'(보험|보험료)',
            r'(카드대금|대출상환)'
        ]
        
        lines = text.split('\n')
        for line in lines:
            date_match = re.search(date_pattern, line)
            amount_match = re.search(amount_pattern, line)
            
            if date_match and amount_match:
                date = date_match.group(1)
                amount_str = amount_match.group(1).replace(',', '')
                try:
                    amount = int(amount_str)
                    
                    # 카테고리 추출
                    category = "기타"
                    for pattern in category_patterns:
                        cat_match = re.search(pattern, line)
                        if cat_match:
                            category = cat_match.group(1)
                            break
                    
                    # 수입/지출 구분
                    transaction_type = "income" if amount > 0 else "expense"
                    
                    transactions.append({
                        'date': date,
                        'category': category,
                        'amount': abs(amount),
                        'type': transaction_type,
                        'description': line.strip()
                    })
                except ValueError:
                    continue
        
        return transactions
    
    def analyze_with_llm(self, text):
        """LangChain을 사용한 고급 분석"""
        # 실제 OpenAI API 키가 필요합니다
        # llm = OpenAI(temperature=0)
        
        # 프롬프트 템플릿
        prompt_template = """
        다음 재무 문서에서 다음 정보를 추출해주세요:
        
        문서 내용:
        {text}
        
        다음 형식으로 JSON 응답을 제공해주세요:
        {{
            "income": 월 수입 금액,
            "expense": 월 지출 금액,
            "savings": 월 저축 금액,
            "credit_score": 신용점수,
            "assets": {{
                "checking": 입출금 계좌 잔액,
                "savings": 적금/저축 계좌 잔액,
                "investment": 투자 계좌 잔액,
                "pension": 연금 계좌 잔액,
                "isa": ISA 계좌 잔액,
                "government": 정부지원계좌 잔액
            }}
        }}
        
        금액은 숫자만 입력하고, 없는 정보는 0으로 설정해주세요.
        """
        
        # 실제 구현에서는 LLM 체인을 사용
        # chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
        # result = chain.run(text=text)
        
        # 임시로 기본 파싱 결과 반환
        return self.parse_financial_data(text)
    
    def convert_to_json(self, data, filename=None):
        """데이터를 JSON 형식으로 변환"""
        json_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_file': filename or 'unknown',
                'version': '1.0'
            },
            'financial_data': data
        }
        return json.dumps(json_data, ensure_ascii=False, indent=2)
    
    def save_json_file(self, data, filename, output_dir='output'):
        """JSON 파일로 저장"""
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 생성
            base_name = os.path.splitext(filename)[0]
            json_filename = f"{base_name}_financial_data.json"
            json_path = os.path.join(output_dir, json_filename)
            
            # JSON 데이터 생성
            json_data = self.convert_to_json(data, filename)
            
            # 파일 저장
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            
            return json_path
        except Exception as e:
            # st.error(f"JSON 파일 저장 오류: {str(e)}")  # Streamlit 의존성 제거
            print(f"JSON 파일 저장 오류: {str(e)}")  # 일반 print로 대체
            return None

def process_pdf_files(uploaded_files):
    """여러 PDF 파일 처리"""
    parser = PDFParser()
    all_data = []
    
    for file in uploaded_files:
        # st.info(f"파일 처리 중: {file.name}")  # Streamlit 의존성 제거
        print(f"파일 처리 중: {file.name}")
        
        # 텍스트 추출
        text = parser.extract_text_from_pdf(file)
        if text:
            # 기본 파싱
            data = parser.parse_financial_data(text)
            
            # 거래 내역 추출
            transactions = parser.extract_transactions(text)
            data['transactions'] = transactions
            
            # LLM 분석 (선택적)
            # llm_data = parser.analyze_with_llm(text)
            # data.update(llm_data)
            
            all_data.append(data)
    
    # 데이터 통합
    if all_data:
        combined_data = combine_financial_data(all_data)
        return combined_data
    
    return None

def process_pdf_to_json(uploaded_files, save_files=True):
    """PDF 파일을 JSON으로 변환하고 저장"""
    parser = PDFParser()
    all_data = []
    json_files = []
    
    for file in uploaded_files:
        # st.info(f"PDF를 JSON으로 변환 중: {file.name}")  # Streamlit 의존성 제거
        print(f"PDF를 JSON으로 변환 중: {file.name}")
        
        # 텍스트 추출
        text = parser.extract_text_from_pdf(file)
        if text:
            # 기본 파싱
            data = parser.parse_financial_data(text)
            
            # 거래 내역 추출
            transactions = parser.extract_transactions(text)
            data['transactions'] = transactions
            
            all_data.append(data)
            
            # JSON 파일로 저장
            if save_files:
                json_path = parser.save_json_file(data, file.name)
                if json_path:
                    json_files.append(json_path)
                    # st.success(f"JSON 파일 저장 완료: {json_path}")  # Streamlit 의존성 제거
                    print(f"JSON 파일 저장 완료: {json_path}")
    
    # 통합 데이터도 JSON으로 저장
    if all_data:
        combined_data = combine_financial_data(all_data)
        if save_files:
            combined_json_path = parser.save_json_file(combined_data, "combined_financial_data")
            if combined_json_path:
                json_files.append(combined_json_path)
                # st.success(f"통합 JSON 파일 저장 완료: {combined_json_path}")  # Streamlit 의존성 제거
                print(f"통합 JSON 파일 저장 완료: {combined_json_path}")
        
        return combined_data, json_files
    
    return None, []

def combine_financial_data(data_list):
    """여러 개월 데이터 통합"""
    if not data_list:
        return None
    
    combined = {
        'income': sum(d.get('income', 0) for d in data_list) / len(data_list),
        'expense': sum(d.get('expense', 0) for d in data_list) / len(data_list),
        'savings': sum(d.get('savings', 0) for d in data_list) / len(data_list),
        'credit_score': max(d.get('credit_score', 0) for d in data_list),
        'assets': {
            'checking': max(d.get('assets', {}).get('checking', 0) for d in data_list),
            'savings': max(d.get('assets', {}).get('savings', 0) for d in data_list),
            'investment': max(d.get('assets', {}).get('investment', 0) for d in data_list),
            'pension': max(d.get('assets', {}).get('pension', 0) for d in data_list),
            'isa': max(d.get('assets', {}).get('isa', 0) for d in data_list),
            'government': max(d.get('assets', {}).get('government', 0) for d in data_list)
        },
        'transactions': []
    }
    
    # 모든 거래 내역 통합
    for data in data_list:
        combined['transactions'].extend(data.get('transactions', []))
    
    return combined

def preview_json_data(data):
    """JSON 데이터 미리보기"""
    if not data:
        return "데이터가 없습니다."
    
    try:
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        return json_str
    except Exception as e:
        return f"JSON 변환 오류: {str(e)}"
