"""
샘플 데이터 생성기
테스트용 PDF 파일과 데이터를 생성합니다.
"""

import json
from datetime import datetime, timedelta

def generate_sample_financial_data():
    """샘플 재무 데이터 생성"""
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
            {
                'date': '2024-01-15',
                'category': '급여',
                'amount': 3500000,
                'type': 'income'
            },
            {
                'date': '2024-01-20',
                'category': '식비',
                'amount': 500000,
                'type': 'expense'
            },
            {
                'date': '2024-01-25',
                'category': '교통비',
                'amount': 150000,
                'type': 'expense'
            },
            {
                'date': '2024-01-30',
                'category': '주거비',
                'amount': 800000,
                'type': 'expense'
            },
            {
                'date': '2024-02-15',
                'category': '급여',
                'amount': 3500000,
                'type': 'income'
            },
            {
                'date': '2024-02-20',
                'category': '식비',
                'amount': 480000,
                'type': 'expense'
            },
            {
                'date': '2024-02-25',
                'category': '교통비',
                'amount': 140000,
                'type': 'expense'
            },
            {
                'date': '2024-02-28',
                'category': '주거비',
                'amount': 800000,
                'type': 'expense'
            },
            {
                'date': '2024-03-15',
                'category': '급여',
                'amount': 3500000,
                'type': 'income'
            },
            {
                'date': '2024-03-20',
                'category': '식비',
                'amount': 520000,
                'type': 'expense'
            },
            {
                'date': '2024-03-25',
                'category': '교통비',
                'amount': 160000,
                'type': 'expense'
            },
            {
                'date': '2024-03-30',
                'category': '주거비',
                'amount': 800000,
                'type': 'expense'
            }
        ]
    }

def generate_sample_pdf_text():
    """샘플 PDF 텍스트 생성"""
    return """
    월별 재무 현황 보고서
    
    기본 정보:
    - 보고 기간: 2024년 1월 ~ 3월
    - 보고자: 홍길동
    
    수입 현황:
    - 월 급여: 3,500,000원
    - 총 수입: 3,500,000원
    
    지출 현황:
    - 월 지출: 2,800,000원
    - 식비: 500,000원
    - 교통비: 150,000원
    - 주거비: 800,000원
    - 통신비: 100,000원
    - 기타: 1,250,000원
    
    저축 현황:
    - 월 저축: 700,000원
    - 저축률: 20%
    
    신용점수:
    - KCB 신용점수: 720점
    - NICE 신용점수: 720점
    
    자산 현황:
    - 입출금 계좌: 5,000,000원
    - 적금 계좌: 15,000,000원
    - 투자 계좌: 8,000,000원
    - 연금 계좌: 3,000,000원
    - ISA 계좌: 2,000,000원
    - 청년도약계좌: 5,000,000원
    
    거래 내역:
    2024-01-15 급여 3,500,000원
    2024-01-20 식비 -500,000원
    2024-01-25 교통비 -150,000원
    2024-01-30 주거비 -800,000원
    2024-02-15 급여 3,500,000원
    2024-02-20 식비 -480,000원
    2024-02-25 교통비 -140,000원
    2024-02-28 주거비 -800,000원
    2024-03-15 급여 3,500,000원
    2024-03-20 식비 -520,000원
    2024-03-25 교통비 -160,000원
    2024-03-30 주거비 -800,000원
    """

def save_sample_data():
    """샘플 데이터를 JSON 파일로 저장"""
    data = generate_sample_financial_data()
    
    with open('sample_financial_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("샘플 데이터가 sample_financial_data.json 파일로 저장되었습니다.")

def create_sample_pdf_text_file():
    """샘플 PDF 텍스트를 파일로 저장"""
    text = generate_sample_pdf_text()
    
    with open('sample_pdf_content.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    print("샘플 PDF 텍스트가 sample_pdf_content.txt 파일로 저장되었습니다.")

if __name__ == "__main__":
    save_sample_data()
    create_sample_pdf_text_file()
