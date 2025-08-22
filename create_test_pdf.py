#!/usr/bin/env python3
"""
테스트용 재무 보고서 PDF 생성 스크립트
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import os

def create_test_financial_report():
    """테스트용 재무 보고서 PDF 생성"""
    
    # PDF 파일명
    filename = "test_financial_report.pdf"
    
    # PDF 문서 생성
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 스타일 정의
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # 중앙 정렬
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    normal_style = styles['Normal']
    
    # 내용 구성
    story = []
    
    # 제목
    story.append(Paragraph("📊 월간 재무 보고서", title_style))
    story.append(Spacer(1, 20))
    
    # 기본 정보
    story.append(Paragraph("📋 기본 재무 정보", heading_style))
    story.append(Paragraph("• 월 수입: 3,500,000원", normal_style))
    story.append(Paragraph("• 월 지출: 2,800,000원", normal_style))
    story.append(Paragraph("• 월 저축: 700,000원", normal_style))
    story.append(Paragraph("• 신용점수: 720점", normal_style))
    story.append(Spacer(1, 15))
    
    # 자산 현황
    story.append(Paragraph("💰 자산 현황", heading_style))
    story.append(Paragraph("• 입출금 계좌: 5,000,000원", normal_style))
    story.append(Paragraph("• 적금/저축: 15,000,000원", normal_style))
    story.append(Paragraph("• 투자 계좌: 8,000,000원", normal_style))
    story.append(Paragraph("• 연금 계좌: 3,000,000원", normal_style))
    story.append(Paragraph("• ISA 계좌: 2,000,000원", normal_style))
    story.append(Paragraph("• 청년도약계좌: 5,000,000원", normal_style))
    story.append(Spacer(1, 15))
    
    # 거래 내역
    story.append(Paragraph("💳 주요 거래 내역", heading_style))
    
    # 거래 내역 테이블
    transaction_data = [
        ['날짜', '항목', '금액', '구분'],
        ['2024-01-15', '급여', '3,500,000', '수입'],
        ['2024-01-20', '식비', '-500,000', '지출'],
        ['2024-01-22', '교통비', '-150,000', '지출'],
        ['2024-01-25', '주거비', '-800,000', '지출'],
        ['2024-01-28', '통신비', '-120,000', '지출'],
        ['2024-01-30', '저축', '700,000', '저축']
    ]
    
    transaction_table = Table(transaction_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1*inch])
    transaction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(transaction_table)
    story.append(Spacer(1, 15))
    
    # 지출 분석
    story.append(Paragraph("📈 지출 분석", heading_style))
    story.append(Paragraph("• 식비: 500,000원 (17.9%)", normal_style))
    story.append(Paragraph("• 주거비: 800,000원 (28.6%)", normal_style))
    story.append(Paragraph("• 교통비: 150,000원 (5.4%)", normal_style))
    story.append(Paragraph("• 통신비: 120,000원 (4.3%)", normal_style))
    story.append(Paragraph("• 기타: 1,230,000원 (43.9%)", normal_style))
    story.append(Spacer(1, 15))
    
    # 투자 현황
    story.append(Paragraph("📊 투자 현황", heading_style))
    story.append(Paragraph("• 주식형 펀드: 5,000,000원", normal_style))
    story.append(Paragraph("• ETF: 2,000,000원", normal_style))
    story.append(Paragraph("• 개별 주식: 1,000,000원", normal_style))
    story.append(Spacer(1, 15))
    
    # 정부지원상품
    story.append(Paragraph("🏛️ 정부지원상품", heading_style))
    story.append(Paragraph("• 청년도약계좌: 5,000,000원 (연 3.5% 금리)", normal_style))
    story.append(Paragraph("• 청년희망적금: 3,000,000원 (연 2.5% 금리)", normal_style))
    story.append(Paragraph("• 청년내일저축계좌: 2,000,000원 (연 2.0% 금리)", normal_style))
    story.append(Spacer(1, 15))
    
    # 목표 및 계획
    story.append(Paragraph("🎯 재무 목표", heading_style))
    story.append(Paragraph("• 단기 목표 (3개월): 비상금 6개월치 확보", normal_style))
    story.append(Paragraph("• 중기 목표 (6개월): 월 저축률 25% 달성", normal_style))
    story.append(Paragraph("• 장기 목표 (1년): 총 자산 50,000,000원 달성", normal_style))
    story.append(Spacer(1, 15))
    
    # 권장사항
    story.append(Paragraph("💡 권장사항", heading_style))
    story.append(Paragraph("• 지출 절약: 월 200,000원 절약 가능", normal_style))
    story.append(Paragraph("• 투자 확대: 월 300,000원 추가 투자 권장", normal_style))
    story.append(Paragraph("• 신용점수 관리: 750점 이상 유지", normal_style))
    story.append(Spacer(1, 15))
    
    # 하단 정보
    story.append(Paragraph("📅 보고서 생성일: 2024년 1월 31일", normal_style))
    story.append(Paragraph("📊 다음 보고서: 2024년 2월 28일", normal_style))
    
    # PDF 생성
    doc.build(story)
    
    print(f"✅ 테스트 PDF 파일이 생성되었습니다: {filename}")
    return filename

if __name__ == "__main__":
    try:
        create_test_financial_report()
        print("🎉 테스트 PDF 생성이 완료되었습니다!")
    except Exception as e:
        print(f"❌ PDF 생성 중 오류가 발생했습니다: {e}")
