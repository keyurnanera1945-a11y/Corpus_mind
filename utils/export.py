import io
import os
import sys
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ExportManager:
    @staticmethod
    def export_to_txt(messages: List[Dict[str, Any]], session_title: str) -> str:
        """Export chat history as plain text."""
        output = f"CHAT HISTORY: {session_title}\n"
        output += "=" * 50 + "\n\n"
        for msg in messages:
            role = "USER" if msg['role'] == 'user' else "ASSISTANT"
            timestamp = msg.get('timestamp', '')
            output += f"[{timestamp}] {role}:\n{msg['content']}\n"
            output += "-" * 50 + "\n\n"
        return output

    @staticmethod
    def export_to_markdown(messages: List[Dict[str, Any]], session_title: str) -> str:
        """Export chat history as Markdown."""
        output = f"# Chat History: {session_title}\n\n"
        for msg in messages:
            role = "### 👤 User" if msg['role'] == 'user' else "### 🤖 Assistant"
            timestamp = msg.get('timestamp', '')
            model_info = f" *({msg.get('model', 'LLM')})*" if msg['role'] != 'user' and msg.get('model') else ""
            output += f"{role}{model_info}\n"
            if timestamp:
                output += f"*{timestamp}*\n\n"
            else:
                output += "\n"
            output += f"{msg['content']}\n\n"
            output += "---\n\n"
        return output

    @staticmethod
    def export_to_pdf(messages: List[Dict[str, Any]], session_title: str) -> bytes:
        """Export chat history as a formatted PDF using reportlab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=54, leftMargin=54,
                topMargin=54, bottomMargin=54
            )
            
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontSize=20,
                leading=24,
                textColor=HexColor('#1E3A8A'),
                spaceAfter=15
            )
            
            meta_style = ParagraphStyle(
                'DocMeta',
                parent=styles['Normal'],
                fontSize=10,
                textColor=HexColor('#6B7280'),
                spaceAfter=25
            )
            
            user_style = ParagraphStyle(
                'UserMsg',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                textColor=HexColor('#111827'),
                spaceAfter=10
            )
            
            user_header = ParagraphStyle(
                'UserHeader',
                parent=styles['Heading3'],
                fontSize=12,
                leading=15,
                textColor=HexColor('#2563EB'),
                spaceAfter=5
            )
            
            assistant_style = ParagraphStyle(
                'AssistantMsg',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                textColor=HexColor('#1F2937'),
                spaceAfter=10
            )
            
            assistant_header = ParagraphStyle(
                'AssistantHeader',
                parent=styles['Heading3'],
                fontSize=12,
                leading=15,
                textColor=HexColor('#059669'),
                spaceAfter=5
            )
            
            story = []
            
            # Title
            story.append(Paragraph(f"Smart AI Assistant Chat Log", title_style))
            story.append(Paragraph(f"Session: {session_title}", meta_style))
            story.append(Spacer(1, 10))
            
            for msg in messages:
                # Header
                if msg['role'] == 'user':
                    story.append(Paragraph("👤 User", user_header))
                    # Content
                    text = msg['content'].replace("\n", "<br/>")
                    story.append(Paragraph(text, user_style))
                else:
                    model_str = f" ({msg['model']})" if msg.get('model') else ""
                    story.append(Paragraph(f"🤖 Assistant{model_str}", assistant_header))
                    # Content
                    text = msg['content'].replace("\n", "<br/>")
                    story.append(Paragraph(text, assistant_style))
                    
                story.append(Spacer(1, 15))
                
            doc.build(story)
            pdf_bytes = buffer.getvalue()
            buffer.close()
            return pdf_bytes
        except Exception as e:
            # Fallback to simple bytes if reportlab errors out
            fallback_text = ExportManager.export_to_txt(messages, session_title)
            return fallback_text.encode('utf-8')
