# backend/app/services/report_generator.py
"""
Report Generator Service for creating comprehensive missing fields reports.

This service generates detailed markdown reports of missing fields, low confidence
mappings, and processing statistics for PA form completion workflows with
specialized formatting for medical form documentation.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from app.models.schemas import (
    FieldMapping, ConfidenceLevel, FieldType, 
    ProcessingResult, MissingField
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ReportSection(str, Enum):
    """Sections available in generated reports."""
    SUMMARY = "summary"
    HIGH_PRIORITY_MISSING = "high_priority_missing"
    MEDIUM_PRIORITY_MISSING = "medium_priority_missing"
    LOW_CONFIDENCE_FIELDS = "low_confidence_fields"
    PROCESSING_STATISTICS = "processing_statistics"
    SOURCE_REFERENCES = "source_references"
    RECOMMENDATIONS = "recommendations"


class ReportFormat(str, Enum):
    """Available report formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"


class ReportGenerator:
    """
    Advanced report generator for PA form processing workflows.
    
    Creates comprehensive missing fields reports with confidence analysis,
    source tracking, and actionable recommendations for manual review
    with specialized formatting for medical documentation.
    """
    
    def __init__(self):
        """Initialize the report generator with medical field patterns."""
        # Field priority mappings for medical forms
        self.field_priorities = {
            "critical": [
                "patient_name", "patient.*name", "first.*name", "last.*name",
                "date.*birth", "dob", "member.*id", "insurance.*id",
                "prescriber.*name", "provider.*name", "diagnosis", "medication"
            ],
            "high": [
                "phone", "address", "npi", "provider.*npi", "license.*number",
                "insurance.*group", "policy.*number", "treatment.*plan"
            ],
            "medium": [
                "fax", "email", "emergency.*contact", "pharmacy",
                "administration.*date", "dosage", "frequency"
            ],
            "low": [
                "comments", "notes", "additional.*info", "optional.*fields"
            ]
        }
        
        # Confidence threshold mappings
        self.confidence_thresholds = {
            "high": 0.85,
            "medium": 0.70,
            "low": 0.50,
            "very_low": 0.30
        }
        
        # Medical terminology for explanations
        self.medical_explanations = {
            "patient_name": "Required for patient identification and insurance claims",
            "date_of_birth": "Critical for patient verification and age-based treatment protocols",
            "member_id": "Essential for insurance authorization and claim processing",
            "npi": "Required for provider identification and billing compliance",
            "diagnosis": "Mandatory for medical necessity determination",
            "medication": "Core requirement for treatment authorization"
        }
        
        logger.info("Report generator initialized with medical field priorities")
    
    def generate_missing_fields_report(
        self,
        processing_result: ProcessingResult,
        field_mappings: Dict[str, FieldMapping],
        missing_fields: List[MissingField],
        output_path: Optional[Union[str, Path]] = None,
        report_format: ReportFormat = ReportFormat.MARKDOWN
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive missing fields report.
        
        Args:
            processing_result: Complete processing results
            field_mappings: Dictionary of successful field mappings
            missing_fields: List of missing or problematic fields
            output_path: Optional path to save the report
            report_format: Format for the generated report
            
        Returns:
            Dictionary containing report content and metadata
        """
        try:
            logger.info(f"Generating missing fields report for session {processing_result.session_id}")
            
            # Analyze field mappings and missing fields
            analysis = self._analyze_fields(field_mappings, missing_fields)
            
            # Generate report sections
            report_sections = {}
            
            if report_format == ReportFormat.MARKDOWN:
                report_sections = {
                    ReportSection.SUMMARY: self._generate_summary_section(processing_result, analysis),
                    ReportSection.HIGH_PRIORITY_MISSING: self._generate_high_priority_section(analysis["high_priority"]),
                    ReportSection.MEDIUM_PRIORITY_MISSING: self._generate_medium_priority_section(analysis["medium_priority"]),
                    ReportSection.LOW_CONFIDENCE_FIELDS: self._generate_low_confidence_section(analysis["low_confidence"]),
                    ReportSection.PROCESSING_STATISTICS: self._generate_statistics_section(processing_result, analysis),
                    ReportSection.SOURCE_REFERENCES: self._generate_source_references_section(field_mappings),
                    ReportSection.RECOMMENDATIONS: self._generate_recommendations_section(analysis)
                }
                
                # Combine sections into full report
                full_report = self.format_as_markdown(report_sections, processing_result)
                
            else:
                # Handle other formats (future implementation)
                full_report = self._generate_plain_text_report(processing_result, analysis)
            
            # Include confidence details
            confidence_analysis = self.include_confidence_details(field_mappings, missing_fields)
            
            # Save report if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(full_report, encoding='utf-8')
                logger.info(f"Report saved to {output_path}")
            
            return {
                "session_id": processing_result.session_id,
                "report_content": full_report,
                "report_format": report_format.value,
                "sections": report_sections,
                "confidence_analysis": confidence_analysis,
                "analysis_summary": analysis,
                "output_path": str(output_path) if output_path else None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "word_count": len(full_report.split()),
                "total_missing_fields": len(missing_fields),
                "critical_missing_count": len(analysis["high_priority"]),
                "low_confidence_count": len(analysis["low_confidence"])
            }
            
        except Exception as e:
            logger.error(f"Failed to generate missing fields report: {e}")
            raise
    
    def format_as_markdown(
        self,
        sections: Dict[ReportSection, str],
        processing_result: ProcessingResult
    ) -> str:
        """
        Format report sections as comprehensive markdown document.
        
        Args:
            sections: Dictionary of report sections
            processing_result: Processing result data
            
        Returns:
            Complete markdown formatted report
        """
        try:
            # Build markdown document
            markdown_content = []
            
            # Header
            markdown_content.append(f"# Prior Authorization Missing Fields Report")
            markdown_content.append(f"")
            markdown_content.append(f"**Session ID:** `{processing_result.session_id}`")
            markdown_content.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%B %d, %Y at %I:%M %p UTC')}")
            markdown_content.append(f"**Processing Status:** {processing_result.status}")
            markdown_content.append(f"")
            markdown_content.append("---")
            markdown_content.append("")
            
            # Add each section in order
            section_order = [
                ReportSection.SUMMARY,
                ReportSection.HIGH_PRIORITY_MISSING,
                ReportSection.MEDIUM_PRIORITY_MISSING,
                ReportSection.LOW_CONFIDENCE_FIELDS,
                ReportSection.SOURCE_REFERENCES,
                ReportSection.PROCESSING_STATISTICS,
                ReportSection.RECOMMENDATIONS
            ]
            
            for section in section_order:
                if section in sections and sections[section]:
                    markdown_content.append(sections[section])
                    markdown_content.append("")
            
            # Footer
            markdown_content.append("---")
            markdown_content.append("")
            markdown_content.append("*This report was generated automatically by the MediLink-AI Prior Authorization system.*")
            markdown_content.append(f"*For questions about this report, please reference session ID: `{processing_result.session_id}`*")
            
            return "\n".join(markdown_content)
            
        except Exception as e:
            logger.error(f"Failed to format markdown report: {e}")
            return f"# Report Generation Error\n\nFailed to format report: {str(e)}"
    
    def include_confidence_details(
        self,
        field_mappings: Dict[str, FieldMapping],
        missing_fields: List[MissingField]
    ) -> Dict[str, Any]:
        """
        Include detailed confidence analysis in the report.
        
        Args:
            field_mappings: Successfully mapped fields
            missing_fields: Fields that could not be mapped
            
        Returns:
            Detailed confidence analysis
        """
        try:
            confidence_analysis = {
                "field_confidence_distribution": {},
                "confidence_statistics": {},
                "low_confidence_details": [],
                "confidence_improvement_suggestions": []
            }
            
            # Analyze confidence distribution
            confidence_counts = {"high": 0, "medium": 0, "low": 0, "very_low": 0}
            confidence_details = []
            
            for field_name, mapping in field_mappings.items():
                confidence = mapping.confidence_score.overall_confidence
                
                if confidence >= self.confidence_thresholds["high"]:
                    confidence_level = "high"
                elif confidence >= self.confidence_thresholds["medium"]:
                    confidence_level = "medium"
                elif confidence >= self.confidence_thresholds["low"]:
                    confidence_level = "low"
                else:
                    confidence_level = "very_low"
                
                confidence_counts[confidence_level] += 1
                
                # Track low confidence fields for detailed analysis
                if confidence < self.confidence_thresholds["medium"]:
                    confidence_details.append({
                        "field_name": field_name,
                        "confidence": confidence,
                        "confidence_level": confidence_level,
                        "mapped_value": mapping.mapped_value,
                        "original_value": mapping.original_value,
                        "requires_review": mapping.requires_review,
                        "improvement_suggestion": self._get_confidence_improvement_suggestion(mapping)
                    })
            
            confidence_analysis["field_confidence_distribution"] = confidence_counts
            confidence_analysis["low_confidence_details"] = confidence_details
            
            # Calculate statistics
            total_fields = len(field_mappings)
            if total_fields > 0:
                confidence_analysis["confidence_statistics"] = {
                    "total_mapped_fields": total_fields,
                    "high_confidence_percentage": round((confidence_counts["high"] / total_fields) * 100, 1),
                    "medium_confidence_percentage": round((confidence_counts["medium"] / total_fields) * 100, 1),
                    "low_confidence_percentage": round((confidence_counts["low"] / total_fields) * 100, 1),
                    "very_low_confidence_percentage": round((confidence_counts["very_low"] / total_fields) * 100, 1),
                    "average_confidence": round(
                        sum(m.confidence_score.overall_confidence for m in field_mappings.values()) / total_fields, 3
                    ),
                    "fields_requiring_review": sum(1 for m in field_mappings.values() if m.requires_review)
                }
            
            return confidence_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze confidence details: {e}")
            return {"error": str(e)}
    
    def _analyze_fields(
        self,
        field_mappings: Dict[str, FieldMapping],
        missing_fields: List[MissingField]
    ) -> Dict[str, Any]:
        """Analyze field mappings and missing fields for report generation."""
        analysis = {
            "high_priority": [],
            "medium_priority": [],
            "low_confidence": [],
            "mapped_successfully": [],
            "total_fields_expected": len(field_mappings) + len(missing_fields),
            "mapping_success_rate": 0.0
        }
        
        # Categorize missing fields by priority
        for missing_field in missing_fields:
            priority = self._determine_field_priority(missing_field.field_name)
            missing_field_data = {
                "field_name": missing_field.field_name,
                "reason": missing_field.reason,
                "priority": priority,
                "explanation": self.medical_explanations.get(
                    missing_field.field_name.lower(), 
                    "Standard field required for complete form submission"
                )
            }
            
            if priority in ["critical", "high"]:
                analysis["high_priority"].append(missing_field_data)
            else:
                analysis["medium_priority"].append(missing_field_data)
        
        # Categorize low confidence mappings
        for field_name, mapping in field_mappings.items():
            if mapping.confidence_score.overall_confidence < self.confidence_thresholds["medium"]:
                analysis["low_confidence"].append({
                    "field_name": field_name,
                    "confidence": mapping.confidence_score.overall_confidence,
                    "mapped_value": mapping.mapped_value,
                    "original_value": mapping.original_value,
                    "requires_review": mapping.requires_review
                })
            else:
                analysis["mapped_successfully"].append({
                    "field_name": field_name,
                    "confidence": mapping.confidence_score.overall_confidence,
                    "mapped_value": mapping.mapped_value
                })
        
        # Calculate success rate
        total_expected = analysis["total_fields_expected"]
        if total_expected > 0:
            successful_mappings = len(analysis["mapped_successfully"])
            analysis["mapping_success_rate"] = round((successful_mappings / total_expected) * 100, 1)
        
        return analysis
    
    def _determine_field_priority(self, field_name: str) -> str:
        """Determine the priority level of a field based on name patterns."""
        field_lower = field_name.lower()
        
        for priority, patterns in self.field_priorities.items():
            for pattern in patterns:
                if pattern.replace(".*", "") in field_lower:
                    return priority
        
        return "low"
    
    def _generate_summary_section(self, processing_result: ProcessingResult, analysis: Dict[str, Any]) -> str:
        """Generate the executive summary section."""
        success_rate = analysis["mapping_success_rate"]
        total_missing = len(analysis["high_priority"]) + len(analysis["medium_priority"])
        low_confidence_count = len(analysis["low_confidence"])
        
        # Determine overall status
        if success_rate >= 90 and total_missing == 0:
            status_emoji = "✅"
            status_text = "Excellent - Form is nearly complete"
        elif success_rate >= 75 and len(analysis["high_priority"]) == 0:
            status_emoji = "⚠️"
            status_text = "Good - Minor fields need attention"
        elif success_rate >= 50:
            status_emoji = "🔍"
            status_text = "Moderate - Manual review recommended"
        else:
            status_emoji = "❌"
            status_text = "Attention Required - Significant gaps identified"
        
        summary = [
            "## 📋 Executive Summary",
            "",
            f"{status_emoji} **Overall Status:** {status_text}",
            f"🎯 **Field Mapping Success Rate:** {success_rate}%",
            f"📊 **Fields Successfully Mapped:** {len(analysis['mapped_successfully'])}",
            f"⚠️ **High Priority Missing Fields:** {len(analysis['high_priority'])}",
            f"📝 **Medium Priority Missing Fields:** {len(analysis['medium_priority'])}",
            f"🔍 **Low Confidence Mappings:** {low_confidence_count}",
            ""
        ]
        
        # Add recommendations based on status
        if len(analysis["high_priority"]) > 0:
            summary.append("🚨 **Action Required:** Critical fields are missing and must be completed manually.")
        elif low_confidence_count > 0:
            summary.append("👀 **Review Recommended:** Some fields have low confidence scores and should be verified.")
        else:
            summary.append("✨ **Status:** Form processing completed successfully with high confidence.")
        
        return "\n".join(summary)
    
    def _generate_high_priority_section(self, high_priority_fields: List[Dict[str, Any]]) -> str:
        """Generate the high priority missing fields section."""
        if not high_priority_fields:
            return "## 🚨 High Priority Missing Fields\n\n✅ **All critical fields have been successfully mapped.**"
        
        section = [
            "## 🚨 High Priority Missing Fields",
            "",
            "These fields are **critical** for PA form submission and must be completed manually:",
            ""
        ]
        
        for i, field in enumerate(high_priority_fields, 1):
            section.extend([
                f"### {i}. `{field['field_name']}`",
                f"- **Reason:** {field['reason']}",
                f"- **Priority:** {field['priority'].title()}",
                f"- **Clinical Importance:** {field['explanation']}",
                f"- **Action Required:** Manual entry required",
                ""
            ])
        
        return "\n".join(section)
    
    def _generate_medium_priority_section(self, medium_priority_fields: List[Dict[str, Any]]) -> str:
        """Generate the medium priority missing fields section."""
        if not medium_priority_fields:
            return "## 📝 Medium Priority Missing Fields\n\n✅ **All standard fields have been successfully mapped.**"
        
        section = [
            "## 📝 Medium Priority Missing Fields",
            "",
            "These fields should be completed for optimal form submission:",
            ""
        ]
        
        for i, field in enumerate(medium_priority_fields, 1):
            section.extend([
                f"### {i}. `{field['field_name']}`",
                f"- **Reason:** {field['reason']}",
                f"- **Priority:** {field['priority'].title()}",
                f"- **Recommendation:** {field['explanation']}",
                ""
            ])
        
        return "\n".join(section)
    
    def _generate_low_confidence_section(self, low_confidence_fields: List[Dict[str, Any]]) -> str:
        """Generate the low confidence mappings section."""
        if not low_confidence_fields:
            return "## 🔍 Low Confidence Field Mappings\n\n✅ **All mapped fields have high confidence scores.**"
        
        section = [
            "## 🔍 Low Confidence Field Mappings",
            "",
            "These fields were mapped but have low confidence scores and should be reviewed:",
            ""
        ]
        
        for i, field in enumerate(low_confidence_fields, 1):
            confidence_percent = round(field['confidence'] * 100, 1)
            review_indicator = "🔴 **REVIEW REQUIRED**" if field['requires_review'] else "🟡 Review Recommended"
            
            section.extend([
                f"### {i}. `{field['field_name']}`",
                f"- **Confidence Score:** {confidence_percent}% {review_indicator}",
                f"- **Mapped Value:** `{field['mapped_value']}`",
                f"- **Original Value:** `{field['original_value']}`",
                f"- **Recommendation:** Verify accuracy before submission",
                ""
            ])
        
        return "\n".join(section)
    
    def _generate_statistics_section(self, processing_result: ProcessingResult, analysis: Dict[str, Any]) -> str:
        """Generate the processing statistics section."""
        section = [
            "## 📊 Processing Statistics",
            "",
            f"- **Session ID:** `{processing_result.session_id}`",
            f"- **Processing Status:** {processing_result.status}",
            f"- **Total Processing Time:** {getattr(processing_result, 'processing_time', 'N/A')}",
            f"- **Expected Fields:** {analysis['total_fields_expected']}",
            f"- **Successfully Mapped:** {len(analysis['mapped_successfully'])}",
            f"- **Missing Fields:** {len(analysis['high_priority']) + len(analysis['medium_priority'])}",
            f"- **Low Confidence Mappings:** {len(analysis['low_confidence'])}",
            f"- **Overall Success Rate:** {analysis['mapping_success_rate']}%",
            ""
        ]
        
        return "\n".join(section)
    
    def _generate_source_references_section(self, field_mappings: Dict[str, FieldMapping]) -> str:
        """Generate the source references section."""
        section = [
            "## 📄 Source Page References",
            "",
            "Fields were extracted from the following source pages:",
            ""
        ]
        
        # Group fields by source page
        page_references = {}
        for field_name, mapping in field_mappings.items():
            source_page = getattr(mapping, 'source_page', 'Unknown')
            if source_page not in page_references:
                page_references[source_page] = []
            page_references[source_page].append(field_name)
        
        for page, fields in sorted(page_references.items()):
            if page != 'Unknown':
                section.append(f"**Page {page}:**")
                for field in fields:
                    section.append(f"  - `{field}`")
                section.append("")
        
        if not page_references or all(page == 'Unknown' for page in page_references.keys()):
            section.append("*Source page information not available*")
        
        return "\n".join(section)
    
    def _generate_recommendations_section(self, analysis: Dict[str, Any]) -> str:
        """Generate the recommendations section."""
        section = [
            "## 💡 Recommendations",
            ""
        ]
        
        high_priority_count = len(analysis["high_priority"])
        low_confidence_count = len(analysis["low_confidence"])
        success_rate = analysis["mapping_success_rate"]
        
        if high_priority_count > 0:
            section.extend([
                "### Immediate Actions Required",
                f"1. **Complete {high_priority_count} critical missing fields** before form submission",
                "2. **Review source documents** for missing information",
                "3. **Contact referring provider** if critical data is unavailable",
                ""
            ])
        
        if low_confidence_count > 0:
            section.extend([
                "### Quality Assurance",
                f"1. **Verify {low_confidence_count} low confidence mappings** for accuracy",
                "2. **Cross-reference with source documents** to confirm values",
                "3. **Update fields** if discrepancies are found",
                ""
            ])
        
        if success_rate >= 80:
            section.extend([
                "### Form Completion",
                "1. **Review missing fields** and complete manually",
                "2. **Submit form** once all required fields are complete",
                "3. **Save completed form** for records",
                ""
            ])
        else:
            section.extend([
                "### Document Review",
                "1. **Review source document quality** - consider rescanning if needed",
                "2. **Check for additional pages** that may contain missing information",
                "3. **Contact document source** for clarification if needed",
                ""
            ])
        
        return "\n".join(section)
    
    def _generate_plain_text_report(self, processing_result: ProcessingResult, analysis: Dict[str, Any]) -> str:
        """Generate a plain text version of the report."""
        return f"""
PRIOR AUTHORIZATION MISSING FIELDS REPORT
Session: {processing_result.session_id}
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

SUMMARY:
- Success Rate: {analysis['mapping_success_rate']}%
- High Priority Missing: {len(analysis['high_priority'])}
- Medium Priority Missing: {len(analysis['medium_priority'])}
- Low Confidence Mappings: {len(analysis['low_confidence'])}

For detailed information, please use the markdown format.
        """.strip()
    
    def _get_confidence_improvement_suggestion(self, mapping: FieldMapping) -> str:
        """Get suggestion for improving confidence in field mapping."""
        confidence = mapping.confidence_score.overall_confidence
        
        if confidence < 0.3:
            return "Consider manual verification - very low confidence"
        elif confidence < 0.5:
            return "Review source document quality and OCR results"
        elif confidence < 0.7:
            return "Cross-reference with additional source pages"
        else:
            return "Minor verification recommended"


# Global report generator instance
report_generator = ReportGenerator()


def get_report_generator() -> ReportGenerator:
    """
    Get the global report generator instance.
    
    Returns:
        ReportGenerator instance for dependency injection
    """
    return report_generator